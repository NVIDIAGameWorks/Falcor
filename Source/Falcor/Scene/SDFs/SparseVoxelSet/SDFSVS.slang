/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "Utils/Math/MathConstants.slangh"

import Scene.SDFs.SDFGridBase;
import Scene.SDFs.SDFVoxelCommon;
import Scene.SDFs.SDFVoxelTypes;
import Scene.SDFs.SDFVoxelHitUtils;
import Utils.Math.AABB;
import Utils.Math.FormatConversion;
import Utils.Geometry.IntersectionHelpers;

struct SDFSVS : SDFGridBase, ISDFGrid
{
    StructuredBuffer<AABB> aabbs;
    StructuredBuffer<SDFSVSVoxel> voxels;

    uint virtualGridLevel;
    uint virtualGridWidth;
    float normalizationFactor;

    /** Unpacks 4 distance values from a 4x4 x-slice.
    */
    void unpackSliceValues(const uint4 packedValues0xx, const uint4 packedValues1xx, const int2 yzMin, out float4 values0xx, out float4 values1xx)
    {
        int shift = 8 * yzMin.y;
        values0xx[0] = unpackSnorm8((packedValues0xx[yzMin.x] >> shift) & 0x000000ff);
        values0xx[1] = unpackSnorm8((packedValues0xx[yzMin.x] >> (shift + 8)) & 0x000000ff);
        values0xx[2] = unpackSnorm8((packedValues0xx[yzMin.x + 1] >> shift) & 0x000000ff);
        values0xx[3] = unpackSnorm8((packedValues0xx[yzMin.x + 1] >> (shift + 8)) & 0x000000ff);

        values1xx[0] = unpackSnorm8((packedValues1xx[yzMin.x] >> shift) & 0x000000ff);
        values1xx[1] = unpackSnorm8((packedValues1xx[yzMin.x] >> (shift + 8)) & 0x000000ff);
        values1xx[2] = unpackSnorm8((packedValues1xx[yzMin.x + 1] >> shift) & 0x000000ff);
        values1xx[3] = unpackSnorm8((packedValues1xx[yzMin.x + 1] >> (shift + 8)) & 0x000000ff);
    }

    /** Unpacks the eight center distance values from 2 4x4 slices.
    */
    void unpackCenterValues(const uint4 packedValues1xx, const uint4 packedValues2xx, out float4 values1xx, out float4 values2xx)
    {
        values1xx[0] = unpackSnorm8((packedValues1xx.y >> 8) & 0x000000ff);
        values1xx[1] = unpackSnorm8((packedValues1xx.y >> 16) & 0x000000ff);
        values1xx[2] = unpackSnorm8((packedValues1xx.z >> 8) & 0x000000ff);
        values1xx[3] = unpackSnorm8((packedValues1xx.z >> 16) & 0x000000ff);

        values2xx[0] = unpackSnorm8((packedValues2xx.y >> 8) & 0x000000ff);
        values2xx[1] = unpackSnorm8((packedValues2xx.y >> 16) & 0x000000ff);
        values2xx[2] = unpackSnorm8((packedValues2xx.z >> 8) & 0x000000ff);
        values2xx[3] = unpackSnorm8((packedValues2xx.z >> 16) & 0x000000ff);
    }

    /** Intersects a ray with an AABB and also checks if the origin of the ray is contained inside the AABB.
    */
    bool intersectRayAABBWithInsideCheck(const float3 rayOrigin, const float3 rayDir, const float3 aabbMin, const float3 aabbMax, out float2 nearFar, out bool origInsideAABB)
    {
        const float3 invDir = 1.f / rayDir;
        const float3 lo = (aabbMin - rayOrigin) * invDir;
        const float3 hi = (aabbMax - rayOrigin) * invDir;
        const float3 tmin = min(lo, hi), tmax = max(lo, hi);
        nearFar.x = max(0.f, max(tmin.x, max(tmin.y, tmin.z)));
        nearFar.y = min(tmax.x, min(tmax.y, tmax.z));
        origInsideAABB = !(any(hi < 0.0f) || any(lo > 0.0f));
        return nearFar.x <= nearFar.y;
    }

    /** Calculates the voxel unit coords of grid local position p using the bounds of a voxel.
    */
    float3 calculteVoxelUnitCoords(const float3 p, const float3 aabbMin, const float3 aabbMax)
    {
        float3 sides = aabbMax - aabbMin;
        float3 delta = p - aabbMin;
        return delta / sides;
    }

    /** Calculates the voxel unit coords of grid local position p using the voxel coordinates.
    */
    float3 calculateVoxelUnitCoords(const float3 p, const uint3 voxelCoords)
    {
        return p * float(virtualGridWidth) - float3(voxelCoords);
    }

    /** Calculates the voxel coordinates of a point p in SDF grid local space.
    */
    uint3 calculateVoxelCoords(const float3 p)
    {
        return uint3(clamp(p, 0.0f, 1.0f) * float(virtualGridWidth));
    }

    bool intersectSDF(const float3 rayOrigin, const float3 rayDir, const float tMin, const float tMax, const uint primitiveID, out float t, out SDFGridHitData hitData)
    {
        // Normalize ray direction.
        float dirLength = length(rayDir);
        float inverseDirLength = 1.0f / dirLength;
        float3 rayDirLocal = rayDir * inverseDirLength;

        // Load the intersected voxel AABB.
        AABB aabb = aabbs[primitiveID];

        // Find near and far plane.
        float2 nearFar;
        bool origInsideAABB;
        if (!intersectRayAABBWithInsideCheck(rayOrigin, rayDirLocal, aabb.minPoint, aabb.maxPoint, nearFar, origInsideAABB))
            return false;

        // Add 0.5 to origin so that it is in [0, 1] instead of [-0.5, 0.5].
        float3 rayOrigLocal = rayOrigin + 0.5f;

        // Compute min and max local position to avoid rounding errors that result in wrong voxel and voxelUnitCoords calculations.
        float3 pMin = aabb.minPoint + 0.5f;
        float3 pMax = aabb.maxPoint + 0.5f;
        float voxelExtent = aabb.maxPoint.x - aabb.minPoint.x;

        // Set up t, tMin, tMax, and pLocal.
        float tMaxLocal = min(tMax * dirLength, nearFar.y);
        float tMinLocal = max(tMin * dirLength, nearFar.x);
        t = tMinLocal;

        // Check that the ray segment overlaps the AABB.
        if (tMaxLocal < tMinLocal) return false;

        float3 pLocal = clamp(rayOrigLocal + t * rayDirLocal, pMin, pMax);

        // Load the voxel corner values.
        SDFSVSVoxel voxel = voxels[primitiveID];
        float4 values0xx;
        float4 values1xx;
        unpackCenterValues(voxel.packedValuesSlices[1], voxel.packedValuesSlices[2], values0xx, values1xx);

        // Convert the ray origin to voxel unit coords.
        float3 voxelUnitSpaceRayOrigin = calculteVoxelUnitCoords(pLocal, pMin, pMax);

        float tLocal;
        if (SDFVoxelHitUtils::intersectSDFVoxel(voxelUnitSpaceRayOrigin, rayDirLocal, origInsideAABB, values0xx, values1xx, kSnormErrorTolerance + (tMaxLocal - t) / voxelExtent, kSolverMaxStepCount, tLocal))
        {
            t += tLocal * voxelExtent;
            t *= inverseDirLength;

            hitData.primitiveID = primitiveID;
            hitData.hitT = t;

            return true;
        }

        return false;
    }

    bool intersectSDFAny(const float3 rayOrigin, const float3 rayDir, const float tMin, const float tMax, const uint primitiveID)
    {
        // Normalize ray direction.
        float dirLength = length(rayDir);
        float inverseDirLength = 1.0f / dirLength;
        float3 rayDirLocal = rayDir * inverseDirLength;

        // Load the intersected voxel AABB.
        AABB aabb = aabbs[primitiveID];

        // Find near and far plane.
        float2 nearFar;
        bool origInsideAABB;
        if (!intersectRayAABBWithInsideCheck(rayOrigin, rayDirLocal, aabb.minPoint, aabb.maxPoint, nearFar, origInsideAABB))
            return false;

        // Add 0.5 to origin so that it is in [0, 1] instead of [-0.5, 0.5].
        float3 rayOrigLocal = rayOrigin + 0.5f;

        // Compute min and max local position to avoid rounding errors that result in wrong voxel and voxelUnitCoords calculations.
        float3 pMin = aabb.minPoint + 0.5f;
        float3 pMax = aabb.maxPoint + 0.5f;
        float voxelExtent = aabb.maxPoint.x - aabb.minPoint.x;

        // Set up t, tMin, tMax, and pLocal.
        float tMaxLocal = min(tMax * dirLength, nearFar.y);
        float tMinLocal = max(tMin * dirLength, nearFar.x);

        // Check that the ray segment overlaps the AABB.
        if (tMaxLocal < tMinLocal) return false;

        float3 pLocal = clamp(rayOrigLocal + tMinLocal * rayDirLocal, pMin, pMax);

        // Load the voxel corner values.
        SDFSVSVoxel voxel = voxels[primitiveID];
        float4 values0xx;
        float4 values1xx;
        unpackCenterValues(voxel.packedValuesSlices[1], voxel.packedValuesSlices[2], values0xx, values1xx);

        // Convert the ray origin to voxel unit coords.
        float3 voxelUnitSpaceRayOrigin = calculteVoxelUnitCoords(pLocal, pMin, pMax);

        return SDFVoxelHitUtils::intersectSDFVoxelAny(voxelUnitSpaceRayOrigin, rayDirLocal, origInsideAABB, values0xx, values1xx, kSnormErrorTolerance + (tMaxLocal - tMinLocal) / voxelExtent, kSolverMaxStepCount);
    }

    void decodeHit(const SDFGridHitData hitData, const float3 hitPosition, out float3 gradient)
    {
        gradient = calculateGradient(hitPosition, hitData.primitiveID);
    }

    float3 calculateGradient(const float3 hitPosition, const uint primitiveID)
    {
        // Add 0.5f to hitPosition so that it is in [0, 1] instead of [-0.5, 0.5].
        float3 hitPosLocal = hitPosition + 0.5f;

        // Load the intersected voxel and AABB.
        AABB aabb = aabbs[primitiveID];
        SDFSVSVoxel voxel = voxels[primitiveID];

        // Find current voxel coords.
        const uint3 voxelCoords = calculateVoxelCoords(aabb.center() + 0.5f);
        const float3 voxelUnitCoords = calculateVoxelUnitCoords(hitPosLocal, voxelCoords);

        float3 gradient;
#if SCENE_SDF_GRADIENT_EVALUATION_METHOD == SCENE_SDF_GRADIENT_NUMERIC_DISCONTINUOUS
        float4 values0xx;
        float4 values1xx;
        unpackCenterValues(voxel.packedValuesSlices[1], voxel.packedValuesSlices[2], values0xx, values1xx);

        const float offset = 0.2f / virtualGridWidth;
        gradient = SDFVoxelHitUtils::computeNumericGradient(voxelUnitCoords, offset, values0xx, values1xx);
#endif

        return gradient * normalizationFactor;
    }


};
