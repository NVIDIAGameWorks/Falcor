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
import Scene.SDFs.SDFVoxelHitUtils;
import Utils.Math.AABB;
import Utils.Math.FormatConversion;
import Utils.Math.PackedFormats;
import Utils.Geometry.IntersectionHelpers;

struct SDFSBS : SDFGridBase, ISDFGrid
{

    StructuredBuffer<AABB> aabbs;
    Texture3D<uint> indirectionBuffer;
    Texture2D<float> bricks;
    SamplerState sampler;

    uint virtualGridWidth;
    uint virtualBricksPerAxis;
    uint2 bricksPerAxis;
    uint2 brickTextureDimensions;
    uint brickWidth;
    float normalizationFactor;

    bool calculateBrickCoords(float3 pLocal, out uint3 virtualBrickCoords, out uint2 brickTexelCoords, out int3 brickLocalVoxelCoords, out float3 voxelUnitCoords)
    {
        float3 virtualGridPos = clamp(pLocal * virtualGridWidth, 0.0f, float(virtualGridWidth));
        uint3 virtualVoxelCoords = uint3(virtualGridPos);
        voxelUnitCoords = virtualGridPos - virtualVoxelCoords;
        brickLocalVoxelCoords = virtualVoxelCoords % brickWidth;

        // Calculate the virtual brick coordinates.
        virtualBrickCoords = virtualVoxelCoords / brickWidth;

        return calculateBrickTexelCoords(virtualBrickCoords, brickTexelCoords);
    }

    bool calculateBrickTexelCoords(const uint3 virtualBrickCoords, out uint2 brickTexelCoords)
    {
        // Load the brick ID from the indirection buffer.
        uint brickID = indirectionBuffer[virtualBrickCoords];
        uint brickWidthInValues = brickWidth + 1;
        brickTexelCoords = uint2(brickID % bricksPerAxis.x, brickID / bricksPerAxis.x) * uint2(brickWidthInValues * brickWidthInValues, brickWidthInValues);

        return brickID != UINT32_MAX;
    }

    void loadVoxelValues(uint2 brickTexelCoords, int3 brickLocalVoxelCoords, out float4 values0xx, out float4 values1xx)
    {
        uint brickWidthInValues = brickWidth + 1;
        uint2 voxelPos = brickTexelCoords + uint2(brickLocalVoxelCoords.x + brickLocalVoxelCoords.z * brickWidthInValues, brickLocalVoxelCoords.y);
        float2 texCoords = (voxelPos + 1.0f) / brickTextureDimensions;

        float4 valuesxx0 = bricks.Gather(sampler, texCoords);
        float4 valuesxx1 = bricks.Gather(sampler, texCoords + float2(float(brickWidthInValues) / brickTextureDimensions.x, 0.0f));
        
        values0xx = float4(valuesxx0.w, valuesxx1.w, valuesxx0.x, valuesxx1.x);
        values1xx = float4(valuesxx0.z, valuesxx1.z, valuesxx0.y, valuesxx1.y);
    }

    bool intersectSDF(const float3 rayOrigin, const float3 rayDir, const float tMin, const float tMax, const uint primitiveID, out float t, out SDFGridHitData hitData)
    {
        // Normalize ray direction.
        float dirLength = length(rayDir);
        float inverseDirLength = 1.0f / dirLength;
        float3 d = rayDir * inverseDirLength;

        // Clamp direction to epsilon to avoid division by zero.
        d.x = abs(d.x) < FLT_EPSILON ? (d.x < 0.0f ? -FLT_EPSILON : FLT_EPSILON) : d.x;
        d.y = abs(d.y) < FLT_EPSILON ? (d.y < 0.0f ? -FLT_EPSILON : FLT_EPSILON) : d.y;
        d.z = abs(d.z) < FLT_EPSILON ? (d.z < 0.0f ? -FLT_EPSILON : FLT_EPSILON) : d.z;

        // Load the intersected brick AABB.
        AABB aabb = aabbs[primitiveID];

        float2 nearFar;
        if (!intersectRayAABB(rayOrigin, d, aabb.minPoint, aabb.maxPoint, nearFar))
            return false;

        uint3 hitBrickWidthInVoxels = uint3(round(aabb.extent() * virtualGridWidth));

        // Add 0.5f to origin so that it is in [0, 1] instead of [-0.5, 0.5].
        float3 rayOrigLocal = rayOrigin + 0.5f;

        // Compute min and max local position to avoid rounding errors.
        float3 brickMin = aabb.minPoint + 0.5f + FLT_EPSILON;
        float3 brickMax = aabb.maxPoint + 0.5f - FLT_EPSILON;

        // Set up t, o, and tVoxelMax.
        t = max(tMin * dirLength, nearFar.x);
        const float tVoxelMax = (min(tMax * dirLength, nearFar.y) - t) * virtualGridWidth;

        // Check that the ray segment overlaps the AABB.
        if (tVoxelMax < 0.0f) return false;

        const float3 o = clamp(rayOrigLocal + t * d, brickMin, brickMax);

        // Find current brick coords.
        uint3 virtualBrickCoords;
        uint2 brickTexelCoords;
        int3 brickLocalVoxelCoords;
        float3 voxelUnitCoords;
        calculateBrickCoords(o, virtualBrickCoords, brickTexelCoords, brickLocalVoxelCoords, voxelUnitCoords);

        int3 stepSign;
        stepSign.x = d.x < 0.0f ? -1 : 1;
        stepSign.y = d.y < 0.0f ? -1 : 1;
        stepSign.z = d.z < 0.0f ? -1 : 1;
 
        float voxelWidth = 1.0f / virtualGridWidth;
        float normalizedVoxelDiagonal = sqrt(3.0f * voxelWidth * voxelWidth) / normalizationFactor;
        float3 invDir = 1.0f / d;
        float3 tDeltas = abs(invDir);
        float3 tVoxels = (step(float3(0.0f), d) - voxelUnitCoords) * invDir;
        int3 voxelIncrementation;
        float tLocal = 0.0f;
        float tLocalNext = min(min(tVoxels.x, tVoxels.y), tVoxels.z);

        // If the ray origin is inside the brick and we intersect with a voxel on the first iteration then we are inside that voxel.
        bool origInsideAABB = all(rayOrigLocal >= brickMin && rayOrigLocal <= brickMax);
        // DDA to surface containing voxel.
        while (all(brickLocalVoxelCoords >= 0) && all(brickLocalVoxelCoords < hitBrickWidthInVoxels) && tLocal <= tVoxelMax)
        {
            // Load the voxel corner values.
            float4 values0xx;
            float4 values1xx;
            loadVoxelValues(brickTexelCoords, brickLocalVoxelCoords, values0xx, values1xx);

            // Todo: Optimize this check, valid bit per voxel?.
            if (SDFVoxelCommon::containsSurface(values0xx, values1xx))
            {
                float3 brickLocalHitP = o + tLocal * voxelWidth * d;                // The hit position in the local space of the brick, i.e., [0, 1).
                float3 brickHitP = (brickLocalHitP - brickMin) * virtualGridWidth;  // The hit position in the grid space of the brick, i.e., [0, hitBrickWidthInVoxels).
                float3 voxelHitP = brickHitP - brickLocalVoxelCoords;               // The hit position in the local space of the voxel, i.e., [0, 1).

                float tVoxelLocal = 0.0f;
                if (SDFVoxelHitUtils::intersectSDFVoxel(voxelHitP, d, origInsideAABB, values0xx, values1xx, kSnormErrorTolerance + min(tVoxelMax, tLocalNext) - tLocal, kSolverMaxStepCount, tVoxelLocal))
                {
                    t += (tLocal + tVoxelLocal) * voxelWidth;
                    t *= inverseDirLength;

                    hitData.primitiveID = primitiveID;
                    hitData.hitT = t;
                    return true;
                }
            }

            voxelIncrementation.x = int(tVoxels.x <= tVoxels.y && tVoxels.x <= tVoxels.z);
            voxelIncrementation.y = int(tVoxels.y <= tVoxels.x && tVoxels.y <= tVoxels.z);
            voxelIncrementation.z = int(tVoxels.z <= tVoxels.x && tVoxels.z <= tVoxels.y);

            tVoxels += tDeltas * voxelIncrementation;
            brickLocalVoxelCoords += voxelIncrementation * stepSign;
            tLocal = tLocalNext;
            tLocalNext = min(min(tVoxels.x, tVoxels.y), tVoxels.z);
            origInsideAABB = false;
        }

        return false;
    }

    bool intersectSDFAny(const float3 rayOrigin, const float3 rayDir, const float tMin, const float tMax, const uint primitiveID)
    {
        // Normalize ray direction.
        float dirLength = length(rayDir);
        float inverseDirLength = 1.0f / dirLength;
        float3 d = rayDir * inverseDirLength;

        // Clamp direction to epsilon to avoid division by zero.
        d.x = abs(d.x) < FLT_EPSILON ? (d.x < 0.0f ? -FLT_EPSILON : FLT_EPSILON) : d.x;
        d.y = abs(d.y) < FLT_EPSILON ? (d.y < 0.0f ? -FLT_EPSILON : FLT_EPSILON) : d.y;
        d.z = abs(d.z) < FLT_EPSILON ? (d.z < 0.0f ? -FLT_EPSILON : FLT_EPSILON) : d.z;

        // Load the intersected brick AABB.
        AABB aabb = aabbs[primitiveID];

        float2 nearFar;
        if (!intersectRayAABB(rayOrigin, d, aabb.minPoint, aabb.maxPoint, nearFar))
            return false;

        uint3 hitBrickWidthInVoxels = uint3(round(aabb.extent() * virtualGridWidth));

        // Add 0.5f to origin so that it is in [0, 1] instead of [-0.5, 0.5].
        float3 rayOrigLocal = rayOrigin + 0.5f;

        // Compute min and max local position to avoid rounding errors.
        float3 brickMin = aabb.minPoint + 0.5f + FLT_EPSILON;
        float3 brickMax = aabb.maxPoint + 0.5f - FLT_EPSILON;

        // Set up t, o, and tVoxelMax.
        float t = max(tMin * dirLength, nearFar.x);
        const float tVoxelMax = (min(tMax * dirLength, nearFar.y) - t) * virtualGridWidth;

        // Check that the ray segment overlaps the AABB.
        if (tVoxelMax < 0.0f) return false;

        const float3 o = clamp(rayOrigLocal + t * d, brickMin, brickMax);

        // Find current brick coords.
        uint3 virtualBrickCoords;
        uint2 brickTexelCoords;
        int3 brickLocalVoxelCoords;
        float3 voxelUnitCoords;
        calculateBrickCoords(o, virtualBrickCoords, brickTexelCoords, brickLocalVoxelCoords, voxelUnitCoords);

        int3 stepSign;
        stepSign.x = d.x < 0.0f ? -1 : 1;
        stepSign.y = d.y < 0.0f ? -1 : 1;
        stepSign.z = d.z < 0.0f ? -1 : 1;
 
        float voxelWidth = 1.0f / virtualGridWidth;
        float normalizedVoxelDiagonal = sqrt(3.0f * voxelWidth * voxelWidth) / normalizationFactor;
        float3 invDir = 1.0f / d;
        float3 tDeltas = abs(invDir);
        float3 tVoxels = (step(float3(0.0f), d) - voxelUnitCoords) * invDir;
        int3 voxelIncrementation;
        float tLocal = 0.0f;
        float tLocalNext = min(min(tVoxels.x, tVoxels.y), tVoxels.z);

        // If the ray origin is inside the brick and we intersect with a voxel on the first iteration then we are inside that voxel.
        bool origInsideAABB = all(rayOrigLocal >= brickMin && rayOrigLocal <= brickMax);

        // DDA to surface containing voxel.
        while (all(brickLocalVoxelCoords >= 0) && all(brickLocalVoxelCoords < hitBrickWidthInVoxels) && tLocal <= tVoxelMax)
        {
            // Load the voxel corner values.
            float4 values0xx;
            float4 values1xx;
            loadVoxelValues(brickTexelCoords, brickLocalVoxelCoords, values0xx, values1xx);

            // Todo: Optimize this check, valid bit per voxel?.
            if (SDFVoxelCommon::containsSurface(values0xx, values1xx))
            {
                float3 brickLocalHitP = o + tLocal * voxelWidth * d;                // The hit position in the local space of the brick, i.e., [0, 1).
                float3 brickHitP = (brickLocalHitP - brickMin) * virtualGridWidth;  // The hit position in the grid space of the brick, i.e., [0, hitBrickWidthInVoxels).
                float3 voxelHitP = brickHitP - brickLocalVoxelCoords;               // The hit position in the local space of the voxel, i.e., [0, 1).

                if (SDFVoxelHitUtils::intersectSDFVoxelAny(voxelHitP, d, origInsideAABB, values0xx, values1xx, kSnormErrorTolerance + min(tVoxelMax, tLocalNext) - tLocal, kSolverMaxStepCount))
                {
                    return true;
                }
            }

            voxelIncrementation.x = uint(tVoxels.x <= tVoxels.y && tVoxels.x <= tVoxels.z);
            voxelIncrementation.y = uint(tVoxels.y <= tVoxels.x && tVoxels.y <= tVoxels.z);
            voxelIncrementation.z = uint(tVoxels.z <= tVoxels.x && tVoxels.z <= tVoxels.y);

            tVoxels += tDeltas * voxelIncrementation;
            brickLocalVoxelCoords += voxelIncrementation * stepSign;
            tLocal = tLocalNext;
            tLocalNext = min(min(tVoxels.x, tVoxels.y), tVoxels.z);
            origInsideAABB = false;
        }

        return false;
    }

    void decodeHit(const SDFGridHitData hitData, const float3 hitPosition, out float3 gradient)
    {
        // Add 0.5f to hitPosition so that it is in [0, 1] instead of [-0.5, 0.5].
        float3 hitPosLocal = hitPosition + 0.5f;

#if SCENE_SDF_GRADIENT_EVALUATION_METHOD == SCENE_SDF_GRADIENT_NUMERIC_DISCONTINUOUS
        // Find current brick coords.
        uint3 virtualBrickCoords;
        uint2 brickTexelCoords;
        int3 brickLocalVoxelCoords;
        float3 voxelUnitCoords;
        calculateBrickCoords(hitPosLocal, virtualBrickCoords, brickTexelCoords, brickLocalVoxelCoords, voxelUnitCoords);

        float4 values0xx;
        float4 values1xx;
        loadVoxelValues(brickTexelCoords, brickLocalVoxelCoords, values0xx, values1xx);

        const float offset = 0.2f / virtualGridWidth;
        gradient = SDFVoxelHitUtils::computeNumericGradient(voxelUnitCoords, offset, values0xx, values1xx);
#endif
        gradient *= normalizationFactor;
    }

};
