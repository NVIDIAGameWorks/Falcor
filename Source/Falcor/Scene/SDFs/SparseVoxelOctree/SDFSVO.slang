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
import Scene.SDFs.SDFVoxelCommon;
import Scene.SDFs.SDFVoxelHitUtils;
import Scene.SDFs.SDFVoxelTypes;
import Scene.SDFs.SDFGridBase;
import Utils.Geometry.IntersectionHelpers;
import Utils.Math.FormatConversion;

struct StackData
{
    uint relationData;
    float tMax;

    static StackData create(uint relationData, float tMax)
    {
        StackData stackData;
        stackData.relationData = relationData;
        stackData.tMax = tMax;
        return stackData;
    }
};

struct SDFSVO : SDFGridBase, ISDFGrid
{
    static const uint kStackSize = 23;
    static const float kEpsilon = 1.0f / exp2(kStackSize);
    static const float kCoarsestLevelNormalizationFactor = 0.5f * sqrt(3.0f);


    StructuredBuffer<SDFSVOVoxel> svo;
    uint levelCount;

    /** Removes the mirroring performed by the intersection algorithm given the octant mask.
    */
    float3 undoMirroring(float3 p, float scale, uint octMask)
    {
        p += (3.0f - scale - 2.0f * p) * (1.0f - float3((octMask & 0x4) >> 2, (octMask & 0x2) >> 1, octMask & 0x1));
        return p;
    }

    bool intersectSDF(const float3 rayOrigin, const float3 rayDir, const float tMin, const float tMax, const uint primitiveID, out float t, out SDFGridHitData hitData)
    {
        // rayOrigin will be in [-0.5, 0.5], offset it so that it is in [1, 2].
        float3 o = rayOrigin + 1.5f;

        // Normalize ray direction.
        float dirLength = length(rayDir);
        float inverseDirLength = 1.0f / dirLength;
        float3 d = rayDir * inverseDirLength;

        // Clamp direction to epsilon to avoid division by zero.
        d.x = abs(d.x) < kEpsilon ? (kEpsilon * step(0.0f, d.x) * 2.0f - 1.0f) : d.x;
        d.y = abs(d.y) < kEpsilon ? (kEpsilon * step(0.0f, d.y) * 2.0f - 1.0f) : d.y;
        d.z = abs(d.z) < kEpsilon ? (kEpsilon * step(0.0f, d.z) * 2.0f - 1.0f) : d.z;

        float2 nearFar;
        if (!intersectRayAABB(o, d, float3(1.0f), float3(2.0f), nearFar)) return false;

        float tMinFinal = max(tMin * dirLength, nearFar.x) - nearFar.x;
        float tMaxFinal = min(tMax * dirLength, nearFar.y) - nearFar.x;

        // Check that the ray segment overlaps the AABB.
        if (tMaxFinal < tMinFinal) return false;

        o += d * nearFar.x;
        bool origInsideAABB = all(o >= 1.0f && o <= 2.0f);

        // The ray will be intersected against axis aligned planes as: t_x(x) = x * (1 / d.x) - (o.x / d.x), therefore, precompute (1 / d.x) and (-o.x / d.x). Similar for other axes.
        float3 tCoef = 1.0f / -abs(d);
        float3 tBias = tCoef * o;

        // Create an oct mask and change tBias to effectively create a new coordinate system where all components of d are negative, this avoids some sign checks in the traversal.
        uint octMask = 0x7;
        if (d.x > 0.0f) octMask ^= 0x4, tBias.x = 3.0f * tCoef.x - tBias.x;
        if (d.y > 0.0f) octMask ^= 0x2, tBias.y = 3.0f * tCoef.y - tBias.y;
        if (d.z > 0.0f) octMask ^= 0x1, tBias.z = 3.0f * tCoef.z - tBias.z;

        // Initialize tMin and tMax and define threshold value h.
        float3 tMinimums = 2.0f * tCoef - tBias;
        float3 tMaximums = tCoef - tBias;
        float tMaxLocal = min(min(tMaximums.x, tMaximums.y), tMaximums.z);
        float tMinLocal = max(max(max(tMinFinal, tMinimums.x), tMinimums.y), tMinimums.z);
        float h = tMaxLocal;

        // Initialize the currently entered voxel.
        uint svoIndex = 0;
        uint nextID = 0;
        SDFSVOVoxel voxel = svo[svoIndex];
        uint relationData = voxel.relationData;

        // Initialize current position and hierarchy.
        float3 p = float3(1.0f);
        int hierarchy = kStackSize - 1;
        float scale = 0.5f;

        // Declare the stack.
        StackData stack[kStackSize + 1];
        stack[kStackSize] = StackData.create(svoIndex, tMaxLocal);

        // Traverse p to the first child and update nextID.
        if (1.5f * tCoef.x - tBias.x > tMinLocal) nextID ^= 0x4, p.x = 1.5f;
        if (1.5f * tCoef.y - tBias.y > tMinLocal) nextID ^= 0x2, p.y = 1.5f;
        if (1.5f * tCoef.z - tBias.z > tMinLocal) nextID ^= 0x1, p.z = 1.5f;

        while (hierarchy < kStackSize)
        {
            float3 tPlane = p * tCoef - tBias;
            float tpMax = min(min(tPlane.x, tPlane.y), tPlane.z);

            // Check if child voxel is valid and the active t-span is non-empty.
            uint childShift = nextID ^ octMask;
            uint childMask = (relationData << childShift);
            if ((childMask & 0x80) != 0 && tMinLocal <= tMaxLocal && hierarchy >= (kStackSize - (levelCount - 1)))
            {
                // INTERSECT

                // Intersect active t-span with the cube and evaluate t at the center of the voxel.
                float halfScale = scale * 0.5f;
                float tvMax = min(tpMax, tMaxLocal);
                float3 tCenter = halfScale * tCoef + tPlane;

                // If the resulting t-span is non-empty, enter the first child.
                if (tMinLocal <= tvMax)
                {
                    // PUSH - We need to enter a child of the current voxel.

                    // Write the current voxel to the stack if necessary and update h.
                    if (tpMax < h) stack[hierarchy] = StackData.create(relationData, tMaxLocal);
                    h = tpMax;

                    // Load the entered voxel.
                    svoIndex = (relationData >> 8) + countbits(childMask & 0x7f);

                    voxel = svo[svoIndex];
                    relationData = voxel.relationData;

                    // If voxel is small enough, perform SDF intersection.
                    if (hierarchy < (kStackSize - levelCount) + 2)
                    {
                        float4 values0xx;
                        float4 values1xx;
                        SDFVoxelCommon::unpackValues(voxel.packedValues, values0xx, values1xx);

                        float3 pHit = o + tMinLocal * d;
                        float3 voxelUnitCoords = (pHit - undoMirroring(p, scale, octMask)) / scale;

                        float tLocal;
                        if (SDFVoxelHitUtils::intersectSDFVoxel(voxelUnitCoords, d, origInsideAABB, values0xx, values1xx, kSnormErrorTolerance + (min(tvMax, tMaxFinal) - tMinLocal) / scale, kSolverMaxStepCount, tLocal))
                        {
                            t = (nearFar.x + tMinLocal + tLocal * scale) * inverseDirLength;

                            hitData.svoIndex = svoIndex;
                            hitData.hitT = t;
                            return true;
                        }
                    }

                    // Select the child that the ray will enter first.
                    nextID = 0;
                    hierarchy--;
                    scale = halfScale;

                    uint3 nextIDIndicator = uint3(tCenter > tMinLocal);
                    nextID ^= (0x4 * nextIDIndicator.x) | (0x2 * nextIDIndicator.y) | (0x1 * nextIDIndicator.z);
                    p += scale * float3(nextIDIndicator);

                    tMaxLocal = tvMax;
                    continue;
                }
            }

            // Check that we're still in the valid ray segment.
            if (tMinLocal > tMaxFinal) return false;

            // ADVANCE - We need to enter a neighbor.
            origInsideAABB = false;
            uint stepMask = 0;
            uint stepDir = 0;

            uint3 stepDirIndicator = uint3(tPlane <= tpMax);
            stepMask ^= (0x4 * stepDirIndicator.x) | (0x2 * stepDirIndicator.y) | (0x1 * stepDirIndicator.z);
            p -= scale * float3(stepDirIndicator);

            // Update active t-span and flip bits of the child ID.
            tMinLocal = tpMax;
            nextID ^= stepMask;

            // Check if we need to pop to enter a neighbor.
            if ((nextID & stepMask) != 0)
            {
                // POP - We need to pop to the parent with the highest differing bits between the two positions.
                uint differingBits = 0;
                if (stepMask & 0x4) differingBits |= asuint(p.x) ^ asuint(p.x + scale);
                if (stepMask & 0x2) differingBits |= asuint(p.y) ^ asuint(p.y + scale);
                if (stepMask & 0x1) differingBits |= asuint(p.z) ^ asuint(p.z + scale);

                hierarchy = ((asint(float(differingBits)) >> 23) - 127);
                scale = asfloat((hierarchy - kStackSize + 127) << 23);

                // Get the correct stack data.
                StackData stackData = stack[hierarchy];
                relationData = stackData.relationData;
                tMaxLocal = stackData.tMax;

                // Round cube position and calculate next ID.
                uint3 shiftedP = asuint(p) >> hierarchy;
                p = asfloat(shiftedP << hierarchy);
                nextID = ((shiftedP.x & 0x1) << 2) | ((shiftedP.y & 0x1) << 1) | (shiftedP.z & 0x1);

                // Prevent same parent from being stored again.
                h = 0.0f;
            }
        }

        return false;
    }

    bool intersectSDFAny(const float3 rayOrigin, const float3 rayDir, float tMin, const float tMax, const uint primitiveID)
    {
        // rayOrigin will be in [-0.5, 0.5], offset it so that it is in [1, 2].
        float3 o = rayOrigin + 1.5f;

        // Normalize ray direction.
        float dirLength = length(rayDir);
        float inverseDirLength = 1.0f / dirLength;
        float3 d = rayDir * inverseDirLength;

        // Clamp direction to epsilon to avoid division by zero.
        d.x = abs(d.x) < kEpsilon ? (kEpsilon * step(0.0f, d.x) * 2.0f - 1.0f) : d.x;
        d.y = abs(d.y) < kEpsilon ? (kEpsilon * step(0.0f, d.y) * 2.0f - 1.0f) : d.y;
        d.z = abs(d.z) < kEpsilon ? (kEpsilon * step(0.0f, d.z) * 2.0f - 1.0f) : d.z;

        float2 nearFar;
        if (!intersectRayAABB(o, d, float3(1.0f), float3(2.0f), nearFar)) return false;

        float tMinFinal = max(tMin * dirLength, nearFar.x) - nearFar.x;
        float tMaxFinal = min(tMax * dirLength, nearFar.y) - nearFar.x;

        // Check that the ray segment overlaps the AABB.
        if (tMaxFinal < tMinFinal) return false;

        o += d * nearFar.x;
        bool origInsideAABB = all(o >= 1.0f && o <= 2.0f);

        // The ray will be intersected against axis aligned planes as: t_x(x) = x * (1 / d.x) - (o.x / d.x), therefore, precompute (1 / d.x) and (-o.x / d.x). Similar for other axes.
        float3 tCoef = 1.0f / -abs(d);
        float3 tBias = tCoef * o;

        // Create an oct mask and change tBias to effectively create a new coordinate system where all components of d are negative, this avoids some sign checks in the traversal.
        uint octMask = 0x7;
        if (d.x > 0.0f) octMask ^= 0x4, tBias.x = 3.0f * tCoef.x - tBias.x;
        if (d.y > 0.0f) octMask ^= 0x2, tBias.y = 3.0f * tCoef.y - tBias.y;
        if (d.z > 0.0f) octMask ^= 0x1, tBias.z = 3.0f * tCoef.z - tBias.z;

        // Initialize tMin and tMax and define threshold value h.
        float3 tMinimums = 2.0f * tCoef - tBias;
        float3 tMaximums = tCoef - tBias;
        float tMaxLocal = min(min(tMaximums.x, tMaximums.y), tMaximums.z);
        float tMinLocal = max(max(max(tMinFinal, tMinimums.x), tMinimums.y), tMinimums.z);
        float h = tMaxLocal;

        // Initialize the currently entered voxel.
        uint svoIndex = 0;
        uint nextID = 0;
        SDFSVOVoxel voxel = svo[svoIndex];
        uint relationData = voxel.relationData;

        // Initialize current position and hierarchy.
        float3 p = float3(1.0f);
        int hierarchy = kStackSize - 1;
        float scale = 0.5f;

        // Declare the stack.
        StackData stack[kStackSize + 1];
        stack[kStackSize] = StackData.create(svoIndex, tMaxLocal);

        // Traverse p to the first child and update nextID.
        if (1.5f * tCoef.x - tBias.x > tMinFinal) nextID ^= 0x4, p.x = 1.5f;
        if (1.5f * tCoef.y - tBias.y > tMinFinal) nextID ^= 0x2, p.y = 1.5f;
        if (1.5f * tCoef.z - tBias.z > tMinFinal) nextID ^= 0x1, p.z = 1.5f;

        while (hierarchy < kStackSize)
        {
            float3 tPlane = p * tCoef - tBias;
            float tpMax = min(min(tPlane.x, tPlane.y), tPlane.z);

            // Check if child voxel is valid and the active t-span is non-empty.
            uint childShift = nextID ^ octMask;
            uint childMask = (relationData << childShift);
            if ((childMask & 0x80) != 0 && tMinFinal <= tMaxLocal && hierarchy >= (kStackSize - (levelCount - 1)))
            {
                // INTERSECT
                // Intersect active t-span with the cube and evaluate t at the center of the voxel.
                float halfScale = scale * 0.5f;
                float tvMax = min(tpMax, tMaxLocal);
                float3 tCenter = halfScale * tCoef + tPlane;

                // If the resulting t-span is non-empty, enter the first child.
                if (tMinFinal <= tvMax)
                {
                    // PUSH - We need to enter a child of the current voxel.

                    // Write the current voxel to the stack if necessary and update h.
                    if (tpMax < h) stack[hierarchy] = StackData.create(relationData, tMaxLocal);
                    h = tpMax;

                    // Load the entered voxel.
                    svoIndex = (relationData >> 8) + countbits(childMask & 0x7f);
                    voxel = svo[svoIndex];
                    relationData = voxel.relationData;

                    // If voxel is small enough, perform SDF intersection.
                    if (hierarchy < (kStackSize - levelCount) + 2)
                    {
                        float4 values0xx;
                        float4 values1xx;
                        SDFVoxelCommon::unpackValues(voxel.packedValues, values0xx, values1xx);

                        float3 pHit = o + tMinFinal * d;
                        float3 voxelUnitCoords = (pHit - undoMirroring(p, scale, octMask)) / scale;

                        if (SDFVoxelHitUtils::intersectSDFVoxelAny(voxelUnitCoords, d, origInsideAABB, values0xx, values1xx, kSnormErrorTolerance + (min(tvMax, tMaxFinal) - tMinLocal) / scale, kSolverMaxStepCount))
                        {
                            break;
                        }
                    }

                    // Select the child that the ray will enter first.
                    nextID = 0;
                    hierarchy--;
                    scale = halfScale;

                    uint3 nextIDIndicator = uint3(tCenter > tMinFinal);
                    nextID ^= (0x4 * nextIDIndicator.x) | (0x2 * nextIDIndicator.y) | (0x1 * nextIDIndicator.z);
                    p += scale * float3(nextIDIndicator);

                    tMaxLocal = tvMax;
                    continue;
                }
            }

            // Check that we're still in the valid ray segment.
            if (tMinFinal > tMaxFinal) return false;

            // ADVANCE - We need to enter a neighbor.
            origInsideAABB = false;
            uint stepMask = 0;
            uint stepDir = 0;

            uint3 stepDirIndicator = uint3(tPlane <= tpMax);
            stepMask ^= (0x4 * stepDirIndicator.x) | (0x2 * stepDirIndicator.y) | (0x1 * stepDirIndicator.z);
            stepDir = stepDirIndicator.y + (2 * stepDirIndicator.z);
            p -= scale * float3(stepDirIndicator);

            // Update active t-span and flip bits of the child ID.
            tMinFinal = tpMax;
            nextID ^= stepMask;

            // Check if we need to pop to enter a neighbor.
            if ((nextID & stepMask) != 0)
            {
                // POP - We need to pop to the parent with the highest differing bits between the two positions.
                uint differingBits = 0;
                if (stepMask & 0x4) differingBits |= asuint(p.x) ^ asuint(p.x + scale);
                if (stepMask & 0x2) differingBits |= asuint(p.y) ^ asuint(p.y + scale);
                if (stepMask & 0x1) differingBits |= asuint(p.z) ^ asuint(p.z + scale);

                hierarchy = ((asint(float(differingBits)) >> 23) - 127);
                scale = asfloat((hierarchy - kStackSize + 127) << 23);

                // Get the correct stack data.
                StackData stackData = stack[hierarchy];
                relationData = stackData.relationData;
                tMaxLocal = stackData.tMax;

                // Round cube position and calculate next ID.
                uint3 shiftedP = asuint(p) >> hierarchy;
                p = asfloat(shiftedP << hierarchy);
                nextID = ((shiftedP.x & 0x1) << 2) | ((shiftedP.y & 0x1) << 1) | (shiftedP.z & 0x1);

                // Prevent same parent from being stored again.
                h = 0.0f;
            }
        }

        return hierarchy < kStackSize;
    }

    void decodeHit(const SDFGridHitData hitData, const float3 hitPosition, out float3 gradient)
    {
        // Add 0.5f to hitPosition so that it is in [0, 1] instead of [-0.5, 0.5].
        float3 hitPosLocal = hitPosition + 0.5f;

        SDFSVOVoxel voxel = svo[hitData.svoIndex];

        float4 values0xx;
        float4 values1xx;
        SDFVoxelCommon::unpackValues(voxel.packedValues, values0xx, values1xx);

        uint3 voxelCoords;
        uint level;
        SDFVoxelCommon::decodeLocation(voxel.locationCode, voxelCoords, level);

        float gridWidth = float(calculateGridWidthFromLevel(level));
        float3 voxelUnitCoords = hitPosLocal * gridWidth - float3(voxelCoords);

#if SCENE_SDF_GRADIENT_EVALUATION_METHOD == SCENE_SDF_GRADIENT_NUMERIC_DISCONTINUOUS
        const float offset = 0.2f / gridWidth;
        gradient = SDFVoxelHitUtils::computeNumericGradient(voxelUnitCoords, offset, values0xx, values1xx);
#endif
        gradient *= (kCoarsestLevelNormalizationFactor / gridWidth);
    }
};
