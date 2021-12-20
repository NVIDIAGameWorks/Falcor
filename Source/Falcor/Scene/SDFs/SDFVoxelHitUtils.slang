/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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

import Scene.SDFs.SDFVoxelCommon;

struct SDFVoxelHitUtils
{
    /** Intersects a ray against the implicit surface surface defined by trilinearly interpolating signed distance values stored at the corner of a voxel.
        \param[in] rayOrigLocal The origin of the ray, in voxel local space, should be inside the voxel.
        \param[in] rayDirLocal The direction of the ray, in voxel local space.
        \param[in] rayOrigInsideVoxel True if the world ray origin is inside the voxel, note that this is note the same as rayOrigLocal.
        \param[in] values0xx The SDF corner values at x=0.
        \param[in] values1xx The SDF corner values at x=1.
        \param[in] tMax The max value of t, in voxel local space.
        \param[in] solverMaxStepCount If a numeric intersection algorithm is seleceted, this is the max number of steps that the algorithm is allowed to take.
        \param[out] t The distance from ray origin to the intersection point in voxel local space, if there is one, otherwise this is undefined.
        \return true if the ray hits the implicit surface, false otherwise.
    */
    static bool intersectSDFVoxel(
        const float3 rayOrigLocal,
        const float3 rayDirLocal,
        const bool rayOrigInsideVoxel,
        const float4 values0xx,
        const float4 values1xx,
        const float tMax,
        const uint solverMaxStepCount,
        out float t)
    {
#if SCENE_SDF_VOXEL_INTERSECTION_METHOD == SCENE_SDF_VOXEL_SPHERE_TRACING
        const float minstep = 0.0001f;
        return Solvers::sphereTraceVoxel(rayOrigLocal, rayDirLocal, values0xx, values1xx, minstep, tMax, solverMaxStepCount, t);
#else
        return false;
#endif
    }

    /** Intersects a ray against the implicit surface surface defined by trilinearly interpolating signed distance values stored at the corner of a voxel, does not return information about the intersection.
        \param[in] rayOrigLocal The origin of the ray, in voxel local space, should be inside the voxel.
        \param[in] rayDirLocal The direction of the ray, in voxel local space.
        \param[in] rayOrigInsideVoxel True if the world ray origin is inside the voxel, note that this is note the same as rayOrigLocal.
        \param[in] values0xx The SDF corner values at x=0.
        \param[in] values1xx The SDF corner values at x=1.
        \param[in] tMax The max value of t, in voxel local space.
        \param[in] solverMaxStepCount If a numeric intersection algorithm is seleceted, this is the max number of steps that the algorithm is allowed to take.
        \return true if the ray hits the implicit surface, false otherwise.
    */
    static bool intersectSDFVoxelAny(
        const float3 rayOrigLocal,
        const float3 rayDirLocal,
        const bool rayOrigInsideVoxel,
        const float4 values0xx,
        const float4 values1xx,
        const float tMax,
        const uint solverMaxStepCount)
    {
#if SCENE_SDF_VOXEL_INTERSECTION_METHOD == SCENE_SDF_VOXEL_SPHERE_TRACING
        const float minstep = 0.0001f;
        return Solvers::sphereTraceVoxelAny(rayOrigLocal, rayDirLocal, values0xx, values1xx, minstep, tMax, solverMaxStepCount);
#else
        return false;
#endif
    }

    /** Computes the gradient at voxelUnitCoords numerically using evaluations in tetrahedron.
        Note that the vector field created by this function is discontinuous between voxels.
        \param[in] voxelUnitCoords Voxel unit coordinates, i.e., [0, 1), where the gradient should be evaluated.
        \param[in] offset The offset to use to find the tetrahedron corners with voxelUnitCoords in the center.
        \param[in] values0xx The SDF corner values at x=0.
        \param[in] values1xx The SDF corner values at x=1.
        \return the computed gradient.
    */
    static float3 computeNumericGradient(const float3 voxelUnitCoords, const float offset, const float4 values0xx, const float4 values1xx)
    {
        float2 e = float2(1.0f, -1.0f) * offset;

        // Calculate offset positions.
        float3 xyyVoxelUnitCoords = voxelUnitCoords + e.xyy;
        float3 yyxVoxelUnitCoords = voxelUnitCoords + e.yyx;
        float3 yxyVoxelUnitCoords = voxelUnitCoords + e.yxy;
        float3 xxxVoxelUnitCoords = voxelUnitCoords + e.xxx;

        float xyyD = SDFVoxelCommon::sdfVoxelTrilin(values0xx, values1xx, xyyVoxelUnitCoords);
        float yyxD = SDFVoxelCommon::sdfVoxelTrilin(values0xx, values1xx, yyxVoxelUnitCoords);
        float yxyD = SDFVoxelCommon::sdfVoxelTrilin(values0xx, values1xx, yxyVoxelUnitCoords);
        float xxxD = SDFVoxelCommon::sdfVoxelTrilin(values0xx, values1xx, xxxVoxelUnitCoords);

        return
            e.xyy * xyyD +
            e.yyx * yyxD +
            e.yxy * yxyD +
            e.xxx * xxxD;
    }


    struct Solvers
    {
        static const float kNumericSolverEpsilon = 1e-3f;

        static bool sphereTraceVoxel(
            const float3 rayOrigLocal,
            const float3 rayDirLocal,
            const float4 values0xx,
            const float4 values1xx,
            const float minstep,
            const float tMax,
            const uint maxIterations,
            out float t)
        {
            float3 p = rayOrigLocal;

            float lastD = 1.0f;
            float currD = SDFVoxelCommon::sdfVoxelTrilin(values0xx, values1xx, p);
            float clampedD = max(currD, minstep);
            t = 0.0f;

            // Check if we're already inside the surface
            if (currD <= 0.0f) return true;

            uint iterations = 0;
            for (; iterations < maxIterations; iterations++)
            {
                // Update t.
                t += clampedD;

                // Check if we're outside farplane.
                if (t > tMax)
                {
                    return false;
                }

                // Update position.
                p = clamp(rayOrigLocal + t * rayDirLocal, 0.0f, 1.0f);

                // Store lastD, and evaluate currD at current position.
                lastD = currD;
                currD = SDFVoxelCommon::sdfVoxelTrilin(values0xx, values1xx, p);
                clampedD = max(currD, minstep);

                // Check if we're inside the surface.
                if (currD <= 0.0f)
                {
                    break;
                }
            }

            // Linear interpolation to approximate intersection point.
            float finalH = currD * clampedD / (currD - lastD);
            t += finalH;

            return iterations < maxIterations;
        }

        static bool sphereTraceVoxelAny(
            const float3 rayOrigLocal,
            const float3 rayDirLocal,
            const float4 values0xx,
            const float4 values1xx,
            const float minstep,
            const float tMax,
            const uint maxIterations)
        {
            float3 p = rayOrigLocal;

            float lastD = 1.0f;
            float currD = SDFVoxelCommon::sdfVoxelTrilin(values0xx, values1xx, p);
            float clampedD = max(currD, minstep);
            float t = 0.0f;

            // Check if we're already inside the surface
            if (currD <= 0.0f) return true;

            uint iterations = 0;
            for (; iterations < maxIterations; iterations++)
            {
                // Update t.
                t += clampedD;

                // Check if we're outside farplane.
                if (t > tMax)
                {
                    return false;
                }

                // Update position.
                p = clamp(rayOrigLocal + t * rayDirLocal, 0.0f, 1.0f);

                // Store lastD, and evaluate currD at current position.
                lastD = currD;
                currD = SDFVoxelCommon::sdfVoxelTrilin(values0xx, values1xx, p);
                clampedD = max(currD, minstep);

                // Check if we're inside the surface.
                if (currD <= 0.0f)
                {
                    return true;
                }

                // Check if we're outside farplane.
                if (t > tMax)
                {
                    return false;
                }
            }

            return iterations < maxIterations;
        }

    };
};
