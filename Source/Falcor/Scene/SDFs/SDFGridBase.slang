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
import Utils.Math.FormatConversion;

__exported import Scene.SDFs.SDFGridHitData;

#ifndef SCENE_SDF_SOLVER_MAX_ITERATION_COUNT
#define SCENE_SDF_SOLVER_MAX_ITERATION_COUNT 0
#endif

interface ISDFGrid
{
    /** Intersect a ray with the SDF grid. The ray must be transformed to the local space of the SDF grid prior to calling this.
        \param[in] rayOrigin The origin of the ray in the local space of the SDF grid.
        \param[in] rayDir The direction of the ray in the local space of the SDF grid, note that this should not be normalized if the SDF grid has been scaled.
        \param[in] tMin Minimum valid value for t.
        \param[in] tMax Maximum valid value for t.
        \param[in] primitiveID The primitive ID of the AABB that was hit.
        \param[out] t Intersection t.
        \param[out] hitData Encodes that required to reconstruct the hit position and/or evaluate the gradient at the hit position.
        \return True if the ray intersects the SDF grid, false otherwise.
    */
    bool intersectSDF(const float3 rayOrigin, const float3 rayDir, const float tMin, const float tMax, const uint primitiveID, out float t, out SDFGridHitData hitData);

    /** Intersect a ray with the SDF grid, does not return information about the intersection. The ray must be transformed to the local space of the SDF grid prior to calling this.
        \param[in] rayOrigin The origin of the ray in the local space of the SDF grid.
        \param[in] rayDir The direction of the ray in the local space of the SDF grid, note that this should not be normalized if the SDF grid has been scaled.
        \param[in] tMin Minimum valid value for t.
        \param[in] tMax Maximum valid value for t.
        \param[in] primitiveID The primitive ID of the AABB that was hit.
        \return True if the ray intersects the SDF grid, false otherwise.
    */
    bool intersectSDFAny(const float3 rayOrigin, const float3 rayDir, const float tMin, const float tMax, const uint primitiveID);

    /** Decodes a hit point and calculates the gradient at that position.
        \param[in] hitData Hit data returned by intersectSDF when the hit point at pLocal was found.
        \param[in] hitPosition The position where the gradient should be calculated, must be transformed to the local space of the SDF grid.
        \param[out] gradient The gradient of the SDF grid at hitPosition, note that this is not normalized.
    */
    void decodeHit(const SDFGridHitData hitData, const float3 hitPosition, out float3 gradient);
}

[open]
struct SDFGridBase
{
    static const uint kSolverMaxStepCount = SCENE_SDF_SOLVER_MAX_ITERATION_COUNT;
    static const float kSnormErrorTolerance = 0.6f / 256.0f; // Errors introduced by the 8-bit snorm format used by all SDF grids should be smaller than this, defined by D3D12_FLOAT32_TO_INTEGER_TOLERANCE_IN_ULP / (2^BITS).

    uint calculateGridWidthFromLevel(const uint level)
    {
        return 1 << level;
    }
};
