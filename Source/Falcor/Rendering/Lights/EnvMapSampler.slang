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

/** Utility functions for environment map sampling.

    Use the class EnvMapSampler on the host to load and prepare the env map.
    The class builds an hierarchical importance map, which is used here
    for importance sampling.
*/

#include "Utils/Math/MathConstants.slangh"
import Scene.Scene;
import Utils.Math.MathHelpers;

/** Struct returned from the sampling functions.
*/
struct EnvMapSample
{
    float3 dir;         ///< Sampled direction towards the light in world space.
    float pdf;          ///< Probability density function for the sampled direction with respect to solid angle.
    float3 Le;          ///< Emitted radiance.
};

/** Struct for sampling and evaluating an environment map.
*/
struct EnvMapSampler
{
    SamplerState        importanceSampler;      ///< Point sampling with clamp to edge.
    Texture2D<float>    importanceMap;          ///< Hierarchical importance map (entire mip chain).
    float2              importanceInvDim;       ///< 1.0 / dimension.
    uint                importanceBaseMip;      ///< Mip level for 1x1 resolution.
    // TODO: Add scalar value for total integrated intensity, i.e., same as 1x1 mip

    /** Evaluates the radiance coming from world space direction 'dir'.
    */
    float3 eval(float3 dir, float lod = 0.f)
    {
        return gScene.envMap.eval(dir, lod);
    }

    /** Importance sampling of the environment map.
    */
    bool sample(const float2 rnd, out EnvMapSample result)
    {
        float2 p = rnd;     // Random sample in [0,1)^2.
        uint2 pos = 0;      // Top-left texel pos of current 2x2 region.

        // Iterate over mips of 2x2...NxN resolution.
        for (int mip = importanceBaseMip - 1; mip >= 0; mip--)
        {
            // Scale position to current mip.
            pos *= 2;

            // Load the four texels at the current position.
            float w[4];
            w[0] = importanceMap.Load(int3(pos, mip));
            w[1] = importanceMap.Load(int3(pos + uint2(1, 0), mip));
            w[2] = importanceMap.Load(int3(pos + uint2(0, 1), mip));
            w[3] = importanceMap.Load(int3(pos + uint2(1, 1), mip));

            float q[2];
            q[0] = w[0] + w[2];
            q[1] = w[1] + w[3];

            uint2 off;

            // Horizontal warp.
            float d = q[0] / (q[0] + q[1]);   // TODO: Do we need to guard against div-by-zero. We should ensure we never go down a path that has p=0.

            if (p.x < d) // left
            {
                off.x = 0;
                p.x = p.x / d;
            }
            else // right
            {
                off.x = 1;
                p.x = (p.x - d) / (1.f - d);
            }

            // Vertical warp.
            // Avoid stack allocation by not using dynamic indexing.
            // float e = w[off.x] / q[off.x];
            float e = off.x == 0 ? (w[0] / q[0]) : (w[1] / q[1]);

            if (p.y < e) // bottom
            {
                off.y = 0;
                p.y = p.y / e;
            }
            else // top
            {
                off.y = 1;
                p.y = (p.y - e) / (1.f - e);
            }

            pos += off;
        }

        // At this point, we have chosen a texel 'pos' in the range [0,dimension) for each component.
        // The 2D sample point 'p' has been warped along the way, and is in the range [0,1) representing sub-texel location.

        // Compute final sample position and map to direction.
        float2 uv = ((float2)pos + p) * importanceInvDim;     // Final sample in [0,1)^2.
        float3 dir = oct_to_ndir_equal_area_unorm(uv);

        // Compute final pdf.
        // We sample exactly according to the intensity of where the final samples lies in the octahedral map, normalized to its average intensity.
        float avg_w = importanceMap.Load(int3(0, 0, importanceBaseMip)); // 1x1 mip holds integral over importance map. TODO: Replace by constant or rescale in setup so that the integral is 1.0
        float pdf = importanceMap[pos] / avg_w;

        result.dir = gScene.envMap.toWorld(dir);
        result.pdf = pdf * M_1_4PI;
        result.Le = gScene.envMap.eval(result.dir);

        return true;
    }

    /** Evaluates the probability density function for a specific direction.
        Note that the sample() function already returns the pdf for the sampled location.
        But, in some cases we need to evaluate the pdf for other directions (e.g. for MIS).

        \param[in] dir World space direction (normalized).
        \return Probability density function evaluated for direction 'dir'.
    */
    float evalPdf(float3 dir)
    {
        float2 uv = ndir_to_oct_equal_area_unorm(gScene.envMap.toLocal(dir));
        float avg_w = importanceMap.Load(int3(0, 0, importanceBaseMip)); // 1x1 mip holds integral over importance map. TODO: Replace by constant or rescale in setup so that the integral is 1.0
        float pdf = importanceMap.SampleLevel(importanceSampler, uv, 0) / avg_w;
        return pdf * (1.f / M_4PI);
    }
};
