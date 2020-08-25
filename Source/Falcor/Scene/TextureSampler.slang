/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include "Scene/Material/MaterialDefines.slangh"

/** Interface for texture sampling techniques.
    Types implementing this interface support sampling using different LOD computation techniques
*/
interface ITextureSampler
{
    /** Sample from a 2D texture using the level of detail computed by this method
    */
    float4 sampleTexture(Texture2D t, SamplerState s, float2 uv);
};

/** Texture sampling using implicit fragment-quad finite differences
*/
struct ImplicitLodTextureSampler : ITextureSampler
{
    float4 sampleTexture(Texture2D t, SamplerState s, float2 uv)
    {
        return t.Sample(s, uv);
    }
};

/** Texture sampling using an explicit scalar level of detail
*/
struct ExplicitLodTextureSampler : ITextureSampler
{
    float lod; ///< The explicit level of detail to use

    float4 sampleTexture(Texture2D t, SamplerState s, float2 uv)
    {
        return t.SampleLevel(s, uv, lod);
    }
};

/** Texture sampling using an explicit scalar level of detail using ray cones (with texture dimensions
    "subtracted" from the LOD value, and added back in before SampleLevel()).
*/
struct ExplicitRayConesLodTextureSampler : ITextureSampler
{
    float rayconesLODWithoutTexDims;    ///< this is \Delta_t, which is texture independent, plus the rest of the terms, except the texture size, which is added below

    float4 sampleTexture(Texture2D t, SamplerState s, float2 uv)
    {
        uint txw, txh;
        t.GetDimensions(txw, txh);
        float lambda = 0.5 * log2(txw * txh) + rayconesLODWithoutTexDims;
        return t.SampleLevel(s, uv, lambda);
    }
};


/** Texture sampling using an explicit scalar level of detail using ray diffs,
    with the final LOD computations done below, since they are dependent on texture dimensions.
*/
enum class RayDiffsLODComputationMode { OpenGLStyle, PBRTStyle };
static const RayDiffsLODComputationMode rayDiffsLODComputationMode = RayDiffsLODComputationMode::OpenGLStyle;

struct ExplicitRayDiffsLodTextureSampler : ITextureSampler
{
    float2 dUVdx;               ///< derivatives in x over uv
    float2 dUVdy;               ///< derivatives in y over uv

    float4 sampleTexture(Texture2D t, SamplerState s, float2 uv)
    {
        uint txw, txh;
        float lambda;
        t.GetDimensions(txw, txh);
        if (rayDiffsLODComputationMode == RayDiffsLODComputationMode::OpenGLStyle)  // sharper, but alias sometimes for sharp edges textures, for example
        {
            const float2 duvdx = dUVdx * txw;
            const float2 duvdy = dUVdy * txh;
            lambda = 0.5 * log2(max(dot(duvdx, duvdx), dot(duvdy, duvdy)));
        }
        else // RayDiffsLODComputationMode::PBRTStyle
        {
            float filterWidth = 2.0 * max(txw * max(abs(dUVdx.x), abs(dUVdy.x)), txh * max(abs(dUVdx.y), abs(dUVdy.y)));    // PBRT style (much blurrier, but never (?) aliases)
            lambda = log2(filterWidth);
        }
        return t.SampleLevel(s, uv, lambda);
    }
};


/** Texture sampling using explicit screen-space gradients
*/
struct ExplicitGradientTextureSampler : ITextureSampler
{
    float2 gradX; ///< Gradient of texture coordinate in the screen-space X direction
    float2 gradY; ///< Gradient of texture coordiante in teh screen-space Y direction

    float4 sampleTexture(Texture2D t, SamplerState s, float2 uv)
    {
        return t.SampleGrad(s, uv, gradX, gradY);
    }
};

/** Load data from a texture

    The `lod` parameter represents the method to use for computing
    texture level of detail, and must implement the `ITextureSampler` interface.
*/
float4 sampleTexture<L:ITextureSampler>(Texture2D t, SamplerState s, float2 uv, float4 factor, uint mode, L lod)
{
    if (mode == ChannelTypeUnused) return 0;
    if (mode == ChannelTypeConst) return factor;
    // else mode == ChannelTypeTexture
    return lod.sampleTexture(t, s, uv);
}
