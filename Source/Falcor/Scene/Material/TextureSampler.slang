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

/** Interface for texture sampling techniques.
    Types implementing this interface support sampling using different LOD computation techniques
*/
interface ITextureSampler
{
    /** Sample from a 2D texture using the level of detail computed by this method
    */
    float4 sampleTexture(Texture2D t, SamplerState s, float2 uv);
};

/** Texture sampling using implicit gradients from finite differences within quads.

    Shader model 6.5 and lower *only* supports gradient operations in pixel shaders.
    Usage in other shader stages will generate a compiler error.

    Shader model 6.6 adds support for gradient operations in other shader stages,
    but the application is responsible for making sure the neighboring threads
    within 2x2 quads have appropriate data for gradient computations to be valid.
*/
struct ImplicitLodTextureSampler : ITextureSampler
{
    __init()
    {
    }

    float4 sampleTexture(Texture2D t, SamplerState s, float2 uv)
    {
        return t.Sample(s, uv);
    }
};

/** Texture sampling using an explicit scalar level of detail.
*/
struct ExplicitLodTextureSampler : ITextureSampler
{
    float lod; ///< The explicit level of detail to use

    __init(float lod)
    {
        this.lod = lod;
    }

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

    __init(float rayconesLODWithoutTexDims)
    {
        this.rayconesLODWithoutTexDims = rayconesLODWithoutTexDims;
    }

    float4 sampleTexture(Texture2D t, SamplerState s, float2 uv)
    {
        uint txw, txh;
        t.GetDimensions(txw, txh);
        float lambda = 0.5 * log2(txw * txh) + rayconesLODWithoutTexDims;
        return t.SampleLevel(s, uv, lambda);
    }
};


/** Texture sampling using an explicit scalar, i.e., isotropic, level of detail using ray diffs,
    with the final LOD computations done below, since they are dependent on texture dimensions.
    Use ExplicitGradientTextureSampler if you want anisotropic filtering with ray diffs.
*/
struct ExplicitRayDiffsIsotropicTextureSampler : ITextureSampler
{
    enum class Mode { IsotropicOpenGLStyle, IsotropicPBRTStyle };
    static const Mode kMode = Mode::IsotropicOpenGLStyle;

    float2 dUVdx;               ///< derivatives in x over uv
    float2 dUVdy;               ///< derivatives in y over uv

    __init(float2 dUVdx, float2 dUVdy)
    {
        this.dUVdx = dUVdx;
        this.dUVdy = dUVdy;
    }

    float4 sampleTexture(Texture2D t, SamplerState s, float2 uv)
    {
        uint2 dim;
        t.GetDimensions(dim.x, dim.y);

        switch (kMode)
        {
        case Mode::IsotropicOpenGLStyle:
            {
                // Sharper, but alias sometimes for sharp edges textures.
                const float2 duvdx = dUVdx * dim.x;
                const float2 duvdy = dUVdy * dim.y;
                const float lambda = 0.5f * log2(max(dot(duvdx, duvdx), dot(duvdy, duvdy)));
                return t.SampleLevel(s, uv, lambda);
            }
        case Mode::IsotropicPBRTStyle:
            {
                // PBRT style (much blurrier, but never (?) aliases).
                const float filterWidth = 2.f * max(dim.x * max(abs(dUVdx.x), abs(dUVdy.x)), dim.y * max(abs(dUVdx.y), abs(dUVdy.y)));
                const float lambda = log2(filterWidth);
                return t.SampleLevel(s, uv, lambda);
            }
        }

        return float4(0.f);
    }
};

/** Texture sampling using explicit screen-space gradients.
*/
struct ExplicitGradientTextureSampler : ITextureSampler
{
    float2 gradX; ///< Gradient of texture coordinate in the screen-space X direction
    float2 gradY; ///< Gradient of texture coordiante in teh screen-space Y direction

    __init(float2 gradX, float2 gradY)
    {
        this.gradX = gradX;
        this.gradY = gradY;
    }

    float4 sampleTexture(Texture2D t, SamplerState s, float2 uv)
    {
        return t.SampleGrad(s, uv, gradX, gradY);
    }
};
