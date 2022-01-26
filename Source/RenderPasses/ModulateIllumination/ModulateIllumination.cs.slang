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

// Static configuration based on which buffers are bound.
#define is_valid(name) (is_valid_##name != 0)

cbuffer CB
{
    uint2   frameDim;
}

Texture2D gEmission;
Texture2D gDiffuseReflectance;
Texture2D gDiffuseRadiance;
Texture2D gSpecularReflectance;
Texture2D gSpecularRadiance;
Texture2D gResidualRadiance;

RWTexture2D gOutput;

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    const uint2 pixel = dispatchThreadId.xy;
    if (any(pixel >= frameDim)) return;

    float4 outputColor = float4(0.f);

    if (is_valid(gEmission))
    {
        outputColor.rgb += gEmission[pixel].rgb;
    }

    if (is_valid(gDiffuseRadiance))
    {
        float3 diffuseColor = gDiffuseRadiance[pixel].rgb;

        if (is_valid(gDiffuseReflectance))
        {
            diffuseColor *= gDiffuseReflectance[pixel].rgb;
        }
        outputColor.rgb += diffuseColor;
    }

    if (is_valid(gSpecularRadiance))
    {
        float3 specularColor = gSpecularRadiance[pixel].rgb;

        if (is_valid(gSpecularReflectance))
        {
            specularColor *= gSpecularReflectance[pixel].rgb;
        }

        outputColor.rgb += specularColor;
    }

    if (is_valid(gResidualRadiance))
    {
        outputColor.rgb += gResidualRadiance[pixel].rgb;
    }

    gOutput[pixel] = outputColor;
}
