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
cbuffer SrcRectCB
{
    float2 gOffset;
    float2 gScale;
}


#ifndef SAMPLE_COUNT
Texture2D gTex;
#else
Texture2DMS<float4> gTex;
#endif
SamplerState gSampler;

struct VsOut
{
    float2 texC : TEXCOORD;
    float4 posH : SV_POSITION;
};

VsOut vs(float4 posS : POSITION, float2 texC : TEXCOORD)
{
    VsOut vOut;
    vOut.texC = texC * gScale + gOffset;
    vOut.posH = posS;
    return vOut;
}

float4 ps(float2 texC : TEXCOORD) : SV_TARGET
{
#ifndef SAMPLE_COUNT
    return gTex.Sample(gSampler, texC);
#else
    uint3 dims;
    gTex.GetDimensions(dims.x, dims.y, dims.z);
    uint2 crd = (uint2)(float2(dims.xy) * texC);
    float4 c = float4(0,0,0,0);

    [unroll]
    for(uint i = 0 ; i < SAMPLE_COUNT ; i++)
    {
        c += gTex.Load(crd, i);
    }

    c /= SAMPLE_COUNT;
    return c;
#endif
}
