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
import SampleTexture;

RWTexture2D<float4> outputTexture;

SamplerState sampler;
Texture2D<float4> metallicTexture;
Texture2D<float4> roughnessTexture;

cbuffer CB {
    int roughnessChannel;
    float roughnessValue;
    int metallicChannel;
    float metallicValue;
    uint2 outDim;
};

[numthreads(16,16,1)]
void createSpecularTexture(uint3 threadId : SV_DispatchThreadID)
{
    if (any(threadId.xy > outDim)) return;

    float metallic;
    if (metallicChannel < 0)
    {
        metallic = metallicValue;
    }
    else
    {
        metallic = sampleTexture(metallicTexture, sampler, threadId.xy, outDim)[metallicChannel];
    }

    float roughness;
    if (roughnessChannel < 0)
    {
        roughness = roughnessValue;
    }
    else
    {
        roughness = sampleTexture(roughnessTexture, sampler, threadId.xy, outDim)[roughnessChannel];
    }

    outputTexture[threadId.xy] = float4(0., saturate(roughness), saturate(metallic), 1.);
}
