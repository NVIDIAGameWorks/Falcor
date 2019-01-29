/***************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
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
***************************************************************************/
import Shading;
import DefaultVS;

struct GBufferOut
{
    float4 posW             : SV_TARGET0;
    float4 normW            : SV_TARGET1;
    float4 bitangentW       : SV_TARGET2;
    float4 texC             : SV_TARGET3;
    float4 diffuseOpacity   : SV_TARGET4;
    float4 specRough        : SV_TARGET5;
    float4 emissive         : SV_TARGET6;
    float4 matlExtra        : SV_TARGET7;
};

/** Entry point for G-buffer rasterization pixel shader.
*/
GBufferOut ps(VertexOut vsOut)
{
    ShadingData sd = prepareShadingData(vsOut, gMaterial, gCamera.posW);

    GBufferOut gOut;
    gOut.posW           = float4(sd.posW, 1.f);
    gOut.normW          = float4(sd.N, 0.f);
    gOut.bitangentW     = float4(sd.B, 0.f);
    gOut.texC           = float4(sd.uv, 0.f, 0.f);

    gOut.diffuseOpacity = float4(sd.diffuse, sd.opacity);
    gOut.specRough      = float4(sd.specular, sd.linearRoughness);
    gOut.emissive       = float4(sd.emissive, 0.f);
    gOut.matlExtra      = float4(normalize(vsOut.normalW), 0.f);

    return gOut;
}
