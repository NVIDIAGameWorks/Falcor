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
SamplerState gSampler;
Texture2D gFont;
Texture2D guiImage;

cbuffer PerFrameCB : register(b0)
{
    float2 scale;
    float2 offset;
    bool useGuiImage;
};

struct VsIn
{
    float2 pos : POSITION;
    float2 texC : TEXCOORD0;
    float4 color : COLOR;
};

struct VsOut
{
    float4 color : COLOR;
    float2 texC : TEXCOORD0;
    float4 pos : SV_POSITION;
};

VsOut vs(VsIn vIn)
{
    VsOut vOut;
    vOut.color = vIn.color;
    vOut.texC = vIn.texC;
    vOut.pos.xy = vIn.pos.xy * scale + offset;
    vOut.pos.zw = float2(0,1);
    return vOut;
}

float4 ps(VsOut vOut) : SV_TARGET
{
    float fontMask = gFont.Sample(gSampler, vOut.texC).r;
    float4 c = (useGuiImage ? float4(guiImage.Sample(gSampler, vOut.texC).xyzw) : vOut.color * fontMask);
    return c;
}