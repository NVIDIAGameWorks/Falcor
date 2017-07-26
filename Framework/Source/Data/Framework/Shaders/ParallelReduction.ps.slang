/***************************************************************************
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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
SamplerState gSampler : register(s0);
texture2D gInputTex : register(t1);

#ifdef _FIRST_ITERATION
float2 compareMinMax(float2 crd, float2 range)
{
    float4 d4 = gInputTex.GatherRed(gSampler, crd);

    for(int i = 0 ; i < 4 ; i++)
    {
        if(d4[i] != 1.0f)
        {
            range.x = min(range.x, d4[i]);
            range.y = max(range.y, d4[i]);
        }
    }
    return range;
}
#else
float2 compareMinMax(float2 crd, float2 range)
{
    float4 min4 = gInputTex.GatherRed(gSampler, crd);
    float4 max4 = gInputTex.GatherGreen(gSampler, crd);
    float prevMin = min(min4.x, min(min4.y, min(min4.z, min4.w)));
    float prevMax = max(max4.x, max(max4.y, max(max4.z, max4.w)));
     
    range.x = min(range.x, prevMin);
    range.y = max(range.y, prevMax);
    return range;
}
#endif

float2 minMaxReduction(float2 posS)
{
    float2 range = float2(1, 0);
    float2 dim;
    gInputTex.GetDimensions(dim.x, dim.y);

    // Using gather4, so need to skip 4 samples everytime
    for(int x = 0 ; x < _TILE_SIZE ; x+=2)
    {
        for(int y = 0 ; y < _TILE_SIZE ; y+=2)
        {
            float2 crd = posS * _TILE_SIZE + float2(x, y);
            float2 normalizedCrd = float2(crd)/float2(dim);

            range = compareMinMax(normalizedCrd, range);
        }
    }
    return range;
}

float2 main(float2 texC : TEXCOORD, float4 posS : SV_POSITION) : SV_TARGET0
{
    return minMaxReduction(posS.xy - 0.5f);
}
