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
SamplerState gSampler : register(s0);
#if defined(_FIRST_ITERATION) && _SAMPLE_COUNT > 1
Texture2DMS gInputTex : register(t1);
#else
Texture2D gInputTex : register(t1);
#endif

#if _SAMPLE_COUNT <= 1 || !defined(_FIRST_ITERATION)
float2 getNormalizedCrd(float2 crd)
{
    float2 dim;
    gInputTex.GetDimensions(dim.x, dim.y);
    return float2(crd)/float2(dim);
}
#endif

#ifdef _FIRST_ITERATION
#if _SAMPLE_COUNT <= 1
float2 compareMinMax(float2 crd, float2 range)
{
    float4 d4 = gInputTex.GatherRed(gSampler, getNormalizedCrd(crd));

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
#else   //_SAMPLE_COUNT <= 1
float2 compareMinMax(float2 crd, float2 range)
{
    uint3 dim;
    gInputTex.GetDimensions(dim.x, dim.y, dim.z);

    for(int s = 0 ; s < _SAMPLE_COUNT ; s++)
    {
        for(int x = 0 ; x < 2 ; x++)
        {
            for(int y = 0 ; y < 2 ; y++)
            {
                int2 icrd = int2(crd) + int2(x, y);
                icrd = min(icrd, dim.xy - 1);
                float d = gInputTex.Load(icrd, s).r;
                if(d != 1.0f)
                {
                    range.x = min(range.x, d);
                    range.y = max(range.y, d);
                }
            }
        }
    }
    return range;
}
#endif  // _SAMPLE_COUNT <= 1
#else // _FIRST_ITERATION
float2 compareMinMax(float2 crd, float2 range)
{
    float2 normalizedCrd = getNormalizedCrd(crd);
    float4 min4 = gInputTex.GatherRed(gSampler, normalizedCrd);
    float4 max4 = gInputTex.GatherGreen(gSampler, normalizedCrd);
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

    // Using gather4, so need to skip 4 samples everytime
    for(int x = 0 ; x < _TILE_SIZE ; x+=2)
    {
        for(int y = 0 ; y < _TILE_SIZE ; y+=2)
        {
            float2 crd = posS * _TILE_SIZE + float2(x, y);         
            range = compareMinMax(crd, range);
        }
    }
    return range;
}

float2 main(float2 texC : TEXCOORD, float4 posS : SV_POSITION) : SV_TARGET0
{
    return minMaxReduction(posS.xy - 0.5f);
}
