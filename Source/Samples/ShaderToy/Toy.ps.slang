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
cbuffer ToyCB
{
    float2 iResolution;
    float iGlobalTime;
};

float rand(float n)
{
    return frac(sin(n) * 43758.5453123);
}

float2 rand2(in float2 p)
{
    return frac(float2(sin(p.x * 591.32 + p.y * 154.077), cos(p.x * 391.32 + p.y * 49.077)));
}

#define v4DarkRed float4(0.2, 0.1, 0.1, 1.0)
#define v4LightRed float4(0.7, 0.1, 0.1, 1.0)

float4 getRBCColor(float x)
{
    float poly = 0.0347197 * x +
        0.247408  * x*x +
        5.69306   * x*x*x -
        19.1026   * x*x*x*x +
        28.6689   * x*x*x*x*x -
        15.5415   * x*x*x*x*x*x + 0.2;
    return lerp(v4DarkRed, v4LightRed, poly);
}

float4 mainImage(in float2 fragCoord)
{
    float2 uv = fragCoord / iResolution.y;

    float freq = 7.0;
    float gap = 1. / freq;
    float2 param_pos = (uv + float2(iGlobalTime / 5.0, 0.0));

    //param_pos = uv;

    float2 closest_center = floor(param_pos * freq + float2(0.5, 0.5)) / freq;

    float ballrad = (0.25 + 0.1 * rand(closest_center.x + 37.0 + closest_center.y)) * gap;
    float jitterrad = 0.5 * gap - ballrad;
    float far = (0.35 * gap - ballrad) / 0.1;

    float black_or_white = 0.5 + 0.5 * sin(
        2.0 * 3.14159 *
        (rand((closest_center.x + 347.0) * (closest_center.y + 129.0)) + iGlobalTime * 1.0));

    closest_center = closest_center + jitterrad * 1.0 *
        sin((iGlobalTime * 0.1 + rand2(closest_center)) * 6.28 +
            sin((iGlobalTime * 0.2 + rand2(closest_center.yx)) * 6.28) +
            sin((iGlobalTime * 0.5 + rand2(closest_center.xy * 93.0 + 127.0)) * 6.28)
            );

    float dist = length(param_pos - closest_center);

    float s = (dist * dist) / (ballrad * ballrad);

    float4 color = lerp(
        lerp(getRBCColor(dist / ballrad), v4DarkRed, far),
        v4DarkRed,
        smoothstep(ballrad*0.95, ballrad*1.05, dist));

    return color;
    //fragColor.rgb = pow(fragColor.rgb, 2.0f);
}

float4 main(in float2 texC : TEXCOORD) : SV_TARGET
{
    return mainImage(texC * iResolution);
}
