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
import RenderPasses.ToneMapper.ToneMapperParams;

SamplerState gLuminanceTexSampler : register(s0);
SamplerState gColorSampler : register(s1);

Texture2D gColorTex;
Texture2D gLuminanceTex;

static const uint kOperator = _TONE_MAPPER_OPERATOR;
static const float kExposureKey = 0.042;
static const float kLuminanceLod = 16.0; // Lookup highest mip level to get average luminance

cbuffer PerImageCB : register(b0)
{
    ToneMapperParams gParams;
};

float calcLuminance(float3 color)
{
    return dot(color, float3(0.299, 0.587, 0.114));
}

// Linear
float3 toneMapLinear(float3 color)
{
    return color;
}

// Reinhard
float3 toneMapReinhard(float3 color)
{
    float luminance = calcLuminance(color);
    float reinhard = luminance / (luminance + 1);
    return color * (reinhard / luminance);
}

// Reinhard with maximum luminance
float3 toneMapReinhardModified(float3 color)
{
    float luminance = calcLuminance(color);
    float reinhard = luminance * (1 + luminance / (gParams.whiteMaxLuminance * gParams.whiteMaxLuminance)) * (1 + luminance);
    return color * (reinhard / luminance);
}

// John Hable's ALU approximation of Jim Heji's operator
// http://filmicgames.com/archives/75
float3 toneMapHejiHableAlu(float3 color)
{
    color = max(float(0).rrr, color - 0.004);
    color = (color*(6.2 * color + 0.5)) / (color * (6.2 * color + 1.7) + 0.06);

    // Result includes sRGB conversion
    return pow(color, float3(2.2));
}

// John Hable's Uncharted 2 filmic tone map
// http://filmicgames.com/archives/75
float3 applyUc2Curve(float3 color)
{
    float A = 0.22; // Shoulder Strength
    float B = 0.3;  // Linear Strength
    float C = 0.1;  // Linear Angle
    float D = 0.2;  // Toe Strength
    float E = 0.01; // Toe Numerator
    float F = 0.3;  // Toe Denominator

    color = ((color * (A*color+C*B)+D*E)/(color*(A*color+B)+D*F))-(E/F);
    return color;
}

float3 toneMapHableUc2(float3 color)
{
    float exposureBias = 2.0f;
    color = applyUc2Curve(exposureBias * color);
    float whiteScale = 1 / applyUc2Curve(float3(gParams.whiteScale)).x;
    color = color * whiteScale;

    return color;
}

float3 toneMapAces(float3 color)
{
    // Cancel out the pre-exposure mentioned in
    // https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
    color *= 0.6;

    float A = 2.51;
    float B = 0.03;
    float C = 2.43;
    float D = 0.59;
    float E = 0.14;

    color = saturate((color*(A*color+B))/(color*(C*color+D)+E));
    return color;
}

float3 toneMap(float3 color)
{
    switch (kOperator)
    {
        case ToneMapperOperator::Linear:
            return toneMapLinear(color);
        case ToneMapperOperator::Reinhard:
            return toneMapReinhard(color);
        case ToneMapperOperator::ReinhardModified:
            return toneMapReinhardModified(color);
        case ToneMapperOperator::HejiHableAlu:
            return toneMapHejiHableAlu(color);
        case ToneMapperOperator::HableUc2:
            return toneMapHableUc2(color);
        case ToneMapperOperator::Aces:
            return toneMapAces(color);
        default:
            return color;
    }
}

float4 main(float2 texC : TEXCOORD) : SV_TARGET0
{
    float4 color = gColorTex.Sample(gColorSampler, texC);
    float3 finalColor = color.rgb;

#ifdef _TONE_MAPPER_AUTO_EXPOSURE
    // apply auto exposure
    float avgLuminance = exp2(gLuminanceTex.SampleLevel(gLuminanceTexSampler, texC, kLuminanceLod).r);
    float pixelLuminance = calcLuminance(finalColor);
    finalColor *= (kExposureKey / avgLuminance);
#endif

    // apply color grading
    finalColor = mul(finalColor, (float3x3) gParams.colorTransform);

    // apply tone mapping
    finalColor = toneMap(finalColor);

#ifdef _TONE_MAPPER_CLAMP
    finalColor = saturate(finalColor);
#endif

    return float4(finalColor, color.a);
}
