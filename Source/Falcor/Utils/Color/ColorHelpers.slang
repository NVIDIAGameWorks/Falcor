/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
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
#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

/**
 * This file contains host/device shared color utility functions.
 */

/**
 * Returns a relative luminance of an input linear RGB color in the ITU-R BT.709 color space
 * @param RGBColor linear HDR RGB color in the ITU-R BT.709 color space
 */
#ifdef HOST_CODE
inline float luminance(float3 rgb)
#else
[Differentiable]
float luminance(float3 rgb)
#endif
{
    return dot(rgb, float3(0.2126f, 0.7152f, 0.0722f));
}

#ifndef HOST_CODE
// TODO: Unify this code with the host-side functions in ColorUtils.h when #175 is solved.
/**
 * Transforms an RGB color in Rec.709 to CIE XYZ.
 */
float3 RGBtoXYZ_Rec709(float3 c)
{
    static const float3x3 M = {
        // clang-format off
        0.4123907992659595, 0.3575843393838780, 0.1804807884018343, // row 0
        0.2126390058715104, 0.7151686787677559, 0.0721923153607337, // row 1
        0.0193308187155918, 0.1191947797946259, 0.9505321522496608, // row 2
        // clang-format off
    };
    return mul(M, c);
}

/**
 * Transforms an XYZ color to RGB in Rec.709.
 */
float3 XYZtoRGB_Rec709(float3 c)
{
    static const float3x3 M = {
        // clang-format off
        3.240969941904522,   -1.537383177570094,  -0.4986107602930032, // row 0
        -0.9692436362808803, 1.875967501507721,   0.04155505740717569, // row 1
        0.05563007969699373, -0.2039769588889765, 1.056971514242878,   // row 2
        // clang-format on
    };
    return mul(M, c);
}
#endif

/**
 * Converts color from RGB to YCgCo space
 * @param RGBColor linear HDR RGB color
 */
inline float3 RGBToYCgCo(float3 rgb)
{
    float Y = dot(rgb, float3(0.25f, 0.50f, 0.25f));
    float Cg = dot(rgb, float3(-0.25f, 0.50f, -0.25f));
    float Co = dot(rgb, float3(0.50f, 0.00f, -0.50f));
    return float3(Y, Cg, Co);
}

/**
 * Converts color from YCgCo to RGB space
 * @param YCgCoColor linear HDR YCgCo color
 */
inline float3 YCgCoToRGB(float3 YCgCo)
{
    float tmp = YCgCo.x - YCgCo.y;
    float r = tmp + YCgCo.z;
    float g = YCgCo.x + YCgCo.y;
    float b = tmp - YCgCo.z;
    return float3(r, g, b);
}

/**
 * Returns a YUV version of an input linear RGB color in the ITU-R BT.709 color space
 * @param RGBColor linear HDR RGB color in the ITU-R BT.709 color space
 */
inline float3 RGBToYUV(float3 rgb)
{
    float3 ret;
    ret.x = dot(rgb, float3(0.2126f, 0.7152f, 0.0722f));
    ret.y = dot(rgb, float3(-0.09991f, -0.33609f, 0.436f));
    ret.z = dot(rgb, float3(0.615f, -0.55861f, -0.05639f));
    return ret;
}

/**
 * Returns a RGB version of an input linear YUV color in the ITU-R BT.709 color space
 * @param YUVColor linear HDR YUV color in the ITU-R BT.709 color space
 */
inline float3 YUVToRGB(float3 yuv)
{
    float3 ret;
    ret.x = dot(yuv, float3(1.0f, 0.0f, 1.28033f));
    ret.y = dot(yuv, float3(1.0f, -0.21482f, -0.38059f));
    ret.z = dot(yuv, float3(1.0f, 2.12798f, 0.0f));
    return ret;
}

/**
 * Returns a linear-space RGB version of an input RGB channel value in the ITU-R BT.709 color space
 * @param sRGBColor sRGB input channel value
 */
inline float sRGBToLinear(float srgb)
{
    if (srgb <= 0.04045f)
    {
        return srgb * (1.0f / 12.92f);
    }
    else
    {
        return pow((srgb + 0.055f) * (1.0f / 1.055f), 2.4f);
    }
}

/**
 * Returns a linear-space RGB version of an input RGB color in the ITU-R BT.709 color space
 * @param sRGBColor sRGB input color
 */
inline float3 sRGBToLinear(float3 srgb)
{
    return float3(sRGBToLinear(srgb.x), sRGBToLinear(srgb.y), sRGBToLinear(srgb.z));
}

/**
 * Returns a sRGB version of an input linear RGB channel value in the ITU-R BT.709 color space
 * @param LinearColor linear input channel value
 */
inline float linearToSRGB(float lin)
{
    if (lin <= 0.0031308f)
    {
        return lin * 12.92f;
    }
    else
    {
        return pow(lin, (1.0f / 2.4f)) * (1.055f) - 0.055f;
    }
}

/**
 * Returns a sRGB version of an input linear RGB color in the ITU-R BT.709 color space
 * @param LinearColor linear input color
 */
inline float3 linearToSRGB(float3 lin)
{
    return float3(linearToSRGB(lin.x), linearToSRGB(lin.y), linearToSRGB(lin.z));
}

/**
 * Returns Michelson contrast given minimum and maximum intensities of an image region
 * @param iMin minimum intensity of an image region
 * @param iMax maximum intensity of an image region
 */
inline float computeMichelsonContrast(float iMin, float iMax)
{
    if (iMin == 0.0f && iMax == 0.0f)
        return 0.0f;
    else
        return (iMax - iMin) / (iMax + iMin);
}

static const float3 kD65ReferenceIlluminant = float3(0.950428545, 1.000000000, 1.088900371);
static const float3 kInvD65ReferenceIlluminant = float3(1.052156925, 1.000000000, 0.918357670);

#ifndef HOST_CODE
[BackwardDifferentiable]
#endif
inline float3 linearRGBToXYZ(float3 linColor)
{
    // Source: https://www.image-engineering.de/library/technotes/958-how-to-convert-between-srgb-and-ciexyz
    // Assumes D65 standard illuminant.
    const float a11 = 10135552.0f / 24577794.0f;
    const float a12 = 8788810.0f / 24577794.0f;
    const float a13 = 4435075.0f / 24577794.0f;
    const float a21 = 2613072.0f / 12288897.0f;
    const float a22 = 8788810.0f / 12288897.0f;
    const float a23 = 887015.0f / 12288897.0f;
    const float a31 = 1425312.0f / 73733382.0f;
    const float a32 = 8788810.0f / 73733382.0f;
    const float a33 = 70074185.0f / 73733382.0f;

    float3 xyzColor;
    xyzColor.r = a11 * linColor.r + a12 * linColor.g + a13 * linColor.b;
    xyzColor.g = a21 * linColor.r + a22 * linColor.g + a23 * linColor.b;
    xyzColor.b = a31 * linColor.r + a32 * linColor.g + a33 * linColor.b;

    return xyzColor;
}

#ifndef HOST_CODE
[BackwardDifferentiable]
#endif
inline float3 XYZToLinearRGB(float3 xyzColor)
{
    // Return values in linear RGB, assuming D65 standard illuminant.
    const float a11 = 3.241003275f;
    const float a12 = -1.537398934f;
    const float a13 = -0.498615861f;
    const float a21 = -0.969224334f;
    const float a22 = 1.875930071f;
    const float a23 = 0.041554224f;
    const float a31 = 0.055639423f;
    const float a32 = -0.204011202f;
    const float a33 = 1.057148933f;

    float3 linColor;
    linColor.r = a11 * xyzColor.r + a12 * xyzColor.g + a13 * xyzColor.b;
    linColor.g = a21 * xyzColor.r + a22 * xyzColor.g + a23 * xyzColor.b;
    linColor.b = a31 * xyzColor.r + a32 * xyzColor.g + a33 * xyzColor.b;

    return linColor;
}

#ifndef HOST_CODE

[BackwardDifferentiable]
float actualPow(float x, float y)
{
    if (x >= 0.f)
        return pow(x, y);
    else
        return -pow(-x, y);
}

[BackwardDifferentiable]
float3 actualPow(float3 in, float y)
{
    return float3(actualPow(in.x, y), actualPow(in.y, y), actualPow(in.z, y));
}

[BackwardDifferentiable]
inline float3 XYZToOklab(float3 xyzColor)
{
    static const float3x3 M1 = {
        0.8189330101f,
        0.3618667424f,
        -0.1288597137f,
        0.0329845436f,
        0.9293118715f,
        0.0361456387f,
        0.0482003018f,
        0.2643662691f,
        0.6338517070f
    };
    float3 lms = mul(M1, xyzColor);
    lms = actualPow(lms, 1.f / 3.f);
    static const float3x3 M2 = {
        0.2104542553f,
        0.7936177850f,
        -0.0040720468f,
        1.9779984951f,
        -2.4285922050f,
        0.4505937099f,
        0.0259040371f,
        0.7827717662f,
        -0.8086757660f
    };
    return mul(M2, lms);
}

[BackwardDifferentiable]
inline float3 OklabToXYZ(float3 oklabColor)
{
    float3x3 M2inv =
        float3x3(1.00000000f, 0.39633779f, 0.21580376f, 1.00000001f, -0.10556134f, -0.06385417f, 1.00000005f, -0.08948418f, -1.29148554f);
    float3 lms = mul(M2inv, oklabColor);
    lms = lms * lms * lms;
    float3x3 M1inv =
        float3x3(1.22701385f, -0.55779998f, 0.28125615f, -0.04058018f, 1.11225687f, -0.07167668f, -0.07638128f, -0.42148198f, 1.58616322f);
    return mul(M1inv, lms);
}

[BackwardDifferentiable]
inline float3 OklabToHCL(float3 oklabColor)
{
    float L = oklabColor.x;
    float a = oklabColor.y;
    float b = oklabColor.z;

    float H = atan2(b, a);
    float C = sqrt(a * a + b * b);
    return float3(H, C, L);
}

[BackwardDifferentiable]
inline float3 HCLToOklab(float3 hclColor)
{
    float H = hclColor.x;
    float C = hclColor.y;
    float L = hclColor.z;

    float a = C * cos(H);
    float b = C * sin(H);
    return float3(L, a, b);
}

[BackwardDifferentiable]
inline float3 RGBToOklab(float3 lColor)
{
    return XYZToOklab(linearRGBToXYZ(lColor));
}

[BackwardDifferentiable]
inline float3 OklabToRGB(float3 lab)
{
    return XYZToLinearRGB(OklabToXYZ(lab));
}

#endif

inline float3 XYZToCIELab(float3 xyzColor, const float3 invReferenceIlluminant = kInvD65ReferenceIlluminant)
{
    // The default illuminant is D65.
    float3 tmpColor = xyzColor * invReferenceIlluminant;

    float delta = 6.0f / 29.0f;
    float deltaSquare = delta * delta;
    float deltaCube = delta * deltaSquare;
    float factor = 1.0f / (3.0f * deltaSquare);
    float term = 4.0f / 29.0f;

    tmpColor.r = (tmpColor.r > deltaCube ? pow(tmpColor.r, 1.0f / 3.0f) : factor * tmpColor.r + term);
    tmpColor.g = (tmpColor.g > deltaCube ? pow(tmpColor.g, 1.0f / 3.0f) : factor * tmpColor.g + term);
    tmpColor.b = (tmpColor.b > deltaCube ? pow(tmpColor.b, 1.0f / 3.0f) : factor * tmpColor.b + term);

    float3 labColor;
    labColor.r = 116.0f * tmpColor.g - 16.0f;
    labColor.g = 500.0f * (tmpColor.r - tmpColor.g);
    labColor.b = 200.0f * (tmpColor.g - tmpColor.b);

    return labColor;
}

inline float3 CIELabToXYZ(float3 labColor, const float3 referenceIlluminant = kD65ReferenceIlluminant)
{
    // The default illuminant is D65.
    float Y = (labColor.r + 16.0f) / 116.0f;
    float X = labColor.g / 500.0f + Y;
    float Z = Y - labColor.b / 200.0f;

    float delta = 6.0f / 29.0f;
    float factor = 3.0f * delta * delta;
    float term = 4.0f / 29.0f;
    X = ((X > delta) ? X * X * X : (X - term) * factor);
    Y = ((Y > delta) ? Y * Y * Y : (Y - term) * factor);
    Z = ((Z > delta) ? Z * Z * Z : (Z - term) * factor);

    return float3(X, Y, Z) * referenceIlluminant;
}

inline float3 XYZToYCxCz(float3 xyzColor, const float3 invReferenceIlluminant = kInvD65ReferenceIlluminant)
{
    // The default illuminant is D65.
    float3 tmpColor = xyzColor * invReferenceIlluminant;

    float3 ycxczColor;
    ycxczColor.x = 116.0f * tmpColor.g - 16.0f;
    ycxczColor.y = 500.0f * (tmpColor.r - tmpColor.g);
    ycxczColor.z = 200.0f * (tmpColor.g - tmpColor.b);

    return ycxczColor;
}

inline float3 YCxCzToXYZ(float3 ycxczColor, const float3 referenceIlluminant = kD65ReferenceIlluminant)
{
    // The default illuminant is D65.
    float Y = (ycxczColor.r + 16.0f) / 116.0f;
    float X = ycxczColor.g / 500.0f + Y;
    float Z = Y - ycxczColor.b / 200.0f;

    return float3(X, Y, Z) * referenceIlluminant;
}

inline float3 linearRGBToCIELab(float3 lColor)
{
    return XYZToCIELab(linearRGBToXYZ(lColor));
}

inline float3 YCxCzToLinearRGB(float3 ycxczColor)
{
    return XYZToLinearRGB(YCxCzToXYZ(ycxczColor));
}

inline float3 linearRGBToYCxCz(float3 lColor)
{
    return XYZToYCxCz(linearRGBToXYZ(lColor));
}

END_NAMESPACE_FALCOR
