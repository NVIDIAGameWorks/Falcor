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
#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

/** This file contains host/device shared color utility functions.
*/

/** Returns a relative luminance of an input linear RGB color in the ITU-R BT.709 color space
    \param RGBColor linear HDR RGB color in the ITU-R BT.709 color space
*/
inline float luminance(float3 rgb)
{
    return dot(rgb, float3(0.2126f, 0.7152f, 0.0722f));
}

#ifndef HOST_CODE
// TODO: Unify this code with the host-side functions in ColorUtils.h when #175 is solved.
/** Transforms an RGB color in Rec.709 to CIE XYZ.
*/
float3 RGBtoXYZ_Rec709(float3 c)
{
    static const float3x3 M =
    {
        0.4123907992659595, 0.3575843393838780, 0.1804807884018343,
        0.2126390058715104, 0.7151686787677559, 0.0721923153607337,
        0.0193308187155918, 0.1191947797946259, 0.9505321522496608
    };
    return mul(M, c);
}

/** Transforms an XYZ color to RGB in Rec.709.
*/
float3 XYZtoRGB_Rec709(float3 c)
{
    static const float3x3 M =
    {
        3.240969941904522, -1.537383177570094, -0.4986107602930032,
        -0.9692436362808803, 1.875967501507721, 0.04155505740717569,
        0.05563007969699373, -0.2039769588889765, 1.056971514242878
    };
    return mul(M, c);
}
#endif

/** Converts color from RGB to YCgCo space
    \param RGBColor linear HDR RGB color
*/
inline float3 RGBToYCgCo(float3 rgb)
{
    float Y = dot(rgb, float3(0.25f, 0.50f, 0.25f));
    float Cg = dot(rgb, float3(-0.25f, 0.50f, -0.25f));
    float Co = dot(rgb, float3(0.50f, 0.00f, -0.50f));
    return float3(Y, Cg, Co);
}

/** Converts color from YCgCo to RGB space
    \param YCgCoColor linear HDR YCgCo color
*/
inline float3 YCgCoToRGB(float3 YCgCo)
{
    float tmp = YCgCo.x - YCgCo.y;
    float r = tmp + YCgCo.z;
    float g = YCgCo.x + YCgCo.y;
    float b = tmp - YCgCo.z;
    return float3(r, g, b);
}

/** Returns a YUV version of an input linear RGB color in the ITU-R BT.709 color space
    \param RGBColor linear HDR RGB color in the ITU-R BT.709 color space
*/
inline float3 RGBToYUV(float3 rgb)
{
    float3 ret;
    ret.x = dot(rgb, float3(0.2126f, 0.7152f, 0.0722f));
    ret.y = dot(rgb, float3(-0.09991f, -0.33609f, 0.436f));
    ret.z = dot(rgb, float3(0.615f, -0.55861f, -0.05639f));
    return ret;
}

/** Returns a RGB version of an input linear YUV color in the ITU-R BT.709 color space
    \param YUVColor linear HDR YUV color in the ITU-R BT.709 color space
*/
inline float3 YUVToRGB(float3 yuv)
{
    float3 ret;
    ret.x = dot(yuv, float3(1.0f, 0.0f, 1.28033f));
    ret.y = dot(yuv, float3(1.0f, -0.21482f, -0.38059f));
    ret.z = dot(yuv, float3(1.0f, 2.12798f, 0.0f));
    return ret;
}

/** Returns a linear-space RGB version of an input RGB channel value in the ITU-R BT.709 color space
    \param sRGBColor sRGB input channel value
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

/** Returns a linear-space RGB version of an input RGB color in the ITU-R BT.709 color space
    \param sRGBColor sRGB input color
*/
inline float3 sRGBToLinear(float3 srgb)
{
    return float3(
        sRGBToLinear(srgb.x),
        sRGBToLinear(srgb.y),
        sRGBToLinear(srgb.z));
}

/** Returns a sRGB version of an input linear RGB channel value in the ITU-R BT.709 color space
    \param LinearColor linear input channel value
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

/** Returns a sRGB version of an input linear RGB color in the ITU-R BT.709 color space
    \param LinearColor linear input color
*/
inline float3 linearToSRGB(float3 lin)
{
    return float3(
        linearToSRGB(lin.x),
        linearToSRGB(lin.y),
        linearToSRGB(lin.z));
}


/** Returns Michelson contrast given minimum and maximum intensities of an image region
    \param iMin minimum intensity of an image region
    \param iMax maximum intensity of an image region
*/
inline float computeMichelsonContrast(float iMin, float iMax)
{
    if (iMin == 0.0f && iMax == 0.0f) return 0.0f;
    else return (iMax - iMin) / (iMax + iMin);
}

END_NAMESPACE_FALCOR
