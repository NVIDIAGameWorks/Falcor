/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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

/** Enums for FLIP -- see FLIPPass.cs.slang and FLIPPass.cpp|h
 */

#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

/** This enum is shared between CPU/GPU.
    It enumerates the different tone mapper options for HDR-FLIP.
*/
enum class FLIPToneMapperType : uint32_t
{
    ACES = 0,
    Hable = 1,
    Reinhard = 2,
};

#ifndef HOST_CODE
float3 toneMap(float3 col, FLIPToneMapperType toneMapper = FLIPToneMapperType::ACES)
{
    float k0, k1, k2, k3, k4, k5;
    if (toneMapper == FLIPToneMapperType::ACES)
    {
        // Source:  ACES approximation : https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
        // Include pre - exposure cancelation in constants
        k0 = 0.6f * 0.6f * 2.51f;
        k1 = 0.6f * 0.03f;
        k2 = 0.0f;
        k3 = 0.6f * 0.6f * 2.43f;
        k4 = 0.6f * 0.59f;
        k5 = 0.14f;
    }
    else if (toneMapper == FLIPToneMapperType::Hable)
    {
        // Source: https://64.github.io/tonemapping/
        const float A = 0.15f;
        const float B = 0.50f;
        const float C = 0.10f;
        const float D = 0.20f;
        const float E = 0.02f;
        const float F = 0.30f;
        k0 = A * F - A * E;
        k1 = C * B * F - B * E;
        k2 = 0.0f;
        k3 = A * F;
        k4 = B * F;
        k5 = D * F * F;

        const float W = 11.2f;
        const float nom = k0 * pow(W, 2.0f) + k1 * W + k2;
        const float denom = k3 * pow(W, 2.0f) + k4 * W + k5;
        const float whiteScale = denom / nom;

        // Include white scale and exposure bias in rational polynomial coefficients
        k0 = 4.0f * k0 * whiteScale;
        k1 = 2.0f * k1 * whiteScale;
        k2 = k2 * whiteScale;
        k3 = 4.0f * k3;
        k4 = 2.0f * k4;
    }
    else if (toneMapper == FLIPToneMapperType::Reinhard)
    {
        float3 luminanceCoefficients = float3(0.2126f, 0.7152f, 0.0722f);
        float Y = dot(col, luminanceCoefficients);

        return clamp(col / (Y + 1.0f), 0.0f, 1.0f);
    }

    float3 colSq = col * col;

    float3 nom = k0 * colSq + k1 * col + k2;
    float3 denom = k3 * colSq + k4 * col + k5;

    denom = isinf(denom) ? 1.0f : denom; // avoid inf / inf division

    float3 toneMappedCol = nom / denom;

    return clamp(toneMappedCol, 0.0f, 1.0f);
}
#endif

END_NAMESPACE_FALCOR
