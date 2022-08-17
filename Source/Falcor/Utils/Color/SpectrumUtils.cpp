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
#include "SpectrumUtils.h"
#include "Utils/Color/ColorUtils.h"

#include <xyzcurves/ciexyzCurves1931_1nm.h>
#include <illuminants/D65_5nm.h>

namespace Falcor
{
    // Initialize static data.
    const SampledSpectrum<float3> SpectrumUtils::sCIE_XYZ_1931_1nm(360.0f, 830.0f, 471, reinterpret_cast<const float3*>(xyz1931_1nm));  // 1 nm between samples.
    const SampledSpectrum<float> SpectrumUtils::sD65_5nm(300.0f, 830.0f, 107, reinterpret_cast<const float*>(D65_1nm));                 // 5 nm between samples.

    float3 SpectrumUtils::wavelengthToXYZ_CIE1931(float lambda)
    {
        return sCIE_XYZ_1931_1nm.eval(lambda);
    }

    float SpectrumUtils::wavelengthToD65(float lambda)
    {
        return sD65_5nm.eval(lambda);
    }

    float3 SpectrumUtils::wavelengthToRGB_Rec709(const float lambda)
    {
        float3 XYZ = wavelengthToXYZ_CIE1931(lambda);
        return XYZtoRGB_Rec709(XYZ);
    }
}
