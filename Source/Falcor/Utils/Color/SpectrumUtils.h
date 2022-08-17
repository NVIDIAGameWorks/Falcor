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
#pragma once
#include "SampledSpectrum.h"
#include "Core/Macros.h"
#include "Core/Assert.h"
#include "Utils/Math/Vector.h"
#include "Utils/Color/ColorUtils.h"
#include <algorithm>
#include <functional>
#include <type_traits>

namespace Falcor
{
    template<typename T> class SampledSpectrum;

    class FALCOR_API SpectrumUtils
    {
    public:
        static const SampledSpectrum<float3> sCIE_XYZ_1931_1nm;
        static const SampledSpectrum<float> sD65_5nm;

        /** Evaluates the 1931 CIE XYZ color matching curves.
            This function uses curves sampled at 1nm and returns XYZ values linearly interpolated from the two nearest samples.
            \param[in] lambda Wavelength in nm.
            \return XYZ tristimulus values.
        */
        static float3 wavelengthToXYZ_CIE1931(float lambda);

        /** Evaluates D65 standard illuminant.
            This function uses curves sampled at 5nm and returns the value linearly interpolated from the two nearest samples.
            \param[in] lambda Wavelength in nm.
            \return D65 value.
        */
        static float wavelengthToD65(float lambda);

        /** Converts from wavelength to XYZ_CIE1931 and then to RGB Rec709.
            \param[in] lambda Wavelength in nm.
            \return RGB color.
        */
        static float3 wavelengthToRGB_Rec709(const float lambda);

        /** Integrate over entire spectrum and apply user-supplied function to each integration.
            \param[in] spectrum The spectrum to be converted.
            \param[in] interpolationType Which type of interpolation that should be used.
            \param[in] func A "ReturnType func(float wavelength)"-function that is applied in each integration step.
            \param[in] componentIndex Which component to evaluate when T is a vector type.
            \param[in] integrationSteps Number of integration steps per sample.
            \return XYZ of the spectrum.
        */
        template<typename T, typename ReturnType>
        static ReturnType integrate(SampledSpectrum<T>& spectrum, const SpectrumInterpolation interpolationType, std::function<ReturnType(float)> func, const uint32_t componentIndex = 0, const uint32_t integrationSteps = 1)
        {
            FALCOR_ASSERT(integrationSteps >= 1);
            float2 wavelengthRange = spectrum.getWavelengthRange();
            uint32_t numEvaluations = uint32_t(spectrum.size() + (integrationSteps - 1) * (spectrum.size() - 1));
            float waveLengthDelta = (wavelengthRange.y - wavelengthRange.x) / (numEvaluations - 1.0f);
            ReturnType sum = ReturnType(0);

            // Riemann sum = integral approximation.
            for (uint32_t q = 0; q < numEvaluations; q++)
            {
                float wavelength = std::min(wavelengthRange.x + waveLengthDelta * q, wavelengthRange.y);
                T spectralIntensity = spectrum.eval(wavelength, interpolationType);
                float s;
                if constexpr (std::is_same_v<T, float>)
                {
                    s = spectralIntensity;
                }
                else
                {
                    s = spectralIntensity[componentIndex];
                }
                sum += func(wavelength) * s * waveLengthDelta * ((q == 0 || q == numEvaluations - 1) ? 0.5f : 1.0f);
            }
            return sum;
        }

        /** Convert entire spectrum to XYZ.
            \param[in] spectrum The spectrum to be converted.
            \param[in] interpolationType Which type of interpolation that should be used.
            \param[in] componentIndex Which component to evaluate when T is a vector type.
            \param[in] integrationSteps Number of integration steps per sample.
            \return XYZ of the spectrum.
        */
        template<typename T>
        static float3 toXYZ(SampledSpectrum<T>& spectrum, const SpectrumInterpolation interpolationType = SpectrumInterpolation::Linear, const uint32_t componentIndex = 0, const uint32_t integrationSteps = 1)
        {
            return integrate<T, float3>(spectrum, interpolationType,
                [](float wavelength) -> float3 { return SpectrumUtils::wavelengthToXYZ_CIE1931(wavelength); },
                componentIndex, integrationSteps);
        }

        /** Convert entire spectrum to XYZ times D65.
            \param[in] spectrum The spectrum to be converted.
            \param[in] interpolationType Which type of interpolation that should be used.
            \param[in] componentIndex Which component to evaluate when T is a vector type..
            \param[in] integrationSteps Number of integration steps per sample.
            \return XYZ of the spectrum times D65.
        */
        template<typename T>
        static float3 toXYZ_D65(SampledSpectrum<T>& spectrum, const SpectrumInterpolation interpolationType = SpectrumInterpolation::Linear, const uint32_t componentIndex = 0, const uint32_t integrationSteps = 1)
        {
            return integrate<T,float3>(spectrum, interpolationType,
                [](float wavelength) -> float3 { return SpectrumUtils::wavelengthToXYZ_CIE1931(wavelength) * SpectrumUtils::wavelengthToD65(wavelength); },
                componentIndex, integrationSteps);
        }

        /** Convert entire spectrum to RGB under the assumption of using the D65 illuminant.
            \param[in] spectrum The spectrum to be converted.
            \param[in] interpolationType Which type of interpolation that should be used.
            \param[in] componentIndex Which component to evaluate when T is a vector type.
            \param[in] integrationSteps Number of integration steps per sample.
            \return An RGB color.
        */
        template<typename T>
        static float3 toRGB_D65(SampledSpectrum<T>& spectrum, const SpectrumInterpolation interpolationType, const uint32_t componentIndex = 0, const uint32_t integrationSteps = 1)
        {
            // Equation 8 from "An OpenEXR Layout for Spectral Images", JCGT.
            // https://jcgt.org/published/0010/03/01/
            float3 XYZ = toXYZ_D65(spectrum, interpolationType, componentIndex, integrationSteps);
            float3 RGB = XYZtoRGB_Rec709(XYZ);
            const float Y_D65 = 10567.0762f;    // Computed as Y_D65 = SpectrumUtils::sD65_5nm.toXYZ(1.0f).y; See Equation 8 in the paper above.
            return RGB * (1.0f / Y_D65);
        }
    };
}
