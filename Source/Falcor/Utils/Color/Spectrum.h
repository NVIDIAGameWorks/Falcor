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
#include "Core/Macros.h"
#include "Utils/Math/Common.h"
#include "Utils/Math/Vector.h"
#include "Utils/Color/ColorUtils.h"
#include <fstd/span.h> // TODO C++20: Replace with <span>
#include <algorithm>
#include <filesystem>
#include <optional>
#include <vector>

namespace Falcor
{
    /** Represents a piecewise linearly interpolated spectrum.
        Stores wavelengths (in increasing order) and a value for each wavelength.
    */
    class FALCOR_API PiecewiseLinearSpectrum
    {
    public:
        /** Create a spectrum.
            \param[in] wavelengths Wavelengths in nm.
            \param[in] values Values.
        */
        PiecewiseLinearSpectrum(fstd::span<const float> wavelengths, fstd::span<const float> values);

        /** Create a spectrum from interleaved data:
            [wavelength_0, value_0, wavelength_1, value_1, .. wavelength_N-1, value_N-1]
            \param[in] interleaved Interleaved data (needs to contain 2*N entries).
            \param[in] normalize Normalize spectrum to have luminance of 1.
            \return The spectrum.
        */
        static PiecewiseLinearSpectrum fromInterleaved(fstd::span<const float> interleaved, bool normalize);

        /** Create a spectrum from a text file that contains interleaved data:
            [wavelength_0, value_0, wavelength_1, value_1, .. wavelength_N-1, value_N-1]
            \param[in] path File path.
            \return The spectrum.
        */
        static std::optional<PiecewiseLinearSpectrum> fromFile(const std::filesystem::path& path);

        /** Scale all values of the spectrum by a constant.
            \param[in] factor Scaling factor.
        */
        void scale(float factor);

        /** Evaluate the spectrum at the given wavelength.
            Note: Returns zero for wavelengths outside the defined range.
            \param wavelength Wavelength in nm.
            \return Interpolated value.
        */
        float eval(float wavelength) const
        {
            if (mWavelengths.empty() || wavelength < mWavelengths.front() || wavelength > mWavelengths.back())
            {
                return 0.f;
            }

            auto it = std::lower_bound(mWavelengths.begin(), mWavelengths.end(), wavelength);
            if (it == mWavelengths.begin())
            {
                return mValues.front();
            }

            size_t index = std::distance(mWavelengths.begin(), it) - 1;
            float t = (wavelength - mWavelengths[index]) / (mWavelengths[index + 1] - mWavelengths[index]);
            float a = mValues[index];
            float b = mValues[index + 1];
            return lerp(a, b, t);
        }

        /** Return the wavelength range.
            \return The wavelength range of the spectrum.
        */
        float2 getWavelengthRange() const
        {
            return { mWavelengths.front(), mWavelengths.back() };
        }

        /** Get the maximum value in the spectrum.
            \return The maximum value.
        */
        float getMaxValue() const
        {
            return mMaxValue;
        }

    private:
        std::vector<float> mWavelengths;    ///< Wavelengths in nm.
        std::vector<float> mValues;         ///< Values at each wavelength.
        float mMaxValue;                    ///< Maximum value in mValues.
    };

    /** Represents a denseley sampled spectrum.
    */
    class FALCOR_API DenseleySampledSpectrum
    {
    public:
        DenseleySampledSpectrum(float minWavelength, float maxWavelength, fstd::span<const float> values)
            : mMinWavelength(minWavelength)
            , mMaxWavelength(maxWavelength)
            , mWavelengthStep((maxWavelength - minWavelength) / (values.size() - 1))
            , mValues(values.begin(), values.end())
            , mMaxValue(*std::max_element(values.begin(), values.end()))
        {}

        template<typename S>
        DenseleySampledSpectrum(const S& spectrum, float wavelengthStep = 1.f)
        {
            auto range = spectrum.getWavelengthRange();
            size_t count = (size_t)std::ceil((range.y - range.x) / wavelengthStep);
            mMinWavelength = range.x;
            mMaxWavelength = range.y;
            // max(1, count - 1) handles edge case where wavelengthStep > wavelength range.
            mWavelengthStep = (mMaxWavelength - mMinWavelength) / std::max(1ul, count - 1);
            mValues.resize(count);
            for (size_t i = 0; i < count; ++i)
            {
                mValues[i] = spectrum.eval(mMinWavelength + i * mWavelengthStep);
            }
            mMaxValue = *std::max_element(mValues.begin(), mValues.end());
        }

        /** Evaluate the spectrum at the given wavelength.
            Note: Returns zero for wavelengths outside the defined range.
            \param wavelength Wavelength in nm.
            \return Interpolated value.
        */
        float eval(float wavelength) const
        {
            int index = std::lroundf((wavelength - mMinWavelength) / mWavelengthStep);
            if (index < 0 || index >= (int)mValues.size()) return 0.f;
            return mValues[index];
        }

        /** Return the wavelength range.
            \return The wavelength range of the spectrum.
        */
        float2 getWavelengthRange() const
        {
            return { mMinWavelength, mMaxWavelength };
        }

        /** Get the maximum value in the spectrum.
            \return The maximum value.
        */
        float getMaxValue() const
        {
            return mMaxValue;
        }

    private:
        float mMinWavelength;
        float mMaxWavelength;
        float mWavelengthStep;
        std::vector<float> mValues;
        float mMaxValue;
    };

    /** Compute blackbody emission.
        \param[in] wavelength Wavelength in nm.
        \param[in] temperature Temperature in K.
        \return The emitted radiance.
    */
    FALCOR_API float blackbodyEmission(float wavelength, float temperature);

    /** Represents a blackbody emission spectrum.
    */
    class FALCOR_API BlackbodySpectrum
    {
    public:
        /** Create blackbody emission spectrum.
            \param[in] temperature Temperature in K.
            \param[in] normalize Normalize spectrum such that the peak value is 1.
        */
        BlackbodySpectrum(float temperature, bool normalize = true);

        /** Evaluate the spectrum at the given wavelength.
            \param wavelength Wavelength in nm.
            \return Value.
        */
        float eval(float wavelength) const
        {
            return blackbodyEmission(wavelength, mTemperature) * mNormalizationFactor;
        }

        /** Return the wavelength range.
            \return The wavelength range of the spectrum.
        */
        float2 getWavelengthRange() const
        {
            return { -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity() };
        }

        /** Get the maximum value in the spectrum.
            \return The maximum value.
        */
        float getMaxValue() const { return mMaxValue; }

    private:
        float mTemperature;             ///< Temperature in K.
        float mNormalizationFactor;
        float mMaxValue;
    };

    /** Collection of useful spectra.
    */
    struct FALCOR_API Spectra
    {
        // CIE 1931
        static const DenseleySampledSpectrum kCIE_X;
        static const DenseleySampledSpectrum kCIE_Y;
        static const DenseleySampledSpectrum kCIE_Z;
        static constexpr float kCIE_Y_Integral = 106.856895f;

        /** Get a named spectrum.
            \param[in] name Spectrum name.
            \return The spectrum or nullptr if not found.
        */
        static const PiecewiseLinearSpectrum* getNamedSpectrum(const std::string& name);
    };

    /** Compute the inner product of two spectra.
    */
    template<typename A, typename B>
    float innerProduct(const A& a, const B& b)
    {
        auto rangeA = a.getWavelengthRange();
        auto rangeB = b.getWavelengthRange();
        float minWavelength = std::max(rangeA.x, rangeB.x);
        float maxWavelength = std::min(rangeA.y, rangeB.y);
        float integral = 0.f;
        for (float wavelength = minWavelength; wavelength <= maxWavelength; wavelength += 1.f)
        {
            integral += a.eval(wavelength) * b.eval(wavelength);
        }
        return integral;
    }

    /** Convert spectrum to CIE 1931 XYZ.
    */
    template<typename S>
    float3 spectrumToXYZ(const S& s)
    {
        return float3(
            innerProduct(s, Spectra::kCIE_X),
            innerProduct(s, Spectra::kCIE_Y),
            innerProduct(s, Spectra::kCIE_Z)
        ) / Spectra::kCIE_Y_Integral;
    }

    /** Convert spectrum to RGB in Rec.709.
    */
    template<typename S>
    float3 spectrumToRGB(const S& s)
    {
        return XYZtoRGB_Rec709(spectrumToXYZ(s));
    }
}
