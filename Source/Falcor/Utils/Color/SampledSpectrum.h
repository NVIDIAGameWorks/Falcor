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
#include "Core/Assert.h"
#include "Core/Errors.h"
#include "Utils/Math/Common.h"
#include "Utils/Math/Vector.h"
#include <type_traits>
#include <vector>
#include <cmath>

namespace Falcor
{
    enum class SpectrumInterpolation
    {
        Linear, ///< Piecewise linear interpolation of the two nearest samples, and zero outside of the end points.
    };

    /** Represents a uniformly sampled spectrum.
        This class is templated on the value type stored at each sample.
        The first sample is centered at lambdaStart and the last sample is centered at lambdaEnd.
        The spectrum is zero outside the stored wavelength range.
        Example: lambdaStart = 400 nm, lambdaEnd = 700, sampleCount = 4. This means that the bins are centered at 400, 500, 600, 700 nm.
                 The bins are [400, 450], [450, 550], [550, 650], [650, 700].
        Evaluation is done by evaluating basis functions at the center of the bins.
    */
    template<typename T>
    class SampledSpectrum
    {
    public:
        using value_type = T;
        static_assert(std::is_floating_point_v<T> || std::is_same_v<T, float2> || std::is_same_v<T, float3> || std::is_same_v<T, float4>, "T must be a floating point scalar or vector");

        /** Create a spectrum initialized to zero.
            \param[in] lambdaStart First sampled wavelength in nm.
            \param[in] lambdaEnd Last sampled wavelength in nm.
            \param[in] sampleCount Number of wavelength samples stored.
        */
        SampledSpectrum(float lambdaStart, float lambdaEnd, size_t sampleCount)
            : mLambdaStart(lambdaStart)
            , mLambdaEnd(lambdaEnd)
        {
            checkArgument(lambdaEnd > lambdaStart, "'lambdaEnd' must be larger than 'lambdaStart'.");
            checkArgument(sampleCount > 0, "'sampleCount' must be at least one.");
            mSamples.resize(sampleCount, value_type(0));
        }

        /** Create a spectrum initialized from sample array.
            \param[in] lambdaStart First sampled wavelength in nm.
            \param[in] lambdaEnd Last sampled wavelength in nm.
            \param[in] sampleCount Number of wavelength samples stored.
            \param[in] pSamples Spectral samples.
        */
        SampledSpectrum(float lambdaStart, float lambdaEnd, size_t sampleCount, const value_type* pSamples)
            : SampledSpectrum(lambdaStart, lambdaEnd, sampleCount)
        {
            set(sampleCount, pSamples);
        }

        /** Set spectrum samples.
            \param[in] sampleCount Size of array in samples.
            \param[in] pSamples Array of spectral samples.
        */
        void set(const size_t sampleCount, const value_type* pSamples)
        {
            checkArgument(pSamples != nullptr, "'pSamples' is nullptr.");
            checkArgument(sampleCount == mSamples.size(), "Sample count mismatch.");
            mSamples.assign(pSamples, pSamples + sampleCount);
        }

        /** Set spectrum samples.
            \param[in] samples Spectral samples.
        */
        void set(const std::vector<value_type>& samples)
        {
            checkArgument(samples.size() == mSamples.size(), "Sample count mismatch.");
            mSamples = samples;
        }

        /** Set spectrum from unsorted spectral data.
            The spectral data is intergrated using piecewise linear interpolation to compute the value of each sample.
            \param[in] sampleCount Size of arrays in samples.
            \param[in] pSamples Array of spectral samples.
            \param[in] pLambdas Array of sample wavelengths.
        */
        void set(const size_t sampleCount, const value_type* pSamples, const float* pLambdas)
        {
            checkArgument(pSamples != nullptr && pLambdas != nullptr, "'pSamples' or 'pLambdas' is nullptr.");
            FALCOR_UNIMPLEMENTED();
        }

        /** Set spectrum from unsorted spectral data.
            The spectral data is intergrated using piecewise linear interpolation to compute the value of each sample.
            \param[in] samples Spectral samples.
            \param[in] lambdas Sample wavelengths.
        */
        void set(const std::vector<value_type>& samples, const std::vector<float>& lambdas)
        {
            checkArgument(!samples.empty() && samples.size() == lambdas.size(), "'samples' and 'lambdas' must be non-empty and of equal length.");
            set(samples.size(), samples.data(), lambdas.data());
        }

        /** Evaluates the spectrum at the given wavelength.
            \param[in] lambda Wavelength in nm.
            \param[in] interpolationType Which type of interpolation should be used. Linear is the default.
            \return Interpolated value.
        */
        value_type eval(const float lambda, const SpectrumInterpolation interpolationType = SpectrumInterpolation::Linear) const;

        /** Return the CIE 1931 tristimulus values for the spectrum.
            This only works on spectra of scalar types.
        */
        float3 toXYZ_CIE1931() const;

        /** Returns the size of the spectrum in number of samples.
            \return Return the number of wavelength samples in the spectrum.
        */
        size_t size() const
        {
            return mSamples.size();
        }

        /** Return the sample with index.
            \param[in] index Index to the sample that is wanted.
            \return The sample.
        */
        value_type get(size_t index) const
        {
            FALCOR_ASSERT(index < size());
            return mSamples[index];
        }

        /** Set the sample with index.
            \param[in] index Index to the sample that should be set.
            \param[in] value Value of the sample to be set.
        */
        void set(size_t index, T value)
        {
            FALCOR_ASSERT(index < size());
            mSamples[index] = value;
        }

        /** Return the wavelength range.
            \return The wavelength range of the spectrum.
        */
        float2 getWavelengthRange() const
        {
            return float2(mLambdaStart, mLambdaEnd);
        }

    private:
        float mLambdaStart;                 ///< First wavelength sample in nm.
        float mLambdaEnd;                   ///< Last wavelength sample in nm.
        std::vector<value_type> mSamples;   ///< Sample values.
    };


    // Implementation

    template<typename T>
    float3 SampledSpectrum<T>::toXYZ_CIE1931() const
    {
        FALCOR_UNIMPLEMENTED();
    }

    template<typename T>
    T SampledSpectrum<T>::eval(const float lambda, const SpectrumInterpolation interpolationType) const
    {
        checkInvariant(interpolationType == SpectrumInterpolation::Linear, "Interpolation type must be 'Linear'");
        if (lambda < mLambdaStart || lambda > mLambdaEnd) return T(0);
        float x = ((lambda - mLambdaStart) / (mLambdaEnd - mLambdaStart))* (size() - 1.0f);
        size_t i = (size_t)std::floor(x);
        if (i + 1 >= mSamples.size()) return mSamples[size() - 1];
        float w = x - (float)i;
        return lerp(mSamples[i], mSamples[i + 1], T(w));
    }
}
