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
#include "stdafx.h"
#include "HaltonSamplePattern.h"

namespace Falcor
{
    namespace
    {
        /** Returns elements of the Halton low-discrepancy sequence.
            \param[in] index Index of the queried element, starting from 0.
            \param[in] base Base for the digit inversion. Should be the next unused prime number.
        */
        float halton(uint32_t index, uint32_t base)
        {
            // Reversing digit order in the given base in floating point.
            float result = 0.0f;
            float factor = 1.0f;

            for (; index > 0; index /= base)
            {
                factor /= base;
                result += factor * (index % base);
            }

            return result;
        }
    }

    HaltonSamplePattern::HaltonSamplePattern(uint32_t sampleCount)
    {
        mSampleCount = sampleCount;
        mCurSample = 0;
    }

    float2 HaltonSamplePattern::next()
    {
        float2 value = {halton(mCurSample, 2), halton(mCurSample, 3)};

        // Modular increment.
        ++mCurSample;
        if (mSampleCount != 0)
        {
            mCurSample = mCurSample % mSampleCount;
        }

        // Map the result so that [0, 1) maps to [-0.5, 0.5) and 0 maps to the origin.
        return glm::fract(value + 0.5f) - 0.5f;
    }
}
