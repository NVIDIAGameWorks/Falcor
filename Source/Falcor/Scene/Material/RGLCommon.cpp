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
#include "RGLCommon.h"
#include "Core/Assert.h"

namespace Falcor
{
    SamplableDistribution4D::SamplableDistribution4D(const float* pdf, uint4 size)
        : mSize(size)
    {
        uint32_t N = size.x * size.y * size.z * size.w;

        mPDF        .reset(new float[N]);
        mConditional.reset(new float[N]);
        mMarginal   .reset(new float[N / size.z]);

        std::memcpy(mPDF.get(), pdf, N * sizeof(float));

        uint32_t sliceStride = size.z * size.w;
        for (uint32_t i = 0; i < N; i += sliceStride)
        {
            build2DSlice(int2(size.z, size.w), mPDF.get() + i, mMarginal.get() + i / size.z, mConditional.get() + i);
        }
    }

    void SamplableDistribution4D::build2DSlice(int2 size, float* pdf, float* marginalCDF, float* conditionalCDF)
    {
        FALCOR_ASSERT(pdf);
        FALCOR_ASSERT(marginalCDF);
        FALCOR_ASSERT(conditionalCDF);

        double tableSum = 0.0;
        for (int i = 0; i < size.x * size.y; ++i)
        {
            tableSum += pdf[i];
        }
        // Edge case: Whole slice is zero. Reset to uniform distribution.
        if (tableSum == 0.0)
        {
            for (int i = 0; i < size.x * size.y; ++i)
            {
                pdf[i] = 1.0f / (size.x * size.y);
            }
            tableSum = 1.0;
        }

        // Step 1: Build row sums.
        for (int y = 0; y < size.y; ++y)
        {
            double rowSum = 0.0;
            conditionalCDF[y * size.x] = float(rowSum);
            for (int x = 1; x < size.x; ++x)
            {
                int idx = x + y * size.x;
                rowSum += (pdf[idx - 1] + pdf[idx]) * 0.5f;
                conditionalCDF[idx] = float(rowSum);
            }
        }

        // Step 2: Build marginal.
        double marginalSum = 0.0;
        marginalCDF[0] = float(marginalSum);
        for (int y = 1; y < size.y; ++y)
        {
            marginalSum += (conditionalCDF[y * size.x - 1] + conditionalCDF[(y + 1) * size.x - 1]) * 0.5f;
            marginalCDF[y] = float(marginalSum);
        }
        // Step 3: Normalize all distributions.
        for (int y = 0; y < size.y; ++y)
        {
            marginalCDF[y] /= float(marginalSum);
        }

        for (int i = 0; i < size.x * size.y; ++i)
        {
            pdf[i] /= float(marginalSum);
            conditionalCDF[i] /= float(marginalSum);
        }
    }
}
