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
#include "StratifiedSamplePattern.h"

namespace Falcor
{
    StratifiedSamplePattern::SharedPtr StratifiedSamplePattern::create(uint32_t sampleCount)
    {
        return SharedPtr(new StratifiedSamplePattern(sampleCount));
    }

    StratifiedSamplePattern::StratifiedSamplePattern(uint32_t sampleCount)
    {
        // Clamp sampleCount to a reasonable number so the permutation table doesn't get too big.
        if (sampleCount < 1) logWarning("StratifiedSamplePattern() requires sampleCount > 0. Using one sample.");
        else if (sampleCount > 1024) logWarning("StratifiedSamplePattern() requires sampleCount <= 1024. Using 1024 samples.");
        sampleCount = std::clamp(sampleCount, 1u, 1024u);

        // Factorize sampleCount into an M x N grid, where M and N are as close as possible.
        // In the worst case sampleCount is prime and we'll end up with a sampleCount x 1 grid.
        mBinsX = (uint32_t)std::sqrt((double)sampleCount);
        mBinsY = sampleCount / mBinsX;
        while (mBinsX * mBinsY != sampleCount)
        {
            mBinsX++;
            mBinsY = sampleCount / mBinsX;
        }
        assert(mBinsX * mBinsY == sampleCount);

        // Create permutation table.
        mPermutation.resize(sampleCount);
        for (uint32_t i = 0; i < sampleCount; i++) mPermutation[i] = i;
    }

    void StratifiedSamplePattern::reset(uint32_t startID)
    {
        if (startID > 0) logWarning("StratifiedSamplePattern::reset() doesn't support restarting at an arbitrary sample. Using startID = 0.");
        mCurSample = 0;
        mRng = std::mt19937();
    }

    float2 StratifiedSamplePattern::next()
    {
        auto dist = std::uniform_real_distribution<float>();
        auto u = [&]() { return dist(mRng); };

        // Create new permutation at the start of each round of sampling.
        if (mCurSample == 0) std::shuffle(mPermutation.begin(), mPermutation.end(), mRng);

        // Compute stratified point in the current bin.
        uint32_t binIdx = mPermutation[mCurSample];
        uint32_t i = binIdx % mBinsX;
        uint32_t j = binIdx / mBinsX;
        mCurSample = (mCurSample + 1) % getSampleCount();

        assert(i < mBinsX && j < mBinsY);
        float x = ((float)i + u()) / mBinsX;
        float y = ((float)j + u()) / mBinsY;
        return float2(x, y) - 0.5f;
    }
}
