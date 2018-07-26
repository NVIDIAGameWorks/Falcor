/***************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
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
***************************************************************************/
#pragma once
#include "PatternGenerator.h"

namespace Falcor
{
    class HaltonSamplePattern : public PatternGenerator, public inherit_shared_from_this<PatternGenerator, HaltonSamplePattern>
    {
    public:
        using SharedPtr = std::shared_ptr<HaltonSamplePattern>;
        virtual ~HaltonSamplePattern() = default;

        static SharedPtr create(uint32_t sampleCount = 8) { return SharedPtr(new HaltonSamplePattern(sampleCount)); }

        virtual uint32_t getSampleCount() const override { return mSampleCount; }

        virtual void reset(uint32_t startID = 0) { mCurSample = 0; }

        virtual vec2 next()
        {
            return kPattern[(mCurSample++) % mSampleCount];
        }
    protected:
        HaltonSamplePattern(uint32_t sampleCount) : mSampleCount(sampleCount)
        {
            assert(sampleCount == 8);
        }

        uint32_t mCurSample = 0;
        const uint32_t mSampleCount = 8;
        static const vec2 kPattern[];
    };
}