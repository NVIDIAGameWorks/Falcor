/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
#include "CPUSampleGenerator.h"
#include "Core/Macros.h"
#include "Utils/Math/Vector.h"

namespace Falcor
{
class FALCOR_API HaltonSamplePattern : public CPUSampleGenerator
{
public:
    /**
     * Create Halton sample pattern generator.
     * @param[in] sampleCount The pattern repeats every 'sampleCount' samples. Zero means no repeating.
     * @return New object, or throws an exception on error.
     */
    static ref<HaltonSamplePattern> create(uint32_t sampleCount = 0) { return make_ref<HaltonSamplePattern>(sampleCount); }

    HaltonSamplePattern(uint32_t sampleCount);
    virtual ~HaltonSamplePattern() = default;

    virtual uint32_t getSampleCount() const override { return mSampleCount; }

    virtual void reset(uint32_t startID = 0) override { mCurSample = 0; }

    virtual float2 next() override;

protected:
    uint32_t mCurSample = 0;
    uint32_t mSampleCount;
};
} // namespace Falcor
