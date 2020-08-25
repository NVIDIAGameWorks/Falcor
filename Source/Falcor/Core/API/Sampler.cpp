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
#include "Sampler.h"

namespace Falcor
{
    struct SamplerData
    {
        uint32_t objectCount = 0;
        Sampler::SharedPtr pDefaultSampler;
    };
    SamplerData gSamplerData;

    Sampler::Sampler(const Desc& desc) : mDesc(desc)
    {
        gSamplerData.objectCount++;
    }

    Sampler::~Sampler()
    {
        gSamplerData.objectCount--;
        if (gSamplerData.objectCount <= 1) gSamplerData.pDefaultSampler = nullptr;
    }

    Sampler::Desc& Sampler::Desc::setFilterMode(Filter minFilter, Filter magFilter, Filter mipFilter)
    {
        mMagFilter = magFilter;
        mMinFilter = minFilter;
        mMipFilter = mipFilter;
        return *this;
    }

    Sampler::Desc& Sampler::Desc::setMaxAnisotropy(uint32_t maxAnisotropy)
    {
        mMaxAnisotropy = maxAnisotropy;
        return *this;
    }

    Sampler::Desc& Sampler::Desc::setLodParams(float minLod, float maxLod, float lodBias)
    {
        mMinLod = minLod;
        mMaxLod = maxLod;
        mLodBias = lodBias;
        return *this;
    }

    Sampler::Desc& Sampler::Desc::setComparisonMode(ComparisonMode mode)
    {
        mComparisonMode = mode;
        return *this;
    }

    Sampler::Desc& Sampler::Desc::setAddressingMode(AddressMode modeU, AddressMode modeV, AddressMode modeW)
    {
        mModeU = modeU;
        mModeV = modeV;
        mModeW = modeW;
        return *this;
    }

    Sampler::Desc& Sampler::Desc::setBorderColor(const float4& borderColor)
    {
        mBorderColor = borderColor;
        return *this;
    }

    Sampler::SharedPtr Sampler::getDefault()
    {
        if (gSamplerData.pDefaultSampler == nullptr)
        {
            gSamplerData.pDefaultSampler = create(Desc());
        }
        return gSamplerData.pDefaultSampler;
    }

    SCRIPT_BINDING(Sampler)
    {
        pybind11::class_<Sampler, Sampler::SharedPtr>(m, "Sampler");

        pybind11::enum_<Sampler::Filter> filter(m, "SamplerFilter");
        filter.value("Linear", Sampler::Filter::Linear);
        filter.value("Point", Sampler::Filter::Point);

        pybind11::enum_<Sampler::AddressMode> addressMode(m, "AddressMode");
        addressMode.value("Wrap", Sampler::AddressMode::Wrap);
        addressMode.value("Mirror", Sampler::AddressMode::Mirror);
        addressMode.value("Clamp", Sampler::AddressMode::Clamp);
        addressMode.value("Border", Sampler::AddressMode::Border);
        addressMode.value("MirrorOnce", Sampler::AddressMode::MirrorOnce);
    }
}
