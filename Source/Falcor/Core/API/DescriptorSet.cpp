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
#include "DescriptorSet.h"

namespace Falcor
{
    DescriptorSet::SharedPtr DescriptorSet::create(const DescriptorPool::SharedPtr& pPool, const Layout& layout)
    {
        return SharedPtr(new DescriptorSet(pPool, layout));
    }

    DescriptorSet::DescriptorSet(DescriptorPool::SharedPtr pPool, const Layout& layout)
        : mpPool(pPool)
        , mLayout(layout)
    {
        apiInit();
    }

    DescriptorSet::~DescriptorSet()
    {
        mpPool->releaseAllocation(mpApiData);
    }

    DescriptorSet::Layout& DescriptorSet::Layout::addRange(DescriptorSet::Type type, uint32_t baseRegIndex, uint32_t descriptorCount, uint32_t regSpace)
    {
        Range r;
        r.descCount = descriptorCount;
        r.baseRegIndex = baseRegIndex;
        r.regSpace = regSpace;
        r.type = type;

        mRanges.push_back(r);
        return *this;
    }
}
