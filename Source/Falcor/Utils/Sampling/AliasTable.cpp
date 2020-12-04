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
#include "AliasTable.h"

namespace Falcor
{
    AliasTable::SharedPtr AliasTable::create(std::vector<float> weights, std::mt19937& rng)
    {
        return SharedPtr(new AliasTable(std::move(weights), rng));
    }

    void AliasTable::setShaderData(const ShaderVar& var) const
    {
        var["items"] = mpItems;
        var["weights"] = mpWeights;
        var["count"] = mCount;
        var["weightSum"] = (float)mWeightSum;
    }

    AliasTable::AliasTable(std::vector<float> weights, std::mt19937& rng)
        : mCount((uint32_t)weights.size())
    {
        if (weights.size() > std::numeric_limits<uint32_t>::max()) throw std::exception("Too many entries for alias table.");

        std::uniform_int_distribution<uint32_t> rngDist;

        mpWeights = Buffer::createStructured(sizeof(float), mCount, Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, weights.data());

        mWeightSum = 0.0;
        for (float f : weights) mWeightSum += f;

        double factor = mCount / mWeightSum;
        for (float& f : weights) f = (float)(f * factor);

        std::vector<uint32_t> permutation(mCount);
        for (uint32_t i = 0; i < mCount; ++i) permutation[i] = i;
        std::sort(permutation.begin(), permutation.end(), [&](uint32_t a, uint32_t b) { return weights[a] < weights[b]; });

        std::vector<float> thresholds(mCount);
        std::vector<uint32_t> redirect(mCount);
        std::vector<uint2> mergedTable(mCount);

        uint32_t head = 0;
        uint32_t tail = mCount - 1;

        while (head != tail)
        {
            int i = permutation[head];
            int j = permutation[tail];

            thresholds[i] = weights[i];
            redirect[i] = j;
            weights[j] -= 1.f - weights[i];

            if (head == tail - 1)
            {
                thresholds[j] = 1.f;
                redirect[j] = j;
                break;
            }
            else if (weights[j] < 1.f)
            {
                std::swap(permutation[head], permutation[tail]);
                tail--;
            }
            else
            {
                head++;
            }
        }

        for (uint32_t i = 0; i < mCount; ++i)
        {
            permutation[i] = i;
        }

        for (uint32_t i = 0; i < mCount; ++i)
        {
            uint32_t dst = i + (rngDist(rng) % (mCount - i));
            std::swap(thresholds[i], thresholds[dst]);
            std::swap(redirect[i], redirect[dst]);
            std::swap(permutation[i], permutation[dst]);
        }

        struct Item
        {
            float threshold;
            uint32_t indexA;
            uint32_t indexB;
            uint32_t _pad;
        };

        std::vector<Item> items(mCount);
        for (uint32_t i = 0; i < mCount; ++i)
        {
            items[i] = { thresholds[i], redirect[i], permutation[i], 0 };
        }

        mpItems = Buffer::createStructured(sizeof(Item), mCount, Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, items.data());
    }
}
