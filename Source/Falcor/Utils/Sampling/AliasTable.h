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
#include "Core/Macros.h"
#include "Core/API/Buffer.h"
#include "Core/Program/ShaderVar.h"
#include <memory>
#include <random>

namespace Falcor
{
/**
 * Implements the alias method for sampling from a discrete probability distribution.
 */
class FALCOR_API AliasTable
{
public:
    /**
     * Create an alias table.
     * The weights don't need to be normalized to sum up to 1.
     * @param[in] pDevice GPU device.
     * @param[in] weights The weights we'd like to sample each entry proportional to.
     * @param[in] rng The random number generator to use when creating the table.
     */
    AliasTable(ref<Device> pDevice, std::vector<float> weights, std::mt19937& rng);

    /**
     * Bind the alias table data to a given shader var.
     * @param[in] var The shader variable to set the data into.
     */
    void bindShaderData(const ShaderVar& var) const;

    /**
     * Get the number of weights in the table.
     */
    uint32_t getCount() const { return mCount; }

    /**
     * Get the total sum of all weights in the table.
     */
    double getWeightSum() const { return mWeightSum; }

private:
    // Item structure for the mpItems buffer.
    struct Item
    {
        float threshold; ///< If rand() < threshold, pick indexB (else pick indexA)
        uint32_t indexA; ///< The "redirect" index, if uniform sampling would overweight indexB.
        uint32_t indexB; ///< The original / permutation index, sampled uniformly in [0...mCount-1]
        uint32_t _pad;
    };

    uint32_t mCount;       ///< Number of items in the alias table.
    double mWeightSum;     ///< Total weight of all elements used to create the alias table.
    ref<Buffer> mpItems;   ///< Buffer containing table items.
    ref<Buffer> mpWeights; ///< Buffer containing item weights.
};
} // namespace Falcor
