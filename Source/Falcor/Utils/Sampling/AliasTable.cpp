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
#include "AliasTable.h"
#include "Core/Errors.h"

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

    // This builds an alias table via the O(N) algorithm from Vose 1991, "A linear algorithm for generating random
    // numbers with a given distribution," IEEE Transactions on Software Engineering 17(9), 972-975.
    //
    // Basic idea:  creating each alias table entry combines one overweighted sample and one underweighted sample
    // into one alias table entry plus a residual sample (the overweighted sample minus some of its weight).
    //
    // By first separating all inputs into 2 temporary buffer (one overweighted set, with weights above the
    // average; one underweighted set, with weights below average), we can simply walk through the lists once,
    // merging the first elements in each temporary buffer.  The residual sample is interted into either the
    // overweighted or underweighted set, depending on its residual weight.
    //
    // The main complexity is dealing with corner cases, thanks to numerical precision issues, where you don't
    // have 2 valid entries to combine.  By definition, in these corner cases, all remaining unhandled samples
    // actually have the average weight (within numerical precision limits)
    AliasTable::AliasTable(std::vector<float> weights, std::mt19937& rng)
        : mCount((uint32_t)weights.size())
    {
        // Use >= since we reserve 0xFFFFFFFFu as an invalid flag marker during construction.
        if (weights.size() >= std::numeric_limits<uint32_t>::max()) throw RuntimeError("Too many entries for alias table.");

        std::uniform_int_distribution<uint32_t> rngDist;

        mpWeights = Buffer::createStructured(sizeof(float), mCount, Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, weights.data());

        // Our working set / intermediate buffers (underweight & overweight); initialize to "invalid"
        std::vector<uint32_t> lowIdx(mCount, 0xFFFFFFFFu);
        std::vector<uint32_t> highIdx(mCount, 0xFFFFFFFFu);

        // Sum element weights, use double to minimize precision issues
        mWeightSum = 0.0;
        for (float f : weights) mWeightSum += f;

        // Find the average weight
        float avgWeight = float(mWeightSum / double(mCount));

        // Initialize working set. Inset inputs into our lists of above-average or below-average weight elements.
        int lowCount = 0;
        int highCount = 0;
        for (uint32_t i = 0; i < mCount; ++i)
        {
            if (weights[i] < avgWeight)
                lowIdx[lowCount++] = i;
            else
                highIdx[highCount++] = i;
        }

        // Create alias table entries by merging above- and below-average samples
        std::vector<AliasTable::Item> items(mCount);
        for (uint32_t i = 0; i < mCount; ++i)
        {
            // Usual case:  We have an above-average and below-average sample we can combine into one alias table entry
            if ((lowIdx[i] != 0xFFFFFFFFu) && (highIdx[i] != 0xFFFFFFFFu))
            {
                // Create an alias table tuple:
                items[i] = { weights[lowIdx[i]] / avgWeight, highIdx[i], lowIdx[i], 0 };

                // We've removed some weight from element highIdx[i]; update it's weight, then re-enter it
                // on the end of either the above-average or below-average lists.
                float updatedWeight = (weights[lowIdx[i]] + weights[highIdx[i]]) - avgWeight;
                weights[highIdx[i]] = updatedWeight;
                if (updatedWeight < avgWeight)
                    lowIdx[lowCount++] = highIdx[i];
                else
                    highIdx[highCount++] = highIdx[i];
            }

            // The next two cases can only occur towards the end of table creation, because either:
            //    (a) all the remaining possible alias table entries have weight *exactly* equal to avgWeight,
            //        which means these alias table entries only have one input item that is selected
            //        with 100% probability
            //    (b) all the remaining alias table entires have *almost* avgWeight, but due to (compounding)
            //        precision issues throughout the process, they don't have *quite* that value.  In this case
            //        treating these entries as having exactly avgWeight (as in case (a)) is the only right
            //        thing to do mathematically (other than re-generating the alias table using higher precision
            //        or trying to reduce catasrophic numerical cancellation in the "updatedWeight" computation above).
            else if (highIdx[i] != 0xFFFFFFFFu)
            {
                items[i] = { 1.0f, highIdx[i], highIdx[i], 0 };
            }
            else if (lowIdx[i] != 0xFFFFFFFFu)
            {
                items[i] = { 1.0f, lowIdx[i], lowIdx[i], 0 };
            }

            // If there is neither a highIdx[i] or lowIdx[i] for some array element(s).  By construction,
            // this cannot occur (without some logic bug above).
            else
            {
                FALCOR_ASSERT(false); // Should not occur
            }
        }

        // TODO: We can simplify the alias table to implicitly store indexB (aka lowIdx[i]), so the AliasTable::Item
        // structure would be 1 float + 1 uint32_t, rather than 128 bits.  This, of course, would change usage in shaders
        // and elsewhere.  To do this, here you'd need to sort elements by indexB so that when looking up mpItems[j],
        // indexB==j.  This works since, by construction, only one element in the table has indexB==j (for any j
        // in [0...mCount-1]).  Alternatively, during the loop above, you could directly enter elements into the
        // correct location in the alias table.

        // Stash the alias table in our GPU buffer
        mpItems = Buffer::createStructured(sizeof(AliasTable::Item), mCount, Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None, items.data());
    }
}
