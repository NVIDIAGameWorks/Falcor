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

/** Implements the alias method for sampling from a discrete probability distribution.
*/
struct AliasTable
{
    struct Item
    {
        uint threshold;
        uint indexA;
        uint indexB;
        uint _pad;

        float getThreshold() { return asfloat(threshold); }
        uint getIndexA() { return indexA; }
        uint getIndexB() { return indexB; }
    };

    StructuredBuffer<Item> items;       ///< List of items used for sampling.
    StructuredBuffer<float> weights;    ///< List of original weights.
    uint count;                         ///< Total number of weights in the table.
    float weightSum;                    ///< Total sum of all weights in the table.
    uint _pad[2];

    /** Sample from the table proportional to the weights.
        \param[in] index Uniform random index in [0..count).
        \param[in] rnd Uniform random number in [0..1).
        \return Returns the sampled item index.
    */
    uint sample(uint index, float rnd)
    {
        Item item = items[index];
        return rnd >= item.getThreshold() ? item.getIndexA() : item.getIndexB();
    }

    /** Sample from the table proportional to the weights.
        \param[in] rnd Two uniform random number in [0..1).
        \return Returns the sampled item index.
    */
    uint sample(float2 rnd)
    {
        uint index = min(count - 1, (uint)(rnd.x * count));
        return sample(index, rnd.y);
    }

    /** Get the original weight at a given index.
        \param[in] index Table index.
        \return Returns the original weight.
    */
    float getWeight(uint index)
    {
        return weights[index];
    }
};
