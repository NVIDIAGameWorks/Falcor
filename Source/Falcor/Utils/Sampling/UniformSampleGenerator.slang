/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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
__exported import Utils.Sampling.SampleGeneratorInterface;
import Utils.Sampling.Pseudorandom.Xoshiro;
import Utils.Sampling.Pseudorandom.SplitMix64;
import Utils.Math.BitTricks;

/** Default uniform pseudorandom number generator.

    This generator has 128 bit state and should have acceptable statistical
    properties for most rendering applications.

    This sample generator requires shader model 6.0 or above.
*/
public struct UniformSampleGenerator : ISampleGenerator
{
    struct Padded
    {
        UniformSampleGenerator internal;
    }

    /** Initializes the sample generator for a given pixel and sample number.
        \param[in] pixel Pixel id.
        \param[in] sampleNumber Sample number.
    */
    __init(uint2 pixel, uint sampleNumber)
    {
        UniformSampleGenerator sampleGenerator;

        // Use SplitMix64 generator to generate a good pseudorandom initial state.
        // The pixel coord is expected to be max 28 bits (16K^2 is the resource limit in D3D12).
        // The sample number is expected to be practically max ~28 bits, e.g. 16spp x 16M samples.
        // As long as both stay <= 32 bits, we will always have a unique initial seed.
        // This is however no guarantee that the generated sequences will never overlap,
        // but it is very unlikely. For example, with these most extreme parameters of
        // 2^56 sequences of length L, the probability of overlap is P(overlap) = L*2^-16.
        SplitMix64 rng = createSplitMix64(interleave_32bit(pixel), sampleNumber);
        uint64_t s0 = nextRandom64(rng);
        uint64_t s1 = nextRandom64(rng);
        uint seed[4] = { uint(s0), uint(s0 >> 32), uint(s1), uint(s1 >> 32) };

        // Create xoshiro128** pseudorandom generator.
        this.rng = createXoshiro128StarStar(seed);
    }

    /** Returns the next sample value. This function updates the state.
    */
    [mutating] uint next()
    {
        return nextRandom(rng);
    }

    Xoshiro128StarStar rng;
};
