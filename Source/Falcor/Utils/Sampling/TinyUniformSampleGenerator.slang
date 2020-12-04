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
import Utils.Math.HashUtils;
import Utils.Math.BitTricks;
import Utils.Sampling.Pseudorandom.LCG;
import Utils.Sampling.SampleGeneratorInterface;

/** Tiny uniform random sample generator.

    This generator has only 32 bit state and sub-optimal statistical properties.
    Do not use for anything critical; correlation artifacts may be prevalent.

    This code works on shader model 5.1 and above.
*/
struct TinyUniformSampleGenerator : ISampleGenerator
{
    struct Padded
    {
        TinyUniformSampleGenerator internal;
        uint3 _pad;
    };

    /** Create sample generator.
        \param[in] seed Seed value.
        \return Return a new sample generator.
    */
    static TinyUniformSampleGenerator create(uint seed)
    {
        TinyUniformSampleGenerator sampleGenerator;
        sampleGenerator.rng = createLCG(seed);
        return sampleGenerator;
    }

    /** Create sample generator for a given pixel and sample number.
        \param[in] pixel Pixel id.
        \param[in] sampleNumber Sample number.
        \return Return a new sample generator.
    */
    static TinyUniformSampleGenerator create(uint2 pixel, uint sampleNumber)
    {
        // Use block cipher to generate a pseudorandom initial seed.
        uint seed = blockCipherTEA(interleave_32bit(pixel), sampleNumber).x;
        return create(seed);
    }

    /** Returns the next sample value. This function updates the state.
    */
    [mutating] uint next()
    {
        return nextRandom(rng);
    }

    LCG rng;    ///< Simple LCG 32-bit pseudorandom number generator.
};
