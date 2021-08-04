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

/** Implementation of the xoshiro128** 32-bit all-purpose, rock-solid generator
    written in 2018 by David Blackman and Sebastiano Vigna (vigna@acm.org).
    The state is 128 bits and the period (2^128)-1. It has a jump function that
    allows you to skip ahead 2^64 in the seqeuence.

    Note: The state must be seeded so that it is not everywhere zero.    
    The recommendation is to initialize the state using SplitMix64.

    See the original public domain code: http://xoshiro.di.unimi.it/xoshiro128starstar.c
*/

struct Xoshiro128StarStar
{
    uint state[4];
};

uint rotl(const uint x, int k)
{
    return (x << k) | (x >> (32 - k));
}

/** Generates the next pseudorandom number in the sequence (32 bits).
*/
uint nextRandom(inout Xoshiro128StarStar rng)
{
    const uint32_t result_starstar = rotl(rng.state[0] * 5, 7) * 9;
	const uint32_t t = rng.state[1] << 9;

    rng.state[2] ^= rng.state[0];
    rng.state[3] ^= rng.state[1];
    rng.state[1] ^= rng.state[2];
    rng.state[0] ^= rng.state[3];

    rng.state[2] ^= t;
    rng.state[3] = rotl(rng.state[3], 11);

	return result_starstar;
}

/** Jump function for the generator. It is equivalent to 2^64 calls to nextRandom().
    It can be used to generate 2^64 non-overlapping subsequences for parallel computations.
*/
void jump(inout Xoshiro128StarStar rng)
{
    static const uint32_t JUMP[] = { 0x8764000b, 0xf542d2d3, 0x6fa035c3, 0x77f2db5b };

    uint32_t s0 = 0;
    uint32_t s1 = 0;
    uint32_t s2 = 0;
    uint32_t s3 = 0;

    for (int i = 0; i < 4; i++)
    {
        for (int b = 0; b < 32; b++)
        {
            if (JUMP[i] & (1u << b))
            {
                s0 ^= rng.state[0];
                s1 ^= rng.state[1];
                s2 ^= rng.state[2];
                s3 ^= rng.state[3];
            }
            nextRandom(rng);
        }
    }

    rng.state[0] = s0;
    rng.state[1] = s1;
    rng.state[2] = s2;
    rng.state[3] = s3;
}

/** Initialize Xoshiro128StarStar pseudorandom number generator.
    The initial state should be pseudorandom and must not be zero everywhere.
    It is recommended to use SplitMix64 for creating the initial state.
    \param[in] s Array of 4x 32-bit values of initial state (seed).
*/
Xoshiro128StarStar createXoshiro128StarStar(uint s[4])
{
    Xoshiro128StarStar rng;
    rng.state[0] = s[0];
    rng.state[1] = s[1];
    rng.state[2] = s[2];
    rng.state[3] = s[3];
    return rng;
}
