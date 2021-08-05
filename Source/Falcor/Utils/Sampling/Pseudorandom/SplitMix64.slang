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

/** SplitMix64 pseudorandom number generator.

    This is a fixed-increment version of Java 8's SplittableRandom generator.
    The period is 2^64 and its state size is 64 bits.
    It is a very fast generator passing BigCrush. It is recommended for use with
    other generators like xoroshiro and xorshift to initialize their state arrays.
    
    Steele Jr, Guy L., Doug Lea, and Christine H. Flood., "Fast Splittable Pseudorandom Number Generators",
    ACM SIGPLAN Notices 49.10 (2014): 453-472. http://dx.doi.org/10.1145/2714064.2660195.

    This code requires shader model 6.0 or above for 64-bit integer support.
*/

struct SplitMix64
{
    uint64_t state;
};

uint64_t asuint64(uint lowbits, uint highbits)
{
    return (uint64_t(highbits) << 32) | uint64_t(lowbits);
}

/** Generates the next pseudorandom number in the sequence (64 bits).
*/
uint64_t nextRandom64(inout SplitMix64 rng)
{
    uint64_t z = (rng.state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

/** Generates the next pseudorandom number in the sequence (low 32 bits).
*/
uint nextRandom(inout SplitMix64 rng)
{
    return (uint)nextRandom64(rng);
}

/** Initialize SplitMix64 pseudorandom number generator.
    \param[in] s0 Low bits of initial state (seed).
    \param[in] s1 High bits of initial state (seed).
*/
SplitMix64 createSplitMix64(uint s0, uint s1)
{
    SplitMix64 rng;
    rng.state = asuint64(s0, s1);
    return rng;
}
