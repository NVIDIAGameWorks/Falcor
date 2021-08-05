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

/** Simple linear congruential generator (LCG).

    The code uses the parameters from the book series "Numerical Recipes".
    The period is 2^32 and its state size is 32 bits.

    Note: Only for basic applications. The generator has poor statistical
    properties and is sensitive to good seeding. If many parallel generators
    are used (e.g. one per pixel) there will be significant correlation
    between the generated pseudorandom sequences. In those cases, it is
    recommended to use one of the generators with larger state.
*/

struct LCG
{
    uint state;
};

/** Generates the next pseudorandom number in the sequence (32 bits).
*/
uint nextRandom(inout LCG rng)
{
    const uint A = 1664525u;
    const uint C = 1013904223u;
    rng.state = (A * rng.state + C);
    return rng.state;
}

/** Initialize LCG pseudorandom number generator.
    \param[in] s0 Initial state (seed).
*/
LCG createLCG(uint s0)
{
    LCG rng;
    rng.state = s0;
    return rng;
}
