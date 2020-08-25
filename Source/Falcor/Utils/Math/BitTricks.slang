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

/** Utility functions for Morton codes.
    This is using the usual bit twiddling. See e.g.: https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/

    The interleave functions are named based to their output size in bits.
    The deinterleave functions are named based on their input size in bits.
    So, deinterleave_16bit(interleave_16bit(x)) == x should hold true.

    TODO: Make this a host/device shared header, ensure code compiles on the host.
    TODO: Add optimized 8-bit and 2x8-bit interleaving functions.
    TODO: Use NvApi intrinsics to optimize the code on NV.
*/

/** 32-bit bit interleave (Morton code).
    \param[in] v 16-bit values in the LSBs of each component (higher bits don't matter).
    \return 32-bit value.
*/
uint interleave_32bit(uint2 v)
{
    uint x = v.x & 0x0000ffff;              // x = ---- ---- ---- ---- fedc ba98 7654 3210
    uint y = v.y & 0x0000ffff;

    x = (x | (x << 8)) & 0x00FF00FF;        // x = ---- ---- fedc ba98 ---- ---- 7654 3210
    x = (x | (x << 4)) & 0x0F0F0F0F;        // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
    x = (x | (x << 2)) & 0x33333333;        // x = --fe --dc --ba --98 --76 --54 --32 --10
    x = (x | (x << 1)) & 0x55555555;        // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0

    y = (y | (y << 8)) & 0x00FF00FF;
    y = (y | (y << 4)) & 0x0F0F0F0F;
    y = (y | (y << 2)) & 0x33333333;
    y = (y | (y << 1)) & 0x55555555;

    return x | (y << 1);
}

/** 16-bit bit interleave (Morton code).
    \param[in] v 8-bit values in the LSBs of each component (higher bits don't matter).
    \return 16-bit value in the lower word, 0 elsewhere.
*/
uint interleave_16bit(uint2 v)
{
    v &= 0xff;
    uint j = (v.y << 16) | v.x;             // j = ---- ---- (   y   ) ---- ---- (   x   )
                                            // j = ---- ---- fedc ba98 ---- ---- 7654 3210
    j = (j ^ (j << 4)) & 0x0f0f0f0f;        // j = ---- fedc ---- ba98 ---- 7654 ---- 3210
    j = (j ^ (j << 2)) & 0x33333333;        // j = --fe --dc --ba --98 --76 --54 --32 --10
    j = (j ^ (j << 1)) & 0x55555555;        // j = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    return (j >> 15) | (j & 0xffff);        // j = ---- ---- ---- ---- f7e6 d5c4 b3a2 9180
}

/** 16-bit bit de-interleave (inverse Morton code).
    \param[in] i 16-bit value in lower word, must be 0 elsewhere.
    \return 8-bit values in the LSBs of each component, 0 elsewhere.
*/
uint2 deinterleave_16bit(uint i)
{
    uint j = ((i >> 1) << 16) | i;          // j = -(     i >> 1     ) (         i       )
    j &= 0x55555555;                        // j = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    j = (j ^ (j >> 1)) & 0x33333333;        // j = --fe --dc --ba --98 --76 --54 --32 --10
    j = (j ^ (j >> 2)) & 0x0f0f0f0f;        // j = ---- fedc ---- ba98 ---- 7654 ---- 3210
    j = (j ^ (j >> 4)) & 0x00ff00ff;        // j = ---- ---- fedc ba98 ---- ---- 7654 3210
    return uint2(j & 0xff, j >> 16);        // x = ---- ---- ---- ---- ---- ---- 7654 3210
                                            // y = ---- ---- ---- ---- ---- ---- fedc ba98
}

/** 8-bit bit de-interleave (inverse Morton code).
    Note: This function has almost exactly the same cost as deinterleave_2x8bit, use the latter if multiple values should be de-interleaved.
    \param[in] i 8-bit value in lower word, must be 0 elsewhere.
    \return 4-bit values in the LSBs of each component, 0 elsewhere.
*/
uint2 deinterleave_8bit(uint i)
{
    uint j = ((i >> 1) << 8) | i;           // j = ---- ---- ---- ---- -(i >> 1) (   i   )
    j &= 0x00005555;                        // j = ---- ---- ---- ---- -7-6 -5-4 -3-2 -1-0
    j = (j ^ (j >> 1)) & 0x33333333;        // j = ---- ---- ---- ---- --76 --54 --32 --10
    j = (j ^ (j >> 2)) & 0x0f0f0f0f;        // j = ---- ---- ---- ---- ---- 7654 ---- 3210
    return uint2(j & 0xf, j >> 8);          // x = ---- ---- ---- ---- ---- ---- ---- 3210
                                            // y = ---- ---- ---- ---- ---- ---- ---- 7654
}

/** 2x 8-bit bit de-interleave (inverse Morton code).
    \param[in] i 8-bit values in the LSBs of each 16-bit word, must be 0 elsewhere.
    \return 4-bit values in each component in the LSBs of each 16-bit word, 0 elsewhere.
*/
uint2 deinterleave_2x8bit(uint i)
{
    uint j = ((i & ~0x00010001) << 7) | i;  // j = -(i1 >> 1)(  i1   ) -(i0 >> 1)(  i0   )
    j &= 0x55555555;                        // j = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    j = (j ^ (j >> 1)) & 0x33333333;        // j = --fe --dc --ba --98 --76 --54 --32 --10
    j = (j ^ (j >> 2)) & 0x0f0f0f0f;        // j = ---- fedc ---- ba98 ---- 7654 ---- 3210                                            
    return uint2(j, j >> 8) & 0x000f000f;   // x = ---- ---- ---- ba98 ---- ---- ---- 3210
                                            // y = ---- ---- ---- fedc ---- ---- ---- 7654
}
