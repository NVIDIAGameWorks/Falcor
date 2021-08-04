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
import Utils.Math.BitTricks;

ByteAddressBuffer testData;
RWStructuredBuffer<uint> result;

uint pack(uint2 v)
{
    return (v.y << 16) | v.x;
}

[numthreads(256, 1, 1)]
void testBitInterleave(uint3 threadId : SV_DispatchThreadID)
{
    const uint idx = threadId.x;
    const uint data = testData.Load(idx * 4);               // data = [ y15..y0 x15..x0 ]

    // Use hi/lo words of each dword in the test set as input:
    const uint2 bits = uint2(data & 0xffff, data >> 16);    // bits.x = [ 0..0 x15..x0 ], bits.y = [ 0..0 y15..y0 ]
    const uint interleavedBits = interleave_32bit(bits);    // interleavedBits = [y15 x15 .. y0 x0]

    // Test bit interleave.
    // Put bogus data in the unused bits to test that it doesn't change the result.
    result[5 * idx + 0] = interleave_32bit(bits ^ 0xbeef0000);
    result[5 * idx + 1] = interleave_16bit(bits ^ 0xbeefed00);

    // Test bit de-interleave.
    // These functions expect zeros for the unused bits.
    result[5 * idx + 2] = pack(deinterleave_16bit(interleavedBits & 0x0000ffff));
    result[5 * idx + 3] = pack(deinterleave_8bit(interleavedBits & 0x000000ff));

    // Test bit de-interleave of two sequences (bits 0-3 and bits 8-11).
    // Note: the result is placed in separate 16-bit words, so shift down by 8 bits.
    uint2 temp = deinterleave_2x8bit(interleavedBits & 0x00ff00ff);
    result[5 * idx + 4] = pack(((temp >> 8) & 0x0f00) | (temp & 0xf));
}
