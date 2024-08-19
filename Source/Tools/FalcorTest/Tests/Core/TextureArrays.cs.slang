/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
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
#define I (1 << BITS_I)
#define J (1 << BITS_J)
#define K (1 << BITS_K)

RWStructuredBuffer<float> result;
RWTexture2D<float> tex[I][J][K];

[numthreads(16, 16, 1)]
void testWrite(uint3 threadId: SV_DispatchThreadID)
{
    if (threadId.x >= 16 || threadId.y >= 16 || threadId.z >= 8)
        return;

    uint i = threadId.z & ((I)-1);
    uint j = (threadId.z >> (BITS_I)) & ((J)-1);
    uint k = (threadId.z >> ((BITS_I) + (BITS_J))) & ((K)-1);
    uint2 p = threadId.xy;

    tex[i][j][k][p] = (float)threadId.x * threadId.y + threadId.z;
}

[numthreads(16, 16, 1)]
void testRead(uint3 threadId: SV_DispatchThreadID)
{
    if (threadId.x >= 16 || threadId.y >= 16 || threadId.z >= 8)
        return;

    uint i = threadId.z & ((I)-1);
    uint j = (threadId.z >> (BITS_I)) & ((J)-1);
    uint k = (threadId.z >> ((BITS_I) + (BITS_J))) & ((K)-1);
    uint2 p = threadId.xy;

    uint idx = threadId.z * 256 + threadId.y * 16 + threadId.x;
    result[idx] = tex[i][j][k][p];
}
