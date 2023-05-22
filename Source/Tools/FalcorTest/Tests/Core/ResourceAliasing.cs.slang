/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
RWStructuredBuffer<float> result;

struct S
{
    float d[8];
};

// These descriptors alias the same raw buffer.
StructuredBuffer<float> bufA1;
StructuredBuffer<float4> bufA2;
StructuredBuffer<S> bufA3;

// These descriptors alias the same raw buffer.
RWStructuredBuffer<float> bufB1;
RWStructuredBuffer<float4> bufB2;
RWStructuredBuffer<S> bufB3;

// These descriptors alias the same structured buffer.
StructuredBuffer<uint> bufStruct1;
StructuredBuffer<uint4> bufStruct2;
StructuredBuffer<S> bufStruct3;

[numthreads(32, 1, 1)]
void testRead(uint3 threadId: SV_DispatchThreadID)
{
    uint i = threadId.x;
    result[i] = bufA1[i];
    result[i + 32] = bufA2[i / 4][i % 4];
    result[i + 64] = bufA3[i / 8].d[i % 8];
}

[numthreads(32, 1, 1)]
void testReadWrite(uint3 threadId: SV_DispatchThreadID)
{
    uint i = threadId.x;
    bufB1[i] = (float)(32 - i);
    bufB2[i / 4 + 8][i % 4] = (float)(32 - i);
    bufB3[i / 8 + 8].d[i % 8] = (float)(32 - i);
}

[numthreads(32, 1, 1)]
void testStructRead(uint3 threadId: SV_DispatchThreadID)
{
    uint i = threadId.x;
    result[i] = asfloat(bufStruct1[i]);
    result[i + 32] = asfloat(bufStruct2[i / 4][i % 4]);
    result[i + 64] = bufStruct3[i / 8].d[i % 8];
}
