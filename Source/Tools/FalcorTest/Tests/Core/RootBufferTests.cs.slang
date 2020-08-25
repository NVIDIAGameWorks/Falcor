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
import Utils.Attributes;

/** Unit tests for binding buffers using root descriptors.
*/

cbuffer CB
{
    uint c0;
    float c1;
}

struct S
{
    float a;
    uint b;
};

// Place the root buffer in the middle of some other buffers to ensure binding the root buffer
// doesn't intefere with the bindings before or after.
RWStructuredBuffer<float> result;
ByteAddressBuffer rawBuffer;
StructuredBuffer<S> structBuffer;

#if USE_UAV
[root] RWByteAddressBuffer testBuffer;      // Buffer bound to root descriptor (UAV)
#else
[root] ByteAddressBuffer testBuffer;        // Buffer bound to root descriptor (SRV)
#endif

RWBuffer<uint> typedBufferUint;
Buffer<float4> typedBufferFloat4;

[numthreads(256, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    const uint i = threadId.x;

    // Add data read from the different buffers together.
    // The values are multiplied by different constants to ensure that aliasing of the buffers to the wrong variables doesn't pass.
    float r = 0.f;
    r += c0;
    r += c1;
    r += rawBuffer.Load(i * 4);
    r += typedBufferUint[i] * 2;
    r += typedBufferFloat4[i].z * 3;
    r += structBuffer[i].a * 4;
    r += structBuffer[i].b * 5;
    r += testBuffer.Load(i * 4) * 6;

    result[i] = r;
}
