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

/** Unit tests for RMW operations on different buffer types.

    The updateBuffer() kernel is run multiple times followed by readBuffer()
    that copies the result to the output buffer.
*/

#if TYPE == 0
RWByteAddressBuffer buffer;
#elif TYPE == 1
RWBuffer<uint> buffer;
#elif TYPE == 2
RWStructuredBuffer<uint> buffer;
#endif

RWStructuredBuffer<uint> result;

// Clear kernel.
// This is necessary for RWStructuredBuffer as we can't use ID3D12CommandList::ClearUnorderedAccessViewUint for structured buffers.
[numthreads(256, 1, 1)]
void clearBuffer(uint3 threadId : SV_DispatchThreadID)
{
    uint i = threadId.x;
#if TYPE == 0
    buffer.Store(i * 4, 0);
#else
    buffer[i] = 0;
#endif
}

[numthreads(256, 1, 1)]
void updateBuffer(uint3 threadId : SV_DispatchThreadID)
{
    uint i = threadId.x;
#if TYPE == 0
    uint val = buffer.Load(i * 4);
    val += i + 1;
    buffer.Store(i * 4, val);
#else
    buffer[i] += i + 1;
#endif
}

[numthreads(256, 1, 1)]
void readBuffer(uint3 threadId : SV_DispatchThreadID)
{
    uint i = threadId.x;
#if TYPE == 0
    result[i] = buffer.Load(i * 4);
#else
    result[i] = buffer[i];
#endif
}
