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
RWStructuredBuffer<uint> result;
RWStructuredBuffer<float> difference;

Texture2D<float4> tex;
Texture2D<float4> ref;

cbuffer CB
{
    uint2 sz;
}

[numthreads(16, 16, 1)]
void readback(uint3 threadId : SV_DispatchThreadID)
{
    if (any(threadId.xy >= sz)) return;

    uint idx = threadId.y * sz.x + threadId.x;
    float4 v = tex[threadId.xy];

    // Convert to uint8_t, and pack into uint32_t
    result[idx] = ((uint)(saturate(v[3]) * 255) << 24) |
                  ((uint)(saturate(v[2]) * 255) << 16) |
                  ((uint)(saturate(v[1]) * 255) <<  8) |
                  ((uint)(saturate(v[0]) * 255));
}

[numthreads(16, 16, 1)]
void diff(uint3 threadId : SV_DispatchThreadID)
{
    if (any(threadId.xy >= sz)) return;

    uint idx = threadId.y * sz.x + threadId.x;
    float4 tv = tex[threadId.xy];
    float4 rv = tex[threadId.xy];
    for (uint32_t i = 0; i < 4; ++i)
    {
        difference[4 * idx + i] = rv[i] - tv[i];
    }
}

