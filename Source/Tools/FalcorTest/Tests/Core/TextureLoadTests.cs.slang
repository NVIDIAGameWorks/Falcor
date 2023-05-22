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
RWStructuredBuffer<uint4> result;

Texture2D<float4> texUnorm;
Texture2D<uint4> texUnormAsUint;
Texture2D<uint4> texUint;

[numthreads(256, 1, 1)]
void testLoadFormat(uint3 threadId: SV_DispatchThreadID)
{
    float4 f = texUnorm[threadId.xy];
    uint4 u = texUnormAsUint[threadId.xy];
    uint4 v = texUint[threadId.xy];

    result[threadId.x] = { asuint(f.x), u.x, (uint)(f.x * 255.f), v.x };
}

[numthreads(1, 1, 1)]
void testLoadMips(uint3 threadId: SV_DispatchThreadID)
{
    result[0] = (uint4)(texUnorm.Load(int3(0, 0, 0)) * 255.f);
    result[1] = (uint4)(texUnorm.Load(int3(0, 0, 1)) * 255.f);
    result[2] = (uint4)(texUnorm.Load(int3(0, 0, 2)) * 255.f);
}
