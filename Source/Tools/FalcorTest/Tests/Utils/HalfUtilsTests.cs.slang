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
import Utils.Math.HalfUtils;

cbuffer CB
{
    uint testSize;
}

StructuredBuffer<float> inputFloat;
StructuredBuffer<uint> inputUint;

RWStructuredBuffer<uint> resultUint;
RWStructuredBuffer<float> resultFloat;

/** Converts a sequence of fp32 to fp16 using default rounding.
*/
[numthreads(256, 1, 1)]
void testFP32ToFP16(uint3 threadId : SV_DispatchThreadID)
{
    uint i = threadId.x;
    if (i >= testSize) return;
    resultUint[i] = f32tof16(inputFloat[i]);
}

/** Converts a sequence of fp16 numbers to fp32.
*/
[numthreads(256, 1, 1)]
void testFP16ToFP32(uint3 threadId : SV_DispatchThreadID)
{
    uint i = threadId.x;
    if (i >= testSize) return;
    resultFloat[i] = f16tof32(inputUint[i]);
}

/** Converts a sequence of fp32 numbers to fp16 using conservative rounding up/down.
*/
[numthreads(256, 1, 1)]
void testFP32ToFP16ConservativeRounding(uint3 threadId : SV_DispatchThreadID)
{
    uint i = threadId.x;
    if (i >= testSize) return;

    float v = inputFloat[i];
    resultUint[2 * i + 0] = f32tof16_roundUp(v);
    resultUint[2 * i + 1] = f32tof16_roundDown(v);
}

#if 0
// TODO: Currently disabled until we figure out the rounding modes and have a matching CPU library.
[numthreads(256, 1, 1)]
void testFP16RoundingMode(uint3 threadId : SV_DispatchThreadID)
{
    uint i = threadId.x;
    if (i >= testSize) return;    
    resultFloat[i] = f16tof32(f32tof16(inputFloat[i])); // TODO: Use precise keyword when available.
}
#endif
