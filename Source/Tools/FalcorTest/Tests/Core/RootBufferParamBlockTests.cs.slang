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
import ParamBlockDefinition;

// Global scope root and non-root resources
RWStructuredBuffer<float> result;
Buffer<uint> globalBufA;

#if USE_UAV
[root] RWByteAddressBuffer globalTestBuffer;    // Buffer bound to root descriptor (UAV)
#else
[root] ByteAddressBuffer globalTestBuffer;      // Buffer bound to root descriptor (SRV)
#endif

// Parameter block containing both root and non-root resources
ParameterBlock<TestBlock> gParamBlock;

[numthreads(256, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    const uint i = threadId.x;

    // Add data read from the different buffers in the parameter block.
    // The values are multiplied by different constants to ensure that aliasing of the buffers to the wrong variables doesn't pass.
    float r = 0.f;
    r += gParamBlock.c0;
    r += gParamBlock.bufA[0].Load(i * 4);
    r += gParamBlock.bufA[1].Load(i * 4) * 2;
    r += gParamBlock.bufB[0][i] * 3;
    r += gParamBlock.bufB[1][i] * 4;
    r += gParamBlock.bufB[2][i] * 5;
    r += gParamBlock.bufC[0][i] * 6;
    r += gParamBlock.bufC[1][i] * 7;
    r += gParamBlock.bufC[2][i] * 8;
    r += gParamBlock.bufC[3][i] * 9;
    r += gParamBlock.testBuffer.Load(i * 4) * 10;

    // Add data read from buffers bound at the global scope.
    r += globalBufA[i] * 11;
    r += globalTestBuffer.Load(i * 4) * 12;

    result[i] = r;
}
