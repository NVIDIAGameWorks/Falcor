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
import Utils.Sampling.SampleGenerator;

cbuffer CB
{
    uint3   gDispatchDim;
    uint    gDimensions;
}

RWStructuredBuffer<float> result;

[numthreads(16, 16, 1)]
void test(uint3 threadId : SV_DispatchThreadID)
{
    if (any(threadId >= gDispatchDim)) return;

    // Create sample generator.
    SampleGenerator sg = SampleGenerator.create(threadId.xy, threadId.z);

    // Generate samples.
    // The output is a 4D tensor of samples, stored in memory as Z instances
    // of 2D tiles XY stored in scanline order, where XYZ is the dispatch dimensions.
    // Each element consists of an N-dimensional sample.
    const uint pixelIdx = threadId.y * gDispatchDim.x + threadId.x;
    const uint tileOffset = threadId.z * (gDispatchDim.x * gDispatchDim.y);
    const uint offset = (tileOffset + pixelIdx) * gDimensions;

    for (uint i = 0; i < gDimensions; i += 8)
    {
        float u0 = sampleNext1D(sg);
        float u1 = sampleNext1D(sg);
        float u2 = sampleNext1D(sg);

        result[offset + i + 0] = u0;
        result[offset + i + 1] = u1;
        result[offset + i + 2] = u2;

        float2 v2 = sampleNext2D(sg);

        result[offset + i + 3] = v2.x;
        result[offset + i + 4] = v2.y;

        float3 v3 = sampleNext3D(sg);

        result[offset + i + 5] = v3.x;
        result[offset + i + 6] = v3.y;
        result[offset + i + 7] = v3.z;
    }
}
