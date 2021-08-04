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
RWStructuredBuffer<uint4> result;
RWBuffer<uint> laneCount;
Buffer<uint> testData;

[numthreads(1, 1, 1)]
void testWaveGetLaneCount(uint3 threadID : SV_DispatchThreadID)
{
    laneCount[0] = WaveGetLaneCount();
}

[numthreads(256, 1, 1)]
void testWaveMatch(uint3 threadID : SV_DispatchThreadID)
{
    const uint i = threadID.x;
    if (i == 0) laneCount[0] = WaveGetLaneCount();

    uint value = testData[i];
    result[i] = WaveMatch(value);
}

[numthreads(256, 1, 1)]
void testWaveMinMax(uint3 threadID : SV_DispatchThreadID)
{
    const uint i = threadID.x;
    if (i == 0) laneCount[0] = WaveGetLaneCount();

    float value = asfloat(testData[i]);

#if CONDITIONAL == 0
    float minVal = WaveActiveMin(value);
    float maxVal = WaveActiveMax(value);
#else
    float minVal = 0.f;
    float maxVal = 0.f;

    if (value - floor(value) < 0.5f)
    {
        minVal = WaveActiveMin(value);
        maxVal = WaveActiveMax(value);
    }
#endif

    result[2 * i + 0] = asuint(minVal);
    result[2 * i + 1] = asuint(maxVal);
}

[numthreads(32, 1, 1)]
void testWaveMaxSimpleFloat(uint3 threadID : SV_DispatchThreadID)
{
    const uint i = threadID.x;

    float value = asfloat(testData[i]);
    if (value <= -2.f) value = WaveActiveMax(value);

    result[i] = asuint(value);
}

[numthreads(32, 1, 1)]
void testWaveMaxSimpleInt(uint3 threadID : SV_DispatchThreadID)
{
    const uint i = threadID.x;

    int value = (int)testData[i];
    if (value <= -2) value = WaveActiveMax(value);

    result[i] = (uint)value;
}
