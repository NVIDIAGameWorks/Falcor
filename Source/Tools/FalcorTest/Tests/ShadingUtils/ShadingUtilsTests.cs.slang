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
import Utils.Helpers;

RWStructuredBuffer<float> result;

StructuredBuffer<float3> sphereCenter;
StructuredBuffer<float> sphereRadius;
StructuredBuffer<float3> rayOrigin;
StructuredBuffer<float3> rayDir;
RWStructuredBuffer<uint> isectResult;
RWStructuredBuffer<float3> isectLoc;

cbuffer TestCB
{
    int resultSize;
};

[numthreads(1, 1, 1)]
void testRadicalInverse()
{
    for (int i = 0; i < resultSize; ++i)
    {
        result[i] = radicalInverse(i);
    }
}

[numthreads(1024, 1, 1)]
void testRand(uint groupIndex : SV_GroupIndex)
{
    uint state = rand_init(0xd00f1337, groupIndex);
    /* We intentionally have a relatively small number of threads, each of
       which generates a bunch of random numbers, rather than having a
       separate thread for each one.  Doing so lets us have a large number
       of successive samples from a relatively small number of random
       contxts; otherwise, we'd basically only be testing the quality of
       the initial seeding.
    */
    int n = resultSize / 1024;
    for (int i = 0; i < n; ++i)
    {
        result[n * groupIndex + i] = rand_next(state);
    }
}

[numthreads(1024, 1, 1)]
void testSphericalCoordinates(uint3 threadId : SV_DispatchThreadID)
{
    uint state = rand_init(0xd00f1337, threadId.x);
    float3 dir = float3(-1 + 2 * rand_next(state),
                        -1 + 2 * rand_next(state),
                        -1 + 2 * rand_next(state));
    float2 uv = dirToSphericalCrd(dir);
    result[threadId.x] = dot(normalize(dir), sphericalCrdToDir(uv));
}

[numthreads(1, 1, 1)]
void testRaySphereIntersection()
{
    for (int i = 0; i < resultSize; ++i)
    {
        if (intersectRaySphere(rayOrigin[i], rayDir[i], sphereCenter[i], sphereRadius[i], isectLoc[i]))
        {
            isectResult[i] = 1;
        }
        else
        {
            isectResult[i] = 0;
        }
    }
}
