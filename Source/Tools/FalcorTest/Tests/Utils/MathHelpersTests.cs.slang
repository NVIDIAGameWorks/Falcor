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
import Utils.Math.MathHelpers;
import Utils.Sampling.TinyUniformSampleGenerator;

RWStructuredBuffer<float> result;
StructuredBuffer<float> input;

[numthreads(1024, 1, 1)]
void testSphericalCoordinates(uint3 threadId: SV_DispatchThreadID)
{
    TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(uint2(0), threadId.x);
    float3 dir = normalize(sampleNext3D(sg) * 2.f - 1.f);
    float2 uv = world_to_latlong_map(dir);
    result[threadId.x] = dot(normalize(dir), latlong_map_to_world(uv));
}

[numthreads(1024, 1, 1)]
void testSphericalCoordinatesRad(uint3 threadId: SV_DispatchThreadID)
{
    TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(uint2(0), threadId.x);
    float3 dir = normalize(sampleNext3D(sg) * 2.f - 1.f);
    float2 sph = cartesian_to_spherical_rad(dir);
    result[threadId.x] = dot(normalize(dir), spherical_to_cartesian_rad(sph));
}

[numthreads(25, 1, 1)]
void testErrorFunction(uint3 threadId: SV_DispatchThreadID)
{
    result[threadId.x] = erf(input[threadId.x]);
}

[numthreads(25, 1, 1)]
void testInverseErrorFunction(uint3 threadId: SV_DispatchThreadID)
{
    result[threadId.x] = erfinv(input[threadId.x]);
}
