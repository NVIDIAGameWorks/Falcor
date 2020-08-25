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
import Utils.Math.MathHelpers;

Buffer<float4> spheres;   // xyz: center, w: radius
RWStructuredBuffer<float> sinTheta, cosTheta;

[numthreads(1, 1, 1)]
void testBoundSphereAngle(uint3 threadId : SV_DispatchThreadID)
{
    const uint index = threadId.x;
    float sqrRadius = spheres[index].w * spheres[index].w;
    boundSphereSubtendedConeAngle(spheres[index].xyz, sqrRadius,
                                  sinTheta[index], cosTheta[index]);
}

Buffer<float3> origin, aabbMin, aabbMax;
RWStructuredBuffer<float3> coneDir;

[numthreads(1, 1, 1)]
void testBoundingConeAngleAverage(uint3 threadId : SV_DispatchThreadID)
{
    const uint index = threadId.x;
    boundBoxSubtendedConeAngleAverage(origin[index], aabbMin[index], aabbMax[index],
                                      coneDir[index], sinTheta[index], cosTheta[index]);
}

[numthreads(1, 1, 1)]
void testBoundingConeAngleCenter(uint3 threadId : SV_DispatchThreadID)
{
    const uint index = threadId.x;
    boundBoxSubtendedConeAngleCenter(origin[index], aabbMin[index], aabbMax[index],
                                     coneDir[index], sinTheta[index], cosTheta[index]);
}
