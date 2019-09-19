/***************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
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
***************************************************************************/
__exported import Helpers;
__exported import Shading;
import Scene;

shared RaytracingAccelerationStructure gRtScene : register(t52);
shared ByteAddressBuffer gAsToInstance          : register(t53);

shared cbuffer DxrPerFrame : register(b13)
{
    uint hitProgramCount;
};

cbuffer DxrPerGeometry : register(b14)
{
    uint geometryIndex;
};

/** Returns the global hit ID (= mesh instance ID in the scene).
    This function can only be called from a ray tracing hit program.
    \return Global hit ID.
*/
uint getGlobalHitID()
{
    uint address = (InstanceID() + geometryIndex) * 4;
    return gAsToInstance.Load(address);
}

/** Returns interpolated vertex attributes in a ray tracing hit program.
    \param[in] triangleIndex Index of the triangle in the current mesh (= PrimitiveIndex()).
    \param[in] attribs Intersection attributes provided by DXR.
    \return Interpolated vertex attributes.
*/
VertexData getVertexData(uint triangleIndex, BuiltInTriangleIntersectionAttributes attribs)
{
    float3 barycentrics = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y, attribs.barycentrics.x, attribs.barycentrics.y);
    return gScene.getVertexData(getGlobalHitID(), triangleIndex, barycentrics);
}

/** Returns interpolated position on a triangle in world space for the previous frame.
    \param[in] triangleIndex Index of the triangle in the current mesh (= PrimitiveIndex()).
    \param[in] attribs Intersection attributes provided by DXR.
    \return Interpolated position in world space for the previous frame.
*/
float3 getPrevPosW(uint triangleIndex, BuiltInTriangleIntersectionAttributes attribs)
{
    float3 barycentrics = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y, attribs.barycentrics.x, attribs.barycentrics.y);
    return gScene.getPrevPosW(getGlobalHitID(), triangleIndex, barycentrics);
}


// TODO: Move below functions into HelloDXR.rt.hlsl where they are used?
// They make assumptions that are not generally applicable (pseudorandom numbers, jitter etc.).

/** Generates a ray given a set of camera parameters, the viewport dimensions, and the location of the pixel to be sampled.
    Code based on Chris Wyman's raytracing tutorial found here: http://cwyman.org/code/dxrTutors/tutors/Tutor4/tutorial04.md.html
*/
RayDesc generateRay(const CameraData camera, uint2 posS, float2 viewportDims)
{
    RayDesc r;
    r.Origin = camera.posW;

    float2 ndc = (((posS + 0.5) / viewportDims) * 2.f - 1.f);
    float3 dir = normalize(ndc.x * camera.cameraU - ndc.y * camera.cameraV + camera.cameraW);
    r.Direction = dir;

    float invCos = 1.f / dot(normalize(camera.cameraW), dir);
    r.TMin = camera.nearZ * invCos;
    r.TMax = camera.farZ * invCos;
    return r;
}

/** Generates a ray from a random position on the camera lens. Code for ray and pseudorandom number generation
    heavily based on Chris Wyman's tutorial found here: http://cwyman.org/code/dxrTutors/tutors/Tutor8/tutorial08.md.html
*/
RayDesc generateDOFRay(const CameraData camera, uint2 posS, float2 viewportDims, inout uint randSeed)
{
    float2 ndc = (((posS + float2(camera.jitterX, camera.jitterY)) / viewportDims) * 2.f - 1.f);
    float3 dir = ndc.x * camera.cameraU - ndc.y * camera.cameraV + camera.cameraW;
    dir /= length(camera.cameraW);
    float3 focalPoint = camera.posW + camera.focalDistance * dir;

    float2 rnd = float2(2.0f * 3.1415926f * rand_next(randSeed), camera.apertureRadius * rand_next(randSeed));
    float2 uv = float2(cos(rnd.x) * rnd.y, sin(rnd.x) * rnd.y);
    float3 randomOrig = camera.posW + uv.x * normalize(camera.cameraU) - uv.y * normalize(camera.cameraV);

    RayDesc r;
    r.Origin = randomOrig;
    r.Direction = normalize(focalPoint - randomOrig);

    float invCos = 1.f / dot(normalize(camera.cameraW), r.Direction);
    r.TMin = camera.nearZ * invCos;
    r.TMax = camera.farZ * invCos;
    return r;
}
