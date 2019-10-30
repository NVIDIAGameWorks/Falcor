/***************************************************************************
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
import Scene;
import Raytracing;
import Utils.Math.MathHelpers;
import Utils.Sampling.SampleGenerator;
import GBufferHelpers;

// GBuffer channels
shared IRWTexture2D gPosW;
shared IRWTexture2D gNormW;
shared IRWTexture2D gBitangentW;
shared IRWTexture2D gTexC;
shared IRWTexture2D gDiffuseOpacity;
shared IRWTexture2D gSpecRough;
shared IRWTexture2D gEmissive;
shared IRWTexture2D gMatlExtra;

// GBufferRT channels
shared IRWTexture2D gFaceNormalW;
shared IRWTexture2D gViewW;
shared RWTexture2D<uint4> gVisBuffer;


/** ***************************** Ray index 0 ****************************** */

/** Payload passed along with primary rays.
    We write the G-buffer from the hit shader, so this struct is very lightweight.
*/
struct PrimaryRayData
{
    bool hit;
};

[shader("miss")]
void primaryMiss(inout PrimaryRayData rayData)
{
}

[shader("anyhit")]
void primaryAnyHit(inout PrimaryRayData rayData : SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs : SV_IntersectionAttributes)
{
    // Alpha test for non-opaque geometry.
    VertexData v = getVertexData(PrimitiveIndex(), attribs);
    if (alphaTest(v, gScene.materials[gScene.getMaterialID(getGlobalHitID())], 0.f)) IgnoreHit();
}

/** Ray differentials for primary hit. Code from RayTracingGems, Chapter 20.
*/
void computeRayDifferentials(uint meshInstanceID, uint triangleIndex, float3 ray_dir, float ray_t, out float2 ddx, out float2 ddy, const CameraData camera)
{
    // TODO: Is this code correct for instance transforms that flip the handedness of the coordinate system?

    // Ray differentials
    float3 P[3];
    gScene.getVertexPositionsW(meshInstanceID, triangleIndex, P);
    float3 e1 = P[1] - P[0];
    float3 e2 = P[2] - P[0];
    float3 d = ray_dir;
    float k = dot(cross(e1, e2), d);
    k = abs(k) > 0.0001f ? rcp(k) : 0.0f;
    float3 cu = cross(e2, d);
    float3 cv = cross(d, e1);
    // Assumes a normalized ray direction
    float3 dx = camera.cameraU * 2.0 * gParams.invFrameSize.x / camera.focalDistance; // dDdx in ray gen
    float3 dy = camera.cameraV * 2.0 * gParams.invFrameSize.y / camera.focalDistance; // dDdy in ray gen
    float3 q = dx * ray_t; // Transfer to primary hit
    float3 r = dy * ray_t;
    float dudx = k * dot(cu, q);
    float dudy = k * dot(cu, r);
    float dvdx = k * dot(cv, q);
    float dvdy = k * dot(cv, r);
    float2 T[3];
    gScene.getVertexTexCoords(meshInstanceID, triangleIndex, T);
    float2 g1 = T[1] - T[0];
    float2 g2 = T[2] - T[0];
    float dsdx = (dudx * g1.x + dvdx * g2.x);
    float dsdy = (dudy * g1.x + dvdy * g2.x);
    float dtdx = (dudx * g1.y + dvdx * g2.y);
    float dtdy = (dudy * g1.y + dvdy * g2.y);
    ddx = float2(dsdx, dtdx);
    ddy = float2(dsdy, dtdy);
}

/** Closest hit shader for primary rays. 
*/
[shader("closesthit")]
void primaryClosestHit(inout PrimaryRayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    rayData.hit = true;
    uint2 launchIndex = DispatchRaysIndex().xy;

    const uint meshInstanceID = getGlobalHitID();
    const uint triangleIndex = PrimitiveIndex();
    VertexData v = getVertexData(triangleIndex, attribs);

#if USE_RAY_DIFFERENTIALS
    float2 ddx, ddy;
    computeRayDifferentials(meshInstanceID, triangleIndex, WorldRayDirection(), RayTCurrent(), ddx, ddy, gScene.camera);
    ShadingData sd = prepareShadingData(v, gScene.materials[gScene.getMaterialID(meshInstanceID)], WorldRayOrigin(), ddx, ddy);
#else
    ShadingData sd = prepareShadingData(v, gScene.materials[gScene.getMaterialID(meshInstanceID)], WorldRayOrigin(), 0.f);
#endif

    // Write the outputs.
    GBuffer gbuf = storeGBufferOutput(sd);

    gPosW[launchIndex]              = gbuf.posW;
    gNormW[launchIndex]             = gbuf.normW;
    gBitangentW[launchIndex]        = gbuf.bitangentW;
    gTexC[launchIndex]              = gbuf.texC;
    gDiffuseOpacity[launchIndex]    = gbuf.diffuseOpacity;
    gSpecRough[launchIndex]         = gbuf.specRough;
    gEmissive[launchIndex]          = gbuf.emissive;
    gMatlExtra[launchIndex]         = gbuf.matlExtra;

    gFaceNormalW[launchIndex]       = float4(v.faceNormalW, 0.f);
    gVisBuffer[launchIndex]         = uint4(meshInstanceID, triangleIndex, asuint(attribs.barycentrics.x), asuint(attribs.barycentrics.y));
}


/** ******************************** RayGen ******************************** */

// p = position on the image plane in [0,1] where (0,0) is top-left corner.
PrimaryRayData shootPrimaryRay(float2 p, inout SampleGenerator sg)
{
    // Compute primary ray's origin and direction.
    float2 ndc = float2(2, -2) * p + float2(-1, 1);
    float3 rayDir = ndc.x * gScene.camera.cameraU + ndc.y * gScene.camera.cameraV + gScene.camera.cameraW; // rayDir = world-space direction to point on image plane (unnormalized)
    float3 origin = gScene.camera.posW;

#if USE_DEPTH_OF_FIELD
    {
        // Draw sample on the lens to compute new origin and direction
        float2 rnd = sampleNext2D(sg);
        float2 apertureSample = sample_disk(rnd);    // Sample lies in the unit disk [-1,1]^2

        float3 rayTarget = origin + rayDir;
        origin += gScene.camera.apertureRadius * (apertureSample.x * normalize(gScene.camera.cameraU) + apertureSample.y * normalize(gScene.camera.cameraV));
        rayDir = rayTarget - origin;
    }
#endif
    rayDir = normalize(rayDir);

    // Setup ray
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = rayDir;
    ray.TMin = 0.f;     // It's OK to start at 0.0 for the camera, as it should not sit exactly on a surface
    ray.TMax = 1e+38f;
    // TODO: Discard hits outside [Z_near, Z_far] to match raster behavior.

    // Setup ray payload
    PrimaryRayData rayData;
    rayData.hit = false;

    TraceRay(gRtScene, gParams.rayFlags, 0xff /* instanceInclusionMask */, 0 /* hitIdx */, hitProgramCount, 0 /* missIdx */, ray, rayData);

    // Write additional outputs.
    uint2 launchIndex = DispatchRaysIndex().xy;
    gViewW[launchIndex] = float4(-rayDir, 0.f);

    if (!rayData.hit) gVisBuffer[launchIndex] = uint4(kInvalidIndex, kInvalidIndex, 0, 0);

    return rayData;
}

[shader("raygeneration")]
void rayGen()
{
    uint2 launchIndex = DispatchRaysIndex().xy;
    uint2 launchDim = DispatchRaysDimensions().xy;

    // Compute sample position in screen space in [0,1] with origin at the top-left corner.
    // The camera jitter offsets the sample by +-0.5 pixels from the pixel center.
    float2 p = (launchIndex + float2(0.5f, 0.5f)) / launchDim + float2(-gScene.camera.jitterX, gScene.camera.jitterY);

    // Initialize random seed
    SampleGenerator sg = SampleGenerator.create(launchIndex, gParams.frameCount);

    // Shoot primary ray. The hit shader writes the G-buffer.
    shootPrimaryRay(p, sg);
}
