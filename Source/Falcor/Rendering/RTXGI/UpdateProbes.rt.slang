/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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
#include "RTXGIDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Scene.Raytracing;
import Utils.Sampling.SampleGenerator;
import RTXGIVolume;
import UpdateProbesDebugData;
import DirectLighting;

RWTexture2D<float4> DDGIProbeRayData;
Texture2D<float4> DDGIProbeData;

cbuffer PerFrameCB
{
    bool gEnableRecursiveIrradiance;
    uint gFrameCount;
    uint2 gDebugThreadID;
    DirectLighting gDirectLighting;
}

RWStructuredBuffer<UpdateProbesDebugData> gDebugData;

/** ********************* Ray index 0: Shadow ray ************************ */

[shader("miss")]
void shadowMiss(inout ShadowRayData rayData : SV_RayPayload)
{
    // The miss shader is executed if the ray misses all geometry. Mark as visible.
    rayData.visible = true;
}

[shader("anyhit")]
void shadowAnyHit(inout ShadowRayData rayData : SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs : SV_IntersectionAttributes)
{
    // Alpha test for non-opaque geometry.
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(instanceID);
    if (gScene.materials.alphaTest(v, materialID, 0.f)) IgnoreHit();
}

/** ********************* Ray index 1: Probe ray ************************ */

struct ProbeRayData
{
    // Ray hit info (8B)
    float               hitT;               ///< Probe ray distance to hit, set to negative value on miss.
    uint                hitKind;            ///< Hit flags, set to 0 on miss.

    // Geometry hit info (16B)
    GeometryInstanceID  instanceID;         ///< Geometry instance ID for ray hit.
    uint                primitiveIndex;     ///< Primitive index for ray hit.
    float2              barycentrics;       ///< Barycentric coordinates at the hit.
};

[shader("closesthit")]
void probeClosestHit(inout ProbeRayData payload, BuiltInTriangleIntersectionAttributes attribs)
{
    payload.hitT = RayTCurrent();
    payload.hitKind = HitKind();
    payload.instanceID = getGeometryInstanceID();
    payload.primitiveIndex = PrimitiveIndex();
    payload.barycentrics = attribs.barycentrics;
}

[shader("anyhit")]
void probeAnyHit(inout ProbeRayData rayData : SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs : SV_IntersectionAttributes)
{
    // Alpha test for non-opaque geometry.
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(instanceID);
    if (gScene.materials.alphaTest(v, materialID, 0.f)) IgnoreHit();
}

[shader("miss")]
void probeMiss(inout ProbeRayData payload)
{
    payload.hitT = -1.f;
    payload.hitKind = 0;
    // No need to set mesh hit info here
}

[shader("raygeneration")]
void rayGen()
{
    const uint2 dispatchIndex = DispatchRaysIndex().xy;

    const int rayIndex = dispatchIndex.x;     // index of ray within a probe
    const int probeIndex = dispatchIndex.y;   // index of current probe

    const DDGIVolumeDescGPU DDGIVolume = gRTXGIVolume.getVolumeDescGPU();

    uint probeState = DDGILoadProbeState(probeIndex, DDGIProbeData, DDGIVolume);

    if (probeState == RTXGIVolume::kProbeStateInactive && rayIndex >= RTXGIVolume::kNumFixedRays)
    {
        return; // If the probe is inactive, do not shoot rays
    }

    int3 probeCoords = DDGIGetProbeCoords(probeIndex, DDGIVolume);
    float3 probeWorldPosition = DDGIGetProbeWorldPosition(probeCoords, DDGIVolume, DDGIProbeData);

    float3 probeRayDirection = DDGIGetProbeRayDirection(rayIndex, DDGIVolume);

    // Setup the probe ray
    RayDesc ray;
    ray.Origin = probeWorldPosition;
    ray.Direction = probeRayDirection;
    ray.TMin = 0.f;
    ray.TMax = DDGIVolume.probeMaxRayDistance;

    // Probe Ray Trace
    ProbeRayData payload = {};
    TraceRay(
        gScene.rtAccel,
        RAY_FLAG_NONE,
        0xff, /* instanceInclusionMask */
        1, /* ray type for probe ray */
        rayTypeCount,
        1, /* miss index for probe ray */
        ray,
        payload);

    // Output debug data
    if (all(dispatchIndex == gDebugThreadID))
    {
        gDebugData[0].rayOrigin = ray.Origin;
        gDebugData[0].rayDir = ray.Direction;
        gDebugData[0].hitT = payload.hitT;
        gDebugData[0].hitKind = payload.hitKind;
        gDebugData[0].instanceID = payload.instanceID.index;
        gDebugData[0].primitiveIndex = payload.primitiveIndex;
        gDebugData[0].barycentrics = payload.barycentrics;
    }

    // Ray miss. Set hit distance to a large value and exit early.
    if (payload.hitT <= 0.f)
    {
        // Some example implementations gather fraction of the env map lighting even on miss,
        // effectively pretending that some of the env map illumination is indirect.
        DDGIStoreProbeRayMiss(DDGIProbeRayData, dispatchIndex, DDGIVolume, float3(0));
        return;
    }

    // Hit a surface backface. Set the radiance to black and exit early.
    // TODO: Should we check our double-sided flag here and treat it as front-facing if the hit is on the back-face of a double-sided material?
    if (payload.hitKind == HIT_KIND_TRIANGLE_BACK_FACE)
    {
        DDGIStoreProbeRayBackfaceHit(DDGIProbeRayData, dispatchIndex, DDGIVolume, payload.hitT);
        return;
    }

    // A ray used only for classification. Its direction is not random (see DDGIGetProbeRayDirection) and
    // would bias the irradiance estimate. We do not perform lighting on these.
    if ((DDGIVolume.probeRelocationEnabled || DDGIVolume.probeClassificationEnabled) && rayIndex < RTXGIVolume::kNumFixedRays)
    {
        DDGIStoreProbeRayFrontfaceHit(DDGIProbeRayData, dispatchIndex, DDGIVolume, payload.hitT);
        return;
    }

    let lod = ExplicitLodTextureSampler(0.f);

    // We hit a front-facing surface. Fetch the shading parameters.
    float3 barycentrics = float3(1.0 - payload.barycentrics.x - payload.barycentrics.y, payload.barycentrics.x, payload.barycentrics.y);
    VertexData v = gScene.getVertexData(payload.instanceID, payload.primitiveIndex, barycentrics);
    const uint materialID = gScene.getMaterialID(payload.instanceID);
    ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -probeRayDirection, lod);

    // Create BSDF instance.
    let bsdf = gScene.materials.getBSDF(sd, lod);

    SampleGenerator sg = SampleGenerator(dispatchIndex, gFrameCount);

    // Evaluate direct diffuse lighting at the probe ray hit.
    float3 Ldirect = gDirectLighting.computeDirectDiffuse(sd, bsdf, sg);

    // Indirect lighting at probe ray hit (recursive via probe lookup from previous frame).
    float3 Lindirect = 0.f;
    if (gEnableRecursiveIrradiance)
    {
        float3 albedo = bsdf.getProperties(sd).diffuseReflectionAlbedo;
        Lindirect = gRTXGIVolume.evalRadiance(sd.posW, sd.N, sd.V, albedo);
    }

    // Compute outgoing radiance as sum of direct and indirect illumination.
    DDGIStoreProbeRayFrontfaceHit(DDGIProbeRayData, dispatchIndex, DDGIVolume, Ldirect + Lindirect, payload.hitT);
}
