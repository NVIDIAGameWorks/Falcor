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
#include "Scene/SceneDefines.slangh"

import Scene.Raytracing;
import Scene.Intersection;
import Scene.SDFs.SDFGridHitData;
import VBufferRT;

ConstantBuffer<VBufferRT> gVBufferRT;

struct RayData
{
    int dummy; // TODO: Passing in an empty payload struct doesn't work. Declare a dummy variable so that the compiler doesn't remove the declaration.
};

//
// Shader entry point for miss shader.
//

[shader("miss")]
void miss(inout RayData rayData)
{
    // Write invalid hit to output buffer.
    uint2 launchIndex = DispatchRaysIndex().xy;
    gVBufferRT.writeMiss(launchIndex, WorldRayOrigin(), WorldRayDirection());
}

//
// Shader entry points for TriangleMesh hit groups.
//

[shader("anyhit")]
void anyHit(inout RayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    if (VBufferRT::kUseAlphaTest)
    {
        // Alpha test for non-opaque geometry.
        GeometryInstanceID instanceID = getGeometryInstanceID();
        VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
        uint materialID = gScene.getMaterialID(instanceID);
        if (gScene.materials.alphaTest(v, materialID, 0.f)) IgnoreHit();
    }
}

[shader("closesthit")]
void closestHit(inout RayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    // Store hit information. Note we don't access the materials here.
    TriangleHit triangleHit;
    triangleHit.instanceID = getGeometryInstanceID();
    triangleHit.primitiveIndex = PrimitiveIndex();
    triangleHit.barycentrics = attribs.barycentrics;

    // Write hit info to output buffer.
    uint2 launchIndex = DispatchRaysIndex().xy;
    gVBufferRT.writeHit(launchIndex, WorldRayOrigin(), WorldRayDirection(), HitInfo(triangleHit));
}

//
// Shader entry points for DisplacedTriangleMesh hit groups.
//

[shader("intersection")]
void displacedTriangleMeshIntersection()
{
    const Ray ray = Ray(WorldRayOrigin(), WorldRayDirection(), RayTMin(), RayTCurrent());
    DisplacedTriangleMeshIntersector::Attribs attribs;
    float t;
    if (DisplacedTriangleMeshIntersector::intersect(ray, getGeometryInstanceID(), PrimitiveIndex(), attribs, t))
    {
        ReportHit(t, 0, attribs);
    }
}

[shader("closesthit")]
void displacedTriangleMeshClosestHit(inout RayData rayData, DisplacedTriangleMeshIntersector::Attribs attribs)
{
    // Store hit information. Note we don't access the materials here.
    DisplacedTriangleHit displacedTriangleHit;
    displacedTriangleHit.instanceID = getGeometryInstanceID();
    displacedTriangleHit.primitiveIndex = PrimitiveIndex();
    displacedTriangleHit.barycentrics = attribs.barycentrics;
    displacedTriangleHit.displacement = attribs.displacement;

    // Write hit info to output buffer.
    uint2 launchIndex = DispatchRaysIndex().xy;
    gVBufferRT.writeHit(launchIndex, WorldRayOrigin(), WorldRayDirection(), HitInfo(displacedTriangleHit));
}

//
// Shader entry points for Curve hit groups.
//

[shader("intersection")]
void curveIntersection()
{
    const Ray ray = Ray(WorldRayOrigin(), WorldRayDirection(), RayTMin(), RayTCurrent());
    CurveIntersector::Attribs attribs;
    float t;
    if (CurveIntersector::intersect(ray, getGeometryInstanceID(), PrimitiveIndex(), attribs, t))
    {
        ReportHit(t, 0, attribs);
    }
}

[shader("closesthit")]
void curveClosestHit(inout RayData rayData, CurveIntersector::Attribs attribs)
{
    // Store hit information. Note we don't access the materials here.
    CurveHit curveHit;
    curveHit.instanceID = getGeometryInstanceID();
    curveHit.primitiveIndex = PrimitiveIndex();
    curveHit.barycentrics = attribs.barycentrics;

    // Write hit info to output buffer.
    uint2 launchIndex = DispatchRaysIndex().xy;
    gVBufferRT.writeHit(launchIndex, WorldRayOrigin(), WorldRayDirection(), HitInfo(curveHit));
}


//
// Shader entry points for SDFGrid hit groups.
//

[shader("intersection")]
void sdfGridIntersection()
{
    const Ray ray = Ray(WorldRayOrigin(), WorldRayDirection(), RayTMin(), RayTCurrent());
    SDFGridHitData sdfGridHitData;
    float t;
    if (SDFGridIntersector::intersect(ray, getGeometryInstanceID(), PrimitiveIndex(), sdfGridHitData, t))
    {
        ReportHit(t, 0, sdfGridHitData);
    }
}

[shader("closesthit")]
void sdfGridClosestHit(inout RayData rayData, SDFGridHitData sdfGridHitData)
{
    // Store hit information. Note we don't access the materials here.
    SDFGridHit sdfGridHit;
    sdfGridHit.instanceID = getGeometryInstanceID();
    sdfGridHit.hitData = sdfGridHitData;

    // Write hit info to output buffer.
    uint2 launchIndex = DispatchRaysIndex().xy;
    gVBufferRT.writeHit(launchIndex, WorldRayOrigin(), WorldRayDirection(), HitInfo(sdfGridHit));
}

//
// Shader entry point for ray generation shader.
//

[shader("raygeneration")]
void rayGen()
{
    uint2 launchIndex = DispatchRaysIndex().xy;

    GpuTimer timer;
    gVBufferRT.beginTime(timer);

    // Trace ray
    const Ray ray = gVBufferRT.generateRay(launchIndex);
    RayData rayData;
    TraceRay(gScene.rtAccel, VBufferRT::kRayFlags, 0xff /* instanceInclusionMask */, 0 /* hitIdx */, rayTypeCount, 0 /* missIdx */, ray.toRayDesc(), rayData);

    gVBufferRT.writeAux(launchIndex, ray);

    gVBufferRT.endTime(launchIndex, timer);
}
