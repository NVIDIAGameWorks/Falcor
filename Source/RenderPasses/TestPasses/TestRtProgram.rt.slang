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
import Scene.Raytracing;
import Utils.Geometry.IntersectionHelpers;

struct TestProgram
{
    uint2 frameDim;
    RWTexture2D<float4> output;

    void execute(const uint2 pixel)
    {
        RayDesc ray = gScene.camera.computeRayPinhole(pixel, frameDim).toRayDesc();
        RayData rayData = {};

        // Select miss shader index in a checkerboard pattern.
        uint missIndex = (pixel.x ^ pixel.y) & 0x10 ? 1 : 0;

        // Select ray type in a diagonal line pattern.
        uint rayType = (pixel.x + pixel.y) & 0x8 ? 1 : 0;

        TraceRay(gScene.rtAccel, RAY_FLAG_NONE, 0xff, rayType, rayTypeCount, missIndex, ray, rayData);

        output[pixel] = float4(rayData.color, 1.f);
    }
};

ConstantBuffer<TestProgram> gTestProgram;

//
// Ray payload and entry points for miss shaders.
//

struct RayData
{
    float t;
    float3 color;
};

[shader("miss")]
void miss0(inout RayData rayData)
{
    rayData.color = float3(0.05f);
}

[shader("miss")]
void miss1(inout RayData rayData)
{
    rayData.color = float3(0.1f);
}

//
// Shader entry points for triangle hit groups.
//

[shader("anyhit")]
void anyHit(inout RayData rayData : SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs : SV_IntersectionAttributes)
{
    // Alpha test for non-opaque geometry.
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    uint materialID = gScene.getMaterialID(instanceID);
    if (alphaTest(v, gScene.materials[materialID], gScene.materialResources[materialID], 0.f)) IgnoreHit();
}

[shader("closesthit")]
void closestHitMtl0(inout RayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    rayData.t = RayTCurrent();
    rayData.color = float3(0.5f);
}

[shader("closesthit")]
void closestHitMtl1(inout RayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    rayData.t = RayTCurrent();
    rayData.color = float3(0.0f, 0.0f, 1.f);
}

[shader("closesthit")]
void closestHitRed(inout RayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    rayData.t = RayTCurrent();
    rayData.color = float3(1.f, 0.f, 0.f);
}

[shader("closesthit")]
void closestHitGreen(inout RayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    rayData.t = RayTCurrent();
    rayData.color = float3(0.f, 1.f, 0.f);
}

//
// Shader entry points for procedural primitive hit groups.
//

struct MyIntersectionAttribs
{
    float t;
};

[shader("intersection")]
void intersectSphere()
{
    const GeometryInstanceID instanceID = getGeometryInstanceID();
    AABB aabb = gScene.getCustomPrimitiveAABB(instanceID);

    // Inscribed sphere radius
    float3 e = aabb.extent();
    float radius = min(e.x, min(e.y, e.z)) * 0.5f;

    float t;
    bool isect = intersectRaySphere(WorldRayOrigin(), WorldRayDirection(), aabb.center(), radius, t);
    if (isect && t < RayTCurrent())
    {
        MyIntersectionAttribs attribs;
        attribs.t = t;
        ReportHit(t, 0, attribs);
    }
}

[shader("closesthit")]
void closestHitSphereMtl0(inout RayData rayData, MyIntersectionAttribs attribs)
{
    rayData.t = RayTCurrent();
    rayData.color = float3(0.25f);
}

[shader("closesthit")]
void closestHitSphereMtl1(inout RayData rayData, MyIntersectionAttribs attribs)
{
    rayData.t = RayTCurrent();
    rayData.color = float3(0.5f, 0.f, 0.5f);
}

[shader("closesthit")]
void closestHitSpherePurple(inout RayData rayData, MyIntersectionAttribs attribs)
{
    rayData.t = RayTCurrent();
    rayData.color = float3(0.75f, 0.f, 1.f);
}

[shader("closesthit")]
void closestHitSphereYellow(inout RayData rayData, MyIntersectionAttribs attribs)
{
    rayData.t = RayTCurrent();
    rayData.color = float3(1.f, 1.f, 0.f);
}

//
// Entry point for ray generation shader.
//

[shader("raygeneration")]
void rayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    gTestProgram.execute(pixel);
}
