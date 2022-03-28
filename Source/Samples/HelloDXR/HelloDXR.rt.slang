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
import Scene.Raytracing;
import Utils.Sampling.TinyUniformSampleGenerator;
import Rendering.Lights.LightHelpers;

RWTexture2D<float4> gOutput;

cbuffer PerFrameCB
{
    float4x4 invView;
    float2 viewportDims;
    float tanHalfFovY;
    uint sampleIndex;
    bool useDOF;
};

struct PrimaryRayData
{
    float4 color;
    uint depth;
    float hitT;
};

struct ShadowRayData
{
    bool hit;
};

[shader("miss")]
void shadowMiss(inout ShadowRayData hitData)
{
    hitData.hit = false;
}

[shader("anyhit")]
void shadowAnyHit(inout ShadowRayData hitData, BuiltInTriangleIntersectionAttributes attribs)
{
    hitData.hit = true;
}

[shader("miss")]
void primaryMiss(inout PrimaryRayData hitData)
{
    hitData.color = float4(0.38f, 0.52f, 0.10f, 1.f);
    hitData.hitT = -1.f;
}

bool traceShadowRay(RayDesc ray)
{
    ShadowRayData rayData;
    rayData.hit = true;
    TraceRay(gScene.rtAccel, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, 1 /* ray index */, rayTypeCount, 1, ray, rayData);
    return rayData.hit;
}

float3 getReflectionColor(const ShadingData sd, const BSDFProperties bsdfProperties, float3 normalW, float3 worldRayDir, uint hitDepth)
{
    float3 reflectColor = float3(0, 0, 0);
    if (hitDepth == 0)
    {
        PrimaryRayData secondaryRay = {};
        secondaryRay.depth = 1;

        RayDesc ray;
        ray.Origin = sd.posW;
        ray.Direction = reflect(worldRayDir, normalW);
        ray.TMin = 0.001f;
        ray.TMax = 100000.f;
        TraceRay(gScene.rtAccel, 0 /*rayFlags*/, 0xFF, 0 /* ray index*/, rayTypeCount, 0, ray, secondaryRay);

        // A very non-PBR inaccurate way to do reflections.
        reflectColor = secondaryRay.hitT < 0.f ? float3(0.f) : secondaryRay.color.rgb;
        float falloff = max(1.f, (secondaryRay.hitT * secondaryRay.hitT));
        reflectColor *= 20.f / falloff;
        float ggxAlpha = min(0.5f, max(1e-8f, bsdfProperties.roughness * bsdfProperties.roughness));
        reflectColor += bsdfProperties.specularReflectance * reflectColor * (ggxAlpha * ggxAlpha);
    }
    return reflectColor;
}

[shader("closesthit")]
void primaryClosestHit(inout PrimaryRayData hitData, BuiltInTriangleIntersectionAttributes attribs)
{
    // Get the hit-point data.
    float3 rayDirW = WorldRayDirection();
    float hitT = RayTCurrent();
    uint triangleIndex = PrimitiveIndex();

    let lod = ExplicitLodTextureSampler(0.f);

    // Prepare the shading data.
    const GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, triangleIndex, attribs);
    uint materialID = gScene.getMaterialID(instanceID);
    ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -rayDirW, lod);

    // Create BSDF instance and query its properties.
    let bsdf = gScene.materials.getBSDF(sd, lod);
    let bsdfProperties = bsdf.getProperties(sd);

    float3 color = bsdfProperties.emission;

    uint3 launchIndex = DispatchRaysIndex();
    TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(launchIndex.xy, sampleIndex);

    // Direct lighting from analytic light sources.
    for (int i = 0; i < gScene.getLightCount(); i++)
    {
        AnalyticLightSample ls;
        if (evalLightApproximate(sd.posW, gScene.getLight(i), ls))
        {
            RayDesc shadowRay = { sd.posW, 0.001f, ls.dir, ls.distance };
            if (!traceShadowRay(shadowRay))
            {
                color += bsdf.eval(sd, ls.dir, sg) * ls.Li;
            }
        }
    }

    // Shoot a reflection ray.
    color += getReflectionColor(sd, bsdfProperties, v.normalW, rayDirW, hitData.depth);

    hitData.color = float4(color, 1.f);
    hitData.hitT = hitT;
}

[shader("anyhit")]
void primaryAnyHit(inout PrimaryRayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    // Alpha test for non-opaque geometry.
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    uint materialID = gScene.getMaterialID(instanceID);
    if (gScene.materials.alphaTest(v, materialID, 0.f))
    {
        IgnoreHit();
    }
}

[shader("raygeneration")]
void rayGen()
{
    uint3 launchIndex = DispatchRaysIndex();
    TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(launchIndex.xy, sampleIndex);

    RayDesc ray;
    if (!useDOF)
    {
        ray = gScene.camera.computeRayPinhole(launchIndex.xy, viewportDims).toRayDesc();
    }
    else
    {
        float2 u = sampleNext2D(sg);
        ray = gScene.camera.computeRayThinlens(launchIndex.xy, viewportDims, u).toRayDesc();
    }

    PrimaryRayData hitData;
    hitData.depth = 0;
    TraceRay(gScene.rtAccel, 0 /*rayFlags*/, 0xFF, 0 /* ray index*/, rayTypeCount, 0, ray, hitData);
    gOutput[launchIndex.xy] = hitData.color;
}
