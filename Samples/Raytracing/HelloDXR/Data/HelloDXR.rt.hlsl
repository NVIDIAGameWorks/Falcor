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
RWTexture2D<float4> gOutput;
__import Raytracing;

shared cbuffer PerFrameCB
{
    float4x4 invView;
    float4x4 invModel;
    float2 viewportDims;
    float tanHalfFovY;
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
void shadowAnyHit(inout ShadowRayData hitData, in BuiltInTriangleIntersectionAttributes attribs)
{
    hitData.hit = true;
}

[shader("miss")]
void primaryMiss(inout PrimaryRayData hitData)
{
    hitData.color = float4(0.38f, 0.52f, 0.10f, 1);
    hitData.hitT = -1;
}

bool checkLightHit(uint lightIndex, float3 origin)
{
    float3 direction = gLights[lightIndex].posW - origin;
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = normalize(direction);
    ray.TMin = 0.001;
    ray.TMax = max(0.01, length(direction));

    ShadowRayData rayData;
    rayData.hit = true;
    TraceRay(gRtScene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, 1 /* ray index */, hitProgramCount, 1, ray, rayData);
    return rayData.hit;
}

float3 getReflectionColor(float3 worldOrigin, VertexOut v, float3 worldRayDir, uint hitDepth)
{
    float3 reflectColor = float3(0, 0, 0);
    if (hitDepth == 0)
    {
        PrimaryRayData secondaryRay;
        secondaryRay.depth.r = 1;
        RayDesc ray;
        ray.Origin = worldOrigin;
        ray.Direction = reflect(worldRayDir, v.normalW);
        ray.TMin = 0.001;
        ray.TMax = 100000;
        TraceRay(gRtScene, 0 /*rayFlags*/, 0xFF, 0 /* ray index*/, hitProgramCount, 0, ray, secondaryRay);
        reflectColor = secondaryRay.hitT == -1 ? 0 : secondaryRay.color.rgb;
        float falloff = max(1, (secondaryRay.hitT * secondaryRay.hitT));
        reflectColor *= 20 / falloff;
    }
    return reflectColor;
}

[shader("closesthit")]
void primaryClosestHit(inout PrimaryRayData hitData, in BuiltInTriangleIntersectionAttributes attribs)
{
    // Get the hit-point data
    float3 rayOrigW = WorldRayOrigin();
    float3 rayDirW = WorldRayDirection(); 
    float hitT = RayTCurrent();
    uint triangleIndex = PrimitiveIndex();

    float3 posW = rayOrigW + hitT * rayDirW;
    // prepare the shading data
    VertexOut v = getVertexAttributes(triangleIndex, attribs);
    ShadingData sd = prepareShadingData(v, gMaterial, rayOrigW, 0);

    // Shoot a reflection ray
    float3 reflectColor = getReflectionColor(posW, v, rayDirW, hitData.depth.r);
    float3 color = 0;

    [unroll]
    for (int i = 0; i < gLightsCount; i++)
    {        
        if (checkLightHit(i, posW) == false)
        {
            color += evalMaterial(sd, gLights[i], 1).color.xyz;
        }
    }

    hitData.color.rgb = color;
    hitData.hitT = hitT;
    // A very non-PBR inaccurate way to do reflections
    float roughness = min(0.5, max(1e-8, sd.roughness));
    hitData.color.rgb += sd.specular * reflectColor * (roughness * roughness);
    hitData.color.rgb += sd.emissive;
    hitData.color.a = 1;
}

[shader("raygeneration")]
void rayGen()
{
    uint3 launchIndex = DispatchRaysIndex();
    float2 d = (((launchIndex.xy + 0.5) / viewportDims) * 2.f - 1.f);
    float aspectRatio = viewportDims.x / viewportDims.y;

    RayDesc ray;
    ray.Origin = invView[3].xyz;

    // We negate the Z exis because the 'view' matrix is generated using a 
    // Right Handed coordinate system with Z pointing towards the viewer
    // The negation of Z axis is needed to get the rays go out in the direction away fromt he viewer.
    // The negation of Y axis is needed because the texel coordinate system, used in the UAV we write into using launchIndex
    // has the Y axis flipped with respect to the camera Y axis (0 is at the top and 1 at the bottom)
    ray.Direction = normalize( (d.x * invView[0].xyz * tanHalfFovY * aspectRatio) - (d.y * invView[1].xyz * tanHalfFovY) - invView[2].xyz );
    
    ray.TMin = 0;
    ray.TMax = 100000;

    PrimaryRayData hitData;
    hitData.depth = 0;
    TraceRay( gRtScene, 0 /*rayFlags*/, 0xFF, 0 /* ray index*/, hitProgramCount, 0, ray, hitData );
    gOutput[launchIndex.xy] = hitData.color;
}
