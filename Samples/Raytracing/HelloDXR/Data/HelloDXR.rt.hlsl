/************************************************************************************************************************************\
|*                                                                                                                                    *|
|*     Copyright © 2017 NVIDIA Corporation.  All rights reserved.                                                                     *|
|*                                                                                                                                    *|
|*  NOTICE TO USER:                                                                                                                   *|
|*                                                                                                                                    *|
|*  This software is subject to NVIDIA ownership rights under U.S. and international Copyright laws.                                  *|
|*                                                                                                                                    *|
|*  This software and the information contained herein are PROPRIETARY and CONFIDENTIAL to NVIDIA                                     *|
|*  and are being provided solely under the terms and conditions of an NVIDIA software license agreement                              *|
|*  and / or non-disclosure agreement.  Otherwise, you have no rights to use or access this software in any manner.                   *|
|*                                                                                                                                    *|
|*  If not covered by the applicable NVIDIA software license agreement:                                                               *|
|*  NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOFTWARE FOR ANY PURPOSE.                                            *|
|*  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.                                                           *|
|*  NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,                                                                     *|
|*  INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.                       *|
|*  IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES,                               *|
|*  OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT,                         *|
|*  NEGLIGENCE OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOURCE CODE.            *|
|*                                                                                                                                    *|
|*  U.S. Government End Users.                                                                                                        *|
|*  This software is a "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT 1995),                                       *|
|*  consisting  of "commercial computer  software"  and "commercial computer software documentation"                                  *|
|*  as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government only as a commercial end item.     *|
|*  Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995),                                          *|
|*  all U.S. Government End Users acquire the software with only those rights set forth herein.                                       *|
|*                                                                                                                                    *|
|*  Any use of this software in individual and commercial software must include,                                                      *|
|*  in the user documentation and internal comments to the code,                                                                      *|
|*  the above Disclaimer (as applicable) and U.S. Government End Users Notice.                                                        *|
|*                                                                                                                                    *|
 \************************************************************************************************************************************/
RWTexture2D<float4> gOutput : register(u1);
__import Raytracing;

cbuffer PerFrameCB : register(b0)
{
    float4x4 invView;
    float4x4 invModel;
    float2 viewportDims;
    float tanHalfFovY;
};

struct PrimaryRayData
{
    float4 color;
    uint4 depth;
    float hitT;
};

struct ShadowRayData
{
    bool hit;
};

[shader("miss")]
void shadowMiss(inout ShadowRayData hitData : SV_RayPayload)
{
    hitData.hit = false;
}

[shader("anyhit")]
void shadowAnyHit(inout ShadowRayData hitData : SV_RayPayload, BuiltinIntersectionAttribs attribs : SV_IntersectionAttributes)
{
    hitData.hit = true;
}

[shader("miss")]
void primaryMiss(inout PrimaryRayData hitData : SV_RayPayload)
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
    ray.TMin = 0.01;
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
        ray.TMin = 0.01;
        ray.TMax = 100000;
        TraceRay(gRtScene, 0 /*rayFlags*/, 0xFF, 0 /* ray index*/, hitProgramCount, 0, ray, secondaryRay);
        reflectColor = secondaryRay.hitT == -1 ? 0 : secondaryRay.color.rgb;
        float falloff = max(1, (secondaryRay.hitT * secondaryRay.hitT));
        reflectColor *= 20 / falloff;
    }
    return reflectColor;
}

[shader("closesthit")]
void primaryClosestHit(inout PrimaryRayData hitData : SV_RayPayload, BuiltinIntersectionAttribs attribs : SV_IntersectionAttributes)
{
    // Get the hit-point data
    float3 rayOrigW = WorldRayOrigin();
    float3 rayDirW = WorldRayDirection(); 
    float hitT = RayTCurrent();
    uint triangleIndex = PrimitiveIndex();

    float3 posW = rayOrigW + hitT * rayDirW;
    // prepare the shading data
    VertexOut v = getVertexAttributes(triangleIndex, attribs);
    ShadingData sd = prepareShadingData(v, gMaterial, gCamera.posW);

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
    uint2 launchIndex = DispatchRaysIndex();
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
