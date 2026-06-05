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
#include "Common.hlsl"

import Scene.Scene;
import Scene.Raytracing;
import Scene.Material.MaterialSystem;
import Rendering.Lights.LightHelpers;

RWStructuredBuffer<Photon> gPhotonBuffer;
RWStructuredBuffer<DrawArguments> gDrawArgument;
RWStructuredBuffer<RayArgument> gRayArgument;
RWStructuredBuffer<RayTask> gRayTask;
RWStructuredBuffer<PixelInfo> gPixelInfo;

StructuredBuffer<uint4> gRayCountQuadTree;
Texture2D<float4> gRayDensityTex;
Texture2D gUniformNoise;

Texture2D gPhotonTexture;
RWTexture2D<uint> gSmallPhotonBuffer;

cbuffer PerFrameCB
{
    float4x4 invView;

    float2 viewportDims;
    uint2 coarseDim;

    float2 randomOffset;
    float emitSize;
    float roughThreshold;

    int gMipmap;
    int rayTaskOffset;
    int maxDepth;
    float iorOverride;

    uint colorPhotonID;
    int photonIDScale;
    float traceColorThreshold;
    float cullColorThreshold;

    uint gAreaType;
    float gIntensity;
    float gSplatSize;
    uint updatePhoton;

    float gMaxScreenRadius;
    float gMinScreenRadius;
    float gMinDrawCount;
    float gSmallPhotonColorScale;

    float3 cameraPos;
};

struct CausticsPackedPayload
{
#ifdef SMALL_COLOR
    uint colorAndFlag;
#else
    float3 color;
    uint isContinue;
#endif
    float hitT;
    float3 nextDir;

#ifdef RAY_DIFFERENTIAL
#ifdef SMALL_RAY_DIFFERENTIAL
        uint3 dP, dD;
#else
        float3 dPdx, dPdy, dDdx, dDdy;  // ray differentials
#endif
#elif defined(RAY_CONE)
    float radius;
    float dRadius;
#elif defined(RAY_NONE)
#endif
};

struct CausticsUnpackedPayload
{
    float3 color;
    float hitT;
    float3 nextDir;
    uint isContinue;
#ifdef RAY_DIFFERENTIAL
    float3 dPdx, dPdy, dDdx, dDdy;  // ray differentials
#elif defined(RAY_CONE)
    float radius;
    float dRadius;
#elif defined(RAY_NONE)
#endif
};

#ifdef SMALL_COLOR

uint colorRayToUint(float3 color, uint isContinue)
{
    uint3 colorI = color * 255;
    return (colorI.r << 24) | (colorI.g << 16) | (colorI.b << 8) | (isContinue & 0x1);
}

void uintToColorRay(uint i,
    out float3 color, out uint isContinue)
{
    isContinue = i & 0x1;
    uint3 colorI = uint3(i >> 24, i >> 16, i >> 8);
    colorI = (colorI & 0xff);
    color = colorI / 255.0;
}
#endif

#ifdef SMALL_RAY_DIFFERENTIAL
uint3 float3ToUint3(float3 a, float3 b)
{
    uint3 res;
    res.x = ((f32tof16(a.x) << 16) | f32tof16(a.y));
    res.y = ((f32tof16(a.z) << 16) | f32tof16(b.x));
    res.z = ((f32tof16(b.y) << 16) | f32tof16(b.z));
    return res;
}

void uint3ToFloat3(uint3 i, out float3 a, out float3 b)
{
    a.x = f16tof32(i.x >> 16);
    a.y = f16tof32(i.x & 0xffff);
    a.z = f16tof32(i.y >> 16);
    b.x = f16tof32(i.y & 0xffff);
    b.y = f16tof32(i.z >> 16);
    b.z = f16tof32(i.z & 0xffff);
}
#endif

void unpackCausticsPayload(CausticsPackedPayload p, out CausticsUnpackedPayload d)
{
#ifdef SMALL_COLOR
    uintToColorRay(p.colorAndFlag, d.color, d.isContinue);
#else
    d.color = p.color;
    d.isContinue = p.isContinue;
#endif
    d.nextDir = p.nextDir;
    d.hitT = p.hitT;

#ifdef RAY_DIFFERENTIAL
#ifdef SMALL_RAY_DIFFERENTIAL
        uint3ToFloat3(p.dP, d.dPdx, d.dPdy);
        uint3ToFloat3(p.dD, d.dDdx, d.dDdy);
#else
        d.dPdx = p.dPdx;
        d.dPdy = p.dPdy;
        d.dDdx = p.dDdx;
        d.dDdy = p.dDdy;
#endif
#elif defined(RAY_CONE)
    d.radius = p.radius;
    d.dRadius = p.dRadius;
#elif defined(RAY_NONE)
#endif
}

void packCausticsPayload(CausticsUnpackedPayload d, out CausticsPackedPayload p)
{
#ifdef SMALL_COLOR
    p.colorAndFlag = colorRayToUint(d.color, d.isContinue);
#else
    p.color = d.color;
    p.isContinue = d.isContinue;
#endif
    p.hitT = d.hitT;
    p.nextDir = d.nextDir;

#ifdef RAY_DIFFERENTIAL
#ifdef SMALL_RAY_DIFFERENTIAL
        p.dP = float3ToUint3(d.dPdx, d.dPdy);
        p.dD = float3ToUint3(d.dDdx, d.dDdy);
#else
        p.dPdx = d.dPdx;
        p.dPdy = d.dPdy;
        p.dDdx = d.dDdx;
        p.dDdy = d.dDdy;
#endif
#elif defined(RAY_CONE)
    p.radius = d.radius;
    p.dRadius = d.dRadius;
#elif defined(RAY_NONE)
#endif
}

struct ShadowRayData
{
    bool hit;
};

[shader("miss")]
void primaryMiss(inout CausticsPackedPayload hitData)
{
#ifdef SMALL_COLOR
    hitData.colorAndFlag = colorRayToUint(float3(0, 0, 0), 0);
#else
    hitData.color = float3(0.0f, 0.0f, 0.0f);
    hitData.isContinue = 0;
#endif
    hitData.hitT = 0.0f;
}

VertexOut getVertexAttributes(const GeometryInstanceID instanceID, uint triangleIndex, BuiltInTriangleIntersectionAttributes attribs,
    out float3 dP1, out float3 dP2,
    out float3 dN1, out float3 dN2)
{
    float3 barycentrics = float3(1.0 - attribs.barycentrics.x - attribs.barycentrics.y, attribs.barycentrics.x, attribs.barycentrics.y);
    uint3 indices = gScene.getIndices(instanceID, triangleIndex);
    VertexOut v;
    v.texC = 0.0f.rr;
    v.normalW = 0.0f.rrr;
    v.bitangentW = 0.0f.rrr;
#ifdef USE_INTERPOLATED_POSITION
    v.posW = 0;
#else
    v.posW = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
#endif
    v.colorV = 0.0f.rrr;
    v.prevPosH = 0.0f.rrr;
    //v.lightmapC = 0;

    float3 P[3], N[3];
    [unroll]
    for (int i = 0; i < 3; i++)
    {
        StaticVertexData svd = gScene.getVertex(indices[i]);
        int address = (indices[i] * 3) * 4;
        P[i] = svd.position; //asfloat(gPositions.Load3(address));
        N[i] = svd.normal; //asfloat(gNormals.Load3(address));

        v.texC += svd.texCrd * barycentrics[i]; //asfloat(gTexCrds.Load2(address)) * barycentrics[i];
        v.normalW += N[i] * barycentrics[i];
        //v.bitangentW += asfloat(gBitangents.Load3(address)) * barycentrics[i]; // there is no bitangent in StaticVertexData
        //v.lightmapC += asfloat(gLightMapUVs.Load2(address)) * barycentrics[i];

#ifdef USE_INTERPOLATED_POSITION
        v.posW += P[i] * barycentrics[i];
#endif
    }

#ifdef USE_INTERPOLATED_POSITION
    v.posW = mul(float4(v.posW, 1.f), gWorldMat[0]).xyz;
#endif

    const float4x4 worldMat = gScene.getWorldMatrix(instanceID);
    const float3x3 worldInvTransposeMat = gScene.getInverseTransposeWorldMatrix(instanceID);
#ifndef _MS_DISABLE_INSTANCE_TRANSFORM
    // Transform normal/bitangent to world space
    v.normalW = mul(v.normalW, (float3x3) worldInvTransposeMat).xyz;
    v.bitangentW = mul(v.bitangentW, (float3x3) worldMat).xyz;
#endif

    dP1 = mul(P[1] - P[0], (float3x3) worldMat).xyz;
    dP2 = mul(P[2] - P[0], (float3x3) worldMat).xyz;
    dN1 = mul(N[1] - N[0], (float3x3) worldInvTransposeMat).xyz;
    dN2 = mul(N[2] - N[0], (float3x3) worldInvTransposeMat).xyz;

    // Handle invalid bitangents gracefully (avoid NaN from normalization).
    v.bitangentW = dot(v.bitangentW, v.bitangentW) > 0.f ? normalize(v.bitangentW) : float3(0.0f, 0.0f, 0.0f);
    return v;
}

void getVerticesAndNormals(const GeometryInstanceID instanceID, uint triangleIndex, BuiltInTriangleIntersectionAttributes attribs,
    out float3 P0, out float3 P1, out float3 P2,
    out float3 N0, out float3 N1, out float3 N2,
    out float3 N)
{
    uint3 indices = gScene.getIndices(instanceID, triangleIndex);
    const float4x4 worldMat = gScene.getWorldMatrix(instanceID);
    const float3x3 worldInvTransposeMat = gScene.getInverseTransposeWorldMatrix(instanceID);

    StaticVertexData svd = gScene.getVertex(indices[0]);
    //int address = (indices[0] * 3) * 4;
    P0 = svd.position; //asfloat(gPositions.Load3(address)).xyz;
    P0 = mul(float4(P0, 1.0f), worldMat).xyz;
    N0 = svd.normal; //asfloat(gNormals.Load3(address)).xyz;
    N0 = normalize(mul(float4(N0, 0.f), worldMat).xyz);

    svd = gScene.getVertex(indices[1]);
    //address = (indices[1] * 3) * 4;
    P1 = svd.position; //asfloat(gPositions.Load3(address)).xyz;
    P1 = mul(float4(P1, 1.0f), worldMat).xyz;
    N1 = svd.normal; //asfloat(gNormals.Load3(address)).xyz;
    N1 = normalize(mul(float4(N1, 0.f), worldMat).xyz);

    svd = gScene.getVertex(indices[2]);
    //address = (indices[2] * 3) * 4;
    P2 = svd.position; //asfloat(gPositions.Load3(address)).xyz;
    P2 = mul(float4(P2, 1.0f), worldMat).xyz;
    N2 = svd.normal; //asfloat(gNormals.Load3(address)).xyz;
    N2 = normalize(mul(float4(N2, 0.f), worldMat).xyz);

    float3 barycentrics = float3(1.0f - attribs.barycentrics.x - attribs.barycentrics.y, attribs.barycentrics.x, attribs.barycentrics.y);
    N = (N0 * barycentrics.x + N1 * barycentrics.y + N2 * barycentrics.z);
}

void updateTransferRayDifferential(float3 N, inout float3 dPdx, float3 dDdx)
{
    float3 D = WorldRayDirection();
    float t = RayTCurrent();
    float dtdx = -1.0f * dot(dPdx + t * dDdx, N) / dot(D, N);
    dPdx = dPdx + t * dDdx + dtdx * D;
}

void calculateDNdx(float3 dP1, float3 dP2, float3 dN1, float3 dN2, float3 N, float3 dPdx, out float3 dNdx)
{
    float P11 = dot(dP1, dP1);
    float P12 = dot(dP1, dP2);
    float P22 = dot(dP2, dP2);

    float Q1 = dot(dP1, dPdx);
    float Q2 = dot(dP2, dPdx);

    float delta = P11 * P22 - P12 * P12;
    float dudx = (Q1 * P22 - Q2 * P12) / delta;
    float dvdx = (Q2 * P11 - Q1 * P12) / delta;

    float n2 = dot(N, N);
    float n1 = sqrt(n2);
    dNdx = dudx * dN1 + dvdx * dN2;
    dNdx = (n2 * dNdx - dot(N, dNdx) * N) / (n2 * n1);
}

void updateReflectRayDifferential(float3 N, float3 dPdx, float3 dNdx, inout float3 dDdx)
{
    N = normalize(N);
    float3 D = WorldRayDirection();
    float dDNdx = dot(dDdx, N) + dot(D, dNdx);
    dDdx = dDdx - 2 * (dot(D, N) * dNdx + dDNdx * N);
}

void getPhotonDifferential(CausticsUnpackedPayload hitData, RayDesc ray, out float3 dPdx, out float3 dPdy)
{
#ifdef RAY_DIFFERENTIAL
    dPdx = hitData.dPdx;
    dPdy = hitData.dPdy;
#elif defined(RAY_CONE)
    float radius = hitData.radius;
    float3 normal = hitData.nextDir;
    float cosVal = dot(normal, ray.Direction);
    dPdx = normalize(ray.Direction - normal * cosVal) * radius;
    dPdy = cross(normal, dPdx);
    dPdx /= cosVal;
#elif defined(RAY_NONE)
    float radius = 0.2f;
    float3 normal = hitData.nextDir;
    float cosVal = dot(normal, ray.Direction);
    dPdx = normalize(ray.Direction - normal * cosVal) * radius;
    dPdy = cross(normal, dPdx);
    dPdx /= cosVal;
#endif
}

float getPhotonScreenArea(float3 posW, float3 dPdx, float3 dPdy, out float3 screenCoord, out bool inFrustum)
{
    const float4x4 viewProj = gScene.camera.getViewProj();
    dPdx = dPdx * gSplatSize;
    dPdy = dPdy * gSplatSize;
    float4 s0 = mul(float4(posW, 1.0f), viewProj);
    float4 s00 = mul(float4(posW + dPdx + dPdy, 1), viewProj);
    float4 s01 = mul(float4(posW + dPdx - dPdy, 1), viewProj);
    float4 s10 = mul(float4(posW - dPdx + dPdy, 1), viewProj);
    float4 s11 = mul(float4(posW - dPdx - dPdy, 1), viewProj);
    inFrustum = isInFrustum(s00) || isInFrustum(s01) || isInFrustum(s10) || isInFrustum(s11);
    s0 /= s0.w;
    s00 /= s00.w;
    s01 /= s01.w;
    float2 dx = (s00.xy - s0.xy) * viewportDims;
    float2 dy = (s01.xy - s0.xy) * viewportDims;
    float area = abs(dx.x * dy.y - dy.x * dx.y) / (gSplatSize * gSplatSize);
    screenCoord = s0.xyz;
    //float zRadius = max(1,length(dPdx) + length(dPdy)) * 0.5 / length(posW - gCamera.posW) * gCamera.nearZ * 0.5 * (viewportDims.x + viewportDims.y);
    //float area = zRadius * zRadius;
    //float area = 0.5 * (dot(dx, dx) + dot(dy, dy)) / (gSplatSize * gSplatSize);
    return area;
}

void updateRefractRayDifferential(float3 D, float3 R, float3 N, float eta, float3 dPdx, float3 dNdx, inout float3 dDdx)
{
    N = normalize(N);
    float DN = dot(D, N);
    float RN = dot(R, N);
    float mu = eta * DN - RN;
    float dDNdx = dot(dDdx, N) + dot(D, dNdx);
    float dmudx = (eta - eta * eta * DN / RN) * dDNdx;
    dDdx = eta * dDdx - (mu * dNdx + dmudx * N);
}

[shader("closesthit")]
void primaryClosestHit(inout CausticsPackedPayload payload, in BuiltInTriangleIntersectionAttributes attribs)
{
    CausticsUnpackedPayload hitData;
    unpackCausticsPayload(payload, hitData);
    // Get the hit-point data
    float3 rayOrigW = WorldRayOrigin();
    float3 rayDirW = WorldRayDirection();
    float hitT = RayTCurrent();
    uint triangleIndex = PrimitiveIndex();

    // prepare the shading data
    float3 dP1, dP2, dN1, dN2;
    const GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexOut v = getVertexAttributes(instanceID, triangleIndex, attribs, dP1, dP2, dN1, dN2);
    float3 N = v.normalW;
    v.normalW = normalize(v.normalW);
#ifdef RAY_DIFFERENTIAL
    updateTransferRayDifferential(v.normalW, hitData.dPdx, hitData.dDdx);
    updateTransferRayDifferential(v.normalW, hitData.dPdy, hitData.dDdy);
#elif defined(RAY_CONE)
    hitData.radius = abs(hitData.radius + hitT * hitData.dRadius);
#endif

    const uint materialID = gScene.getMaterialID(instanceID);
    let lod = ExplicitLodTextureSampler(0.0f);
    VertexData v2 = getVertexData(instanceID, triangleIndex, attribs);
    ShadingData sd = gScene.materials.prepareShadingData(v2, materialID, rayOrigW /*, 0*/);

    uint hints = 0u;
    let mi = gScene.materials.getMaterialInstance(sd, lod, hints);
    BSDFProperties bsdfProperties = mi.getProperties(sd);

    hitData.hitT = hitT;
    bool isSpecular = (bsdfProperties.roughness > roughThreshold || /*sd.opacity < 1*/bsdfProperties.isTransmissive);
    hitData.isContinue = isSpecular;
    if (isSpecular)
    {
        float3 N_ = v.normalW;
        //bool isReflect = (sd.opacity == 1);
        bool isReflect = !bsdfProperties.isTransmissive; // probably?
        float3 R;
        float eta = iorOverride > 0 ? 1.0 / iorOverride : 1.0 / sd.IoR;
        if (!isReflect)
        {
            if (dot(N_, rayDirW) > 0)
            {
                eta = 1.0 / eta;
                N *= -1;
                N_ *= -1;
#ifdef RAY_DIFFERENTIAL
                dN1 *= -1;
                dN2 *= -1;
#endif
            }
            isReflect = isTotalInternalReflection(rayDirW, N_, eta);
        }

#ifdef RAY_DIFFERENTIAL
        float3 dNdx = 0;
        float3 dNdy = 0;
        calculateDNdx(dP1, dP2, dN1, dN2, N, hitData.dPdx, dNdx);
        calculateDNdx(dP1, dP2, dN1, dN2, N, hitData.dPdy, dNdy);
        if (isReflect)
        {
            updateReflectRayDifferential(N_, hitData.dPdx, dNdx, hitData.dDdx);
            updateReflectRayDifferential(N_, hitData.dPdy, dNdy, hitData.dDdy);
            R = reflect(rayDirW, N_);
        }
        else
        {
            getRefractVector(rayDirW, N_, R, eta);
            updateRefractRayDifferential(rayDirW, R, N_, eta, hitData.dPdx, dNdx, hitData.dDdx);
            updateRefractRayDifferential(rayDirW, R, N_, eta, hitData.dPdy, dNdy, hitData.dDdy);
        }
        float area = (dot(hitData.dPdx, hitData.dPdx) + dot(hitData.dPdy, hitData.dPdy)) * 0.5;
#elif defined(RAY_CONE)
        float cosVal = abs(dot(N_, rayDirW));
        float area = hitData.radius * hitData.radius / cosVal;
        float triArea = length(cross(dP1, dP2)) * 0.5;
        float triNormalArea = max(dot(dN1, dN1),dot(dN2, dN2));
        float isConcave = (dot(dP1, dN1) + dot(dP2, dN2) >= 0) ? 1.0 : -1.0;
        float dH = area / triArea * triNormalArea * isConcave * 2;
        float dO;
        if (isReflect)
        {
            R = reflect(rayDirW, N_);
            dO = dH * 4 * cosVal;
        }
        else
        {
            getRefractVector(rayDirW, N_, R, eta);
            float etaI = eta;
            float etaO = 1;
            float3 ht = -1 * (-etaI * rayDirW + etaO * R);
            float dHdO = dot(ht, ht) / (etaO * etaO * abs(dot(ht, R)));
            dO = dH * dHdO;
        }
        hitData.dRadius += sqrt(dO);
#elif defined(RAY_NONE)
        if (isReflect)
        {
            R = reflect(rayDirW, N_);
        }
        else
        {
            getRefractVector(rayDirW, N_, R, eta);
        }
        float area = 0.0001f;
#endif
        hitData.nextDir = R;
        float3 baseColor = bsdfProperties.diffuseReflectionAlbedo; /*lerp(1, sd.diffuse, sd.opacity);*/
        hitData.color = baseColor * hitData.color; // *float4(sd.specular, 1);
    }
    else
    {
        //hitData.color.rgb = dot(-rayDirW, sd.N) * bsdfProperties.diffuseReflectionAlbedo/*sd.diffuse*/ * hitData.color.rgb;
        hitData.color.rgb = dot(-rayDirW, bsdfProperties.guideNormal) * bsdfProperties.diffuseReflectionAlbedo/*sd.diffuse*/ * hitData.color.rgb;
        //hitData.nextDir = sd.N;
        hitData.nextDir = bsdfProperties.guideNormal;
    }

    packCausticsPayload(hitData, payload);
}

float getArea(float3 dPdx, float3 dPdy)
{
    float area;
    if (gAreaType == 0)
    {
        area = (dot(dPdx, dPdx) + dot(dPdy, dPdy)) * 0.5;
    }
    else if (gAreaType == 1)
    {
        area = (length(dPdx) + length(dPdy));
        area *= area;
    }
    else if (gAreaType == 2)
    {
        area = max(dot(dPdx, dPdx), dot(dPdy, dPdy));
    }
    else
    {
        float3 areaVector = cross(dPdx, dPdy);
        area = length(areaVector);
    }
    return area;
}


void initFromLight(float2 lightUV, float2 pixelSize0, out RayDesc ray, out CausticsUnpackedPayload hitData)
{
    lightUV = lightUV * 2.0f - 1.0f;
    float2 pixelSize = pixelSize0 * emitSize / float2(coarseDim.xy);

    const LightData light = gScene.getLight(0);
    float3 lightOrigin = cameraPos + light.dirW * (-100.0f); // gLights[0].posW;
    float3 lightDirZ = light.dirW;
    float3 lightDirX = normalize(float3(-lightDirZ.z, 0, lightDirZ.x));
    float3 lightDirY = normalize(cross(lightDirZ, lightDirX));

    ray.Origin = lightOrigin + (lightDirX * lightUV.x + lightDirY * lightUV.y) * emitSize;
    ray.Direction = lightDirZ;
    ray.TMin = 0.001f;
    ray.TMax = 1e10;

    float3 color0 = 1;
    if (colorPhotonID)
    {
        uint3 launchIndex = DispatchRaysIndex();
        color0.xyz = frac(launchIndex.xyz / float(photonIDScale)) * 0.8 + 0.2;
    }
    hitData.color = color0 * pixelSize.x * pixelSize.y * 512 * 512 * 0.5 * gIntensity; // / (gSplatSize * gSplatSize);
    hitData.nextDir = ray.Direction;
    hitData.isContinue = 1;
#ifdef RAY_DIFFERENTIAL
    hitData.dDdx = 0;// lightDirX* pixelSize.x * 2.0;
    hitData.dDdy = 0;// lightDirY* pixelSize.y * 2.0;
    hitData.dPdx = lightDirX * pixelSize.x * 2.0;
    hitData.dPdy = -lightDirY * pixelSize.y * 2.0;
#elif defined(RAY_CONE)
    hitData.radius = 0.5 * (pixelSize.x + pixelSize.y);
    hitData.dRadius = 0.0;
#endif
}

void StorePhoton(RayDesc ray, CausticsUnpackedPayload hitData, uint2 pixelCoord)
{
    bool isInFrustum;
    float3 dPdx, dPdy;
    float3 posW = ray.Origin;
    getPhotonDifferential(hitData, ray, dPdx, dPdy);
    float area = getArea(dPdx, dPdy);
    float3 color = hitData.color.rgb / area;
    float3 screenCoord;
    float pixelArea = getPhotonScreenArea(posW, dPdx, dPdy, screenCoord, isInFrustum);
    if (dot(color, float3(0.299, 0.587, 0.114)) > cullColorThreshold && isInFrustum)
    {
        uint pixelLoc = pixelCoord.y * coarseDim.x + pixelCoord.x;

        bool storePhoton = (pixelArea < gMaxScreenRadius * gMaxScreenRadius);
        uint oldV;
#ifdef FAST_PHOTON_PATH
        if (pixelArea < gMinScreenRadius * gMinScreenRadius)
        {
            float2 uv = (screenCoord.xy + 1) * 0.5;
            int2 dstPixel = uv * viewportDims;
            float4 lastClr = gPhotonTexture.Load(int3(dstPixel,0));
            if (lastClr.a >= gMinDrawCount)
            {
                uint clr = compressColor(color, gSmallPhotonColorScale);
                InterlockedAdd(gSmallPhotonBuffer[dstPixel], clr, oldV);
                storePhoton = false;
            }
        }
#endif
        if (storePhoton)
        {
            uint instanceIdx = 0;
            InterlockedAdd(gDrawArgument[0].InstanceCount, 1, instanceIdx);

            Photon photon;
            photon.posW = posW;
            //photon.normalW = hitData.nextDir;
            if (dot(cross(dPdx, dPdy), hitData.nextDir) < 0)
            {
                dPdy *= -1;
            }
            photon.color = color;
            photon.dPdx = dPdx;
            photon.dPdy = dPdy;
            gPhotonBuffer[instanceIdx] = photon;
#ifdef TRACE_FIXED
            gPixelInfo[pixelLoc].photonIdx = instanceIdx;
#endif
        }

        pixelArea = clamp(pixelArea, 1, 100 * 100);
        InterlockedAdd(gPixelInfo[pixelLoc].screenArea, uint(pixelArea.x), oldV);
        InterlockedAdd(gPixelInfo[pixelLoc].screenAreaSq, uint(pixelArea.x * pixelArea.x), oldV);
        InterlockedAdd(gPixelInfo[pixelLoc].count, 1, oldV);
    }
}

#ifdef TRACE_ADAPTIVE_RAY_MIP_MAP
bool getSamplePos(uint threadId, out uint2 pixelPos, out uint sampleIdx)
{
    pixelPos = 0;
    sampleIdx = threadId;
    uint4 value = gRayCountQuadTree[0];
    if (threadId >= value.w)
        return false;
    for (int mip = 1; mip <= gMipmap; mip++)
    {
        pixelPos <<= 1;
        if(sampleIdx >= value.b)
        {
            pixelPos += int2(1, 1);
            sampleIdx -= value.b;
        }
        else if (sampleIdx >= value.g)
        {
            pixelPos += int2(0, 1);
            sampleIdx -= value.g;
        }
        else if (sampleIdx >= value.r)
        {
            pixelPos += int2(1, 0);
            sampleIdx -= value.r;
        }

        int nodeOffset = getTextureOffset(pixelPos, mip);
        value = gRayCountQuadTree[nodeOffset];
    }
    return true;
}

void getRaySample(uint2 pixel00, uint sampleIdx, inout float2 screenCoord, inout float2 pixelSize)
{
    float v00 = gRayDensityTex.Load(int3(pixel00 + int2(0, 0), 0)).r;
    float v10 = gRayDensityTex.Load(int3(pixel00 + int2(1, 0), 0)).r;
    float v01 = gRayDensityTex.Load(int3(pixel00 + int2(0, 1), 0)).r;
    float v11 = gRayDensityTex.Load(int3(pixel00 + int2(1, 1), 0)).r;
    float sampleCountF = 0.25 * (v00 + v10 + v01 + v11);

    int sampleDim = (int)ceil(sqrt(sampleCountF));
    int sampleCount = sampleDim * sampleDim;
    float sampleWeight = 1.0 / sqrt(float(sampleCount)) * sqrt(sampleCountF / sampleCount);
    pixelSize = sqrt(sampleCountF / sampleCount);
    if (sampleCount == 1)
    {
        v00 = v10 = v01 = v11 = 1;
        pixelSize = 1;
    }
    uint yi = sampleIdx / sampleDim;
    uint xi = sampleIdx - yi * sampleDim;
    float x = (xi + 0.5) / sampleDim;
    float y = (yi + 0.5) / sampleDim;
    float2 rnd = float2(x, y);
    float2 uv = bilinearSample(v00, v10, v01, v11, rnd);
    float2 duv = bilinearSample(v00, v10, v01, v11, rnd + 1e-3) - uv;
    float aniso = sqrt(duv.y / duv.x);

    screenCoord = pixel00 + uv + 0.5;
    pixelSize = pixelSize * sqrt(1 / (bilinearIntepolation(v00, v10, v01, v11, uv)));// *float2(1 / aniso, aniso);
}

uint part1By1(uint x)
{
    x &= 0x0000ffff;                  // x = ---- ---- ---- ---- fedc ba98 7654 3210
    x = (x ^ (x << 8)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
    x = (x ^ (x << 4)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
    x = (x ^ (x << 2)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
    x = (x ^ (x << 1)) & 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    return x;
}

uint encodeMorton2(uint2 idx)
{
    return (part1By1(idx.y) << 1) + part1By1(idx.x);
}

#endif

bool getTask(out float2 lightUV, out uint2 pixelCoord, out float2 pixelSize)
{
    uint3 launchIndex = DispatchRaysIndex();
    uint3 launchDimension = DispatchRaysDimensions();
    uint taskIdx = launchIndex.y * launchDimension.x + launchIndex.x;
#ifdef TRACE_ADAPTIVE
    {
        if (taskIdx >= gRayArgument[0].rayTaskCount)
        {
            return false;
        }
        RayTask task = gRayTask[taskIdx];
        pixelCoord = task.screenCoord;
        lightUV = (task.screenCoord + randomOffset * task.pixelSize) / float2(coarseDim);
        pixelSize = task.pixelSize;
    }
#elif defined(TRACE_FIXED)
    {
        pixelSize = 1;
        pixelCoord = launchIndex.xy;
        lightUV = float2(launchIndex.xy) / float2(coarseDim.xy);
        lightUV += randomOffset / float2(coarseDim.xy);
        if (any(launchIndex.xy >= coarseDim))
        {
            return false;
        }
    }
#elif defined(TRACE_NONE)
    {
        return false;
    }
#elif defined(TRACE_ADAPTIVE_RAY_MIP_MAP)
    {
        uint2 pixelPosI;
        uint sampleIdx;
        float2 screenCoord = 0;
        pixelSize = 1;
        pixelCoord = 0;
        //taskIdx = encodeMorton2(launchIndex.xy);
        if (!getSamplePos(taskIdx, pixelPosI, sampleIdx))
            return false;
        getRaySample(pixelPosI, sampleIdx, screenCoord, pixelSize);
        pixelCoord = screenCoord;
        lightUV = (screenCoord + randomOffset * pixelSize) / float2(coarseDim);
        //lightUV = pixelPosI / float2(coarseDim); //
    }
#endif

    if (updatePhoton)
    {
        gPixelInfo[taskIdx].photonIdx = -1;
    }
#ifdef TRACE_FIXED
    {
        gRayTask[taskIdx].screenCoord = launchIndex.xy;
        gRayTask[taskIdx].pixelSize = 1;
    }
#endif
    return true;
}

[shader("raygeneration")]
void rayGen()
{
    // fetch task
    float2 lightUV = 0.0f.xx;
    uint2 pixelCoord = 0u.xx;
    float2 pixelSize = 0.0f.xx;
    if (!getTask(lightUV, pixelCoord, pixelSize)) return;

    // Init ray and hit data
    RayDesc ray;
    CausticsUnpackedPayload hitData;
    initFromLight(lightUV, pixelSize, ray, hitData);
    float colorIntensity = max(hitData.color.r, max(hitData.color.g, hitData.color.b));
    hitData.color /= colorIntensity;

    // Photon trace
    int depth;
    for (depth = 0; depth < maxDepth && hitData.isContinue; depth++)
    {
        ray.Direction = hitData.nextDir;

        CausticsPackedPayload payload;
        packCausticsPayload(hitData, payload);
        TraceRay(gScene.rtAccel, 0, 0xFF, 0, rayTypeCount /*hitProgramCount*/, 0, ray, payload);
        unpackCausticsPayload(payload, hitData);

        float area = (dot(hitData.dPdx, hitData.dPdx) + dot(hitData.dPdy, hitData.dPdy)) * 0.5;
        if (hitData.isContinue && dot(hitData.color, float3(0.299f, 0.587f, 0.114f)) * colorIntensity < traceColorThreshold * area)
        {
            hitData.isContinue = 0;
            hitData.color = 0;
        }

        ray.Origin = ray.Origin + ray.Direction * hitData.hitT;
    }

    // write result
#ifdef UPDATE_PHOTON
    if (any(hitData.color > 0) && depth > 1)
    {
        hitData.color *= colorIntensity;
        StorePhoton(ray, hitData, pixelCoord);
    }
#endif
}
