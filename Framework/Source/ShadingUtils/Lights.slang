/***************************************************************************
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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
#include "HostDeviceData.h"

import Helpers;
import Interfaces;
import Effects.CascadedShadowMap;

cbuffer InternalLightingParams : register(b21)
{
    LightProbeSharedResources gProbeShared;
};

// work around compiler issue
#define calcCommonLightProperties(geom, ls) \
    { float3 H = normalize(geom.V + ls.L); \
      ls.NdotH = dot(geom.N, H); \
      ls.NdotL = dot(geom.N, ls.L); \
      ls.LdotH = dot(ls.L, H);   }

/*
void calcCommonLightProperties(SurfaceGeometry geom, inout LightSample ls)
{
    float3 H = normalize(geom.V + ls.L);
    ls.NdotH = dot(geom.N, H);
    ls.NdotL = dot(geom.N, ls.L);
    ls.LdotH = dot(ls.L, H);
};
*/

float getDistanceFalloff(float distSquared)
{
    float falloff = 1 / ((0.01 * 0.01) + distSquared); // The 0.01 is to avoid infs when the light source is close to the shading point
    return falloff;
}

/** Evaluate a directional light source intensity/direction at a shading point
*/
LightSample evalDirectionalLight(in LightData light, in float3 surfacePosW)
{
    LightSample ls;
    ls.diffuse = light.intensity;
    ls.specular = light.intensity;
    ls.L = -normalize(light.dirW);
    float dist = length(surfacePosW - light.posW);
    ls.posW = surfacePosW - light.dirW * dist;
    return ls;
}

/** Evaluate a point light source intensity/direction at a shading point
*/
LightSample evalPointLight(in LightData light, in float3 surfacePosW)
{
    LightSample ls;
    ls.posW = light.posW;
    ls.L = light.posW - surfacePosW;
    // Avoid NaN
    float distSquared = dot(ls.L, ls.L);
    ls.distance = (distSquared > 1e-5f) ? length(ls.L) : 0;
    ls.L = (distSquared > 1e-5f) ? normalize(ls.L) : 0;

    // Calculate the falloff
    float falloff = getDistanceFalloff(distSquared);

    // Calculate the falloff for spot-lights
    float cosTheta = -dot(ls.L, light.dirW); // cos of angle of light orientation
    if(cosTheta < light.cosOpeningAngle)
    {
        falloff = 0;
    }
    else if(light.penumbraAngle > 0)
    {
        float deltaAngle = light.openingAngle - acos(cosTheta);
        falloff *= saturate((deltaAngle - light.penumbraAngle) / light.penumbraAngle);
    }
    ls.diffuse = light.intensity * falloff;
    ls.specular = ls.diffuse;
    return ls;
}

/** Evaluate a area light source intensity/direction at a shading point
*/
LightSample evalAreaLight(in AreaLightData light, in float3 surfacePosW)
{
    LightSample ls;
    ls.posW = mul(float4(light.posW, 1.0), light.transMat).xyz;
    ls.L = light.posW - surfacePosW;
    float distSquared = dot(ls.L, ls.L);
    ls.distance = (distSquared > 1e-5f) ? length(ls.L) : 0;
    ls.L = (distSquared > 1e-5f) ? normalize(ls.L) : 0;

    // Calculate the falloff
    float cosTheta = -dot(ls.L, light.dirW); // cos of angle of light orientation 
    float falloff = max(0.f, cosTheta) * light.surfaceArea;
    falloff *= getDistanceFalloff(distSquared);
    ls.diffuse = falloff * light.intensity;
    ls.specular = ls.diffuse;
    return ls;
}

float linearRoughnessToLod(float linearRoughness, float mipCount)
{
    return sqrt(linearRoughness) * (mipCount - 1);
}

/** Evaluate a light source intensity/direction at a shading point
*/
LightSample evalLight(LightData light, SurfaceGeometry geom)
{
    LightSample ls;
    if(light.type == LightDirectional) ls = evalDirectionalLight(light, geom.posW);
    else if(light.type == LightPoint)  ls = evalPointLight(light, geom.posW);
    calcCommonLightProperties(geom, ls);
    return ls;
};

float3 getDiffuseDominantDir(float3 N, float3 V, float roughness)
{
    float a = 1.02341 * roughness - 1.51174;
    float b = -0.511705 * roughness + 0.755868;
    float factor = saturate((saturate(dot(N, V)) * a + b) * roughness);
    return normalize(lerp(N, V, factor));
}

float3 getSpecularDominantDir(float3 N, float3 R, float roughness)
{
    float smoothness = 1 - roughness;
    float factor = smoothness * (sqrt(smoothness) + roughness);
    return normalize(lerp(N, R, factor));
}

float3 evalLightProbeDiffuse(LightProbeData probe, SurfaceGeometry sd, SamplerState lightProbeSampler, float roughness)
{
    float3 N = getDiffuseDominantDir(sd.N, sd.V, roughness);

    // Interpret negative radius as global light probe with infinite distance
    // Otherwise simulate the light probe as covering a finite spherical area
    float2 uv;
    if(probe.radius < 0.0f)
    {
        uv = dirToSphericalCrd(N);
    }
    else
    {
        float3 intersectPosW;
        intersectRaySphere(sd.posW, N, probe.posW, probe.radius, intersectPosW);
        uv = dirToSphericalCrd(normalize(intersectPosW));
    }

    float3 diffuseLighting = probe.resources.diffuseTexture.SampleLevel(lightProbeSampler, uv, 0).rgb;
    float diffF = gProbeShared.dfgTexture.SampleLevel(gProbeShared.dfgSampler, float2(sd.NdotV, roughness), 0).z;

    return diffuseLighting * diffF;
}

float3 evalLightProbeSpecular(LightProbeData probe, SurfaceGeometry sd, SamplerState lightProbeSampler, float3 L, float roughness, float3 specular)
{
    float dfgWidth, dfgHeight;
    gProbeShared.dfgTexture.GetDimensions(dfgWidth, dfgHeight);

    float width, height, mipCount;
    probe.resources.specularTexture.GetDimensions(0, width, height, mipCount);

    float3 dominantDir = getSpecularDominantDir(sd.N, L, roughness);
    float NdotV = max(sd.NdotV, 0.5 / dfgWidth);

    // #TODO This looks a lot closer to reference when passing in roughness instead of linearRoughness. Why?
    float mipLevel = linearRoughnessToLod(roughness, mipCount);

    float2 uv = dirToSphericalCrd(dominantDir);
    float3 ld = probe.resources.specularTexture.SampleLevel(lightProbeSampler, uv, mipLevel).rgb;

    float2 dfg = gProbeShared.dfgTexture.SampleLevel(gProbeShared.dfgSampler, float2(NdotV, roughness), 0).xy;
    
    // ld * (f0 * Gv * (1 - Fc)) + (f90 * Gv * Fc)
    return ld * (specular * dfg.x + dfg.y);
}

/** Evaluate a 2D light-probe filtered using linear-filtering
*/
LightSample evalLightProbeLinear2D(SurfaceGeometry geom, LightProbeData probe, SamplerState lightProbeSampler, float roughness, float linearRoughness, float3 specular)
{
    LightSample ls;

    // Calculate the reflection vector
    ls.L = reflect(-geom.V, geom.N);

    // Evaluate diffuse component
    ls.diffuse = evalLightProbeDiffuse(probe, geom, lightProbeSampler, roughness);

    // Get the specular component
    ls.specular = evalLightProbeSpecular(probe, geom, lightProbeSampler, ls.L, roughness, specular);

    ls.diffuse *= probe.intensity;
    ls.specular *= probe.intensity;
    ls.posW = probe.posW;
    ls.distance = length(probe.posW - geom.posW);
    calcCommonLightProperties(geom, ls);

    return ls;
}

/** Evaluate the properties of a light-probe
*/
LightSample evalLightProbe(SurfaceGeometry geom, LightProbeData probe, SamplerState lightProbeSampler, float roughness, float linearRoughness, float3 specular)
{
    return evalLightProbeLinear2D(geom, probe, lightProbeSampler, roughness, linearRoughness, specular);
}

interface IShadow
{
    float getShadowFactor(SurfaceGeometry geom);
}

struct NoShadow : IShadow
{
    float getShadowFactor(SurfaceGeometry geom)
    {
        return 1.0;
    }
};

struct CsmShadow : IShadow
{
    CsmData csm;
    float getShadowFactor(SurfaceGeometry geom)
    {
        return calcShadowFactor(csm, geom.shadowDepthC, geom.posW, geom.posH.xy/geom.posH.w);
    }
};


struct PointLight<TShadow : IShadow> : ILight
{
    LightData light;
    TShadow shadow;
    float3 getLightDir(float3 p)
    {
        return light.posW - p;
    }
    ShadingResult evalLighting<TBxDF:IBxDF>(ShadingPoint<TBxDF> p, GlobalLightParams gParams)
    {
        LightSample ls = evalPointLight(light, p.geom.posW);
        calcCommonLightProperties(p.geom, ls);
        float shadowFactor = shadow.getShadowFactor(p.geom);
        /* Evaluate material with lighting attributes */
        return p.bxdf.eval(p.geom, ls, shadowFactor);
    }
};

struct DirectionalLight<TShadow : IShadow> : ILight
{
    LightData light;
    TShadow shadow;
    float3 getLightDir(float3 p)
    {
        return light.dirW;
    }
    ShadingResult evalLighting<TBxDF:IBxDF>(ShadingPoint<TBxDF> p, GlobalLightParams gParams)
    {
        LightSample ls = evalDirectionalLight(light, p.geom.posW);
        calcCommonLightProperties(p.geom, ls);
        float shadowFactor = shadow.getShadowFactor(p.geom);
        /* Evaluate material with lighting attributes */
        return p.bxdf.eval(p.geom, ls, shadowFactor);
    }
};

struct ProbeLight : ILight
{
    LightProbeData probeData;
    float3 getLightDir(float3 p)
    {
        return float3(0,0,0);
    }
    ShadingResult evalLighting<TBxDF:IBxDF>(ShadingPoint<TBxDF> p, GlobalLightParams gParams)
    {
        return p.bxdf.evalProbe(p.geom, probeData, gParams);
    }
};

struct AmbientLight : ILight
{
    float3 ambientLight;
    float3 getLightDir(float3 p)
    {
        return float3(0,0,0);
    }
    ShadingResult evalLighting<TBxDF:IBxDF>(ShadingPoint<TBxDF> p, GlobalLightParams gParams)
    {
        return p.bxdf.evalAmbient(ambientLight);
    }
};

struct LightArray<TLight:ILight, let N:int> : ILightCollection
{   
    TLight lights[N];
    int getLightCount()
    {
        return N;
    }
    int filter<TLightFilter:ILightFilter>(int i0, TLightFilter filter, RWStructuredBuffer<int> indexRange, int indexStart)
    {
        int count = 0;
        for (int i = 0; i < N; i++)
        {
            if (filter.filter(lights[i]))
            {
                indexRange[indexStart + count] = i0 + i;
                count++;
            }
        }
        return count;
    }

    TLightIntegrator.AggregateResult integrateRanged<TLightIntegrator:ILightIntegrator>(TLightIntegrator integrator, TLightIntegrator.AggregateResult initVal, int i0, StructuredBuffer<int> range, inout int indexStart, int indexEnd)
    {
        TLightIntegrator.AggregateResult rs = initVal;
        for (; indexStart < indexEnd; indexStart++)
        {
            int index = range[indexStart] - i0;
            if (index >= N)
                break;
            rs = integrator.processLight(rs, lights[index]);
        }
        return rs;
    }

    TLightIntegrator.AggregateResult integrateAll<TLightIntegrator:ILightIntegrator>(TLightIntegrator integrator, TLightIntegrator.AggregateResult initVal)
    {
        TLightIntegrator.AggregateResult rs = initVal;
        for (int i = 0; i < N; i++)
        {
            rs = integrator.processLight(rs, lights[i]);
        }
        return rs;
    }
};

struct LightSingleton<TLight : ILight> : ILightCollection
{
    TLight light;
    int getLightCount() { return 1; }
    int filter<TLightFilter:ILightFilter>(int i0, TLightFilter filter, RWStructuredBuffer<int> indexRange, int indexStart)
    {
        if (filter.filter(light))
        {
            indexRange[indexStart] = i0;
            return 1;
        }
        return 0;
    }

    TLightIntegrator.AggregateResult integrateRanged<TLightIntegrator:ILightIntegrator>(TLightIntegrator integrator, TLightIntegrator.AggregateResult initVal, int i0, StructuredBuffer<int> range, inout int indexStart, int indexEnd)
    {
        TLightIntegrator.AggregateResult rs = initVal;
        if (range[indexStart] == i0)
        {
            rs = integrator.processLight(rs, light);
            indexStart++;
        }
        return rs;
    }
    TLightIntegrator.AggregateResult integrateAll<TLightIntegrator:ILightIntegrator>(TLightIntegrator integrator, TLightIntegrator.AggregateResult initVal)
    {
        TLightIntegrator.AggregateResult rs = integrator.processLight(initVal, light);
        return rs;
    }

};

struct LightPair<TLight1 : ILightCollection, TLight2 : ILightCollection> : ILightCollection
{
    TLight1 light1;
    TLight2 light2;

    int getLightCount() { return light1.getLightCount() + light2.getLightCount(); }

    int filter<TLightFilter:ILightFilter>(int i0, TLightFilter filter, RWStructuredBuffer<int> indexRange, int indexStart)
    {
        int count1 = light1.filter(i0, filter, indexRange, indexStart);
        indexStart += count1;
        int count2 = light2.filter(i0 + light1.getLightCount(), filter, indexRange, indexStart);
        return count1 + count2;
    }

    TLightIntegrator.AggregateResult integrateRanged<TLightIntegrator:ILightIntegrator>(TLightIntegrator integrator, TLightIntegrator.AggregateResult initVal, int i0, StructuredBuffer<int> range, inout int indexStart, int indexEnd)
    {
        TLightIntegrator.AggregateResult rs = initVal;
        rs = light1.integrateRanged(integrator, rs, i0, range, indexStart, indexEnd);
        rs = light2.integrateRanged(integrator, rs, i0, range, indexStart, indexEnd);
        return rs;
    }

    TLightIntegrator.AggregateResult integrateAll<TLightIntegrator:ILightIntegrator>(TLightIntegrator integrator, TLightIntegrator.AggregateResult initVal)
    {
        TLightIntegrator.AggregateResult rs = initVal;
        rs = light1.integrateAll(integrator, rs);
        rs = light2.integrateAll(integrator, rs);
        return rs;
    }
};