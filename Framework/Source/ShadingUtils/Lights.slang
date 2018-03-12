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

// work around compiler issue
#define calcCommonLightProperties(geom, ls) \
    float3 H = normalize(geom.V + ls.L); \
    ls.NdotH = dot(geom.N, H); \
    ls.NdotL = dot(geom.N, ls.L); \
    ls.LdotH = dot(ls.L, H);

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

/** Evaluate a 2D light-probe filtered using linear-filtering
*/
LightSample evalLightProbeLinear2D(SurfaceGeometry geom, LightProbeData probe, SamplerState lightProbeSampler, float roughness, float linearRoughness)
{
    LightSample ls;

    // Calculate the reflection vector
    ls.L = reflect(-geom.V, geom.N);

    // Get the mip-level count
    uint h, w, mipCount;
    probe.resources.origTexture.GetDimensions(0, h, w, mipCount);

    // For the diffuse component, we use the last mip-level
    ls.diffuse = probe.resources.origTexture.SampleLevel(lightProbeSampler, 0, mipCount - 2).rgb;

    // Get the specular component
    float3 specDir = getSpecularDominantDir(geom.N, ls.L, roughness);
    float2 specCrd = dirToSphericalCrd(specDir);
    float lod = linearRoughnessToLod(linearRoughness, mipCount);
    ls.specular = probe.resources.origTexture.SampleLevel(lightProbeSampler, specCrd, lod).rgb;
    
    ls.diffuse *= probe.intensity;
    ls.specular *= probe.intensity;
    ls.posW = probe.posW;
    ls.distance = length(probe.posW - geom.posW);
    calcCommonLightProperties(geom, ls);

    return ls;
}

/** Evaluate the properties of a light-probe
*/
LightSample evalLightProbe(SurfaceGeometry geom, LightProbeData probe, SamplerState lightProbeSampler, float roughness, float linearRoughness)
{
    return evalLightProbeLinear2D(geom, probe, lightProbeSampler, roughness, linearRoughness);
}

struct PointLight : ILight
{
    LightData light;
    ShadingResult evalLighting<TBxDF:IBxDF>(ShadingPoint<TBxDF> p, float shadowFactor, GlobalLightParams gParams)
    {
        LightSample ls = evalPointLight(light, p.geom.posW);
        calcCommonLightProperties(p.geom, ls);

        /* Evaluate material with lighting attributes */
        return p.bxdf.eval(p.geom, ls, shadowFactor);
    }
};

struct DirectionalLight : ILight
{
    LightData light;
    ShadingResult evalLighting<TBxDF:IBxDF>(ShadingPoint<TBxDF> p, float shadowFactor, GlobalLightParams gParams)
    {
        LightSample ls = evalDirectionalLight(light, p.geom.posW);
        calcCommonLightProperties(p.geom, ls);
        
        /* Evaluate material with lighting attributes */
        return p.bxdf.eval(p.geom, ls, shadowFactor);
    }
};

struct ProbeLight : ILight
{
    LightProbeData probeData;

    ShadingResult evalLighting<TBxDF:IBxDF>(ShadingPoint<TBxDF> p, float shadowFactor, GlobalLightParams gParams)
    {
        return p.bxdf.evalProbe(p.geom, probeData, gParams);
    }
};

struct AmbientLight : ILight
{
    float3 ambientLight;
    ShadingResult evalLighting<TBxDF:IBxDF>(ShadingPoint<TBxDF> p, float shadowFactor, GlobalLightParams gParams)
    {
        return p.bxdf.evalAmbient(ambientLight);
    }
};

struct LightArray<TLight:ILight, let N:int> : ILight
{   
    TLight lights[N];
    ShadingResult evalLighting<TBxDF:IBxDF>(ShadingPoint<TBxDF> shAttr, float shadowFactor, GlobalLightParams gParams)
    {
        ShadingResult rs = initShadingResult();
        for (int i = 0; i < N; i++)
        {
            ShadingResult r = lights[i].evalLighting(shAttr, shadowFactor, gParams);
            rs.diffuse += r.diffuse;
            rs.specular += r.specular;
            rs.color += r.color;
            rs.color.a = r.color.a;
        }
        return rs;
    }
};

struct LightPair<TLight1 : ILight, TLight2 : ILight> : ILight
{
    TLight1 light1;
    TLight2 light2;
    ShadingResult evalLighting<TBxDF:IBxDF>(ShadingPoint<TBxDF> shAttr, float shadowFactor, GlobalLightParams gParams)
    {
        ShadingResult rs = initShadingResult();
        ShadingResult r1 = light1.evalLighting(shAttr, shadowFactor, gParams);
        ShadingResult r2 = light2.evalLighting(shAttr, shadowFactor, gParams);
        rs.diffuse = r1.diffuse + r2.diffuse;
        rs.specular = r1.specular + r2.specular;
        rs.color = r1.color + r2.color;
        rs.color.a = r1.color.a;
        return rs;
    }
};