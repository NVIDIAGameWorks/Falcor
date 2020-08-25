/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

/** This file contains legacy light evaluation functions used by rasterization.
    It should be merged/replaced with the BSDF code for ray tracing.
*/

#include "Utils/Math/MathConstants.slangh"

import Utils.Helpers;
import Scene.ShadingData;
__exported import Scene.Lights.LightData;
__exported import Scene.Lights.LightProbeData;

struct LightSample
{
    float3 diffuse;   // The light intensity at the surface location used for the diffuse term
    float3 specular;  // The light intensity at the surface location used for the specular term. For light probes, the diffuse and specular components are different
    float3 L;         // The direction from the surface to the light source
    float3 posW;      // The world-space position of the light
    float NdotH;      // Unclamped, can be negative
    float NdotL;      // Unclamped, can be negative
    float LdotH;      // Unclamped, can be negative
    float distance;   // Distance from the light-source to the surface
};

void calcCommonLightProperties(ShadingData sd, inout LightSample ls)
{
    float3 H = normalize(sd.V + ls.L);
    ls.NdotH = dot(sd.N, H);
    ls.NdotL = dot(sd.N, ls.L);
    ls.LdotH = dot(ls.L, H);
};

float getDistanceFalloff(float distSquared)
{
    float falloff = 1 / ((0.01 * 0.01) + distSquared); // The 0.01 is to avoid infs when the light source is close to the shading point
    return falloff;
}

RayDesc getShadowRay(LightData light, float3 origin)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.TMin = 0.f;
    ray.TMax = 0.f;
    if (light.type == uint(LightType::Directional))
    {
        ray.Direction = -light.dirW;
        ray.TMax = FLT_MAX;
    }
    else if (light.type == uint(LightType::Point))
    {
        float3 direction = light.posW - origin;
        ray.Direction = normalize(direction);
        ray.TMax = FLT_MAX;
    }
    return ray;
}

/** Evaluate a directional light source intensity/direction at a shading point
*/
LightSample evalDirectionalLight(in LightData light, in float3 surfacePosW)
{
    LightSample ls;
    ls.diffuse = light.intensity;
    ls.specular = light.intensity;
    ls.L = -normalize(light.dirW);
    ls.posW = float3(0.f);
    ls.distance = FLT_MAX;
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
    if (cosTheta < light.cosOpeningAngle)
    {
        falloff = 0;
    }
    else if (light.penumbraAngle > 0)
    {
        float deltaAngle = light.openingAngle - acos(cosTheta);
        falloff *= saturate((deltaAngle - light.penumbraAngle) / light.penumbraAngle);
    }
    ls.diffuse = light.intensity * falloff;
    ls.specular = ls.diffuse;
    return ls;
}

float linearRoughnessToLod(float linearRoughness, float mipCount)
{
    return sqrt(linearRoughness) * (mipCount - 1);
}

/** Evaluate a light source intensity/direction at a shading point
*/
LightSample evalLight(LightData light, ShadingData sd)
{
    LightSample ls = {};
    if (light.type == uint(LightType::Directional))
    {
        ls = evalDirectionalLight(light, sd.posW);
    }
    else if (light.type == uint(LightType::Point))
    {
        ls = evalPointLight(light, sd.posW);
    }
    else
    {
        return ls;
    }
    calcCommonLightProperties(sd, ls);
    return ls;
};

float3 getDiffuseDominantDir(float3 N, float3 V, float ggxAlpha)
{
    float a = 1.02341 * ggxAlpha - 1.51174;
    float b = -0.511705 * ggxAlpha + 0.755868;
    float factor = saturate((saturate(dot(N, V)) * a + b) * ggxAlpha);
    return normalize(lerp(N, V, factor));
}

float3 getSpecularDominantDir(float3 N, float3 R, float ggxAlpha)
{
    float smoothness = 1 - ggxAlpha;
    float factor = smoothness * (sqrt(smoothness) + ggxAlpha);
    return normalize(lerp(N, R, factor));
}

float3 evalLightProbeDiffuse(LightProbeData probe, ShadingData sd)
{
    float3 N = getDiffuseDominantDir(sd.N, sd.V, sd.ggxAlpha);

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
        uv = dirToSphericalCrd(normalize(intersectPosW - probe.posW));
    }

    float width, height, mipCount;
    probe.resources.diffuseTexture.GetDimensions(0, width, height, mipCount);

    float3 diffuseLighting = probe.resources.diffuseTexture.SampleLevel(probe.resources.sampler, uv, 0).rgb;
    float preintegratedDisneyBRDF = probe.sharedResources.dfgTexture.SampleLevel(probe.sharedResources.dfgSampler, float2(sd.NdotV, sd.ggxAlpha), 0).z;

    return diffuseLighting * preintegratedDisneyBRDF * sd.diffuse.rgb;
}

float3 evalLightProbeSpecular(LightProbeData probe, ShadingData sd, float3 L)
{
    float dfgWidth, dfgHeight;
    probe.sharedResources.dfgTexture.GetDimensions(dfgWidth, dfgHeight);

    float width, height, mipCount;
    probe.resources.specularTexture.GetDimensions(0, width, height, mipCount);

    float3 dominantDir = getSpecularDominantDir(sd.N, L, sd.ggxAlpha);
    float mipLevel = linearRoughnessToLod(sd.ggxAlpha, mipCount);

    float2 uv;
    if (probe.radius < 0.0f)
    {
        uv = dirToSphericalCrd(dominantDir);
    }
    else
    {
        float3 intersectPosW;
        intersectRaySphere(sd.posW, dominantDir, probe.posW, probe.radius, intersectPosW);
        uv = dirToSphericalCrd(normalize(intersectPosW - probe.posW));
    }

    float3 ld = probe.resources.specularTexture.SampleLevel(probe.resources.sampler, uv, mipLevel).rgb;

    float2 dfg = probe.sharedResources.dfgTexture.SampleLevel(probe.sharedResources.dfgSampler, float2(sd.NdotV, sd.ggxAlpha), 0).xy;

    // ld * (f0 * Gv * (1 - Fc)) + (f90 * Gv * Fc)
    return ld * (sd.specular * dfg.x + dfg.y);
}

/** Evaluate a 2D light-probe filtered using linear-filtering
*/
LightSample evalLightProbeLinear2D(LightProbeData probe, ShadingData sd)
{
    LightSample ls;

    // Calculate the reflection vector
    ls.L = reflect(-sd.V, sd.N);

    // Evaluate diffuse component
    ls.diffuse = evalLightProbeDiffuse(probe, sd);

    // Get the specular component
    ls.specular = evalLightProbeSpecular(probe, sd, ls.L);

    ls.diffuse *= probe.intensity;
    ls.specular *= probe.intensity;
    ls.posW = probe.posW;
    ls.distance = length(probe.posW - sd.posW);
    calcCommonLightProperties(sd, ls);

    return ls;
}

/** Evaluate the properties of a light-probe
*/
LightSample evalLightProbe(LightProbeData probe, ShadingData sd)
{
    return evalLightProbeLinear2D(probe, sd);
}
