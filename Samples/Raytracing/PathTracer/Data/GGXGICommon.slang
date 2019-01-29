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
import ShaderCommon;
import Raytracing;
import Shading;
import BRDF;
import Lights;
import Helpers;
#include "HostDeviceSharedMacros.h"
#include "HostDeviceData.h"

// A constant buffer we'll populate from our C++ code  (used for our ray generation shader)
shared cbuffer GlobalCB
{
    float gMinT; // Min distance to start a ray to avoid self-occlusion
    uint gFrameCount; // An integer changing every frame to update the random number
    bool gDoIndirectGI; // A boolean determining if we should shoot indirect GI rays
    bool gDoDirectGI; // A boolean determining if we should compute direct lighting
    uint gMaxDepth; // Maximum number of recursive bounces to allow
    float gEmitMult; // Emissives are currently hacked.  Multiply emissive amount by this factor
}

shared ITexture2D gPos;
shared ITexture2D gNorm;
shared ITexture2D gDiffuseMatl;
shared ITexture2D gSpecMatl;
shared ITexture2D gExtraMatl;
shared ITexture2D gEmissive;
shared IRWTexture2D gOutput;
shared ITexture2D gEnvMap;

// Payload for our shadow rays.
struct ShadowRayPayload
{
    float visFactor; // Will be 1.0 for fully lit, 0.0 for fully shadowed
};

// The payload structure for our indirect rays
struct IndirectRayPayload
{
    float3 color; // The (returned) color in the ray's direction
    uint rndSeed; // Our random seed, so we pick uncorrelated RNGs along our ray
    uint rayDepth; // What is the depth of our current ray?
};

/** Returns true if surface is transparent
*/
bool evalRtAlphaTest(BuiltInTriangleIntersectionAttributes attribs)
{
    VertexOut vsOut = getVertexAttributes(PrimitiveIndex(), attribs);

    // Extracts the diffuse color from the material (the alpha component is opacity)
    ExplicitLodTextureSampler lodSampler = { 0 };
    float4 baseColor = sampleTexture(gMaterial.resources.baseColor, gMaterial.resources.samplerState,
        vsOut.texC, gMaterial.baseColor, EXTRACT_DIFFUSE_TYPE(gMaterial.flags), lodSampler);

    return evalBasicAlphaTest(baseColor.a, gMaterial.alphaThreshold);
}

/** Shoots a shadow ray using given parameters. Assumes hit group #0 and miss shader #0;
    \return Visibility factor: 1 if no shadow, and 0 if shadowed.
*/
float shootShadowRay(float3 origin, float3 direction, float minT, float maxT)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = minT;
    ray.TMax = maxT;

    // Our shadow rays are *assumed* to hit geometry; this miss shader changes this to 1.0 for "visible"
    ShadowRayPayload payload = { 0.0f };

    // Query if anything is between the current point and the light
    TraceRay(gRtScene,
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
        0xFF, 0, hitProgramCount, 0, ray, payload);

    // Return our ray payload (which is 1 for visible, 0 for occluded)
    return payload.visFactor;
}

/** Evaluates direct lighting and shadow from a single randomly selected light in the scene.

    \return Final evaluated color
*/
float3 ggxDirect(inout uint rndSeed, ShadingData sd)
{
    // Pick a random light from our scene to sample
    int lightIndex = min(int(rand_next(rndSeed) * gLightsCount), gLightsCount - 1);

    LightData ld = gLights[lightIndex];
    ShadingResult sr = evalMaterial(sd, gLights[lightIndex], 1.0f);

    // Shoot our shadow ray to our randomly selected light
    float distToLight = length(ld.posW - sd.posW);
    float3 L = -normalize(ld.dirW);
    float shadowMult = float(gLightsCount) * shootShadowRay(sd.posW, L, gMinT, distToLight);

    return shadowMult * sr.color.rgb;
}

/** Helper for shooting a ray for indirect lighting. Assumes hit group #1 and miss shader #1

    \return Color sampled by the ray.
*/
float3 shootIndirectRay(float3 rayOrigin, float3 rayDir, float minT, uint curPathLen, uint seed, uint curDepth)
{
    // Setup our indirect ray
    RayDesc rayColor;
    rayColor.Origin = rayOrigin;
    rayColor.Direction = rayDir;
    rayColor.TMin = minT;
    rayColor.TMax = 1.0e38f;

    // Initialize the ray's payload data with black return color and the current rng seed
    IndirectRayPayload payload;
    payload.color = float3(0, 0, 0);
    payload.rndSeed = seed;
    payload.rayDepth = curDepth + 1;

    // Trace our ray to get a color in the indirect direction.  Use hit group #1 and miss shader #1
    TraceRay(gRtScene, 0, 0xFF, 1, hitProgramCount, 1, rayColor, payload);

    // Return the color we got from our ray
    return payload.color;
}

/** Our material has have both a diffuse and a specular lobe.
    Calculates what probability the diffuse one should be sampled.

    \return Probability between 0 and 1
*/
float probabilityToSampleDiffuse(float3 difColor, float3 specColor)
{
    float lumDiffuse = max(0.01f, luminance(difColor.rgb));
    float lumSpecular = max(0.01f, luminance(specColor.rgb));
    return lumDiffuse / (lumDiffuse + lumSpecular);
}

/** Evaluates indirect lighting up to a bounce count specified by gMaxDepth. Initial call should leave rayDepth = 0.
*/
float3 ggxIndirect(inout uint rndSeed, ShadingData sd, float3 geomN, uint rayDepth = 0)
{
    // We have to decide whether we sample our diffuse or specular/ggx lobe.
    float probDiffuse = probabilityToSampleDiffuse(sd.diffuse, sd.specular);
    float chooseDiffuse = (rand_next(rndSeed) < probDiffuse);
    float2 randVal = float2(rand_next(rndSeed), rand_next(rndSeed));

    // We'll need NdotV for both diffuse and specular
    float NdotV = saturate(dot(sd.N, sd.V));

    // If we randomly selected to sample our diffuse lobe
    if (chooseDiffuse)
    {
        // Shoot a randomly selected cosine-sampled diffuse ray.
        float3 L = getCosHemisphereSample(randVal, sd.N, getPerpendicularStark(sd.N));
        float3 bounceColor = shootIndirectRay(sd.posW, L, gMinT, 0, rndSeed, rayDepth);

        // Check to make sure our randomly selected, normal mapped diffuse ray didn't go below the surface.
        if (dot(geomN, L) <= 0.0f) bounceColor = float3(0, 0, 0);

        // Accumulate the color: (NdotL * incomingLight * diff / pi)
        // Probability of sampling:  (NdotL / pi) * probDiffuse
        return bounceColor * sd.diffuse / probDiffuse;
    }
    // Otherwise we randomly selected to sample our GGX lobe
    else
    {
        // Randomly sample the NDF to get a microfacet in our BRDF to reflect off of
        float3 H = getGGXMicrofacet(randVal, sd.N, sd.roughness);

        // Compute the outgoing direction based on this (perfectly reflective) microfacet
        float3 L = reflect(-sd.V, H);

        // Compute our color by tracing a ray in this direction
        float3 bounceColor = shootIndirectRay(sd.posW, L, gMinT, 0, rndSeed, rayDepth);

        // Check to make sure our randomly selected, normal mapped diffuse ray didn't go below the surface.
        if (dot(geomN, L) <= 0.0f) bounceColor = float3(0, 0, 0);

        // Compute some dot products needed for shading
        float NdotL = saturate(dot(sd.N, L));
        float NdotH = saturate(dot(sd.N, H));
        float LdotH = saturate(dot(L, H));

        // Cannot use evalSpecularBrdf() because we need the D term below
        float D = evalGGX(sd.roughness, NdotH) * M_INV_PI; // Our GGX function does not include division by PI
        float G = evalSmithGGX(NdotL, NdotV, sd.roughness); // Includes division by 4 * NdotL * NdotV
        float3 F = fresnelSchlick(sd.specular, 1, max(0, LdotH));
        float3 brdf = D * G * F;

        // Probability of sampling vector H from getGGXMicrofacet()
        float ggxProb = D * NdotH / (4 * LdotH);

        // Accumulate the color:  ggx-BRDF * incomingLight * NdotL / probability-of-sampling
        return NdotL * bounceColor * brdf / (ggxProb * (1.0f - probDiffuse));
    }
}
