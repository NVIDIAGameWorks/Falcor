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

#ifndef _FALCOR_SHADING_H_
#define _FALCOR_SHADING_H_

// Make sure we get the macros like `_fn` and `_ref`
// TODO: just eliminate these since we know this is pure Slang.
#include "HostDeviceData.h"

__import Helpers;

/*******************************************************************
Documentation
*******************************************************************/

// Possible user-defined global macros to configure shading 
//#define _MS_DISABLE_ALPHA_TEST    ///< Disables alpha test (handy for early-z etc.)
//#define _MS_DISABLE_TEXTURES      ///< Disables material texture fetches, all materials are solid color
//#define _MS_USER_DERIVATIVES      ///< Use user-passed dx/dy derivatives for texture filtering during shading
//#define _MS_USER_NORMAL_MAPPING   ///< Use user-defined callback of a form void PerturbNormal(SMaterial mat, inout ShadingAttribs ShAttr) to perform normal mapping
//#define _MS_NUM_LAYERS            ///< Use a pre-specified number of layers to use in the material
//#define _MS_IMPORTANCE_SAMPLING   ///< Use BSDF importance sampling for path tracer next event estimation

/*******************************************************************
Shading routines
*******************************************************************/

/**
This is an output resulting radiance of a full pass over all material layers
after evaluating the material for a single light source.
This result can be accumulated to a total shading result for all light sources.
*/
struct PassOutput
{
    float3  value;                  ///< Outgoing radiance after evaluating the complete material for a single light
    float3  diffuseAlbedo;          ///< Total view-independent albedo color component of the material
    float3  diffuseIllumination;    ///< Total view-independent outgoing radiance from one light before multiplication by albedo
    float3  specularAlbedo;         ///< Total view-dependent albedo (specular) color component of the material
    float3  specularIllumination;   ///< Total view-dependent outgoing radiance from one light before multiplication by albedo

    float2  roughness;              ///< Roughness from the last layer
    float2  effectiveRoughness;     ///< Accumulated effective roughness
};

/**
This is an accumulative structure that collects the resulting output
of material evaluations with all light sources.
*/
struct ShadingOutput
{
    float3  diffuseAlbedo;          ///< Total view-independent albedo color component of the material
    float3  diffuseIllumination;    ///< Total view-independent outgoing radiance from all lights before multiplication by albedo
    float3  specularAlbedo;         ///< Total view-dependent albedo (specular) color component of the material
    float3  specularIllumination;   ///< Total view-dependent outgoing radiance from all light before multiplication by albedo

    float3  finalValue;             ///< Outgoing radiance after evaluating the complete material for all lights

    float3  wi;                     ///< Incident direction after importance sampling the BRDF
    float   pdf;                    ///< Probability density function for choosing the incident direction
    float3  thp;                    ///< Current path throughput

    float2  effectiveRoughness;     ///< Sampled brdf roughness, or for evaluated material, an effective roughness
};

/* Include all shading routines, including endpoints (camera and light) */

__import Cameras;
__import Lights;
__import BSDFs;

/*******************************************************************
Material shading building blocks
*******************************************************************/

/**
Prepares all light-independent attributes needed to perform shading at a hit point.
This includes fetching all textures and computing final shading attributes, like albedo and roughness.
After this step, one can save these shading attributes, e.g., in a G-Buffer to perform lighting afterwards.
This routine also applies all material modifiers, like performs alpha test and applies a normal map.
*/
void _fn prepareShadingAttribs(MaterialData material, in float3 P, in float3 camPos,
    in float3 normal, in float3 bitangent, in float2 uv,
#ifdef _MS_USER_DERIVATIVES
    float2 dPdx, float2 dPdy,
#else
    float lodBias,
#endif
    _ref(ShadingAttribs) shAttr)
{
    /* Prepare shading attributes */
    shAttr.P = P;
    shAttr.E = normalize(camPos - P);
    shAttr.N = normalize(normal);
    shAttr.B = normalize(bitangent - shAttr.N * (dot(bitangent, shAttr.N)));
    shAttr.T = normalize(cross(shAttr.B, shAttr.N));
    shAttr.UV = uv;
#ifdef _MS_USER_DERIVATIVES
    shAttr.DPDX = dPdx;
    shAttr.DPDY = dPdy;
#else
    shAttr.lodBias = lodBias;
#endif


    /* Copy the input material parameters */
#ifdef _MS_STATIC_MATERIAL_DESC
    MaterialDesc desc = _MS_STATIC_MATERIAL_DESC;
#else
    MaterialDesc desc = material.desc;
#endif

    shAttr.preparedMat.values = material.values;
    shAttr.preparedMat.desc = desc;

    /* Evaluate alpha test material modifier */
    applyAlphaTest(material, shAttr, P);
    shAttr.aoFactor = 1;

    /* Prefetch textures */
    loop_unroll
    for(uint iLayer = 0 ; iLayer < MatMaxLayers ; iLayer++)
    {
        if(shAttr.preparedMat.desc.layers[iLayer].type == MatNone) break;

        shAttr.preparedMat.values.layers[iLayer].albedo = 
            evalWithColor(desc.layers[iLayer].hasTexture, material.textures.layers[iLayer], material.samplerState, material.values.layers[iLayer].albedo, shAttr);
    }

    /* Perturb shading normal is needed */
    perturbNormal(material, shAttr);
}

/**
Another overload of PrepareShadingAttribs(), which does not require tangents.
Instead it constructs a world-space axis-aligned tangent frame on the fly.
*/
void _fn prepareShadingAttribs(MaterialData material, in float3 P, in float3 camPos,
    in float3 normal, in float2 uv,
#ifdef _MS_USER_DERIVATIVES
    float2 dPdx, float2 dPdy,
#else
    float lodBias,
#endif
    _ref(ShadingAttribs) shAttr)
{
    /* Generate an axis-aligned tangent frame */
    float3 bitangent;
    createTangentFrame(normal, bitangent);

    prepareShadingAttribs(material, P, camPos, normal, bitangent, uv,
#ifdef _MS_USER_DERIVATIVES
        dPdx, dPdy,
#else
        lodBias,
#endif
        shAttr);
}

#ifndef _MS_USER_DERIVATIVES
// Legacy version of prepareShadingAttribs() without LOD bias
void _fn prepareShadingAttribs(MaterialData material, in float3 P, in float3 camPos, in float3 normal, in float2 uv, _ref(ShadingAttribs) shAttr)
{
    prepareShadingAttribs(material, P, camPos, normal, uv, 0, shAttr);
}

void _fn prepareShadingAttribs(MaterialData material, in float3 P, in float3 camPos, in float3 normal, in float3 bitangent, in float2 uv, _ref(ShadingAttribs) shAttr)
{
    prepareShadingAttribs(material, P, camPos, normal, bitangent, uv, 0, shAttr);
}
#endif

float4 _fn evalEmissiveLayer(MaterialLayerValues layer, _ref(PassOutput) result)
{
    result.diffuseAlbedo += 1.f;
    result.diffuseIllumination += layer.albedo.rgb;
    return (1).rrrr;
}

float4 _fn evalDiffuseLayer(MaterialLayerValues layer, float3 lightIntensity, float3 lightDir, float3 normal, _ref(PassOutput) result)
{
    float3 value = lightIntensity;
    float weight = 0;
    float4 albedo = layer.albedo;
    result.roughness = 1;
#ifndef _MS_DISABLE_DIFFUSE
    value *= evalDiffuseBSDF(normal, lightDir);
    weight = albedo.w;
    result.diffuseAlbedo += albedo.rgb * layer.pmf;
    result.diffuseIllumination += value;
#else
    value = 0;
#endif
    return float4(value, weight);
}

// Implementation of NDF filtering code from the HPG'16 submission
// The writeup is here: //research/graphics/projects/halfvectorSpace/SAA/paper/specaa-sub.pdf
float2 _fn filterRoughness(ShadingAttribs shAttr, LightAttribs lAttr, in float2 roughness)
{
#ifdef _MS_USER_HALF_VECTOR_DERIVATIVES
    float2  hppDx = shAttr.DHDX;
    float2  hppDy = shAttr.DHDY;
#else
    // Compute half-vector derivatives
    float3  H = normalize(shAttr.E + lAttr.L);
    float2  hpp = float2(dot(H, shAttr.T), dot(H, shAttr.B));
    float2  hppDx = ddx_fine(hpp);
    float2  hppDy = ddy_fine(hpp);
#endif
    // Compute filtering region
    float2 rectFp = (abs(hppDx) + abs(hppDy)) * 0.5f;

    // For grazing angles where the first-order footprint goes to very high values
    // Usually you don’t need such high values and the maximum value of 1.0 or even 0.1 is enough for filtering.
    rectFp = min(0.7f, rectFp);

    // Covariance matrix of pixel filter's Gaussian (remapped in roughness units)
    float2 covMx = rectFp * rectFp * 2.f;   // Need to x2 because roughness = sqrt(2) * pixel_sigma_hpp

    roughness = sqrt(roughness*roughness + covMx);          // Beckmann proxy convolution for GGX

    return roughness;
}

float4 _fn evalSpecularLayer(MaterialLayerDesc desc, MaterialLayerValues data, ShadingAttribs shAttr, LightAttribs lAttr, _ref(PassOutput) result)
{
#ifndef _MS_DISABLE_SPECULAR
    /* Add albedo regardless of facing */
    result.specularAlbedo += data.albedo.rgb * data.pmf;

    /* Ignore the layer if it's a transmission or backfacing */
    if (dot(lAttr.L, shAttr.N) * dot(shAttr.E, shAttr.N) <= 0.f)
    {
        return 0;
    }

    float3 value = lAttr.lightIntensity;

    float2 roughness;
    if(desc.hasTexture & ROUGHNESS_CHANNEL_BIT)
    {
        roughness = float2(data.albedo.w, data.albedo.w);
    }
    else
    {
        roughness = data.roughness.rg;
    }

#ifndef _MS_DISABLE_ROUGHNESS_FILTERING
    roughness = filterRoughness(shAttr, lAttr, roughness);
#endif

    // Respect perfect specular cutoff
    if(max(roughness.x, roughness.y) < 1e-3f)
    {
        return 0;
    }
    
    // compute halfway vector
    float3 hW = normalize(shAttr.E + lAttr.L);
    float3 h = normalize(float3(dot(hW, shAttr.T), dot(hW, shAttr.B), dot(hW, shAttr.N)));

    result.roughness = roughness;

    switch (desc.ndf)
    {
    case NDFBeckmann: /* Beckmann microfacet distribution */
    {
        value *= evalBeckmannDistribution(h, roughness);
    }
    break;
    case NDFGGX: /* GGX */
    {
        value *= evalGGXDistribution(h, roughness);
    }
    break;
    }

    /* Evaluate standard microfacet model terms */
    value *= evalMicrofacetTerms(shAttr.T, shAttr.B, shAttr.N, h, shAttr.E, lAttr.L, roughness, desc.ndf, (desc.type) == MatDielectric);

    /* Cook-Torrance Jacobian */
    value /= 4.f * dot(shAttr.E, shAttr.N);

    /* Fresnel conductor/dielectric term */
    float HoE = dot(hW, shAttr.E);
    float IoR = data.extraParam.x;
    float kappa = data.extraParam.y;
    float F_term = (desc.type == MatConductor) ? conductorFresnel(HoE, IoR, kappa) : 1.f - dielectricFresnel(HoE, IoR);
    value *= F_term;
    float weight = F_term;

    result.specularIllumination += value;

    return float4(value, weight);
#else
    return 0;
#endif
}

float3 _fn blendLayer(float4 albedo, float4 layerOut, uint blendType, float3 currentValue)
{
    /* Account for albedo */
    float3 scaledLayerOut = layerOut.rgb * albedo.rgb;
    float weight = layerOut.w;

    /* Perform layer blending */
    if(blendType == BlendConstant)
    {
        weight = albedo.w;
    }

    float3 result;
    if(blendType != BlendAdd)
    {
        result = lerp(currentValue, scaledLayerOut, weight);
    }
    else
    {
        result = currentValue + scaledLayerOut;
    }
    return result;
}

/**
A helper routine for evaluating a single layer of a layered material, given a shading point and a point on a light source.
The routine takes a layer index, as well as shading attributes, including prepared layers of the material;
it also takes prepared attributes of a light source, such as incident radiance and a world-space direction to the light.
The output is the illumination of the current layer, blended into the results of the previous layers with a specified blending mode.
*/
void _fn evalMaterialLayer(int iLayer, ShadingAttribs attr, LightAttribs lAttr,
    _ref(PassOutput) result)
{
    float4 value = 0;

    MaterialLayerValues values = attr.preparedMat.values.layers[iLayer];
    MaterialLayerDesc desc = attr.preparedMat.desc.layers[iLayer];
    switch(desc.type)
    {
    case MatLambert: /* Diffuse BRDF */
        value = evalDiffuseLayer(values, lAttr.lightIntensity, lAttr.L, attr.N, result) * lAttr.shadowFactor;
        break;
    case MatEmissive:
        value = evalEmissiveLayer(values, result);
        break;
    case MatConductor:
    case MatDielectric:
        value = evalSpecularLayer(desc, values, attr, lAttr, result) * lAttr.shadowFactor;
        break;
    };

    float3 oldValue = result.value;
    result.value = blendLayer(values.albedo, value, desc.blending, result.value);

    float delta = max(1e-3f, luminance(result.value - oldValue));
    result.effectiveRoughness += result.roughness * delta;
}


/**	The highest-level material evaluation function.
This is the main routing for evaluating a complete PBR material, given a shading point and a light source.
Should be called once per light.
*/
void _fn evalMaterial(
    ShadingAttribs shAttr,
    LightAttribs lAttr,
    _ref(ShadingOutput) result,
    bool initializeShadingOut DEFAULTS(false))
{
    /* If it's the first pass, initialize all the aggregates to zero */
    if(initializeShadingOut)
    {
        result.diffuseAlbedo = 0;
        result.diffuseIllumination = 0;
        result.specularAlbedo = 0;
        result.specularIllumination = 0;
        result.finalValue = 0;
        result.effectiveRoughness = 0;
        result.wi = 0;
        result.pdf = 0;
        result.thp = 0;
    }

    /* Go through all layers and perform a layer-by-layer shading and compositing */
    PassOutput passResult;
    passResult.value = 0;
    passResult.diffuseAlbedo = 0;
    passResult.diffuseIllumination = 0;
    passResult.specularAlbedo = 0;
    passResult.specularIllumination = 0;
    passResult.roughness = 0;
    passResult.effectiveRoughness = 0;
    [unroll]
    for(uint iLayer = 0 ; iLayer < MatMaxLayers ; iLayer++)
    {
        if(shAttr.preparedMat.desc.layers[iLayer].type == MatNone) break;
        evalMaterialLayer(iLayer, shAttr, lAttr, passResult);
    }

    /* Accumulate the results of the pass */
    result.finalValue += passResult.value;
    result.effectiveRoughness += passResult.effectiveRoughness / max(1e-3f, luminance(passResult.value));
    result.diffuseIllumination += passResult.diffuseIllumination;
    result.specularIllumination += passResult.specularIllumination;
    result.diffuseAlbedo = passResult.diffuseAlbedo;
    result.specularAlbedo = passResult.specularAlbedo;
}

void _fn evalMaterial(
    ShadingAttribs shAttr,
    LightData light,
    float shadowFactor,
    _ref(ShadingOutput) result,
    bool initializeShadingOut DEFAULTS(false))
{
    /* Prepare lighting attributes */
    LightAttribs LAttr;
    prepareLightAttribs(light, shAttr, shadowFactor, LAttr);

    /* Evaluate material with lighting attributes */
    evalMaterial(shAttr, LAttr, result, initializeShadingOut);
}

/**
Another overload of material evaluation function, which prepares light attributes internally.
*/
void _fn evalMaterial(
    ShadingAttribs shAttr,
    LightData light,
    _ref(ShadingOutput) result,
    bool initializeShadingOut DEFAULTS(false))
{
    evalMaterial(shAttr, light, 1, result, initializeShadingOut);
}

/*******************************************************************
Material building helpers
*******************************************************************/

/**
Initializes a material layer with an empty layer
*/
void _fn initNullLayer(_ref(MaterialLayerDesc) layer)
{
    layer.type = MatNone;
    layer.blending = BlendAdd;
    layer.hasTexture = false;
    layer.ndf = NDFUser;
}

/**
Initializes a material layer with diffuse BRDF
*/
void _fn initDiffuseLayer(_ref(MaterialLayerDesc) desc, _ref(MaterialLayerValues) data, float3 albedo)
{
    desc.type = MatLambert;
    desc.blending = BlendAdd;
    desc.hasTexture = false;
    desc.ndf = NDFGGX;
    data.albedo = float4(albedo, 1.f);
    data.extraParam = float4(0,0,0,0);
    data.roughness = float4(0,0,0,0);
    data.pad = float3(0,0,0);
    data.pmf = 1;
}

/**
Initializes a material layer with conductor BRDF. IoR and Kappa are set to the values of chrome by default.
*/
void _fn initConductorLayer(_ref(MaterialLayerDesc) desc, _ref(MaterialLayerValues) data, float3 color, float roughness, float IoR = 3.f, float kappa = 4.2f)
{
    desc.type = MatConductor;
    desc.blending = BlendAdd;
    data.albedo = float4(color, 1.f);
    data.roughness = roughness.rrrr;
    data.extraParam.x = IoR;
    data.extraParam.y = kappa;
    data.pmf = 1;
}

/**
Initializes a material layer with dielectric BRDF.
*/
void _fn initDielectricLayer(_ref(MaterialLayerDesc) desc, _ref(MaterialLayerValues) data, float3 color, float roughness, float IoR)
{
    desc.type = MatDielectric;
    desc.blending = BlendFresnel;
    data.albedo = float4(color, 1.f);
    data.roughness = roughness.rrrr;
    data.extraParam.x = IoR;
    data.pmf = 1;
}

/*******************************************************************
Simple material helpers
*******************************************************************/

/**
Tries to find a layer data for a given material type (diffuse/conductor/etc).
\param[in] material         Material to look in
\param[in] layerType        Material layer Type (MatLambert/MatConductor etc.)
\param[out] data           Layer data, if found
returns false if the layer is not found, true otherwise
*/
bool _fn getLayerByType(PreparedMaterialData material, uint layerType, _ref(MaterialLayerValues) data, _ref(MaterialLayerDesc) desc)
{
    int layerId = material.desc.layerIdByType[layerType].id;
    if(layerId != -1)
    {
        data = material.values.layers[layerId];
        desc = material.desc.layers[layerId];
        return true;
    }
    return false;
}

/**
Tries to find a diffuse albedo color for a given material
\param[in] Material Material to look in
returns black if the layer is not found, diffuse albedo color otherwise
*/
float4 _fn getDiffuseColor(ShadingAttribs shAttr)
{
    float4 ret = 0;
    // This is here because the HLSL compiler complains about 'data' not being completely initialized when used
    int layerId = shAttr.preparedMat.desc.layerIdByType[MatLambert].id;
    if (layerId != -1)
    {
        MaterialLayerValues data;
        MaterialLayerDesc   desc;
        getLayerByType(shAttr.preparedMat, MatLambert, data, desc);
        ret = data.albedo;
    }
    return ret;
}

/**
Tries to override a diffuse albedo color for all layers within a given material
\param[in] Material Material to look in
returns true if succeeded
*/
bool _fn overrideDiffuseColor(_ref(ShadingAttribs) shAttr, float4 albedo, bool allLayers = true)
{
    bool found = false;
    if(!allLayers)
    {
        int layerId = shAttr.preparedMat.desc.layerIdByType[MatLambert].id;
        if(layerId != -1)
        {
            found = true;
            shAttr.preparedMat.values.layers[layerId].albedo = albedo;
        }
    }
    else
    {
        for(uint iLayer = 0 ; iLayer < MatMaxLayers ; iLayer++)
        {
            if(shAttr.preparedMat.desc.layers[iLayer].type == MatNone) break;

            if(shAttr.preparedMat.desc.layers[iLayer].type == MatLambert)
            {
                shAttr.preparedMat.values.layers[iLayer].albedo = albedo;
                found = true;
            }
        }
    }
    return found;
}

/**
Tries to find a specular albedo color for a given material
\param[in] material Material to look in
returns black if the layer is not found, specular color otherwise
*/
float4 _fn getSpecularColor(ShadingAttribs shAttr)
{
    float4 ret = 0;
    MaterialLayerValues data;
    MaterialLayerDesc   desc;
    if(getLayerByType(shAttr.preparedMat, MatConductor, data, desc))
    {
        ret = data.albedo;
    }
    return ret;
}

/**
    This routine importance samples the BRDF for path tracer next event
    estimation. A new direction is chosen based on the material properties.

    \param[in] shAttr Complete information about the shading point
    \param[in] rSample Random numbers for sampling
    \param[out] result Gather incident direction, probability density function, and path throughput
*/
void _fn sampleMaterial(
    ShadingAttribs shAttr,
    in float2 rSample,
    _ref(ShadingOutput) result)
{
    bool sampleDiffuse = true;

    // Set the initial path throughput
    result.thp = 0;
    result.diffuseAlbedo = 0;
    result.specularAlbedo = 0;

    // Sample specular layer if it exists
    MaterialLayerValues specData;
    MaterialLayerDesc   specDesc;
    if(getLayerByType(shAttr.preparedMat, MatConductor, specData, specDesc))
    {
        // Randomly sample the specular component by using the probability mass function
        if (rSample.x < specData.pmf)
        {
            // Rescale random number value
            rSample.x /= specData.pmf;

            sampleDiffuse = false;

            float3 m;
            float3 wo = toLocal(shAttr.E, shAttr.T, shAttr.B, shAttr.N);
            float2 roughness = float2(specData.roughness.x, specData.roughness.y);

            // Set the specular reflectivity
            result.specularAlbedo = specData.albedo.rgb;
            result.thp = result.specularAlbedo;

            // Choose the appropriate normal distribution function
            switch(specDesc.ndf)
            {
            case NDFBeckmann: // Beckmann normal distribution function 
            {
                result.thp *= sampleBeckmannDistribution(wo, roughness, rSample, m, result.wi, result.pdf);
            }
            break;

            case NDFGGX: // GGX normal distribution function
            {
                result.thp *= sampleGGXDistribution(wo, roughness, rSample, m, result.wi, result.pdf);
            }
            break;
            }

            // Convert direction vector from local to global frame
            // global space coordinates
            result.wi = fromLocal(result.wi, shAttr.T, shAttr.B, shAttr.N);

            // Evaluate standard microfacet model terms
            result.thp *= evalMicrofacetTerms(shAttr.T, shAttr.B, shAttr.N, m, shAttr.E, result.wi, roughness, specDesc.ndf, (specDesc.type) == MatDielectric);

            // Fresnel conductor/dielectric term
            float HoE = dot(m, shAttr.E);
            float IoR = specData.extraParam.x;
            float kappa = specData.extraParam.y;
            float F_term = (specDesc.type == MatConductor) ? conductorFresnel(HoE, IoR, kappa) : 1.f - dielectricFresnel(HoE, IoR);
            result.thp *= F_term;

            result.effectiveRoughness = roughness;
        }
        else
        {
            // Rescale random number value
            rSample.x -= specData.pmf;
            rSample.x /= 1.f - specData.pmf;
        }
    }

    // Sample the diffuse component
    if (sampleDiffuse)
    {
        // Cosine lobe sampling
        result.wi = cosine_sample_hemisphere(rSample.x, rSample.y);

        // Convert direction vector from local to global frame
        // global space coordinates
        result.wi = fromLocal(result.wi, shAttr.T, shAttr.B, shAttr.N);

        // Ideally thp = (\rho / \pi) * |\omega_i . n| / bsdfPdf
        // By importance sampling the cosine lobe, we can set bsdfPdf = |\omega_i . n| / \pi
        // Thus, we can simplify thp = \rho
        result.thp = getDiffuseColor(shAttr).rgb;
        result.diffuseAlbedo = result.thp;

        // Probability density function for perfect importance sampling of cosine lobe 
        result.pdf = M_1_PIf;

        // Record roughness for diffuse brdf
        result.effectiveRoughness = 1;
    }
}

#endif	// _FALCOR_SHADING_H_
