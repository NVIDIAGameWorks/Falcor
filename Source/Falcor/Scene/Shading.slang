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
#include "Scene/Material/MaterialDefines.slangh"

__exported import Scene.ShadingData;
import Scene.SceneTypes;
import Scene.TextureSampler;
import Scene.Material.MaterialData;
import Experimental.Scene.Material.BxDFTypes;
import Experimental.Scene.Lights.EnvMapLighting;
import Utils.Math.MathHelpers;
import Utils.Helpers;

/** Convert RGB to normal (unnormalized).
*/
float3 RgbToNormal(float3 rgb)
{
    return rgb * 2.f - 1.f;
}

/** Convert RG to normal (unnormalized).
*/
float3 RgToNormal(float2 rg)
{
    float3 n;
    n.xy = rg * 2 - 1;

    // Saturate because error from BC5 can break the sqrt
    n.z = saturate(dot(rg, rg)); // z = r*r + g*g
    n.z = sqrt(1 - n.z);
    return n;
}

/** Apply normal map

    The `lod` parameter represents the method to use for computing
    texture level of detail, and must implement the `ITextureSampler` interface.
*/
void applyNormalMap<L:ITextureSampler>(MaterialData md, MaterialResources mr, inout ShadingData sd, float4 tangentW, L lod)
{
    uint mapType = EXTRACT_NORMAL_MAP_TYPE(md.flags);
    if (mapType == NormalMapUnused) return;

    float3 mapN = lod.sampleTexture(mr.normalMap, mr.samplerState, sd.uv).xyz;
    switch(mapType)
    {
    case NormalMapRGB:
        mapN = RgbToNormal(mapN);
        break;
    case NormalMapRG:
        mapN = RgToNormal(mapN.rg);
        break;
    default:
        return;
    }

    // Note if the normal ends up being parallel to the tangent, the tangent frame cannot be orthonormalized.
    // That case is rare enough that it is probably not worth the runtime cost to check for it here.
    // If it occurs we should foremost fix the asset, or if problems persist add a check here.

    // Apply the transformation.
    sd.N = normalize(sd.T * mapN.x + sd.B * mapN.y + sd.N * mapN.z);
    sd.T = normalize(tangentW.xyz - sd.N * dot(tangentW.xyz, sd.N));
    sd.B = cross(sd.N, sd.T) * tangentW.w;
}

/** Internal implementation of `alphaTest`.
    The `lod` parameter represents the method to use for computing texture level of detail, and must implement the `ITextureSampler` interface.
    \return True if hit should be ignored/discarded.
*/
bool _alphaTest<L:ITextureSampler>(VertexData v, MaterialData md, MaterialResources mr, L lod)
{
    if (EXTRACT_ALPHA_MODE(md.flags) != AlphaModeMask) return false;

    // Load opacity from the alpha channel of the diffuse texture.
    float alpha = sampleTexture(mr.baseColor, mr.samplerState, v.texC, md.baseColor, EXTRACT_DIFFUSE_TYPE(md.flags), lod).a;
    return evalAlphaTest(md.flags, alpha, md.alphaThreshold, v.posW);
}

/** Apply alpha test and return the result.
    This version samples alpha using implicit gradients and only works in pixel shaders.
    \return True if hit should be ignored/discarded.
*/
bool alphaTest(VertexData v, MaterialData md, MaterialResources mr)
{
    ImplicitLodTextureSampler lod = {};
    return _alphaTest(v, md, mr, lod);
}

/** Apply alpha test and return the result.
    This version samples alpha at a level of detail specified by the `lod` parameter.
    \return True if hit should be ignored/discarded.
*/
bool alphaTest(VertexData v, MaterialData md, MaterialResources mr, float lod)
{
    ExplicitLodTextureSampler explicitLOD = { lod };
    return _alphaTest(v, md, mr, explicitLOD);
}

/** Apply alpha test and return the result.
    This version samples alpha at a level of detail computed from screen-space gradients `gradX` and `gradY`.
    \return True if hit should be ignored/discarded.
*/
bool alphaTest(VertexData v, MaterialData md, MaterialResources mr, float2 gradX, float2 gradY)
{
    ExplicitGradientTextureSampler lod = { gradX, gradY };
    return _alphaTest(v, md, mr, lod);
}

void applyAlphaTest(uint matFlags, float alpha, float threshold, float3 posW)
{
#ifndef _MS_DISABLE_ALPHA_TEST
    if (evalAlphaTest(matFlags, alpha, threshold, posW))
        discard;
#endif
}

/** Internal implementation of `prepareShadingData`

    The `lod` parameter represents the method to use for computing
    texture level of detail, and must implement the `ITextureSampler` interface.
*/
ShadingData _prepareShadingData<L:ITextureSampler>(VertexData v, uint materialID, MaterialData md, MaterialResources mr, float3 viewDir, L lod, bool useNormalMap)
{
    ShadingData sd = {};

#ifdef _MS_STATIC_MATERIAL_FLAGS
    md.flags = _MS_STATIC_MATERIAL_FLAGS;
#endif

    // Sample the diffuse texture and apply the alpha test.
    float4 baseColor = sampleTexture(mr.baseColor, mr.samplerState, v.texC, md.baseColor, EXTRACT_DIFFUSE_TYPE(md.flags), lod);
    sd.opacity = baseColor.a;
    applyAlphaTest(md.flags, baseColor.a, md.alphaThreshold, v.posW);

    sd.posW = v.posW;
    sd.uv = v.texC;
    sd.V = viewDir;
    sd.N = v.normalW;

    sd.faceN = v.faceNormalW;
    sd.frontFacing = dot(sd.V, sd.faceN) >= 0.f;
    sd.doubleSided = EXTRACT_DOUBLE_SIDED(md.flags);
    sd.activeLobes = (uint)LobeType::All;
    sd.materialID = materialID;

    sd.IoR = md.IoR;
    sd.specularTransmission = sampleTexture(mr.specularTransmission, mr.samplerState, v.texC, md.specularTransmission, EXTRACT_SPEC_TRANS_TYPE(md.flags), lod).r;
    sd.eta = sd.frontFacing ? (1 / sd.IoR) : sd.IoR;

    // Setup tangent space.
    bool validTangentSpace = computeTangentSpace(sd, v.tangentW);

    // Sample the specular texture.
    // Depending on the shading model this encodes the material parameters differently.
    sd.occlusion = 1.0f;
    bool sampleOcclusion = EXTRACT_OCCLUSION_MAP(md.flags) > 0;
    float4 spec = sampleTexture(mr.specular, mr.samplerState, v.texC, md.specular, EXTRACT_SPECULAR_TYPE(md.flags), lod);

    if (EXTRACT_SHADING_MODEL(md.flags) == ShadingModelMetalRough)
    {
        // R - Occlusion; G - Roughness; B - Metallic
        sd.diffuse = lerp(baseColor.rgb, float3(0), spec.b);

        // Calculate the specular reflectance for dielectrics from the IoR, as in the Disney BSDF [Burley 2015].
        // UE4 uses 0.08 multiplied by a default specular value of 0.5, hence F0=0.04 as default. The default IoR=1.5 gives the same result.
        float f = (sd.IoR - 1.f) / (sd.IoR + 1.f);
        float F0 = f * f;

        sd.specular = lerp(float3(F0), baseColor.rgb, spec.b);
        sd.linearRoughness = spec.g;
        sd.metallic = spec.b;
        if (sampleOcclusion) sd.occlusion = spec.r;
    }
    else // if (EXTRACT_SHADING_MODEL(md.flags) == ShadingModelSpecGloss)
    {
        sd.diffuse = baseColor.rgb;
        sd.specular = spec.rgb;
        sd.linearRoughness = 1 - spec.a;
        sd.metallic = getMetallic(sd.diffuse, sd.specular);

        if (sampleOcclusion)
        {
            sd.occlusion = sampleTexture(mr.occlusionMap, mr.samplerState, v.texC, 1, ChannelTypeTexture, lod);
        }
    }

    // Compute GGX alpha value.
    // TODO: ggxAlpha should be removed and computed where it's needed.
    // The threshold of 0.0064 is because in previous versions we used to clamp linearRoughness to 0.08.
    sd.ggxAlpha = max(0.0064, sd.linearRoughness * sd.linearRoughness);

    // Sample the emissive texture. Note that triangles are emissive only on the front-facing side.
    if (sd.frontFacing)
    {
        sd.emissive = sampleTexture(mr.emissive, mr.samplerState, v.texC, float4(md.emissive, 1), EXTRACT_EMISSIVE_TYPE(md.flags), lod).rgb * md.emissiveFactor;
    }

    // Apply normal mapping only if we have a valid tangent space.
    if (useNormalMap && validTangentSpace) applyNormalMap(md, mr, sd, v.tangentW, lod);
    sd.NdotV = dot(sd.N, sd.V);

    // Flip the shading normal for back-facing hits on double-sided materials.
    if (!sd.frontFacing && sd.doubleSided)
    {
        sd.N = -sd.N;
        sd.NdotV = -sd.NdotV;
    }

    return sd;
}

/** Prepare the hit-point data
*/
ShadingData prepareShadingData(VertexData v, uint materialID, MaterialData md, MaterialResources mr, float3 viewDir)
{
    ImplicitLodTextureSampler lod = { };
    return _prepareShadingData(v, materialID, md, mr, viewDir, lod, true);
}

/** Prepare the hit-point data

    The `lod` parameter represents the level of detail to use for all material
    texture fetches.
*/
ShadingData prepareShadingData(VertexData v, uint materialID, MaterialData md, MaterialResources mr, float3 viewDir, float lod)
{
    ExplicitLodTextureSampler explicitLOD = { lod };
    return _prepareShadingData(v, materialID, md, mr, viewDir, explicitLOD, true);
}

/** Prepare the hit-point data when using ray cones for texture LOD.
    \param[in] v Vertex data.
    \param[in] materialID Material identifier.
    \param[in] md Material data.
    \param[in] mr Material resources.
    \param[in] camPosW Position of camera in world space.
    \param[in] lodMinusTexDim Texture level-of-details for ray cones with the texture dimensions subtracted out. See TextureSampler and TexLODHelpers.
*/
ShadingData prepareShadingDataUsingRayConesLOD(VertexData v, uint materialID, MaterialData md, MaterialResources mr, float3 camPosW, float lodMinusTexDim)
{
    ExplicitRayConesLodTextureSampler explicitRayConesLOD = { lodMinusTexDim };
    return _prepareShadingData(v, materialID, md, mr, camPosW, explicitRayConesLOD, true);
}

/** Prepare the hit-point data when using ray differentials for texture LOD.
    \param[in] v Vertex data.
    \param[in] materialID Material identifier.
    \param[in] md Material data.
    \param[in] mr Material resources.
    \param[in] camPosW Position of camera in world space.
    \param[in] dUVdx Differential texture coordinates in x.
    \param[in] dUVdy Differential texture coordinates in y.
*/
ShadingData prepareShadingDataUsingRayDiffsLOD(VertexData v, uint materialID, MaterialData md, MaterialResources mr, float3 camPosW, float2 dUVdx, float2 dUVdy)
{
    ExplicitRayDiffsLodTextureSampler explicitRayDiffsLOD = { dUVdx, dUVdy };
    return _prepareShadingData(v, materialID, md, mr, camPosW, explicitRayDiffsLOD, true);
}

/** Prepare the hit-point data

    The `gradX` and `gradY` parameters should be the screen-space gradients of
    `v.texC` with respect to screen-space X and Y, respectively. These gradient
    values will be used for all material texture fetches.
*/
ShadingData prepareShadingData(VertexData v, uint materialID, MaterialData md, MaterialResources mr, float3 viewDir, float2 gradX, float2 gradY)
{
    ExplicitGradientTextureSampler lod = { gradX, gradY };
    return _prepareShadingData(v, materialID, md, mr, viewDir, lod, true);
}

/** Computes an orthonormal tangent space based on the normal and given tangent.
    \param[in,out] sd ShadingData struct that is updated.
    \param[in] tangent Interpolated tangent in world space (xyz) and bitangent sign (w). The tangent is *only* valid when w is != 0.
    \return True if a valid tangent space was computed based on the supplied tangent.
*/
bool computeTangentSpace(inout ShadingData sd, const float4 tangentW)
{
    // Check that tangent space exists and can be safely orthonormalized.
    // Otherwise invent a tanget frame based on the normal.
    // We check that:
    //  - Tangent exists, this is indicated by a nonzero sign (w).
    //  - It has nonzero length. Zeros can occur due to interpolation or bad assets.
    //  - It is not parallel to the normal. This can occur due to normal mapping or bad assets.
    //  - It does not have NaNs. These will propagate and trigger the fallback.

    float NdotT = dot(tangentW.xyz, sd.N);
    bool nonParallel = abs(NdotT) < 0.9999f;
    bool nonZero = dot(tangentW.xyz, tangentW.xyz) > 0.f;

    bool valid = tangentW.w != 0.f && nonZero && nonParallel;
    if (valid)
    {
        sd.T = normalize(tangentW.xyz - sd.N * NdotT);
        sd.B = cross(sd.N, sd.T) * tangentW.w;
    }
    else
    {
        sd.T = perp_stark(sd.N);
        sd.B = cross(sd.N, sd.T);
    }

    return valid;
}

/** Helper function to adjust the shading normal for avoding black pixels due to back-facing view direction.
    Note: This breaks the reciprocity of the BSDF!
*/
void adjustShadingNormal(inout ShadingData sd, VertexData v)
{
    float3 Ng = sd.frontFacing ? v.faceNormalW : -v.faceNormalW;
    float3 Ns = sd.N;

    // Blend the shading normal towards the geometric normal at grazing angles.
    // This is to avoid the view vector from becoming back-facing.
    const float kCosThetaThreshold = 0.1f;
    float cosTheta = dot(sd.V, Ns);
    if (cosTheta <= kCosThetaThreshold)
    {
        float t = saturate(cosTheta * (1.f / kCosThetaThreshold));
        sd.N = normalize(lerp(Ng, Ns, t));
        sd.NdotV = dot(sd.N, sd.V);
        computeTangentSpace(sd, v.tangentW);
    }
}

// ----------------------------------------------------------------------------
// Legacy raster shading code
// ----------------------------------------------------------------------------

import Scene.Lights.Lights;
import Scene.Material.BRDF;

/** Shading result struct
*/
struct ShadingResult
{
    float3 diffuseBrdf;             // The result of the diffuse BRDF
    float3 specularBrdf;            // The result of the specular BRDF
    float3 diffuse;                 // The diffuse component of the result
    float3 specular;                // The specular component of the result
    float3 color;                   // The final color
};

ShadingResult evalMaterial(ShadingData sd, LightData light, float shadowFactor)
{
    ShadingResult sr = {};
    LightSample ls = evalLight(light, sd);

    // If the light doesn't hit the surface or we are viewing the surface from the back, return
    if (ls.NdotL <= 0) return sr;
    sd.NdotV = saturate(sd.NdotV);

    // Calculate the diffuse term
    sr.diffuseBrdf = evalDiffuseBrdf(sd, ls);
    sr.diffuse = ls.diffuse * sr.diffuseBrdf * ls.NdotL;
    sr.color = sr.diffuse;

    // Calculate the specular term
    sr.specularBrdf = evalSpecularBrdf(sd, ls);
    sr.specular = ls.specular * sr.specularBrdf * ls.NdotL;
    sr.color += sr.specular;

    // Apply the shadow factor
    sr.color *= shadowFactor;

    return sr;
};

ShadingResult evalMaterial(ShadingData sd, EnvMapLighting envMapLighting)
{
    ShadingResult sr = {};

    // Calculate the reflection vector
    float3 L = reflect(-sd.V, sd.N);

    sr.diffuse = envMapLighting.evalDiffuse(sd);
    sr.specular = envMapLighting.evalSpecular(sd, L);
    sr.color = sr.diffuse + sr.specular;

    return sr;
}
