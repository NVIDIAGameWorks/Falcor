#include "HostDeviceSharedMacros.h"

import Interfaces;
import BRDF;
import TextureChannel;
import Helpers;
import Lights;

// This file defines StandardMaterial, an implementation of the IMaterial interface

// this is what used to be the ShadingData class
struct StandardBxDF : IBxDF
{
    // Pre-loaded texture data
    float3 diffuse;
    float  opacity;
    float3 specular;
    float  linearRoughness; // This it the original roughness, before re-mapping. It is required for the Disney diffuse term
    float  roughness;       // This is the re-mapped roughness value, which should be used for GGX computations
    float3 emissive;
    float4 occlusion;
    float3 lightMap;
    float2 height;
    float IoR;
    bool doubleSidedMaterial;
    
    float3 getIntrinsicLighting()
    {
        return lightMap * diffuse + emissive;
    }

    float getOpacity()
    {
        return opacity;
    }

    ShadingResult eval(SurfaceGeometry geom, LightSample ls, float shadowFactor)
    {
        ShadingResult sr = initShadingResult();

        // If the light doesn't hit the surface or we are viewing the surface from the back, return
        if(ls.NdotL <= 0 || geom.NdotV <= 0) return sr;

        // Calculate the diffuse term
        sr.diffuseBrdf = evalDiffuseBrdf(geom.NdotV, ls);
        sr.diffuse = ls.diffuse * sr.diffuseBrdf * ls.NdotL;
        sr.color.rgb = sr.diffuse;
        sr.color.a = this.opacity;

        // Calculate the specular term
        sr.specularBrdf = evalSpecularBrdf(geom.NdotV, ls);
        sr.specular = ls.specular * sr.specularBrdf * ls.NdotL;
        sr.color.rgb += sr.specular;

        // Apply the shadow factor
        sr.color.rgb *= shadowFactor;

        return sr;
    }
    ShadingResult evalAmbient(float3 ambientLight)
    {
        ShadingResult sr = initShadingResult();
        sr.diffuse = diffuse * ambientLight;
        sr.color.rgb = sr.diffuse;
        sr.color.a = this.opacity;
        return sr;
    }
    ShadingResult evalProbe(SurfaceGeometry geom, LightProbeData probe)
    {
        ShadingResult sr = initShadingResult();
        LightSample ls = evalLightProbe(geom, probe, roughness, linearRoughness);

        sr.diffuse = this.diffuse.rgb * ls.diffuse;
        sr.color.rgb = sr.diffuse;
        sr.specular = saturate(evalGGX(this.roughness, ls.NdotH)) * ls.specular * this.specular;
        sr.color.rgb += sr.specular;
        return sr;
    }
    
    float3 evalDiffuseBrdf(float NdotV, LightSample ls)
    {
    #if DiffuseBrdf == DiffuseBrdfLambert
        return evalDiffuseLambertBrdf(diffuse.rgb);
    #elif DiffuseBrdf == DiffuseBrdfDisney
        return evalDiffuseDisneyBrdf(diffuse.rgb, NdotV, linearRoughness, ls);
    #elif DiffuseBrdf == DiffuseBrdfFrostbite
        return evalDiffuseFrostbiteBrdf(diffuse.rgb, NdotV, linearRoughness, ls);
    #endif
    }

    float3 evalSpecularBrdf(float NdotV, LightSample ls)
    {
        float D = evalGGX(roughness, ls.NdotH);
        float G = evalSmithGGX(ls.NdotL, NdotV, roughness);
        float3 F = fresnelSchlick(specular, 1, max(0, ls.LdotH));
        return D * G * F * (1 / M_PI);
    }
};

StandardBxDF initStandardBxDF()
{
    StandardBxDF rs;
    rs.diffuse = 0;
    rs.opacity = 1;
    rs.specular = 0;
    rs.linearRoughness = 0;
    rs.roughness = 0;
    rs.emissive = 0;
    rs.occlusion = 0;
    rs.lightMap = 0;
    rs.height = 0;
    rs.IoR = 0;
    rs.doubleSidedMaterial = false;
    return rs;
}

struct StandardMaterial : IMaterial
{
    typedef StandardBxDF BxDF;
    MaterialData materialData;

    ShadingPoint<StandardBxDF> evalShadingPoint(VertexOut v, float3 camPosW)
    {
        ShadingPoint<StandardBxDF> rs;
        
        SurfaceGeometry geom = initSurfaceGeometry();

        StandardBxDF sd = initStandardBxDF();

        // Sample the diffuse texture and apply the alpha test
        float4 diffuse = sampleTexture(materialData.resources.diffuse, materialData.resources.samplerState, v.texC, materialData.diffuse, EXTRACT_DIFFUSE_TYPE(materialData.flags));
        sd.diffuse = diffuse.rgb;
        sd.opacity = materialData.diffuse.a;
        alphaTest(materialData.flags, diffuse.a, materialData.alphaThreshold, v.posW);

        geom.posW = v.posW;
        geom.uv = v.texC;
        geom.V = normalize(camPosW - v.posW);
        geom.N = normalize(v.normalW);

        geom.B = normalize(v.bitangentW - geom.N * (dot(v.bitangentW, geom.N)));
        geom.T = normalize(cross(geom.B, geom.N));

        // Sample the rest of the texture
        float4 spec = sampleTexture(materialData.resources.specular, materialData.resources.samplerState, v.texC, materialData.specular, EXTRACT_SPECULAR_TYPE(materialData.flags));
        sd.specular = spec.rgb;
        sd.linearRoughness = max(0.08, 1 - spec.a); // Clamp the roughness so that the BRDF won't explode
        sd.roughness = sd.linearRoughness * sd.linearRoughness;
        sd.emissive = sampleTexture(materialData.resources.emissive, materialData.resources.samplerState, v.texC, float4(materialData.emissive, 1), EXTRACT_EMISSIVE_TYPE(materialData.flags)).rgb;
        sd.IoR = materialData.IoR;
        sd.doubleSidedMaterial = EXTRACT_DOUBLE_SIDED(materialData.flags);

        #define channel_type(extract) (extract(materialData.flags) ? ChannelTypeTexture : ChannelTypeUnused)

        sd.occlusion = sampleTexture(materialData.resources.occlusionMap, materialData.resources.samplerState, v.texC, 1, channel_type(EXTRACT_OCCLUSION_MAP));
        sd.lightMap = sampleTexture(materialData.resources.lightMap, materialData.resources.samplerState, v.lightmapC, 1, channel_type(EXTRACT_LIGHT_MAP)).rgb;
        sd.height = sampleTexture(materialData.resources.heightMap, materialData.resources.samplerState, v.texC, 1, channel_type(EXTRACT_HEIGHT_MAP)).xy;
        sd.height = sd.height * materialData.heightScaleOffset.x + materialData.heightScaleOffset.y;
        
        #undef channel_type
    
        applyNormalMap(materialData, geom);
        geom.NdotV = dot(geom.N, geom.V);

        // Flip the normal if it's backfacing
        if(geom.NdotV <= 0 && sd.doubleSidedMaterial)
        {
            geom.N = -geom.N;
            geom.NdotV = -geom.NdotV;
        }

        rs.bxdf = sd;
        rs.geom = geom;

        return rs;
    }
};

/** Apply alpha-test
*/
void applyAlphaTest(MaterialData m, float alpha)
{
    if(EXTRACT_ALPHA_MODE(m.flags) == AlphaModeMask)
    {
        if(alpha < m.alphaThreshold) discard;
    }
}


/** Convert RGB to normal
*/
float3 RgbToNormal(float3 rgb)
{
    float3 n = rgb * 2 - 1;
    return normalize(n);
}

/** Convert RG to normal
*/
float3 RgToNormal(float2 rg)
{
    float3 n;
    n.xy = rg * 2 - 1;

    // Saturate because error from BC5 can break the sqrt
    n.z = saturate(dot(rg, rg)); // z = r*r + g*g
    n.z = sqrt(1 - n.z);
    return normalize(n);
}

/** Apply normal map
*/
void applyNormalMap(MaterialData m, inout SurfaceGeometry geom)
{
    uint mapType = EXTRACT_NORMAL_MAP_TYPE(m.flags);
    if(mapType == NormalMapUnused) return;

    float3 mapN = m.resources.normalMap.Sample(m.resources.samplerState, geom.uv).rgb;
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

    // Apply the transformation. Everything is normalized already
    geom.N = geom.T * mapN.x + geom.B * mapN.y + geom.N * mapN.z;
    geom.B = normalize(geom.B - geom.N * dot(geom.B, geom.N));
    geom.T = normalize(cross(geom.B, geom.N));
}