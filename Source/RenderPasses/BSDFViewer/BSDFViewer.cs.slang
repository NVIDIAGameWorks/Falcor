/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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
#include "Utils/Math/MathConstants.slangh"
#include "Scene/Material/MaterialDefines.slangh"

import Scene.Scene;
import Scene.Shading;
import Scene.TextureSampler;
import Experimental.Scene.Material.StandardMaterial;
import Utils.Sampling.SampleGenerator;
import Utils.Debug.PixelDebug;
import Utils.Math.BitTricks;
import Utils.Math.MathHelpers;
import Utils.Geometry.IntersectionHelpers;
import BSDFViewerParams;

cbuffer PerFrameCB
{
    BSDFViewerParams gParams;
    EnvMap gEnvMap;
}

RWTexture2D<float4> gOutput;
RWStructuredBuffer<PixelData> gPixelData;

static const float3 kGroundPlaneColor = float3(0.05f);

struct SurfaceData
{
    ShadingData sd;

    // Additional fields we want to inspect that are not part of Falcor's ShadingData.
    float3 baseColor;
    float3 wi;
};


/** Get normalized viewport coordinate.
    The viewport is centered on the image with square aspect and height 1.0. The y-axis points down.
    TODO: Change to a more standard definition.
    \return Viewport coordinate.
*/
float2 getViewportCoord(uint2 pixel)
{
    float2 p = pixel + float2(0.5f);
    return (p - gParams.viewportOffset) * gParams.viewportScale;
}

/** Setup geometric frame of reference for BRDF slice.
    \param[in] uv Viewport coordinate in [0,1].
    \param[out] v Interpolated attributes for the point on the sphere.
    \param[out] viewDir View direction.
    \return Normalized incident direction (light vector).
*/
float3 calculateSliceGeometry(float2 uv, out VertexData v, out float3 viewDir)
{
    // Setup local surface frame as T,B,N.
    v.posW = float3(0, 0, 0);
    v.normalW = float3(0, 0, 1);
    v.tangentW = float4(1, 0, 0, 1);
    v.texC = gParams.texCoords;
    v.faceNormalW = v.normalW;

    // Compute dot products.
    // These are based on the axes in the 2D slice (theta_h, theta_d) with origin in lower-left corner.
    // This is the same format as the slices in Burley et al. 2012, 2015.
    float theta_h = uv.x * (M_PI / 2);
    float theta_d = (1.f - uv.y) * (M_PI / 2);

    float NdotH = cos(theta_h);
    float HdotL = cos(theta_d);     // Note: HdotL = HdotV

    // Place the H vector at (0,0,1) to start.
    // Compute L, V that are mirrored about the yz-plane.
    float3 L = float3(sqrt(1.f - HdotL * HdotL), 0, HdotL);
    float3 V = float3(-L.x, 0.f, L.z);

    // Rotate L, V about the x-axis by an angle theta_h.
    float cos_h = NdotH;
    float sin_h = sqrt(1 - NdotH * NdotH);
    L = float3(L.x, cos_h * L.y - sin_h * L.z, sin_h * L.y + cos_h * L.z);
    V = float3(V.x, cos_h * V.y - sin_h * V.z, sin_h * V.y + cos_h * V.z);

    // Return vectors.
    viewDir = V;
    return normalize(L);
}

/** Calculate sphere geometry for the given viewport coordinate.
    \param[in] uv Viewport coordinate in [0,1].
    \param[out] v Interpolated attributes for the point on the sphere (if hit).
    \param[out] rayDir Ray direction for the camera ray (normalized).
    \return True if we're on the sphere.
*/
bool calculateSphereGeometry(float2 uv, out VertexData v, out float3 rayDir)
{
    const float2 ndc = float2(2.f * uv.x - 1.f, -2.f * uv.y + 1.f);

    if (gParams.orthographicCamera)
    {
        // Calculate intersection with the unit sphere.
        // The orthographic camera's viewport is +-1 units vertically so the sphere fits exactly.
        float3 p = float3(ndc, 0);
        float d = 1.f - p.x * p.x - p.y * p.y;
        rayDir = float3(0, 0, -1);

        if (d < 0.f) return false;
        p.z = sqrt(d);
        v.posW = p;
    }
    else // Projective camera
    {
        // Setup camera ray and calculate ray-sphere intersection.
        float3 origin = { 0, 0, gParams.cameraDistance };
        float3 target = float3(ndc * gParams.cameraViewportScale, 0);
        rayDir = normalize(target - origin);

        float t;
        if (!intersectRaySphere(origin, rayDir, float3(0.f), 1.f, t)) return false;
        v.posW = origin + t * rayDir;
    }

    // Setup surface attributes for the unit sphere.
    v.normalW = v.posW;
    v.tangentW = float4(perp_stark(v.normalW), 1.f); // Make up a tangent
    v.faceNormalW = v.normalW;

    if (gParams.useFixedTexCoords)
    {
        v.texC = gParams.texCoords;
    }
    else
    {
        // Compute texture coords using cylindrical mapping of the visible hemisphere.
        // We place u=0 on the left side and and u=1 on the right, and v=0 at the bottom and v=1 at the top.
        float3 p = v.posW;
        float texU = atan2(p.z, -p.x) / M_PI;
        float texV = acos(-p.y) / M_PI;
        v.texC = float2(texU, texV);
    }

    return true;
}

/** Prepare SurfaceData struct with material parameters.
    All unused fields are initialized to their default values.
*/
SurfaceData prepareMaterial(VertexData v, float3 viewDir)
{
    SurfaceData data = {};

    if (gParams.useSceneMaterial)
    {
        // Setup Falcor's ShadingData based on scene material.
        ExplicitLodTextureSampler lod = { 0.f };
        data.sd = _prepareShadingData(v, gParams.materialID, gScene.materials[gParams.materialID], gScene.materialResources[gParams.materialID], viewDir, lod, gParams.useNormalMapping);

        // Setup additional fields not currently available in ShadingData.
        MaterialData md = gScene.getMaterial(gParams.materialID);
        MaterialResources mr = gScene.materialResources[gParams.materialID];
        data.baseColor = sampleTexture(mr.baseColor, mr.samplerState, v.texC, md.baseColor, EXTRACT_BASE_COLOR_TYPE(md.flags), lod).rgb;
    }
    else
    {
        ShadingData sd = {};

        // Set geometric parameters.
        sd.posW = v.posW;
        sd.uv = v.texC;
        sd.V = normalize(viewDir);
        sd.N = normalize(v.normalW);
        sd.NdotV = dot(sd.N, sd.V);
        sd.faceN = v.faceNormalW;
        sd.frontFacing = dot(sd.V, sd.faceN) >= 0.f;

        computeTangentSpace(sd, v.tangentW);

        // Set material data
        sd.setMaterialType(MaterialType::Standard);
        sd.setDoubleSided(false);
        sd.setThinSurface(false);

        // Set material parameters.
        // Calculate the specular reflectance for dielectrics from the IoR.
        sd.IoR = gParams.IoR;
        float f = (sd.IoR - 1.f) / (sd.IoR + 1.f);
        float F0 = f * f;
        sd.diffuse = lerp(gParams.baseColor.rgb, float3(0), gParams.metallic);
        sd.specular = lerp(float3(F0), gParams.baseColor.rgb, gParams.metallic);
        sd.linearRoughness = gParams.linearRoughness;
        sd.ggxAlpha = sd.linearRoughness * sd.linearRoughness;
        sd.metallic = gParams.metallic;

        // Unused
        sd.opacity = 1;

        // Store outputs
        data.sd = sd;
        data.baseColor = gParams.baseColor;
    }

    data.sd.setActiveLobes(getActiveLobes());

    return data;
}

/** Returns the color to use for background pixels.
    \param[in] uv Viewport coordinates.
    \param[in] dir Normalized ray direction.
*/
float3 evalBackground(float2 uv, float3 dir)
{
    if (gParams.useGroundPlane)
    {
        bool hitGround = gParams.orthographicCamera ? (uv.y >= 0.5f) : (dir.y < 0.f);
        if (hitGround) return kGroundPlaneColor;
    }
    if (gParams.useDirectionalLight) return float3(0);

    float3 L = gParams.useEnvMap ? gEnvMap.eval(dir) : gParams.lightColor;
    return L * gParams.lightIntensity;
}

/** Evaluates the incident lighting from a given direction.
    If directional lighting is enabled, it can be assumed 'dir' is light's direction.
*/
float3 evalLighting(float3 dir)
{
    if (gParams.useGroundPlane && dir.y < 0.f)
    {
        return float3(0.f);
    }

    float3 L = gParams.useEnvMap ? gEnvMap.eval(dir) : gParams.lightColor;
    return L * gParams.lightIntensity;
}

/** Returns the BSDF lobe mask for the currently enabled lobes.
*/
uint getActiveLobes()
{
    uint lobes = 0;
    if (gParams.enableDiffuse) lobes |= (uint)LobeType::DiffuseReflection;
    if (gParams.enableSpecular) lobes |= (uint)LobeType::SpecularReflection | (uint)LobeType::DeltaReflection;
    // TODO: Viewer doesn't support transmission lobes yet
    return lobes;
}

/** Evaluates the BSDF slice for a given viewport coordinate.
    \return Evaluated BSDF value.
*/
float3 evalBSDFSlice<Material : IMaterial>(const Material mtl, const float2 uv, inout SurfaceData data)
{
    // Calculate geometry and incident/outgoing directions.
    VertexData v;
    float3 viewDir;
    float3 lightDir = calculateSliceGeometry(uv, v, viewDir);

    // Setup shading data based on the current material.
    data = prepareMaterial(v, viewDir);
    data.wi = lightDir;

    // Evaluate BRDF at this point.
    float3 f = mtl.eval(data.sd, data.wi);

    // Remove cosine term if it's disabled in the viewer.
    if (!gParams.applyNdotL)
    {
        float NdotL = dot(data.sd.N, data.wi);
        f = NdotL > 0.f ? f / NdotL : float3(0);
    }

    return f;
}

/** Samples the BSDF to evaluate incident illumination.
    This is done differently depending on the configuration.
    \param[in] sd Shading point data.
    \param[in] sg Sample generator.
    \param[out] s Generated sample. Only valid if true is returned.
    \return True if a sample was generated, false otherwise.
*/
bool generateBSDFSample<Material : IMaterial>(const ShadingData sd, const Material mtl, inout SampleGenerator sg, out BSDFSample s)
{
    if (gParams.useDirectionalLight)
    {
        // With directional light, disable BSDF sampling and just return a sample in the light's direction.
        s.wo = -normalize(gParams.lightDir);
        s.weight = mtl.eval(sd, s.wo);
        s.pdf = 1.f;
        return dot(sd.N, s.wo) > 0.f;
    }
    else
    {
        return mtl.sample(sd, sg, s, gParams.useBrdfSampling);
    }
}

/** Evaluates the lit sphere for a given viewport coordinate.
    The viewport shows an analytic sphere of the specified material at infinite distance.
    When each pixel is evaluated using a random light direction and omnidirectional white light,
    the result converges to the total reflectance (integral of BSDF times the dot(N,L) factor.
    \return Outgoing radiance value.
*/
float3 evalSphere<Material : IMaterial>(const Material mtl, const float2 uv, inout SurfaceData data, inout SampleGenerator sg)
{
    // Calculate the local surface frame.
    VertexData v;
    float3 rayDir;
    if (!calculateSphereGeometry(uv, v, rayDir)) return evalBackground(uv, rayDir);

    // Setup shading data based on the current material.
    data = prepareMaterial(v, -rayDir);

    float3 output = 0;
    BSDFSample s = {};
    if (generateBSDFSample(data.sd, mtl, sg, s))
    {
        data.wi = s.wo;
        float3 L = evalLighting(s.wo);

        // Use computed pdf explicitly (for debugging).
        if (gParams.usePdf)
        {
            output = L * mtl.eval(data.sd, s.wo) / s.pdf;
        }
        else
        {
            output = L * s.weight;
        }
    }

    return output;
}

/** BSDF viewer pass entry point.
*/
[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    const uint2 pixel = dispatchThreadID.xy;
    if (any(pixel >= gParams.frameDim)) return;

    printSetPixel(pixel);

    StandardMaterial mtl; // Instance of the standard material.
    SurfaceData data = {};
    float3 output = 0;
    float2 uv = getViewportCoord(pixel);

    if (gParams.sliceViewer)
    {
        if (all(uv >= 0.f && uv < 1.f))
        {
            output = evalBSDFSlice(mtl, uv, data);
        }
    }
    else
    {
        // Create pseudorandom number generator.
        SampleGenerator sg = SampleGenerator.create(pixel, gParams.frameCount);
        output = evalSphere(mtl, uv, data, sg);
    }

    // Write output data.
    gOutput[pixel] = float4(output, 1);

    if (gParams.readback && all(pixel == gParams.selectedPixel))
    {
        PixelData px;
        px.texC = data.sd.uv;
        px.baseColor = data.baseColor;
        px.diffuse = data.sd.diffuse;
        px.specular = data.sd.specular;
        px.linearRoughness = data.sd.linearRoughness;
        px.metallic = data.sd.metallic;
        px.N = data.sd.N;
        px.T = data.sd.T;
        px.B = data.sd.B;
        px.wo = data.sd.V;
        px.wi = data.wi;
        px.output = output;
        gPixelData[0] = px;
    }
}
