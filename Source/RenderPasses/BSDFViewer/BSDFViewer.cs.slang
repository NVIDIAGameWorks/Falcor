/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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

import Scene.Shading;
import Scene.Material.ShadingUtils;
import Utils.Sampling.SampleGenerator;
import Utils.Debug.PixelDebug;
import Utils.Math.BitTricks;
import Utils.Math.MathHelpers;
import Utils.Geometry.IntersectionHelpers;
import BSDFViewerParams;

struct BSDFViewer
{
    BSDFViewerParams params;
    EnvMap envMap;
    RWTexture2D<float4> outputColor;
    RWStructuredBuffer<PixelData> pixelData;

    static const float3 kGroundPlaneColor = float3(0.05f);

    struct SurfaceData
    {
        ShadingData sd;
        BSDFProperties bsdfProperties;
        float3 wo;
    };

    /** Get normalized viewport coordinate.
        The viewport is centered on the image with square aspect and height 1.0. The y-axis points down.
        TODO: Change to a more standard definition.
        \return Viewport coordinate.
    */
    float2 getViewportCoord(uint2 pixel)
    {
        float2 p = pixel + float2(0.5f);
        return (p - params.viewportOffset) * params.viewportScale;
    }

    /** Setup geometric frame of reference for BSDF slice.
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
        v.texC = params.texCoords;
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

        if (params.orthographicCamera)
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
            float3 origin = { 0, 0, params.cameraDistance };
            float3 target = float3(ndc * params.cameraViewportScale, 0);
            rayDir = normalize(target - origin);

            float t;
            if (!intersectRaySphere(origin, rayDir, float3(0.f), 1.f, t)) return false;
            v.posW = origin + t * rayDir;
        }

        // Setup surface attributes for the unit sphere.
        v.normalW = v.posW;
        v.tangentW = float4(perp_stark(v.normalW), 1.f); // Make up a tangent
        v.faceNormalW = v.normalW;

        if (params.useFixedTexCoords)
        {
            v.texC = params.texCoords;
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

    /** Prepare SurfaceData struct with shading data.
        All unused fields are initialized to their default values.
    */
    SurfaceData prepareShadingData(const VertexData v, const float3 viewDir, const ITextureSampler lod)
    {
        SurfaceData data = {};

        // Setup Falcor's ShadingData based on the selected scene material and lobes.
        data.sd = gScene.materials.prepareShadingData(v, params.materialID, viewDir, lod, params.useNormalMapping);
        data.sd.mtl.setActiveLobes(getActiveLobes());

        return data;
    }

    /** Returns the color to use for background pixels.
        \param[in] uv Viewport coordinates.
        \param[in] dir Normalized ray direction.
    */
    float3 evalBackground(float2 uv, float3 dir)
    {
        if (params.useGroundPlane)
        {
            bool hitGround = params.orthographicCamera ? (uv.y >= 0.5f) : (dir.y < 0.f);
            if (hitGround) return kGroundPlaneColor;
        }
        if (params.useDirectionalLight) return float3(0);

        float3 L = params.useEnvMap ? envMap.eval(dir) : params.lightColor;
        return L * params.lightIntensity;
    }

    /** Evaluates the incident lighting from a given direction.
        If directional lighting is enabled, it can be assumed 'dir' is light's direction.
    */
    float3 evalLighting(float3 dir)
    {
        if (params.useGroundPlane && dir.y < 0.f)
        {
            return float3(0.f);
        }

        float3 L = params.useEnvMap ? envMap.eval(dir) : params.lightColor;
        return L * params.lightIntensity;
    }

    /** Returns the BSDF lobe mask for the currently enabled lobes.
    */
    uint getActiveLobes()
    {
        uint lobes = 0;
        if (params.enableDiffuse) lobes |= (uint)LobeType::DiffuseReflection;
        if (params.enableSpecular) lobes |= (uint)LobeType::SpecularReflection | (uint)LobeType::DeltaReflection;
        // TODO: Viewer doesn't support transmission lobes yet
        return lobes;
    }

    /** Evaluates the BSDF slice for a given viewport coordinate.
        \return Evaluated BSDF value.
    */
    float3 evalBSDFSlice(const float2 uv, const ITextureSampler lod, inout SurfaceData data, inout SampleGenerator sg)
    {
        // Calculate geometry and incident/outgoing directions.
        VertexData v;
        float3 viewDir;
        float3 lightDir = calculateSliceGeometry(uv, v, viewDir);

        // Setup shading data.
        data = prepareShadingData(v, viewDir, lod);
        data.wo = lightDir;

        // Create BSDF instance.
        let bsdf = gScene.materials.getBSDF(data.sd, lod);
        data.bsdfProperties = bsdf.getProperties(data.sd);

        // Evaluate BSDF at shading point.
        float3 f = bsdf.eval(data.sd, data.wo, sg);

        // Remove cosine term if it's disabled in the viewer.
        if (!params.applyNdotL)
        {
            float NdotL = dot(data.sd.N, data.wo);
            f = NdotL > 0.f ? f / NdotL : float3(0);
        }

        return f;
    }

    /** Samples the BSDF to evaluate incident illumination.
        This is done differently depending on the configuration.
        \param[in] sd Shading data.
        \param[in] bsdf BSDF instance.
        \param[in,out] sg Sample generator.
        \param[out] s Generated sample. Only valid if true is returned.
        \return True if a sample was generated, false otherwise.
    */
    bool generateBSDFSample(const ShadingData sd, const IBSDF bsdf, inout SampleGenerator sg, out BSDFSample s)
    {
        if (params.useDirectionalLight)
        {
            // With directional light, disable BSDF sampling and just return a sample in the light's direction.
            s.wo = -normalize(params.lightDir);
            s.weight = bsdf.eval(sd, s.wo, sg);
            s.pdf = 1.f;
            return dot(sd.N, s.wo) > 0.f;
        }
        else
        {
            return bsdf.sample(sd, sg, s, params.useImportanceSampling);
        }
    }

    float3 getAlbedo(const BSDFProperties bsdfProperties)
    {
        float3 albedo = {};
        if (params.outputAlbedo & (uint)AlbedoSelection::DiffuseReflection) albedo += bsdfProperties.diffuseReflectionAlbedo;
        if (params.outputAlbedo & (uint)AlbedoSelection::DiffuseTransmission) albedo += bsdfProperties.diffuseTransmissionAlbedo;
        if (params.outputAlbedo & (uint)AlbedoSelection::SpecularReflection) albedo += bsdfProperties.specularReflectionAlbedo;
        if (params.outputAlbedo & (uint)AlbedoSelection::SpecularTransmission) albedo += bsdfProperties.specularTransmissionAlbedo;
        return albedo;
    }

    /** Evaluates the lit sphere for a given viewport coordinate.
        The viewport shows an analytic sphere of the specified material at infinite distance.
        When each pixel is evaluated using a random light direction and omnidirectional white light,
        the result converges to the total reflectance (integral of BSDF times the dot(N,L) factor.
        \return Outgoing radiance value.
    */
    float3 evalSphere(const float2 uv, const ITextureSampler lod, inout SurfaceData data, inout SampleGenerator sg)
    {
        // Calculate the local surface frame.
        VertexData v;
        float3 rayDir;
        if (!calculateSphereGeometry(uv, v, rayDir)) return evalBackground(uv, rayDir);

        // Setup shading data.
        data = prepareShadingData(v, -rayDir, lod);

        // Create BSDF instance.
        let bsdf = gScene.materials.getBSDF(data.sd, lod);
        data.bsdfProperties = bsdf.getProperties(data.sd);

        float3 output = 0;
        if (params.outputAlbedo & (uint)AlbedoSelection::ShowAlbedo)
        {
            output = getAlbedo(data.bsdfProperties);
        }
        else
        {
            // Sample BSDF at shading point.
            BSDFSample s = {};

            if (generateBSDFSample(data.sd, bsdf, sg, s))
            {
                data.wo = s.wo;
                float3 L = evalLighting(s.wo);

                // Use computed pdf explicitly (for debugging).
                if (params.usePdf)
                {
                    output = L * bsdf.eval(data.sd, s.wo, sg) / s.pdf;
                }
                else
                {
                    output = L * s.weight;
                }
            }
        }

        return output;
    }

    /** BSDF viewer entry point.
    */
    void execute(const uint2 pixel)
    {
        if (any(pixel >= params.frameDim)) return;

        printSetPixel(pixel);

        SurfaceData data = {};
        float3 output = 0;
        float2 uv = getViewportCoord(pixel);

        // TODO: Implement texture level-of-detail. For now sample at mip 0.
        let lod = ExplicitLodTextureSampler(0.f);

        SampleGenerator sg = SampleGenerator(pixel, params.frameCount);
        if (params.viewerMode == BSDFViewerMode::Slice)
        {
            if (all(uv >= 0.f && uv < 1.f))
            {
                output = evalBSDFSlice(uv, lod, data, sg);
            }
        }
        else if (params.viewerMode == BSDFViewerMode::Material)
        {
            // Create pseudorandom number generator.
            output = evalSphere(uv, lod, data, sg);
        }

        // Write output data.
        outputColor[pixel] = float4(output, 1);

        if (all(pixel == params.selectedPixel))
        {
            PixelData px = {};

            px.texC = data.sd.uv;
            px.N = data.sd.N;
            px.T = data.sd.T;
            px.B = data.sd.B;
            px.wi = data.sd.V;
            px.wo = data.wo;
            px.output = output;

            // BSDF properties
            px.emission = data.bsdfProperties.emission;
            px.roughness = data.bsdfProperties.roughness;
            px.diffuseReflectionAlbedo = data.bsdfProperties.diffuseReflectionAlbedo;
            px.diffuseTransmissionAlbedo = data.bsdfProperties.diffuseTransmissionAlbedo;
            px.specularReflectionAlbedo = data.bsdfProperties.specularReflectionAlbedo;
            px.specularTransmissionAlbedo = data.bsdfProperties.specularTransmissionAlbedo;
            px.specularReflectance = data.bsdfProperties.specularReflectance;
            px.isTransmissive = data.bsdfProperties.isTransmissive ? 1 : 0;

            pixelData[0] = px;
        }
    }
};

ParameterBlock<BSDFViewer> gBSDFViewer;

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    gBSDFViewer.execute(dispatchThreadID.xy);
}
