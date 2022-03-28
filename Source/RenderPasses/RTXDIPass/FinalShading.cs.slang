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
import LoadShadingData;
import Scene.RaytracingInline;
import Rendering.Materials.Microfacet;
import Rendering.RTXDI.RTXDI;
import Utils.Sampling.TinyUniformSampleGenerator;

RWTexture2D<float4> gColor;
RWTexture2D<float4> gEmission;
RWTexture2D<float4> gDiffuseIllumination;
RWTexture2D<float4> gDiffuseReflectance;
RWTexture2D<float4> gSpecularIllumination;
RWTexture2D<float4> gSpecularReflectance;

#define is_valid(name) (is_valid_##name != 0)

/** Perform shading with final samples from ReSTIR.
*/
struct FinalShading
{
    // Static configuration.
    static const bool kUseEnvBackground = USE_ENV_BACKGROUND;

    Texture2D<PackedHitInfo> vbuffer;

    uint2 frameDim;

    void execute(const uint2 pixel)
    {
        if (any(pixel >= frameDim)) return;

        float3 color = {};
        float3 emission = {};
        float3 diffuseReflectance = {};
        float3 diffuseIllumination = {};
        float3 specularReflectance = {};
        float3 specularIllumination = {};
        float hitT = 10000.f; // TODO: Remove magic number also used in SpatioTemporalResampling pass.

        ShadingData sd;
        let lod = ExplicitLodTextureSampler(0.f); // TODO: Implement texture level-of-detail.
        if (loadShadingData(pixel, frameDim, gScene.camera, vbuffer, lod, sd))
        {
            // Create BSDF instance and query its properties.
            let bsdf = gScene.materials.getBSDF(sd, lod);
            let bsdfProperties = bsdf.getProperties(sd);

            // Get final sample from RTXDI.
            float3 dir;
            float distance;
            float3 Li;
            bool valid = gRTXDI.getFinalSample(pixel, dir, distance, Li);

            // Check light visibility.
            if (valid)
            {
                // Create a DXR 1.1 query object to trace a ray (the <1> means use alpha testing)
                SceneRayQuery<1> rayQuery;
                const Ray ray = Ray(sd.computeNewRayOrigin(), dir, 0.f, distance);
                if (!rayQuery.traceVisibilityRay(ray, RAY_FLAG_NONE, 0xff))
                {
                    valid = false;
                }
            }

            // Evaluate diffuse and specular lobes.
            float3 diffuse = {};
            float3 specular = {};
            if (valid)
            {
                TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(pixel, gRTXDI.frameIndex);

                sd.mtl.setActiveLobes((uint)LobeType::DiffuseReflection);
                diffuse = bsdf.eval(sd, dir, sg) * Li;

                sd.mtl.setActiveLobes((uint)LobeType::SpecularReflection);
                specular = bsdf.eval(sd, dir, sg) * Li;

                hitT = distance;
            }

            emission = bsdfProperties.emission;
            color = diffuse + specular + emission;

            // RTXDI uses a simple material model with only diffuse and specular reflection lobes.
            // We query the BSDF for the respective albedos and use them for demodulation below.

            // Demodulate diffuse reflectance (albedo) from diffuse color.
            diffuseReflectance = bsdfProperties.diffuseReflectionAlbedo;
            float3 diffuseFactor = diffuseReflectance <= 0.f ? 0.f : 1.f / diffuseReflectance;
            diffuseIllumination = diffuse * diffuseFactor;

            // Demodulate preintegrated specular reflectance from specular color.
            float NdotV = saturate(dot(sd.N, sd.V));
            float ggxAlpha = bsdfProperties.roughness * bsdfProperties.roughness;
            specularReflectance = approxSpecularIntegralGGX(bsdfProperties.specularReflectance, ggxAlpha, NdotV);
            float3 specularFactor = specularReflectance <= 0.f ? 0.f : 1.f / specularReflectance;
            specularIllumination = specular * specularFactor;
        }
        else
        {
            if (kUseEnvBackground)
            {
                float3 dir = gScene.camera.computeRayPinhole(pixel, frameDim).dir;
                emission = gScene.envMap.eval(dir);
                color = emission;
            }
        }

        // Write active outputs.
        if (is_valid(gColor)) gColor[pixel] = float4(color, 1.f);
        if (is_valid(gEmission)) gEmission[pixel] = float4(emission, 1.f);
        if (is_valid(gDiffuseReflectance)) gDiffuseReflectance[pixel] = float4(diffuseReflectance, 1.f);
        if (is_valid(gDiffuseIllumination)) gDiffuseIllumination[pixel] = float4(diffuseIllumination, hitT);
        if (is_valid(gSpecularReflectance)) gSpecularReflectance[pixel] = float4(specularReflectance, 1.f);
        if (is_valid(gSpecularIllumination)) gSpecularIllumination[pixel] = float4(specularIllumination, hitT);
    }
};

ParameterBlock<FinalShading> gFinalShading;

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gFinalShading.execute(dispatchThreadId.xy);
}
