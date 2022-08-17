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
#pragma once
#if FALCOR_HAS_D3D12
#include "Core/Macros.h"
#include "Core/API/Texture.h"
#include "Core/API/Buffer.h"
#include "Core/API/ParameterBlock.h"
#include "Core/API/Shared/D3D12DescriptorSet.h"
#include "Utils/Scripting/Dictionary.h"
#include "Utils/Sampling/SampleGenerator.h"
#include "Rendering/Lights/EnvMapSampler.h"
#include "Rendering/Lights/EmissiveLightSampler.h"

#include "RTXGIDefines.slangh"
#include <rtxgi/ddgi/DDGIVolume.h>
#include <rtxgi/ddgi/gfx/DDGIVolume_D3D12.h>

#include <memory>

namespace Falcor
{
    /** This is a wrapper around the RTXGI SDK. It provides an implementation to update
        the light probes using analytic and emissive light sampling (LightBVH by default)
        and allows shader code to query the probes through the RTXGIVolume class in slang.
    */
    class FALCOR_API RTXGIVolume
    {
    public:
        using SharedPtr = std::shared_ptr<RTXGIVolume>;

        /** Configuration options.
            These mostly reflect the options in the DDGIVolumeDesc struct.
        */
        struct Options
        {
            bool useAutoGrid = true;                    ///< Compute probe grid parameters based on scene bounds.

            int3 gridSize = { 8, 8, 8 };                ///< Number of probes in the grid.
            float3 gridOrigin = { 0.f, 0.f, 0.f };      ///< Origin of the probe grid in world-space.
            float3 gridSpacing = { 1.f, 1.f, 1.f };     ///< Spacing between probes in world-space.

            uint32_t numIrradianceTexels = 8;           ///< Number of texels used in one dimension of the irradiance texture, not including the 1-pixel border on each side.
            uint32_t numDistanceTexels = 16;            ///< Number of texels used in one dimension of the distance texture, not including the 1-pixel border on each side.
            uint32_t numRaysPerProbe = 256;             ///< Number of rays cast per probe per frame. Independent of the number of probes or resolution of probe textures.

            bool useAutoMaxRayDistance = false;         ///< Compute probe ray distance automatically.
            float probeMaxRayDistance = 1e27f;          ///< Maximum distance a probe ray can travel.

            float probeHysteresis = 0.97f;              ///< Controls the influence of new rays when updating each probe.
            float probeDistanceExponent = 50.f;         ///< Exponent for depth testing. A high value will rapidly react to depth discontinuities, but risks causing banding.
            float probeIrradianceEncodingGamma = 5.f;   ///< Irradiance blending happens in post-tonemap space.

            float probeIrradianceThreshold = 0.25f;     ///< A threshold ratio used during probe radiance blending that determines if a large lighting change has happened.
            float probeBrightnessThreshold = 0.10f;     ///< A threshold value used during probe radiance blending that determines the maximum allowed difference in brightness between the previous and current irradiance values.

            float viewBias = 0.1f;                      ///< Bias values for Indirect Lighting.
            float normalBias = 0.1f;                    ///< Bias values for Indirect Lighting.

            float probeMinFrontfaceDistance = 1.f;      ///< Probe relocation moves probes that see front facing triangles closer than this value.
            float probeBackfaceThreshold = 0.25f;       ///< Probe relocation assumes probes with more than this ratio of backface hits are in walls, and will attempt to move them.

            bool enableProbeRelocation = true;          ///< Enable automatic probe reolocation.
            bool enableProbeClassification = true;      ///< Enable automatic probe state handling. If turned off, probes are always active.

            bool enableRecursiveIrradiance = true;      ///< Accumulate irradiance from previous frames.
            bool enableEnvMap = true;                   ///< Compute direct lighting from environment map.
            bool enableAnalyticLights = true;           ///< Compute direct lighting from analytic lights.
            bool enableEmissiveLights = true;           ///< Compute direct lighting from emissive lights.
            uint32_t emissiveSampleCount = 1;           ///< Number of samples for emissive lights.

            // Note: Empty constructor needed for clang due to the use of the nested struct constructor in the parent constructor.
            Options() {}
        };

        /** Creates a RTXGIVolume for a given scene.
            \param[in] pRenderContext Render context.
            \param[in] pScene Scene.
            \param[in] pEnvMapSampler Environment map sampler for updating probes (optional, creating a new instance otherwise).
            \param[in] pEmissiveSampler Emissive light sampler for updating probes (optional, using LightBVH by default otherwise).
            \param[in] options Options to override the default behavior.
            \return Returns a new RTXGIVolume or throws an exception on error.
        */
        static SharedPtr create(RenderContext* pRenderContext, Scene::SharedPtr pScene, EnvMapSampler::SharedPtr pEnvMapSampler = nullptr, EmissiveLightSampler::SharedPtr pEmissiveSampler = nullptr, const Options& options = Options());

        /** Get the current options.
        */
        const Options& getOptions() const { return mOptions; }

        /** Get the current number of probes.
        */
        uint32_t getProbeCount() const { return mOptions.gridSize.x * mOptions.gridSize.y * mOptions.gridSize.z; }

        /** Update the RTXGI volume.
            This includes a ray tracing pass to generate new probe samples.
            \param[in] pRenderContext Render context.
        */
        void update(RenderContext* pRenderContext);

        /** Render the GUI.
            \return True if the options were edited, false otherwise.
        */
        bool renderUI(Gui::Widgets& widget);

        /** Bind the volume data to a given shader variable.
            \param[in] var Shader variable.
        */
        void setShaderData(const ShaderVar& var) const { var["gRTXGIVolume"] = mpParameterBlock; }

        // Accessors to textures.
        const Texture::SharedPtr& getRayDataTexture() const { return mpRayDataTex; }
        const Texture::SharedPtr& getIrradianceTexture() const { return mpIrradianceTex; }

        // Accessors to direct lighting resources.
        const EnvMapSampler::SharedPtr& getEnvMapSampler() const { return mpEnvMapSampler; }
        const EmissiveLightSampler::SharedPtr& getEmissiveSampler() const { return mpEmissiveSampler; }
        const SampleGenerator::SharedPtr& getSampleGenerator() const { return mpSampleGenerator; }

        /** Get shader defines for using `DirectLighting.slang`.
            This method is exposed for debugging/visualization purposes only.
            \return Returns a list of shader defines.
        */
        Program::DefineList getDirectLightingDefines() const;

        /** Bind the shader data for using `DirectLighting.slang`.
            This method is exposed for debugging/visualization purposes only.
            \param[in] var Shader variable.
        */
        void setDirectLightingShaderData(ShaderVar var) const;

    private:
        RTXGIVolume(RenderContext* pRenderContext, Scene::SharedPtr pScene, EnvMapSampler::SharedPtr pEnvMapSampler, EmissiveLightSampler::SharedPtr pEmissiveSampler, const Options& options);

        void initRTXGIShaders();
        void initRTXGI();
        void initRTXGIResources();
        void initRTXGIPipelineStates();
        void destroyRTXGI();

        void updateParameterBlock();

        void parseDictionary(const Dictionary& dict);
        void validateOptions();
        void computeProbeGrid();
        void probeUpdatePass(RenderContext* pRenderContext);

        // Internal state
        Scene::SharedPtr mpScene;
        AABB mSceneBounds = AABB({ 0,0,0 }, { 0,0,0 });

        Options mOptions;                               ///< Configuration options.
        bool mOptionsDirty = true;                      ///< True if the options have changed. This will trigger a re-creation of all RTXGI resources.

        uint32_t mFrameCount = 0;                       ///< Frames rendered. This is used as random seed.

        // RTXGI resources
        // These need to be recreated every time there is a new desc and the DDGIVolume is recreated.
        rtxgi::DDGIVolumeDesc mDDGIDesc;
        rtxgi::d3d12::DDGIVolumeResources mDDGIResources;
        std::unique_ptr<rtxgi::d3d12::DDGIVolume> mpDDGIVolume;
        bool mIsDDGIVolumeValid = false;                ///< True when DDGIVolume is valid and ready for use.

        D3D12DescriptorSet::SharedPtr mpSet;
        ID3D12RootSignaturePtr mpRootSig;
        Texture::SharedPtr mpRayDataTex;
        Texture::SharedPtr mpIrradianceTex;
        Texture::SharedPtr mpProbeDistanceTex;
        Texture::SharedPtr mpProbeDataTex;
        RenderTargetView::SharedPtr mpProbeIrradianceRTV;
        RenderTargetView::SharedPtr mpProbeDistanceRTV;


        ComputeVars::SharedPtr mpDummyVars;

        // Container for SDK inputs, grouped by shader.
        // RTXGI takes 1 PSO per shader.
        struct RTXGIShader
        {
            ComputeProgram::SharedPtr pCS;
            ID3D12PipelineStatePtr pPSO;
        };

        RTXGIShader mDistanceBlendingCS;                ///< RTXGI's ProbeBlendingCS shader compiled with RTXGI_DDGI_BLEND_RADIANCE=0.
        RTXGIShader mRadianceBlendingCS;                ///< RTXGI's ProbeBlendingCS shader compiled with RTXGI_DDGI_BLEND_RADIANCE=1.
        RTXGIShader mDistanceBorderRowUpdateCS;
        RTXGIShader mDistanceBorderColUpdateCS;
        RTXGIShader mRadianceBorderRowUpdateCS;
        RTXGIShader mRadianceBorderColUpdateCS;
        RTXGIShader mProbeRelocationUpdateCS;
        RTXGIShader mProbeRelocationResetCS;
        RTXGIShader mProbeClassificationUpdateCS;
        RTXGIShader mProbeClassificationResetCS;

        ParameterBlock::SharedPtr mpDDGIVolumeBlock;    ///< ParameterBlock holding the DDGIVolumeDescGPU constant buffer.
        Buffer::SharedPtr         mpDDGIVolumeBlockSDK; ///< A buffer holding the DDGIVolumeDescGPU constant buffer for the SDK (needs to bind without reflections).

        ParameterBlock::SharedPtr mpParameterBlock;     ///< Parameter block for RTXGIVolume.

        // Probe update pass
        uint3 mProbeUpdateDispatchDims = { 0, 0, 1 };   ///< Update pass dispatch dimensions. These are set upon initialization of RTXGI resources.
        RtProgram::SharedPtr mpProbeUpdateProgram;
        RtBindingTable::SharedPtr mpRtBindingTable;
        RtProgramVars::SharedPtr mpProbeUpdateVars;

        EnvMapSampler::SharedPtr mpEnvMapSampler;       ///< Environment map sampler.
        bool mUseEmissiveSampler = true;                ///< Use emissive light sampler for the current frame. This field is updated automatically based on the scene.
        EmissiveLightSampler::SharedPtr mpEmissiveSampler; ///< Emissive light sampler.
        SampleGenerator::SharedPtr mpSampleGenerator;   ///< Sample generator used for direct lighting computation.

        // Debugging
        struct
        {
            bool enable = false;
            uint2 threadID = { 0, 0 };
            Buffer::SharedPtr pData;
        } mDebug;
    };
}

#endif // FALCOR_HAS_D3D12
