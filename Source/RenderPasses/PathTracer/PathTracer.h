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
#include "Falcor.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "Utils/Debug/PixelDebug.h"
#include "Utils/Sampling/SampleGenerator.h"
#include "Rendering/Lights/LightBVHSampler.h"
#include "Rendering/Lights/EmissivePowerSampler.h"
#include "Rendering/Lights/EnvMapSampler.h"
#include "Rendering/Materials/TexLODTypes.slang"
#include "Rendering/Utils/PixelStats.h"
#include "Rendering/RTXDI/RTXDI.h"

#include "Params.slang"

using namespace Falcor;

/** Fast path tracer.
*/
class PathTracer : public RenderPass
{
public:
    using SharedPtr = std::shared_ptr<PathTracer>;

    static const Info kInfo;

    static SharedPtr create(RenderContext* pRenderContext, const Dictionary& dict);

    virtual Dictionary getScriptingDictionary() override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override;
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }

    const PixelStats::SharedPtr& getPixelStats() const { return mpPixelStats; }

    static void registerBindings(pybind11::module& m);

private:
    struct TracePass
    {
        std::string name;
        std::string passDefine;
        RtProgram::SharedPtr pProgram;
        RtBindingTable::SharedPtr pBindingTable;
        RtProgramVars::SharedPtr pVars;

        TracePass(const std::string& name, const std::string& passDefine, const Scene::SharedPtr& pScene, const Program::DefineList& defines, const Program::TypeConformanceList& globalTypeConformances);
        void prepareProgram(const Program::DefineList& defines);
    };

    PathTracer(const Dictionary& dict);

    void parseDictionary(const Dictionary& dict);
    void validateOptions();
    void updatePrograms();
    void setFrameDim(const uint2 frameDim);
    void prepareResources(RenderContext* pRenderContext, const RenderData& renderData);
    void preparePathTracer(const RenderData& renderData);
    void resetLighting();
    void prepareMaterials(RenderContext* pRenderContext);
    bool prepareLighting(RenderContext* pRenderContext);
    void prepareRTXDI(RenderContext* pRenderContext);
    void setNRDData(const ShaderVar& var, const RenderData& renderData) const;
    void setShaderData(const ShaderVar& var, const RenderData& renderData, bool useLightSampling = true) const;
    bool renderRenderingUI(Gui::Widgets& widget);
    bool renderDebugUI(Gui::Widgets& widget);
    void renderStatsUI(Gui::Widgets& widget);
    bool beginFrame(RenderContext* pRenderContext, const RenderData& renderData);
    void endFrame(RenderContext* pRenderContext, const RenderData& renderData);
    void generatePaths(RenderContext* pRenderContext, const RenderData& renderData);
    void tracePass(RenderContext* pRenderContext, const RenderData& renderData, TracePass& tracePass);
    void resolvePass(RenderContext* pRenderContext, const RenderData& renderData);

    /** Static configuration. Changing any of these options require shader recompilation.
    */
    struct StaticParams
    {
        // Rendering parameters
        uint32_t    samplesPerPixel = 1;                        ///< Number of samples (paths) per pixel, unless a sample density map is used.
        uint32_t    maxSurfaceBounces = 0;                      ///< Max number of surface bounces (diffuse + specular + transmission), up to kMaxPathLenth. This will be initialized at startup.
        uint32_t    maxDiffuseBounces = 3;                      ///< Max number of diffuse bounces (0 = direct only), up to kMaxBounces.
        uint32_t    maxSpecularBounces = 3;                     ///< Max number of specular bounces (0 = direct only), up to kMaxBounces.
        uint32_t    maxTransmissionBounces = 10;                ///< Max number of transmission bounces (0 = none), up to kMaxBounces.

        // Sampling parameters
        uint32_t    sampleGenerator = SAMPLE_GENERATOR_TINY_UNIFORM; ///< Pseudorandom sample generator type.
        bool        useBSDFSampling = true;                     ///< Use BRDF importance sampling, otherwise cosine-weighted hemisphere sampling.
        bool        useRussianRoulette = false;                 ///< Use russian roulette to terminate low throughput paths.
        bool        useNEE = true;                              ///< Use next-event estimation (NEE). This enables shadow ray(s) from each path vertex.
        bool        useMIS = true;                              ///< Use multiple importance sampling (MIS) when NEE is enabled.
        MISHeuristic misHeuristic = MISHeuristic::Balance;      ///< MIS heuristic.
        float       misPowerExponent = 2.f;                     ///< MIS exponent for the power heuristic. This is only used when 'PowerExp' is chosen.
        EmissiveLightSamplerType emissiveSampler = EmissiveLightSamplerType::LightBVH;  ///< Emissive light sampler to use for NEE.
        bool        useRTXDI = false;                           ///< Use RTXDI for direct illumination.

        // Material parameters
        bool        useAlphaTest = true;                        ///< Use alpha testing on non-opaque triangles.
        bool        adjustShadingNormals = false;               ///< Adjust shading normals on secondary hits.
        uint32_t    maxNestedMaterials = 2;                     ///< Maximum supported number of nested materials.
        bool        useLightsInDielectricVolumes = false;       ///< Use lights inside of volumes (transmissive materials). We typically don't want this because lights are occluded by the interface.
        bool        disableCaustics = false;                    ///< Disable sampling of caustics.
        TexLODMode  primaryLodMode = TexLODMode::Mip0;          ///< Use filtered texture lookups at the primary hit.

        // Output parameters
        ColorFormat colorFormat = ColorFormat::LogLuvHDR;       ///< Color format used for internal per-sample color and denoiser buffers.

        // Denoising parameters
        bool        useNRDDemodulation = true;                  ///< Global switch for NRD demodulation.

        Program::DefineList getDefines(const PathTracer& owner) const;
    };

    // Configuration
    PathTracerParams                mParams;                    ///< Runtime path tracer parameters.
    StaticParams                    mStaticParams;              ///< Static parameters. These are set as compile-time constants in the shaders.
    LightBVHSampler::Options        mLightBVHOptions;           ///< Current options for the light BVH sampler.
    RTXDI::Options                  mRTXDIOptions;              ///< Current options for the RTXDI sampler.

    bool                            mEnabled = true;            ///< Switch to enable/disable the path tracer. When disabled the pass outputs are cleared.
    RenderPassHelpers::IOSize       mOutputSizeSelection = RenderPassHelpers::IOSize::Default;  ///< Selected output size.
    uint2                           mFixedOutputSize = { 512, 512 };                            ///< Output size in pixels when 'Fixed' size is selected.

    // Internal state
    Scene::SharedPtr                mpScene;                    ///< The current scene, or nullptr if no scene loaded.
    SampleGenerator::SharedPtr      mpSampleGenerator;          ///< GPU pseudo-random sample generator.
    EnvMapSampler::SharedPtr        mpEnvMapSampler;            ///< Environment map sampler or nullptr if not used.
    EmissiveLightSampler::SharedPtr mpEmissiveSampler;          ///< Emissive light sampler or nullptr if not used.
    RTXDI::SharedPtr                mpRTXDI;                    ///< RTXDI sampler for direct illumination or nullptr if not used.
    PixelStats::SharedPtr           mpPixelStats;               ///< Utility class for collecting pixel stats.
    PixelDebug::SharedPtr           mpPixelDebug;               ///< Utility class for pixel debugging (print in shaders).

    ParameterBlock::SharedPtr       mpPathTracerBlock;          ///< Parameter block for the path tracer.

    bool                            mRecompile = false;         ///< Set to true when program specialization has changed.
    bool                            mVarsChanged = true;        ///< This is set to true whenever the program vars have changed and resources need to be rebound.
    bool                            mOptionsChanged = false;    ///< True if the config has changed since last frame.
    bool                            mGBufferAdjustShadingNormals = false; ///< True if GBuffer/VBuffer has adjusted shading normals enabled.
    bool                            mFixedSampleCount = true;   ///< True if a fixed sample count per pixel is used. Otherwise load it from the pass sample count input.
    bool                            mOutputGuideData = false;   ///< True if guide data should be generated as outputs.
    bool                            mOutputNRDData = false;     ///< True if NRD diffuse/specular data should be generated as outputs.
    bool                            mOutputNRDAdditionalData = false;   ///< True if NRD data from delta and residual paths should be generated as designated outputs rather than being included in specular NRD outputs.

    ComputePass::SharedPtr          mpGeneratePaths;            ///< Fullscreen compute pass generating paths starting at primary hits.
    ComputePass::SharedPtr          mpResolvePass;              ///< Sample resolve pass.
    ComputePass::SharedPtr          mpReflectTypes;             ///< Helper for reflecting structured buffer types.

    std::unique_ptr<TracePass>      mpTracePass;                ///< Main trace pass.
    std::unique_ptr<TracePass>      mpTraceDeltaReflectionPass; ///< Delta reflection trace pass (for NRD).
    std::unique_ptr<TracePass>      mpTraceDeltaTransmissionPass;   ///< Delta transmission trace pass (for NRD).

    Texture::SharedPtr              mpSampleOffset;             ///< Output offset into per-sample buffers to where the samples for each pixel are stored (the offset is relative the start of the tile). Only used with non-fixed sample count.
    Buffer::SharedPtr               mpSampleColor;              ///< Compact per-sample color buffer. This is used only if spp > 1.
    Buffer::SharedPtr               mpSampleGuideData;          ///< Compact per-sample denoiser guide data.
    Buffer::SharedPtr               mpSampleNRDRadiance;        ///< Compact per-sample NRD radiance data.
    Buffer::SharedPtr               mpSampleNRDHitDist;         ///< Compact per-sample NRD hit distance data.
    Buffer::SharedPtr               mpSampleNRDEmission;        ///< Compact per-sample NRD emission data.
    Buffer::SharedPtr               mpSampleNRDReflectance;     ///< Compact per-sample NRD reflectance data.
};
