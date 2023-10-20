/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
#include "RenderGraph/RenderPass.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "Utils/Sampling/SampleGenerator.h"
#include "Utils/Debug/PixelDebug.h"
#include "Rendering/Lights/EmissiveUniformSampler.h"
#include "DiffRendering/SceneGradients.h"
#include "DiffRendering/SharedTypes.slang"

#include "Params.slang"

using namespace Falcor;

/**
 * Warped-Area Reparameterized differentiable path tracer.
 * This pass implements a differentiable path tracer using warped-area reparameterization for handle geometric discontinuities.
 */
class WARDiffPathTracer : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(
        WARDiffPathTracer,
        "WARDiffPathTracer",
        "Warped-area reparameterized differentiable path tracer using DXR 1.1 TraceRayInline."
    );

    static ref<WARDiffPathTracer> create(ref<Device> pDevice, const Properties& props)
    {
        return make_ref<WARDiffPathTracer>(pDevice, props);
    }

    WARDiffPathTracer(ref<Device> pDevice, const Properties& props);

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override;
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }

    static void registerBindings(pybind11::module& m);

    // Python bindings
    const ref<SceneGradients>& getSceneGradients() const { return mpSceneGradients; }
    void setSceneGradients(const ref<SceneGradients>& sg) { mpSceneGradients = sg; }
    uint32_t getRunBackward() const { return mParams.runBackward; }
    void setRunBackward(uint32_t value) { mParams.runBackward = value; }
    const ref<Buffer>& getdLdI() const { return mpdLdI; }
    void setdLdI(const ref<Buffer>& buf) { mpdLdI = buf; }

    void setDiffDebugParams(DiffVariableType varType, uint2 id, uint32_t offset, float4 grad);

private:
    struct TracePass
    {
        std::string name;
        std::string passDefine;
        ref<Program> pProgram;
        ref<RtBindingTable> pBindingTable;
        ref<RtProgramVars> pVars;

        TracePass(
            ref<Device> pDevice,
            const std::string& name,
            const std::string& passDefine,
            const ref<Scene>& pScene,
            const DefineList& defines,
            const TypeConformanceList& globalTypeConformances
        );
        void prepareProgram(ref<Device> pDevice, const DefineList& defines);
    };

    void parseProperties(const Properties& props);
    void updatePrograms();
    void setFrameDim(const uint2 frameDim);
    void prepareResources(RenderContext* pRenderContext, const RenderData& renderData);
    void prepareDiffPathTracer(const RenderData& renderData);
    void resetLighting();
    void prepareMaterials(RenderContext* pRenderContext);
    bool prepareLighting(RenderContext* pRenderContext);
    void bindShaderData(const ShaderVar& var, const RenderData& renderData, bool useLightSampling = true) const;
    bool renderRenderingUI(Gui::Widgets& widget);
    bool renderDebugUI(Gui::Widgets& widget);
    bool beginFrame(RenderContext* pRenderContext, const RenderData& renderData);
    void endFrame(RenderContext* pRenderContext, const RenderData& renderData);
    void tracePass(RenderContext* pRenderContext, const RenderData& renderData, TracePass& tracePass);

    /**
     * Static configuration. Changing any of these options require shader recompilation.
     */
    struct StaticParams
    {
        // Rendering parameters

        /// Number of samples (paths) per pixel, unless a sample density map is used.
        uint32_t samplesPerPixel = 1;
        /// Max number of indirect bounces (0 = none).
        uint32_t maxBounces = 0;

        // Differentiable rendering parameters

        /// Differentiation mode.
        DiffMode diffMode = DiffMode::ForwardDiffDebug;
        /// Name of the variable to differentiate. Used for rendering forward-mode gradient images.
        std::string diffVarName = "";

        // Sampling parameters

        /// Pseudorandom sample generator type.
        uint32_t sampleGenerator = SAMPLE_GENERATOR_TINY_UNIFORM;
        /// Use BRDF importance sampling, otherwise cosine-weighted hemisphere sampling.
        bool useBSDFSampling = true;
        /// Use next-event estimation (NEE). This enables shadow ray(s) from each path vertex.
        bool useNEE = true;
        /// Use multiple importance sampling (MIS) when NEE is enabled.
        bool useMIS = true;

        // WAR parameters

        /// Use warped-area reparameterization (required if there are geometric discontinuities).
        bool useWAR = true;
        /// Number of auxiliary samples per primary sample for warped-area reparameterization.
        uint32_t auxSampleCount = 16;
        /// Log10 of the VMF concentration parameter.
        float log10vMFConcentration = 5.f;
        /// Log10 of the VMF concentration parameter.
        float log10vMFConcentrationScreen = 5.f;
        /// Beta parameter for the boundary term.
        float boundaryTermBeta = 0.01f;
        /// Use antithetic sampling for variance reduction.
        bool useAntitheticSampling = true;
        /// Gamma parameter for the harmonic weights.
        float harmonicGamma = 2.f;

        DefineList getDefines(const WARDiffPathTracer& owner) const;
    };

    // Configuration

    /// Runtime path tracer parameters.
    WARDiffPathTracerParams mParams;
    /// Static parameters. These are set as compile-time constants in the shaders.
    StaticParams mStaticParams;
    /// Differentiable rendering debug parameters.
    DiffDebugParams mDiffDebugParams;

    /// Switch to enable/disable the path tracer. When disabled the pass outputs are cleared.
    bool mEnabled = true;

    // Internal state

    ref<Scene> mpScene;
    ref<SampleGenerator> mpSampleGenerator;
    std::unique_ptr<EmissiveLightSampler> mpEmissiveSampler;
    std::unique_ptr<PixelDebug> mpPixelDebug;

    ref<SceneGradients> mpSceneGradients;
    /// Derivatives of loss function w.r.t. image pixel values.
    ref<Buffer> mpdLdI;

    ref<ParameterBlock> mpDiffPTBlock;

    // Runtime data

    /// Set to true when program specialization has changed.
    bool mRecompile = false;
    /// This is set to true whenever the program vars have changed and resources need to be rebound.
    bool mVarsChanged = true;
    /// True if the config has changed since last frame.
    bool mOptionsChanged = false;

    std::unique_ptr<TracePass> mpTracePass;
};
