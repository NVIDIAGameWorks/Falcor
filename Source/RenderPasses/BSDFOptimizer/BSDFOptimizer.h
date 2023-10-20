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
#include "Utils/Sampling/SampleGenerator.h"
#include "DiffRendering/SceneGradients.h"
#include "BSDFOptimizerParams.slang"
#include <fstd/span.h>

using namespace Falcor;

class BSDFOptimizer : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(BSDFOptimizer, "BSDFOptimizer", "Optimizing BSDF parameters with differentiable materials.");

    static ref<BSDFOptimizer> create(ref<Device> pDevice, const Properties& props) { return make_ref<BSDFOptimizer>(pDevice, props); }

    BSDFOptimizer(ref<Device> pDevice, const Properties& props);

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }

    static void registerBindings(pybind11::module& m);
    uint32_t getInitMaterialID() const { return mParams.initMaterialID; }
    uint32_t getRefMaterialID() const { return mParams.refMaterialID; }
    uint32_t getBSDFSliceResolution() const;
    void setBSDFSliceResolution(uint32_t reso);
    ref<Buffer> computeBSDFGrads();

private:
    void parseProperties(const Properties& props);

    void initOptimization();

    void executeOptimizerPass(RenderContext* pRenderContext);
    void step(RenderContext* pRenderContext);
    void executeViewerPass(RenderContext* pRenderContext, const RenderData& renderData);

    struct AdamOptimizer
    {
        std::vector<float> lr;
        float beta1;
        float beta2;
        float epsilon;
        int steps;

        std::vector<float> m;
        std::vector<float> v;

        AdamOptimizer() {}

        AdamOptimizer(fstd::span<float> lr, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-6f)
            : lr(lr.begin(), lr.end()), beta1(beta1), beta2(beta2), epsilon(epsilon), steps(0)
        {}

        void step(fstd::span<float> dx, fstd::span<float> x);
    };

    // Internal state
    ref<Scene> mpScene; ///< Loaded scene if any, nullptr otherwise.
    std::unique_ptr<SceneGradients> mpSceneGradients;

    SerializedMaterialParams mInitBSDFParams;
    SerializedMaterialParams mRefBSDFParams;

    SerializedMaterialParams mCurBSDFParams;
    SerializedMaterialParams mBSDFGrads;
    AdamOptimizer mAdam;

    /// Parameters shared with the shaders.
    BSDFOptimizerParams mParams;
    ref<SampleGenerator> mpSampleGenerator;
    bool mOptionsChanged = false;

    /// GPU fence for synchronizing readback.
    ref<Fence> mpFence;

    ref<ComputePass> mpOptimizerPass;
    ref<ComputePass> mpViewerPass;

    // UI variables
    Gui::DropdownList mMaterialList;
    bool mRunOptimization = false;
};
