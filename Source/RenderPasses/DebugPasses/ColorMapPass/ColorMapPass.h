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
#include "Core/Pass/FullScreenPass.h"
#include "RenderGraph/RenderPass.h"
#include "Utils/Algorithm/ParallelReduction.h"
#include "ColorMapParams.slang"

using namespace Falcor;

class ColorMapPass : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(ColorMapPass, "ColorMapPass", "Pass that applies a color map to the input.");

    static ref<ColorMapPass> create(ref<Device> pDevice, const Properties& props) { return make_ref<ColorMapPass>(pDevice, props); }

    ColorMapPass(ref<Device> pDevice, const Properties& props);

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;

private:
    ColorMap mColorMap = ColorMap::Jet;
    uint32_t mChannel = 0;
    bool mAutoRange = true;
    float mMinValue = 0.f;
    float mMaxValue = 1.f;

    ref<FullScreenPass> mpColorMapPass;
    ref<Fbo> mpFbo;
    bool mRecompile = true;

    class AutoRanging
    {
    public:
        AutoRanging(ref<Device> pDevice);

        std::optional<std::pair<double, double>> getMinMax(RenderContext* pRenderContext, const ref<Texture>& texture, uint32_t channel);

    private:
        std::unique_ptr<ParallelReduction> mpParallelReduction;
        ref<Buffer> mpReductionResult;
        ref<Fence> mpFence;
        bool mReductionAvailable = false;
    };

    std::unique_ptr<AutoRanging> mpAutoRanging;
    double mAutoMinValue;
    double mAutoMaxValue;
};
