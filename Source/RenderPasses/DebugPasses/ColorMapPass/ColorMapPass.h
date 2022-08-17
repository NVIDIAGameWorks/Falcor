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
#include "Utils/Algorithm/ComputeParallelReduction.h"
#include "RenderGraph/BasePasses/FullScreenPass.h"
#include "ColorMapParams.slang"

using namespace Falcor;

class ColorMapPass : public RenderPass
{
public:
    using SharedPtr = std::shared_ptr<ColorMapPass>;

    static const Info kInfo;

    /** Create a new object
    */
    static SharedPtr create(RenderContext* pRenderContext, const Dictionary& dict);

    virtual Dictionary getScriptingDictionary() override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;

    static void registerScriptBindings(pybind11::module& m);

private:
    ColorMapPass(const Dictionary& dict);

    ColorMap mColorMap = ColorMap::Jet;
    uint32_t mChannel = 0;
    bool mAutoRange = true;
    float mMinValue = 0.f;
    float mMaxValue = 1.f;

    FullScreenPass::SharedPtr mpColorMapPass;
    Fbo::SharedPtr mpFbo;
    bool mRecompile = true;

    class AutoRanging
    {
    public:
        AutoRanging();

        std::optional<std::pair<double, double>> getMinMax(RenderContext* pRenderContext, const Texture::SharedPtr& texture, uint32_t channel);

    private:
        ComputeParallelReduction::SharedPtr mpParallelReduction;
        Buffer::SharedPtr mpReductionResult;
        GpuFence::SharedPtr mpFence;
        bool mReductionAvailable = false;
    };

    std::unique_ptr<AutoRanging> mpAutoRanging;
    double mAutoMinValue;
    double mAutoMaxValue;
};
