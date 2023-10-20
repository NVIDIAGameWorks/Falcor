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
#include "Utils/UI/TextRenderer.h"

using namespace Falcor;

class ComparisonPass : public RenderPass
{
public:
    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;

protected:
    ComparisonPass(ref<Device> pDevice);
    virtual void createProgram() = 0;
    bool parseKeyValuePair(const std::string key, const Properties::ConstValue& val);

    ref<FullScreenPass> mpSplitShader;
    ref<Texture> pLeftSrcTex;
    ref<Texture> pRightSrcTex;
    ref<Fbo> pDstFbo;
    std::unique_ptr<TextRenderer> mpTextRenderer;

    // Screen parameters

    /// Is the left input on the left side
    bool mSwapSides = false;

    // Divider parameters

    /// Location of the divider as a fraction of screen width, values < 0 are initialized to 0.5
    float mSplitLoc = -1.0f;
    /// Size of the divider (in pixels: 2*mDividerSize+1)
    uint32_t mDividerSize = 2;

    // Label Parameters

    /// Show text labels for two images?
    bool mShowLabels = false;
    /// Left label.  Set in Python script with "leftLabel"
    std::string mLeftLabel = "Left side";
    /// Right label.  Set in Python script with "rightLabel"
    std::string mRightLabel = "Right side";
};
