/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include "ComparisonPass.h"

namespace
{
    const std::string kSplitLocation = "splitLocation";
    const std::string kShowTextLabels = "showTextLabels";
    const std::string kLeftLabel = "leftLabel";
    const std::string kRightLabel = "rightLabel";

    const std::string kLeftInput = "leftInput";
    const std::string kRightInput = "rightInput";
    const std::string kOutput = "output";
}

bool ComparisonPass::parseKeyValuePair(const std::string key, const Dictionary::Value val)
{
    if (key == kSplitLocation)
    {
        mSplitLoc = val;
        return true;
    }
    else if (key == kShowTextLabels)
    {
        mShowLabels = val;
        return true;
    }
    else if (key == kLeftLabel)
    {
        std::string str = val;
        mLeftLabel = str;
        return true;
    }
    else if (key == kRightLabel)
    {
        std::string str = val;
        mRightLabel = str;
        return true;
    }
    else return false;
}

Dictionary ComparisonPass::getScriptingDictionary()
{
    Dictionary dict;
    dict[kSplitLocation] = mSplitLoc;
    dict[kShowTextLabels] = mShowLabels;
    dict[kLeftLabel] = mLeftLabel;
    dict[kRightLabel] = mRightLabel;
    return dict;
}

RenderPassReflection ComparisonPass::reflect(const CompileData& compileData)
{
    RenderPassReflection r;
    r.addInput(kLeftInput, "Left side image").bindFlags(Falcor::Resource::BindFlags::ShaderResource).texture2D(0, 0);
    r.addInput(kRightInput, "Right side image").bindFlags(Falcor::Resource::BindFlags::ShaderResource).texture2D(0, 0);
    r.addOutput(kOutput, "Output image").bindFlags(Falcor::Resource::BindFlags::RenderTarget).texture2D(0, 0);
    return r;
}

void ComparisonPass::execute(RenderContext* pContext, const RenderData& renderData)
{
    // Get references to our input, output, and temporary accumulation texture
    pLeftSrcTex = renderData[kLeftInput]->asTexture();
    pRightSrcTex = renderData[kRightInput]->asTexture();
    pDstFbo = Fbo::create({ renderData[kOutput]->asTexture() });

    // If we haven't initialized the split location, split the screen in half by default
    if (mSplitLoc < 0) mSplitLoc = 0.5f;

    // Set shader parameters
    mpSplitShader["GlobalCB"]["gSplitLocation"] = int32_t(mSplitLoc * renderData.getDefaultTextureDims().x);
    mpSplitShader["GlobalCB"]["gDividerSize"] = mDividerSize;
    mpSplitShader["gLeftInput"] = mSwapSides ? pRightSrcTex : pLeftSrcTex;
    mpSplitShader["gRightInput"] = mSwapSides ? pLeftSrcTex : pRightSrcTex;

    // Execute the accumulation shader
    mpSplitShader->execute(pContext, pDstFbo);

    // Render some labels
    if (mShowLabels)
    {
        int32_t screenLoc = int32_t(mSplitLoc * renderData.getDefaultTextureDims().x);

        // Draw text labeling the right side image
        std::string rightSide = mSwapSides ? mLeftLabel : mRightLabel;
        TextRenderer::render(pContext, rightSide.c_str(), pDstFbo, float2(screenLoc + 16, 16));

        // Draw text labeling the left side image
        std::string leftSide = mSwapSides ? mRightLabel : mLeftLabel;
        uint32_t leftLength = uint32_t(leftSide.length()) * 9;
        TextRenderer::render(pContext, leftSide.c_str(), pDstFbo, float2(screenLoc - 16 - leftLength, 16));
    }
}

void ComparisonPass::renderUI(Gui::Widgets& widget)
{
    widget.checkbox("Swap Sides", mSwapSides);
    widget.checkbox("Show Labels", mShowLabels);
}
