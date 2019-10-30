/***************************************************************************
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
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
***************************************************************************/
#pragma once
#include "ComparisonPass.h"

namespace
{
    const std::string kSplitLocation = "splitLocation";
    const std::string kDividerSize = "dividerSize";
    const std::string kShowTextLabels = "showTextLabels";
    const std::string kLeftLabel = "leftLabel";
    const std::string kRightLabel = "rightLabel";

    const std::string kLeftInput = "leftInput";
    const std::string kRightInput = "rightInput";
    const std::string kOutput = "output";

    // Divider colors
    vec4 kColorUnselected = vec4(0, 0, 0, 1);
    vec4 kColorSelected = vec4(1, 1, 1, 1);

    // A simple character array representing a 16x16 grayscale arrow
    const unsigned char kArrowArray[256] = {
        0, 0, 0, 0,  0, 0, 0, 0,    87, 13, 0, 0,       0, 0, 0, 0,
        0, 0, 0, 0,  0, 0, 0, 212,  255, 255, 34, 0,    0, 0, 0, 0,
        0, 0, 0, 0,  0, 0, 0, 255,  255, 255, 255, 32,  0, 0, 0, 0,
        0, 0, 0, 0,  0, 0, 0, 78,   255, 255, 255, 255, 33, 0, 0, 0,
        0, 0, 0, 0,  0, 0, 0, 0,    81, 255, 255, 255,  255, 32, 0, 0,
        0, 0, 0, 0,  0, 0, 0, 0,    0, 72, 255, 255,    255, 255, 34, 0,
        31, 158, 156, 156,   156, 156, 156, 156,  156, 146, 212, 255,  255, 255, 255, 34,
        241, 255, 255, 255,  255, 255, 255, 255,  255, 255, 255, 255,  255, 255, 255, 240,
        241, 255, 255, 255,  255, 255, 255, 255,  255, 255, 255, 255,  255, 255, 255, 240,
        31, 158, 156, 156,   156, 156, 156, 156,  156, 146, 212, 255,  255, 255, 255, 33,
        0, 0, 0, 0,  0, 0, 0, 0,   0, 73, 255, 255,     255, 255, 34, 0,
        0, 0, 0, 0,  0, 0, 0, 0,   81, 255, 255, 255,   255, 31, 0, 0,
        0, 0, 0, 0,  0, 0, 0, 79,  255, 255, 255 ,255,  32, 0, 0, 0,
        0, 0, 0, 0,  0, 0, 0, 255, 255, 255, 255, 31,   0, 0, 0, 0,
        0, 0, 0, 0,  0, 0, 0, 212, 255, 255, 33, 0,     0, 0, 0, 0,
        0, 0, 0, 0,  0, 0, 0, 0,   87, 12, 0, 0,        0, 0, 0, 0
    };
}

ComparisonPass::ComparisonPass()
{
    mpArrowTex = Texture::create2D(16, 16, ResourceFormat::R8Unorm, 1, Texture::kMaxPossible, kArrowArray);
    mClock = gpFramework->getGlobalClock();
}

void ComparisonPass::parseDictionary(const Dictionary& dict)
{
    for (const auto& v : dict)
    {
        if (v.key() == kSplitLocation) mSplitLoc = v.val();
        if (v.key() == kDividerSize) mDividerSize = v.val();
        if (v.key() == kShowTextLabels) mShowLabels = v.val();
        if (v.key() == kLeftLabel)
        {
            std::string str = v.val();
            mLeftLabel = str;
        }
        if (v.key() == kRightLabel)
        {
            std::string str = v.val();
            mRightLabel = str;
        }
        else logWarning("Unknown field `" + v.key() + "` in a ComparisonPass dictionary");
    }
}

Dictionary ComparisonPass::getScriptingDictionary()
{
    Dictionary dict;
    dict[kSplitLocation] = mSplitLoc;
    dict[kDividerSize] = mDividerSize;
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
    mpSplitShader["GlobalCB"]["gDividerColor"] = mMouseOverDivider ? kColorSelected : kColorUnselected;
    mpSplitShader["GlobalCB"]["gMousePosition"] = mMousePos;
    mpSplitShader["GlobalCB"]["gDrawArrows"] = mDrawArrows && mMouseOverDivider;
    mpSplitShader["gLeftInput"] = mSwapSides ? pRightSrcTex : pLeftSrcTex;
    mpSplitShader["gRightInput"] = mSwapSides ? pLeftSrcTex : pRightSrcTex;
    mpSplitShader["gArrowTex"] = mpArrowTex;

    // Execute the accumulation shader
    mpSplitShader->execute(pContext, pDstFbo);

    // Render some labels
    if (mShowLabels)
    {
        // Can optionally only show the labels when hovering over the divider.
        if (!mLabelsOnlyWhenHovering || mMouseOverDivider)
        {
            int32_t screenLoc = int32_t(mSplitLoc * renderData.getDefaultTextureDims().x);

            // Draw text labeling the right side image
            std::string rightSide = mSwapSides ? mLeftLabel : mRightLabel;
            TextRenderer::render(pContext, rightSide.c_str(), pDstFbo, vec2(screenLoc + 16, 16));

            // Draw text labeling the left side image
            std::string leftSide = mSwapSides ? mRightLabel : mLeftLabel;
            uint32_t leftLength = uint32_t(leftSide.length()) * 9;
            TextRenderer::render(pContext, leftSide.c_str(), pDstFbo, vec2(screenLoc - 16 - leftLength, 16));
        }
    }
}

bool ComparisonPass::onMouseEvent(const MouseEvent& mouseEvent)
{
    // If we have the divider grabbed, claim *all* mouse movements for ourself
    bool handled = mDividerGrabbed;

    // Find out where on the screen we are
    mMousePos = ivec2(mouseEvent.screenPos.x, mouseEvent.screenPos.y);

    // If we're outside the window, stop.
    mMousePos = glm::clamp(mMousePos, ivec2(0, 0), ivec2(pDstFbo->getWidth() - 1, pDstFbo->getHeight() - 1));

    // Actually process our events
    if (mMouseOverDivider && mouseEvent.type == MouseEvent::Type::LeftButtonDown)
    {
        mDividerGrabbed = true;
        handled = true;

        if (mClock.now() - mTimeOfLastClick < 0.1f) mSplitLoc = 0.5f;
        else mTimeOfLastClick = mClock.now();
    }
    else if (mDividerGrabbed)
    {
        if (mouseEvent.type == MouseEvent::Type::LeftButtonUp)
        {
            mDividerGrabbed = false;
            handled = true;
        }
        else if (mouseEvent.type == MouseEvent::Type::Move)
        {
            mSplitLoc = (float)mMousePos.x / (float)pDstFbo->getWidth();
            handled = true;
        }
    }

    // Update whether the mouse if over the divider.  To ensure selecting the slider isn't a pain,
    //    have a minimum landing size (13 pixels, 2*6+1) that counts as hovering over the slider.
    mMouseOverDivider = (glm::abs(int32_t(mSplitLoc * pDstFbo->getWidth()) - mMousePos.x) < glm::max(6, int32_t(mDividerSize)));

    return handled;
}

void ComparisonPass::renderUI(Gui::Widgets& widget)
{
    widget.var("Split location", mSplitLoc, 0.0f, 1.0f, 0.001f);
    widget.checkbox("Swap Sides", mSwapSides);
    widget.checkbox("Show Arrows", mDrawArrows, true);
    widget.checkbox("Show Labels", mShowLabels);
    if (mShowLabels)
    {
        widget.checkbox("Show Only On Hover", mLabelsOnlyWhenHovering, true);
    }
}
