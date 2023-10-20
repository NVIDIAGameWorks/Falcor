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
#include "SplitScreenPass.h"

namespace
{
// Divider colors
const float4 kColorUnselected = float4(0, 0, 0, 1);
const float4 kColorSelected = float4(1, 1, 1, 1);

// A simple character array representing a 16x16 grayscale arrow
const unsigned char kArrowArray[256] = {
    // clang-format off
    0,  0,  0,  0,  0,  0,  0,  0,  87, 13, 0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  212,255,255,34, 0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  255,255,255,255,32, 0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  78, 255,255,255,255,33, 0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  81, 255,255,255,255,32, 0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  72, 255,255,255,255,34, 0,
    31, 158,156,156,156,156,156,156,156,146,212,255,255,255,255,34,
    241,255,255,255,255,255,255,255,255,255,255,255,255,255,255,240,
    241,255,255,255,255,255,255,255,255,255,255,255,255,255,255,240,
    31, 158,156,156,156,156,156,156,156,146,212,255,255,255,255,33,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  73, 255,255,255,255,34, 0,
    0,  0,  0,  0,  0,  0,  0,  0,  81, 255,255,255,255,31, 0,  0,
    0,  0,  0,  0,  0,  0,  0,  79, 255,255,255,255,32, 0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  255,255,255,255,31, 0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  212,255,255,33, 0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  87, 12, 0,  0,  0,  0,  0,  0
    // clang-format on
};

// Where is our shader located?
const std::string kSplitShader = "RenderPasses/DebugPasses/SplitScreenPass/SplitScreen.ps.slang";
} // namespace

SplitScreenPass::SplitScreenPass(ref<Device> pDevice, const Properties& props) : ComparisonPass(pDevice)
{
    mpArrowTex = mpDevice->createTexture2D(16, 16, ResourceFormat::R8Unorm, 1, Texture::kMaxPossible, kArrowArray);
    createProgram();

    for (const auto& [key, value] : props)
    {
        if (!parseKeyValuePair(key, value))
        {
            logWarning("Unknown property '{}' in a SplitScreenPass properties.", key);
        }
    }
}

void SplitScreenPass::createProgram()
{
    // Create our shader that splits the screen.
    mpSplitShader = FullScreenPass::create(mpDevice, kSplitShader);
}

void SplitScreenPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto var = mpSplitShader->getRootVar();
    var["GlobalCB"]["gDividerColor"] = mMouseOverDivider ? kColorSelected : kColorUnselected;
    var["GlobalCB"]["gMousePosition"] = mMousePos;
    var["GlobalCB"]["gDrawArrows"] = mDrawArrows && mMouseOverDivider;
    var["gArrowTex"] = mpArrowTex;

    ComparisonPass::execute(pRenderContext, renderData);
}

bool SplitScreenPass::onMouseEvent(const MouseEvent& mouseEvent)
{
    // If we have the divider grabbed, claim *all* mouse movements for ourself
    bool handled = mDividerGrabbed;

    // Find out where on the screen we are
    mMousePos = int2(mouseEvent.screenPos.x, mouseEvent.screenPos.y);

    // If we're outside the window, stop.
    mMousePos = clamp(mMousePos, int2(0, 0), int2(pDstFbo->getWidth() - 1, pDstFbo->getHeight() - 1));

    // Actually process our events
    if (mMouseOverDivider && mouseEvent.type == MouseEvent::Type::ButtonDown && mouseEvent.button == Input::MouseButton::Left)
    {
        mDividerGrabbed = true;
        handled = true;

        if (CpuTimer::calcDuration(mTimeOfLastClick, CpuTimer::getCurrentTimePoint()) < 100.0)
            mSplitLoc = 0.5f;
        else
            mTimeOfLastClick = CpuTimer::getCurrentTimePoint();
    }
    else if (mDividerGrabbed)
    {
        if (mouseEvent.type == MouseEvent::Type::ButtonUp && mouseEvent.button == Input::MouseButton::Left)
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
    // have a minimum landing size (13 pixels, 2*6+1) that counts as hovering over the slider.
    mMouseOverDivider = (std::abs(int32_t(mSplitLoc * pDstFbo->getWidth()) - mMousePos.x) < std::max(6, int32_t(mDividerSize)));

    return handled;
}

void SplitScreenPass::renderUI(Gui::Widgets& widget)
{
    widget.var("Split location", mSplitLoc, 0.0f, 1.0f, 0.001f);
    widget.checkbox("Show Arrows", mDrawArrows, true);
    ComparisonPass::renderUI(widget);
}
