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
#pragma once
#include "Falcor.h"
#include "../ComparisonPass.h"

using namespace Falcor;

class SplitScreenPass : public ComparisonPass
{
public:
    using SharedPtr = std::shared_ptr<SplitScreenPass>;

    static SharedPtr create(RenderContext* pRenderContext = nullptr, const Dictionary& dict = {});
    virtual std::string getDesc() override { return kDesc; }
    virtual void execute(RenderContext* pContext, const RenderData& renderData) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override;
    virtual void renderUI(Gui::Widgets& widget) override;

    static const char* kDesc;

private:
    SplitScreenPass();
    virtual void createProgram() override;
    Texture::SharedPtr mpArrowTex; // A texture storing a 16x16 grayscale arrow

    // Mouse parameters
    bool mMouseOverDivider = false; ///< Is the mouse over the divider?
    int2 mMousePos = int2(0, 0); ///< Where was mouse in last mouse event processed
    bool mDividerGrabbed = false; ///< Are we grabbing the divider?

    bool mDrawArrows = false; ///< When hovering over divider, show arrows?

    // Double-click detection Parameters
    Clock mClock; ///< Global clock used to track click times
    double mTimeOfLastClick = 0; ///< Time since mouse was last clicked
};
