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
#include "Utils/Timing/CpuTimer.h"
#include "../ComparisonPass.h"

using namespace Falcor;

class SplitScreenPass : public ComparisonPass
{
public:
    FALCOR_PLUGIN_CLASS(SplitScreenPass, "SplitScreenPass", "Allows the user to split the screen between two inputs.");

    static ref<SplitScreenPass> create(ref<Device> pDevice, const Properties& props) { return make_ref<SplitScreenPass>(pDevice, props); }

    SplitScreenPass(ref<Device> pDevice, const Properties& props);

    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override;
    virtual void renderUI(Gui::Widgets& widget) override;

private:
    virtual void createProgram() override;

    /// A texture storing a 16x16 grayscale arrow
    ref<Texture> mpArrowTex;

    // Mouse parameters

    /// Is the mouse over the divider?
    bool mMouseOverDivider = false;
    /// Where was mouse in last mouse event processed
    int2 mMousePos = int2(0, 0);
    /// Are we grabbing the divider?
    bool mDividerGrabbed = false;

    /// When hovering over divider, show arrows?
    bool mDrawArrows = false;

    /// Time of last mouse click (double-click detection)
    CpuTimer::TimePoint mTimeOfLastClick{};
};
