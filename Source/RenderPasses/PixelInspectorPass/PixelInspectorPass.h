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

using namespace Falcor;

/**
 * Pass extracting material information for the currently selected pixel.
 */
class PixelInspectorPass : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(
        PixelInspectorPass,
        "PixelInspectorPass",
        {"Inspect geometric and material properties at a given pixel.\n"
         "Left-mouse click on a pixel to select it.\n"}
    );

    static ref<PixelInspectorPass> create(ref<Device> pDevice, const Properties& props)
    {
        return make_ref<PixelInspectorPass>(pDevice, props);
    }

    PixelInspectorPass(ref<Device> pDevice, const Properties& props);

    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override;

private:
    void recreatePrograms();

    // Internal state
    ref<Scene> mpScene;
    ref<Program> mpProgram;
    ref<ComputeState> mpState;
    ref<ProgramVars> mpVars;

    ref<Buffer> mpPixelDataBuffer;

    float2 mCursorPosition = float2(0.0f);
    float2 mSelectedCursorPosition = float2(0.0f);
    std::unordered_map<std::string, bool> mAvailableInputs;
    std::unordered_map<std::string, bool> mIsInputInBounds;

    // UI variables
    uint2 mSelectedPixel = uint2(0u);
    bool mScaleInputsToWindow = false;
    bool mUseContinuousPicking = false;
};
