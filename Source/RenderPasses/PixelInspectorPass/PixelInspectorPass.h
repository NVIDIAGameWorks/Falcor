/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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

using namespace Falcor;

/** Pass extracting material information for the currently selected pixel.
*/
class PixelInspectorPass : public RenderPass
{
public:
    using SharedPtr = std::shared_ptr<PixelInspectorPass>;

    static const Info kInfo;

    /** Create a new object
    */
    static SharedPtr create(RenderContext* pRenderContext, const Dictionary& dict = {});

    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override;

private:
    PixelInspectorPass();

    // Internal state
    Scene::SharedPtr                      mpScene;
    ComputeProgram::SharedPtr             mpProgram;
    ComputeState::SharedPtr               mpState;
    ComputeVars::SharedPtr                mpVars;

    Buffer::SharedPtr                     mpPixelDataBuffer;

    float2                                mCursorPosition = float2(0.0f);
    float2                                mSelectedCursorPosition = float2(0.0f);
    std::unordered_map<std::string, bool> mAvailableInputs;
    std::unordered_map<std::string, bool> mIsInputInBounds;

    // UI variables
    uint2                                 mSelectedPixel = uint2(0u);
    bool                                  mScaleInputsToWindow = false;
    bool                                  mUseContinuousPicking = false;
};
