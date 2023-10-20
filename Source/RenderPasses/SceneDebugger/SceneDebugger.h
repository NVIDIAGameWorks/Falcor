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
#include "Scene/HitInfoType.slang"
#include "SharedTypes.slang"

using namespace Falcor;

/**
 * Scene debugger render pass.
 *
 * This pass helps identify asset issues such as incorrect normals.
 */
class SceneDebugger : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(SceneDebugger, "SceneDebugger", "Scene debugger for identifying asset issues.");

    static ref<SceneDebugger> create(ref<Device> pDevice, const Properties& props) { return make_ref<SceneDebugger>(pDevice, props); }

    SceneDebugger(ref<Device> pDevice, const Properties& props);

    Properties getProperties() const override;
    RenderPassReflection reflect(const CompileData& compileData) override;
    void compile(RenderContext* pRenderContext, const CompileData& compileData) override;
    void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    void renderUI(Gui::Widgets& widget) override;
    bool onMouseEvent(const MouseEvent& mouseEvent) override;
    bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }

    // Scripting functions
    SceneDebuggerMode getMode() const { return (SceneDebuggerMode)mParams.mode; }
    void setMode(SceneDebuggerMode mode) { mParams.mode = (uint32_t)mode; }

private:
    void renderPixelDataUI(Gui::Widgets& widget);
    void initInstanceInfo();

    // Internal state

    ref<Scene> mpScene;
    SceneDebuggerParams mParams;
    ref<ComputePass> mpDebugPass;
    ref<Fence> mpFence;
    /// Buffer for recording pixel data at the selected pixel.
    ref<Buffer> mpPixelData;
    /// Readback buffer.
    ref<Buffer> mpPixelDataStaging;
    ref<Buffer> mpMeshToBlasID;
    ref<Buffer> mpInstanceInfo;
    bool mPixelDataAvailable = false;
};
