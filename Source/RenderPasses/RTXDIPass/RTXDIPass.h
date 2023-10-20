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
#include "Rendering/RTXDI/RTXDI.h"

using namespace Falcor;

/**
 * This RenderPass provides a simple example of how to use the RTXDI module
 * available in the "Source/Falcor/Rendering/RTXDI/" directory.
 *
 * See the RTXDI.h header for more explicit instructions.
 *
 * This pass consists of two compute passes:
 *
 * - PrepareSurfaceData.slang takes in a Falcor VBuffer (e.g. from the GBuffer
 * render pass) and sets up the surface data required by RTXDI to perform
 * light sampling.
 * - FinalShading.slang takes the final RTXDI light samples, checks visiblity
 * and shades the pixels by evaluating the actual material's BSDF.
 *
 * Please see the README on how to install the RTXDI SDK.
 */
class RTXDIPass : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(RTXDIPass, "RTXDIPass", {"Standalone pass for direct lighting using RTXDI."})

    static ref<RTXDIPass> create(ref<Device> pDevice, const Properties& props) { return make_ref<RTXDIPass>(pDevice, props); }

    RTXDIPass(ref<Device> pDevice, const Properties& props);

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;

    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override;

private:
    void parseProperties(const Properties& props);
    void recreatePrograms();
    void prepareSurfaceData(RenderContext* pRenderContext, const ref<Texture>& pVBuffer);
    void finalShading(RenderContext* pRenderContext, const ref<Texture>& pVBuffer, const RenderData& renderData);

    ref<Scene> mpScene;

    std::unique_ptr<RTXDI> mpRTXDI;
    RTXDI::Options mOptions;

    ref<ComputePass> mpPrepareSurfaceDataPass;
    ref<ComputePass> mpFinalShadingPass;

    uint2 mFrameDim = {0, 0};
    bool mOptionsChanged = false;
    bool mGBufferAdjustShadingNormals = false;
};
