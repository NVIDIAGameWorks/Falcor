/***************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
#include "Falcor.h"

using namespace Falcor;

class GBufferRaster : public RenderPass, inherit_shared_from_this<RenderPass, GBufferRaster>
{
public:
    using SharedPtr = std::shared_ptr<GBufferRaster>;

    static SharedPtr create(const Dictionary& dict = {});

    RenderPassReflection reflect() const override;
    void execute(RenderContext* pContext, const RenderData* pRenderData) override;
    void renderUI(Gui* pGui, const char* uiGroup) override;
    Dictionary getScriptingDictionary() const override;
    void onResize(uint32_t width, uint32_t height) override;
    void setScene(const std::shared_ptr<Scene>& pScene) override;
    std::string getDesc(void) override { return "Raster GBuffer generation"; }
private:
    GBufferRaster();
    void setCullMode(RasterizerState::CullMode mode);
    bool parseDictionary(const Dictionary& dict);

    GraphicsState::SharedPtr                mpGraphicsState;
    SceneRenderer::SharedPtr                mpSceneRenderer;
    Fbo::SharedPtr                          mpFbo;
    RasterizerState::CullMode               mCullMode = RasterizerState::CullMode::Back;

    // Rasterization resources
    struct
    {
        GraphicsState::SharedPtr pState;
        GraphicsProgram::SharedPtr pProgram;
        GraphicsVars::SharedPtr pVars;
    } mRaster;
};
