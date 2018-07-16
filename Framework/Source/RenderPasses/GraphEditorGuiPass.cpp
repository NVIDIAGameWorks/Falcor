/***************************************************************************
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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
#include "Framework.h"
#include "GraphEditorGuiPass.h"

namespace Falcor
{
    static std::string kColor = "color";
    static std::string kDepth = "depth";
// 
//     static GraphEditorGuiPass::PassData createRenderPassData()
//     {
//         RenderPass::PassData data;
//         RenderPass::PassData::Field output;
//         output.bindFlags = Resource::BindFlags::RenderTarget;
//         output.name = kColor;
//         output.pType = ReflectionResourceType::create(ReflectionResourceType::Type::Texture, ReflectionResourceType::Dimensions::Texture2D, ReflectionResourceType::StructuredType::Invalid, ReflectionResourceType::ReturnType::Unknown, ReflectionResourceType::ShaderAccess::Read);
//         data.outputs.push_back(output);
// 
//         output.name = kDepth;
//         output.format = ResourceFormat::D32Float;
//         output.bindFlags = Resource::BindFlags::DepthStencil;
//         data.outputs.push_back(output);
// 
//         return data;
//     }
// 
//     const GraphEditorGuiPass::PassData GraphEditorGuiPass::kRenderPassData = createRenderPassData();
// 
//     GraphEditorGuiPass::SharedPtr GraphEditorGuiPass::create()
//     {
//         try
//         {
//             return SharedPtr(new GraphEditorGuiPass);
//         }
//         catch (const std::exception&)
//         {
//             return nullptr;
//         }
//     }
// 
//     GraphEditorGuiPass::GraphEditorGuiPass() : RenderPass("GraphEditorGuiPass", nullptr)
//     {
//         mpState = GraphicsState::create();
//         recreateShaders();
//         mpFbo = Fbo::create();
//     }
// 
//     void GraphEditorGuiPass::recreateShaders()
//     {
//     }
// 
//     void GraphEditorGuiPass::sceneChangedCB()
//     {
//         mpSceneRenderer = nullptr;
//         if (mpScene)
//         {
//             mpSceneRenderer = SceneRenderer::create(mpScene);
//         }
//     }
// 
//     bool GraphEditorGuiPass::isValid(std::string& log)
//     {
//         bool b = true;
//         if (mpSceneRenderer == nullptr)
//         {
//             log += "GraphEditorGuiPass must have a scene attached to it\n";
//             b = false;
//         }
// 
//         const auto& pColor = mpFbo->getColorTexture(0).get();
//         if (!pColor)
//         {
//             log += "GraphEditorGuiPass must have a color texture attached\n";
//             b = false;
//         }
//         const auto& pDepth = mpFbo->getDepthStencilTexture().get();
//         if (!pDepth)
//         {
//             log += "GraphEditorGuiPass must have a depth texture attached\n";
//             b = false;
//         }
// 
//         if (mpFbo->checkStatus() == false)
//         {
//             log += "GraphEditorGuiPass FBO is invalid, probably because the depth and color textures have different dimensions";
//             b = false;
//         }
// 
//         return b;
//     }
// 
//     bool GraphEditorGuiPass::setInput(const std::string& name, const std::shared_ptr<Resource>& pResource)
//     {
//         logError("GraphEditorGuiPass::setInput() - trying to set `" + name + "` but this render-pass requires no inputs");
//         return false;
//     }
// 
//     bool GraphEditorGuiPass::setOutput(const std::string& name, const std::shared_ptr<Resource>& pResource)
//     {
//         if (!mpFbo)
//         {
//             logError("GraphEditorGuiPass::setOutput() - please call onResizeSwapChain() before setting an input");
//             return false;
//         }
// 
//         if (name == kColor)
//         {
//             Texture::SharedPtr pColor = std::dynamic_pointer_cast<Texture>(pResource);
//             mpFbo->attachColorTarget(pColor, 0);
//         }
//         else if (name == kDepth)
//         {
//             Texture::SharedPtr pDepth = std::dynamic_pointer_cast<Texture>(pResource);
//             mpFbo->attachDepthStencilTarget(pDepth);
//         }
//         else
//         {
//             logError("GraphEditorGuiPass::setOutput() - trying to set `" + name + "` which doesn't exist in this render-pass");
//             return false;
//         }
// 
//         return true;
//     }
// 
//     void GraphEditorGuiPass::execute(RenderContext* pContext)
//     {
//         pContext->clearFbo(mpFbo.get(), mClearColor, 1, 0);
//         if (mpSceneRenderer)
//         {
//             mpState->setFbo(mpFbo);
//             pContext->pushGraphicsState(mpState);
//             pContext->pushGraphicsVars(mpVars);
//             mpSceneRenderer->renderScene(pContext);
//             pContext->popGraphicsState();
//             pContext->popGraphicsVars();
//         }
//     }
}