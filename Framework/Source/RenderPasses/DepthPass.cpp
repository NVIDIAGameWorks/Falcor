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
#include "DepthPass.h"

namespace Falcor
{
    static const std::string& kDepth = "depth";

    DepthPass::SharedPtr DepthPass::create()
    {
        try
        {
            return SharedPtr(new DepthPass);
        }
        catch (const std::exception&)
        {
            return nullptr;
        }
    }

    DepthPass::DepthPass() : RenderPass("DepthPass")
    {
        GraphicsProgram::SharedPtr pProgram = GraphicsProgram::create({});
        mpState = GraphicsState::create();
        mpState->setProgram(pProgram);
        mpVars = GraphicsVars::create(pProgram->getReflector());
        mpFbo = Fbo::create();
        
        DepthStencilState::Desc dsDesc;
        dsDesc.setDepthTest(false).setStencilTest(false);
    }

    void DepthPass::describe(RenderPassReflection& reflector) const
    {
        auto pType = ReflectionResourceType::create(ReflectionResourceType::Type::Texture, ReflectionResourceType::Dimensions::Texture2D);
        reflector.addField(kDepth, RenderPassReflection::Field::Type::Output).setResourceType(pType).setBindFlags(Resource::BindFlags::DepthStencil).setFormat(ResourceFormat::D32Float);
    }

    void DepthPass::setScene(const Scene::SharedPtr& pScene)
    {
        mpSceneRenderer = nullptr;
        if (pScene)
        {
            mpSceneRenderer = SceneRenderer::create(pScene);
        }
    }

    void DepthPass::execute(RenderContext* pContext, const RenderData* pData)
    {
        if(mpSceneRenderer)
        {
            const auto& pDepth = std::dynamic_pointer_cast<Texture>(pData->getResource(kDepth));
            mpFbo->attachDepthStencilTarget(pDepth);

            pContext->clearDsv(pDepth->getDSV().get(), 1, 0);
            if (mpSceneRenderer)
            {
                mpState->setFbo(mpFbo);
                pContext->pushGraphicsState(mpState);
                pContext->pushGraphicsVars(mpVars);
                mpSceneRenderer->renderScene(pContext);
                pContext->popGraphicsState();
                pContext->popGraphicsVars();
            }
        }
    }
}