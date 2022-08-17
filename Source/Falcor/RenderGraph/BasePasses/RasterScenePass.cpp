/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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
#include "RasterScenePass.h"
#include "Scene/Scene.h"

namespace Falcor
{
    RasterScenePass::RasterScenePass(const Scene::SharedPtr& pScene, const Program::Desc& progDesc, const Program::DefineList& programDefines)
        : BaseGraphicsPass(progDesc, programDefines), mpScene(pScene)
    {
        FALCOR_ASSERT(pScene);
    }

    RasterScenePass::SharedPtr RasterScenePass::create(const Scene::SharedPtr& pScene, const Program::Desc& progDesc, const Program::DefineList& programDefines)
    {
        if (pScene == nullptr) throw ArgumentError("Can't create a RasterScenePass object without a scene");

        Program::DefineList dl = programDefines;
        dl.add(pScene->getSceneDefines());

        return SharedPtr(new RasterScenePass(pScene, progDesc, dl));
    }

    RasterScenePass::SharedPtr RasterScenePass::create(const Scene::SharedPtr& pScene, const std::filesystem::path& path, const std::string& vsEntry, const std::string& psEntry, const Program::DefineList& programDefines)
    {
        Program::Desc d;
        d.addShaderLibrary(path).vsEntry(vsEntry).psEntry(psEntry);
        return create(pScene, d, programDefines);
    }

    void RasterScenePass::renderScene(RenderContext* pRenderContext, const Fbo::SharedPtr& pDstFbo)
    {
        mpState->setFbo(pDstFbo);
        mpScene->rasterize(pRenderContext, mpState.get(), mpVars.get());
    }

    bool RasterScenePass::onMouseEvent(const MouseEvent& mouseEvent)
    {
        return mpScene->onMouseEvent(mouseEvent);
    }

    bool RasterScenePass::onKeyEvent(const KeyboardEvent& keyEvent)
    {
        return mpScene->onKeyEvent(keyEvent);
    }
}
