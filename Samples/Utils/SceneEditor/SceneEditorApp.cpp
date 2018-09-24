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
#include "SceneEditorApp.h"
#include "Graphics/Scene/SceneImporter.h"

void SceneEditorApp::onGuiRender(SampleCallbacks* pSample, Gui* pGui)
{
	pGui->addSeparator();

    if (pGui->addButton("Create New Scene"))
    {
        createScene();
    }
    if (pGui->addButton("Load Scene"))
    {
        loadScene();
    }

    if(mpEditor)
    {
        mpEditor->renderGui(pGui);
        if(mpScene->getCameraCount())
        {
			pGui->addCheckBox("Preview Camera", mCameraLiveViewMode);
        }
    }
}

void SceneEditorApp::onResizeSwapChain(SampleCallbacks* pSample, uint32_t width, uint32_t height)
{
    if (mpEditor)
    {
        mpEditor->onResizeSwapChain();
    }
}

void SceneEditorApp::onLoad(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext)
{
}

void SceneEditorApp::reset()
{
    mpProgram = nullptr;
    mpVars = nullptr;
}

void SceneEditorApp::initNewScene()
{
    if(mpScene)
    {
        mpRenderer = SceneRenderer::create(mpScene);
        mpEditor = SceneEditor::create(mpScene);

        initShader();
    }
}

void SceneEditorApp::initShader()
{
    mpProgram = GraphicsProgram::createFromFile("SceneEditorApp.slang", "", "ps");
    mpVars = GraphicsVars::create(mpProgram->getReflector());
}

void SceneEditorApp::loadScene()
{
    std::string Filename;
    if(openFileDialog(Scene::kFileFormatString, Filename))
    {
        reset();

        mpScene = Scene::loadFromFile(Filename, Model::LoadFlags::None, Scene::LoadFlags::None);
        initNewScene();
    }
}

void SceneEditorApp::createScene()
{
    reset();
    mpScene = Scene::create();
    initNewScene();
}

void SceneEditorApp::onFrameRender(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext, const Fbo::SharedPtr& pTargetFbo)
{
    const glm::vec4 clearColor(0.38f, 0.52f, 0.10f, 1);
    pRenderContext->clearFbo(pTargetFbo.get(), clearColor, 1.0f, 0, FboAttachmentType::All);

    if(mpScene)
    {
		GraphicsState::SharedPtr pState = pRenderContext->getGraphicsState();
		pState->setBlendState(nullptr);
		pState->setDepthStencilState(nullptr);
        pRenderContext->setGraphicsVars(mpVars);
		pState->setProgram(mpProgram);

		auto currentTime = pSample->getCurrentTime();
        mpEditor->update(currentTime);
        mpRenderer->update(currentTime);

        const auto& pCamera = mCameraLiveViewMode ? mpScene->getActiveCamera() : mpEditor->getEditorCamera();
        mpRenderer->renderScene(pRenderContext.get(), pCamera.get());
    }

    if (mpEditor && mCameraLiveViewMode == false)
    {
        mpEditor->render(pRenderContext.get());
    }
}

void SceneEditorApp::onShutdown(SampleCallbacks* pSample)
{
    reset();
}

bool SceneEditorApp::onKeyEvent(SampleCallbacks* pSample, const KeyboardEvent& keyEvent)
{
    if (mCameraLiveViewMode)
    {
        return mpRenderer->onKeyEvent(keyEvent);
    }

    return mpEditor ? mpEditor->onKeyEvent(keyEvent) : false;
}

bool SceneEditorApp::onMouseEvent(SampleCallbacks* pSample, const MouseEvent& mouseEvent)
{
    if (mCameraLiveViewMode)
    {
        return mpRenderer->onMouseEvent(mouseEvent);
    }

    return mpEditor ? mpEditor->onMouseEvent(pSample->getRenderContext().get(), mouseEvent) : false;
}

#ifdef _WIN32
int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd)
#else
int main(int argc, char** argv)
#endif
{
	SceneEditorApp::UniquePtr pRenderer = std::make_unique<SceneEditorApp>();
    SampleConfig config;
    config.windowDesc.title = "Scene Editor";
    config.freezeTimeOnStartup = true;
#ifdef _WIN32
    Sample::run(config, pRenderer);
#else
    config.argc = (uint32_t)argc;
    config.argv = argv;
    Sample::run(config, pRenderer);
#endif
    return 0;
}
