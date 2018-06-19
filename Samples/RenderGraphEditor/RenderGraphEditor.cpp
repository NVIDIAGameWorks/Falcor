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
#include "RenderGraphEditor.h"

const std::string gkDefaultScene = "Arcade/Arcade.fscene";

RenderGraphEditor::RenderGraphEditor()
    : mCurrentGraphIndex(0), mCreatingRenderGraph(false), mPreviewing(false)
{
    mNextGraphString.resize(255, 0);
}

void RenderGraphEditor::onGuiRender(SampleCallbacks* pSample, Gui* pGui)
{
    if (mPreviewing)
    {
        return;
    }

    if (pGui->addButton("Load Scene"))
    {
        std::string filename;
        if (openFileDialog(Scene::kFileFormatString, filename)) loadScene(filename, true);
    }

    uint32_t screenHeight = pSample->getWindow()->getClientAreaHeight();
    uint32_t screenWidth  =  pSample->getWindow()->getClientAreaWidth();

    pGui->pushWindow("Graph Editor Settings", screenWidth / 8, screenHeight - 1, screenWidth - screenWidth / 8, 0, false);

    pGui->addTextBox("Graph Name", mNextGraphString);
    if (mCreatingRenderGraph)
    {
        pGui->addText("Creating Graph ...");
    }
    else
    {
        if (mNextGraphString[0] && pGui->addButton("Create New Graph"))
        {
            createRenderGraph(mNextGraphString);
        }
    }

    pGui->addButton("Load Graph");
    pGui->addButton("Save Graph");

    uint32_t selection = false;
    if (mOpenGraphNames.size() && pGui->addDropdown("Open Graph", mOpenGraphNames, selection))
    {
        // Display graph
         mCurrentGraphIndex = selection;
    }

    if (pGui->addButton("Preview Graph"))
    {
        mPreviewing = true;
    }

    pGui->popWindow();

    mpGraphs[mCurrentGraphIndex]->renderUI(pGui);
}

void RenderGraphEditor::loadScene(const std::string& filename, bool showProgressBar)
{
    ProgressBar::SharedPtr pBar;
    if (showProgressBar)
    {
        pBar = ProgressBar::create("Loading Scene", 100);
    }

    mpGraphs[mCurrentGraphIndex]->setScene(nullptr);
    Scene::SharedPtr pScene = Scene::loadFromFile(filename);
    mpGraphs[mCurrentGraphIndex]->setScene(pScene);
    mCamControl.attachCamera(pScene->getCamera(0));
}

// takes string by value for async calls to ensure copy
void RenderGraphEditor::createRenderGraph(std::string renderGraphName)
{
    mCreatingRenderGraph = true;

    Gui::DropdownValue nextGraphID;
    nextGraphID.value = static_cast<int32_t>(mOpenGraphNames.size());
    nextGraphID.label = renderGraphName;
    mOpenGraphNames.push_back(nextGraphID);
    
    RenderGraph::SharedPtr newGraph = RenderGraph::create();
    mpGraphs.push_back(newGraph);

    mCurrentGraphIndex = mOpenGraphNames.size() - static_cast<size_t>(1);

    newGraph->addRenderPass(SceneRenderPass::create(), "SceneRenderer");
    newGraph->addRenderPass(BlitPass::create(), "BlitPass");

    newGraph->addEdge("SceneRenderer.color", "BlitPass.src");

    // only load the scene for the first graph for now
    if (mCurrentGraphIndex >= 1)
    {
        mpGraphs[mCurrentGraphIndex]->setScene(mpGraphs[0]->getScene());
    }
    else
    {
        loadScene(gkDefaultScene, false);
    }
    
    mCreatingRenderGraph = false;
}

void RenderGraphEditor::onLoad(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext)
{
    mpEditorGraph = RenderGraph::create();
    createRenderGraph("DefaultRenderGraph");
}

void RenderGraphEditor::onFrameRender(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext, const Fbo::SharedPtr& pTargetFbo)
{
    if (mPreviewing && pSample->isKeyPressed(KeyboardEvent::Key::E))
    {
        mPreviewing = false;
    }

    if (!mPreviewing)
    {
        // render the editor gui graph
        const glm::vec4 clearColor(0.38f, 0.52f, 0.10f, 1);
        pRenderContext->clearFbo(pTargetFbo.get(), clearColor, 1.0f, 0, FboAttachmentType::All);

    }
    else
    {
        mpGraphs[mCurrentGraphIndex]->getScene()->update(pSample->getCurrentTime(), &mCamControl);

        // render the editor gui graph
        const glm::vec4 clearColor(0.38f, 0.52f, 0.10f, 1);
        pRenderContext->clearFbo(pTargetFbo.get(), clearColor, 1.0f, 0, FboAttachmentType::All);

        mpGraphs[mCurrentGraphIndex]->execute(pRenderContext.get());
    }
}

bool RenderGraphEditor::onKeyEvent(SampleCallbacks* pSample, const KeyboardEvent& keyEvent)
{
    return mCamControl.onKeyEvent(keyEvent);
}

bool RenderGraphEditor::onMouseEvent(SampleCallbacks* pSample, const MouseEvent& mouseEvent)
{
    return mCamControl.onMouseEvent(mouseEvent);
}

void RenderGraphEditor::onResizeSwapChain(SampleCallbacks* pSample, uint32_t width, uint32_t height)
{
    auto pColor = Texture::create2D(width, height, pSample->getCurrentFbo()->getColorTexture(0)->getFormat(), 1, 1, nullptr, Resource::BindFlags::RenderTarget | Resource::BindFlags::ShaderResource);
    auto pDepth = Texture::create2D(width, height, ResourceFormat::D32Float, 1, 1, nullptr, Resource::BindFlags::DepthStencil);
    mpGraphs[mCurrentGraphIndex]->setOutput("BlitPass.dst", pSample->getCurrentFbo()->getColorTexture(0));
    mpGraphs[mCurrentGraphIndex]->onResizeSwapChain(pSample, width, height);
}

int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd)
{
    RenderGraphEditor::UniquePtr pEditor = std::make_unique<RenderGraphEditor>();
    SampleConfig config;
    config.windowDesc.title = "Render Graph Editor";
    config.windowDesc.resizableWindow = true;
    Sample::run(config, pEditor);
    return 0;
}
