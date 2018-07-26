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
#include "RenderGraphViewer.h"
#include "Utils/RenderGraphLoader.h"

const std::string gkDefaultScene = "SunTemple/SunTemple.fscene";

void RenderGraphViewer::onGuiRender(SampleCallbacks* pSample, Gui* pGui)
{
    if (pGui->addButton("Load Scene"))
    {
        std::string filename;
        if (openFileDialog(Scene::kFileFormatString, filename)) loadScene(filename, true, pSample);

        if (pGui->addCheckBox("Depth Pass", mEnableDepthPrePass))
        {
            createGraph(pSample);
        }
    }

    if (mTempRenderGraphLiveEditor.isOpen())
    {
        mTempRenderGraphLiveEditor.updateGraph(*mpGraph, pSample->getLastFrameTime());
    }
    else
    {
        if (pGui->addButton("Edit RenderGraph"))
        {
            mTempRenderGraphLiveEditor.openEditor(*mpGraph);
        }
    }

    if (mpGraph)
    {
        Gui::DropdownList renderGraphOutputs;
        for (int32_t i = 0; i < static_cast<int32_t>(mpGraph->getGraphOutputCount()); ++i)
        {
            Gui::DropdownValue graphOutput;
            graphOutput.label = mpGraph->getGraphOutputName(i);
            graphOutput.value = i;
            renderGraphOutputs.push_back(graphOutput);
        }
        
        if (renderGraphOutputs.size() && pGui->addDropdown("Render Graph Output", renderGraphOutputs, mGraphOutputIndex))
        {
            mpGraph->setOutput(mOutputString, nullptr);
            mOutputString = renderGraphOutputs[mGraphOutputIndex].label;
            mpGraph->setOutput(mOutputString, pSample->getCurrentFbo()->getColorTexture(0));
        }
        
        
        mpGraph->renderUI(pGui, "Render Graph");
    }
}

void RenderGraphViewer::createGraph(SampleCallbacks* pSample)
{
    mpGraph = RenderGraph::create();
    auto pLightingPass = RenderPassLibrary::createRenderPass("SceneLightingPass");
    mpGraph->addRenderPass(pLightingPass, "LightingPass");

    mpGraph->addRenderPass(DepthPass::deserialize({}), "DepthPrePass");
    mpGraph->addRenderPass(CascadedShadowMaps::deserialize({}), "ShadowPass");
    mpGraph->addRenderPass(BlitPass::deserialize({}), "BlitPass");
    mpGraph->addRenderPass(ToneMapping::deserialize({}), "ToneMapping");
    mpGraph->addRenderPass(SSAO::deserialize({}), "SSAO");
    mpGraph->addRenderPass(FXAA::deserialize({}), "FXAA");

    // Add the skybox
    Scene::UserVariable var = mpScene->getUserVariable("sky_box");
    assert(var.type == Scene::UserVariable::Type::String);
    std::string skyBox = getDirectoryFromFile(mSceneFilename) + '/' + var.str;
    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
    mpGraph->addRenderPass(SkyBox::createFromTexture(skyBox, true, Sampler::create(samplerDesc)), "SkyBox");

    mpGraph->addEdge("DepthPrePass.depth", "ShadowPass.depth");
    mpGraph->addEdge("DepthPrePass.depth", "LightingPass.depth");
    mpGraph->addEdge("DepthPrePass.depth", "SkyBox.depth");

    mpGraph->addEdge("SkyBox.target", "LightingPass.color");
    mpGraph->addEdge("ShadowPass.visibility", "LightingPass.visibilityBuffer");

    mpGraph->addEdge("LightingPass.color", "ToneMapping.src");
    mpGraph->addEdge("ToneMapping.dst", "SSAO.colorIn");
    mpGraph->addEdge("LightingPass.normals", "SSAO.normals");
    mpGraph->addEdge("LightingPass.depth", "SSAO.depth");

    mpGraph->addEdge("SSAO.colorOut", "FXAA.src");
    mpGraph->addEdge("FXAA.dst", "BlitPass.src");

    mpGraph->setScene(mpScene);

    mpGraph->markGraphOutput("BlitPass.dst");
    mpGraph->onResizeSwapChain(pSample->getCurrentFbo().get());
}

void RenderGraphViewer::loadScene(const std::string& filename, bool showProgressBar, SampleCallbacks* pSample)
{
    ProgressBar::SharedPtr pBar;
    if (showProgressBar)
    {
        pBar = ProgressBar::create("Loading Scene", 100);
    }

    mpScene = Scene::loadFromFile(filename);
    mSceneFilename = filename;
    mCamControl.attachCamera(mpScene->getCamera(0));
    mpScene->getActiveCamera()->setAspectRatio((float)pSample->getCurrentFbo()->getWidth() / (float)pSample->getCurrentFbo()->getHeight());
}

void RenderGraphViewer::onLoad(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext)
{
#ifdef _WIN32
    // if editor opened from running render graph, get memory view for live update
    std::string commandLine(GetCommandLineA());
    size_t firstSpace = commandLine.find_first_of(' ') + 1;
    std::string filePath = (commandLine.substr(firstSpace, commandLine.size() - firstSpace));
    if (filePath.size())
    {
        mpGraph = RenderGraph::create();
        RenderGraphLoader::LoadAndRunScript(filePath, *mpGraph);
        mTempRenderGraphLiveEditor.openUpdatesFile(filePath);
        mpScene = mpGraph->getScene();
        if (!mpScene)
        {
            loadScene(gkDefaultScene, false, pSample);
        }
        else
        {
            mCamControl.attachCamera(mpScene->getCamera(0));
            mpScene->getActiveCamera()->setAspectRatio((float)pSample->getCurrentFbo()->getWidth() / (float)pSample->getCurrentFbo()->getHeight());
        }
        mpGraph->onResizeSwapChain(pSample->getCurrentFbo().get());
        return;
    }
#endif
    loadScene(gkDefaultScene, false, pSample);
    createGraph(pSample);
}

void RenderGraphViewer::onFrameRender(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext, const Fbo::SharedPtr& pTargetFbo)
{
    const glm::vec4 clearColor(0.38f, 0.52f, 0.10f, 1);
    pRenderContext->clearFbo(pTargetFbo.get(), clearColor, 1.0f, 0, FboAttachmentType::All);

    if (mpGraph)
    {
        mpGraph->setOutput(mOutputString, pSample->getCurrentFbo()->getColorTexture(0));
        mpGraph->getScene()->update(pSample->getCurrentTime(), &mCamControl);
        mpGraph->execute(pRenderContext.get());
    }
}

bool RenderGraphViewer::onKeyEvent(SampleCallbacks* pSample, const KeyboardEvent& keyEvent)
{
    return mCamControl.onKeyEvent(keyEvent);
}

bool RenderGraphViewer::onMouseEvent(SampleCallbacks* pSample, const MouseEvent& mouseEvent)
{
    return mCamControl.onMouseEvent(mouseEvent);
}

void RenderGraphViewer::onResizeSwapChain(SampleCallbacks* pSample, uint32_t width, uint32_t height)
{
    if(mpGraph)
    {
        mpGraph->onResizeSwapChain(pSample->getCurrentFbo().get());
    }
}

int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd)
{
    RenderGraphViewer::UniquePtr pRenderer = std::make_unique<RenderGraphViewer>();
    SampleConfig config;
    config.windowDesc.title = "Render Graph Viewer";
    config.windowDesc.resizableWindow = true;
    Sample::run(config, pRenderer);
    return 0;
}
