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
#include "RenderGraphViewer.h"
#include "Utils/RenderGraphScriptContext.h"

const std::string gkDefaultScene = "Arcade/Arcade.fscene";
const char* kEditorExecutableName = "RenderGraphEditor";

RenderGraphViewer::~RenderGraphViewer()
{
    closeSharedFile(mTempFilePath);

    if (mEditorProcess)
    {
        terminateProcess(mEditorProcess);
        mEditorProcess = 0;
    }
}

void RenderGraphViewer::onGuiRender(SampleCallbacks* pSample, Gui* pGui)
{
    if (pGui->addButton("Load Graph"))
    {
        std::string filename;
        if (openFileDialog("", filename)) fileWriteCallback(pSample, filename);

        if (pGui->addCheckBox("Depth Pass", mEnableDepthPrePass))
        {
            createGraph(pSample);
        }
    }

//     if (!mEditorRunning && pGui->addButton("Edit RenderGraph"))
//     {
//         // reset outputs to original state
//         mpGraph->unmarkGraphOutput(mOutputString);
//         for (const std::string& output : mOriginalOutputs)
//         {
//             mpGraph->markGraphOutput(output);
//         }
// 
//         std::string renderGraphScript = RenderGraphLoader::saveRenderGraphAsScriptBuffer(*mpGraph);
//         if (!renderGraphScript.size())
//         {
//             logError("No graph data to display in editor.");
//         }
//         
//         char* result = nullptr;
//         mTempFilePath = std::tmpnam(result);
//         std::ofstream updatesFileOut(mTempFilePath);
//         assert(updatesFileOut.is_open());
// 
//         updatesFileOut.write(renderGraphScript.c_str(), renderGraphScript.size());
//         updatesFileOut.close();
// 
//         openSharedFile(mTempFilePath, std::bind(&RenderGraphViewer::fileWriteCallback, this, std::placeholders::_1));
// 
//         // load application for the editor given it the name of the mapped file
//         std::string commandLine = std::string("-tempFile ") + mTempFilePath;
//         mEditorProcess = executeProcess(kEditorExecutableName, commandLine);
// 
//         assert(mEditorProcess);
//         mEditorRunning = true;
// 
//         mpGraph->setOutput(mOutputString, pSample->getCurrentFbo()->getColorTexture(0));
//     }
    
    if (mEditorProcess && mEditorRunning)
    {
        if (!isProcessRunning(mEditorProcess))
        {
            terminateProcess(mEditorProcess);
            mEditorProcess = 0;
            mEditorRunning = false;
        }
    }

    if (mpGraph)
    {
        pGui->addCheckBox("Show All Outputs", mShowAllOutputs);

        Gui::DropdownList renderGraphOutputs;
        if (mShowAllOutputs)
        {
            std::vector<std::string> outputs = mpGraph->getAllOutputs();
            int32_t i = 0;

            for (const std::string& outputName : outputs)
            {
                Gui::DropdownValue graphOutput;
                graphOutput.label = outputName;
                if (outputName == mOutputString)
                {
                    mGraphOutputIndex = i;
                }
                graphOutput.value = i++;
                renderGraphOutputs.push_back(graphOutput);
            }

        }
        else
        {
            for (int32_t i = 0; i < static_cast<int32_t>(mpGraph->getGraphOutputCount()); ++i)
            {
                Gui::DropdownValue graphOutput;
                graphOutput.label = mpGraph->getGraphOutputName(i);
                graphOutput.value = i;
                renderGraphOutputs.push_back(graphOutput);
            }
        }
        
        // with switching between all outputs and only graph outputs
        if (mGraphOutputIndex > renderGraphOutputs.size())
        {
            mGraphOutputIndex = static_cast<uint32_t>(renderGraphOutputs.size()) - 1;
        }

        if (renderGraphOutputs.size() && pGui->addDropdown("Render Graph Output", renderGraphOutputs, mGraphOutputIndex))
        {
            mpGraph->setOutput(mOutputString, nullptr);
            mpGraph->unmarkGraphOutput(mOutputString);
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

    mpGraph->addRenderPass(DepthPass::create(), "DepthPrePass");
    mpGraph->addRenderPass(CascadedShadowMaps::create(Dictionary()), "ShadowPass");
    mpGraph->addRenderPass(BlitPass::create(), "BlitPass");
    mpGraph->addRenderPass(ToneMapping::create(Dictionary()), "ToneMapping");
    mpGraph->addRenderPass(SSAO::create(Dictionary()), "SSAO");
    mpGraph->addRenderPass(FXAA::create(), "FXAA");

    // Add the skybox
    Scene::UserVariable var = mpScene->getUserVariable("sky_box");
    assert(var.getType() == Scene::UserVariable::Type::String);
    std::string skyBox = getDirectoryFromFile(mSceneFilename) + '/' + var.asString();
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

void RenderGraphViewer::fileWriteCallback(SampleCallbacks* pSample, const std::string& filename)
{
    RenderGraphScriptContext::SharedPtr pScript = RenderGraphScriptContext::create(filename);
    mpGraph = pScript->getGraph("g");
    mpGraph->setScene(mpScene);
    mpGraph->onResizeSwapChain(pSample->getCurrentFbo().get());
}

void RenderGraphViewer::onLoad(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext)
{
    // if editor opened from running render graph, get the name of the file to read
    std::vector<ArgList::Arg> commandArgs = pSample->getArgList().getValues("tempFile");
    std::string filePath;
    
    if (commandArgs.size())
    {
        filePath = commandArgs.front().asString();
        mEditorRunning = true;

        if (filePath.size())
        {
            mpGraph = RenderGraph::create();
            mpGraph = RenderGraphScriptContext::importGraphsFromFile(filePath)[0].obj;
            mpScene = mpGraph->getScene();
            if (!mpScene)
            {
                loadScene(gkDefaultScene, false, pSample);
                mpGraph->setScene(mpScene);
            }
            else
            {
                mCamControl.attachCamera(mpScene->getCamera(0));
                mpScene->getActiveCamera()->setAspectRatio((float)pSample->getCurrentFbo()->getWidth() / (float)pSample->getCurrentFbo()->getHeight());
            }
            mpGraph->onResizeSwapChain(pSample->getCurrentFbo().get());

//            openSharedFile(filePath, std::bind(&RenderGraphViewer::fileWriteCallback, this, std::placeholders::_1));
        }
        else
        {
            msgBox("No path to temporary file provided");
        }
    }
    else
    {
        loadScene(gkDefaultScene, false, pSample);
        createGraph(pSample);
    }

    for (int32_t i = 0; i < static_cast<int32_t>(mpGraph->getGraphOutputCount()); ++i)
    {
        mOriginalOutputs.push_back(mpGraph->getGraphOutputName(i));
    }
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
