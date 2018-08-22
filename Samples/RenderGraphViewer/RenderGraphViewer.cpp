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
#include "Utils/RenderGraphLoader.h"

const std::string gkDefaultScene = "SunTemple/SunTemple.fscene";// "testScenes/tree.fscene";
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
    if (pGui->addButton("Load Scene"))
    {
        std::string filename;
        if (openFileDialog(Scene::kFileFormatString, filename)) loadScene(filename, true, pSample);

        if (pGui->addCheckBox("Depth Pass", mEnableDepthPrePass))
        {
            createGraph(pSample);
        }
    }

    // allows for editing of lights while viewing graph output
    if (mpScene && pGui->beginGroup("Scene Settings"))
    {
        uint32_t lightCount = mpScene->getLightCount();
        if (lightCount && pGui->beginGroup("Light Sources"))
        {
            for (uint32_t i = 0; i < lightCount; i++)
            {
                Light* pLight = mpScene->getLight(i).get();
                pLight->renderUI(pGui, pLight->getName().c_str());
            }
            pGui->endGroup();
        }

        // allow detaching the camera
        if (mpScene->getPathCount() && pGui->addCheckBox("Use Camera Path", mUseCameraPath))
        {
            if (mUseCameraPath)
            {
                mpScene->getPath(0)->attachObject(mpScene->getActiveCamera());
            }
            else
            {
                mpScene->getPath(0)->detachObject(mpScene->getActiveCamera());
            }
        }

        pGui->endGroup();
    }

    if (!mEditorRunning && pGui->addButton("Edit RenderGraph"))
    {
        // reset outputs to original state
        mpGraph->unmarkGraphOutput(mOutputString);
        for (const std::string& output : mOriginalOutputs)
        {
            mpGraph->markGraphOutput(output);
        }

        std::string renderGraphScript = RenderGraphLoader::saveRenderGraphAsScriptBuffer(*mpGraph);
        if (!renderGraphScript.size())
        {
            logError("No graph data to display in editor.");
        }
        
        char* result = nullptr;
        mTempFilePath = std::tmpnam(result);
        std::ofstream updatesFileOut(mTempFilePath);
        assert(updatesFileOut.is_open());

        updatesFileOut.write(renderGraphScript.c_str(), renderGraphScript.size());
        updatesFileOut.close();

        openSharedFile(mTempFilePath, std::bind(&RenderGraphViewer::fileWriteCallback, this, std::placeholders::_1));

        // load application for the editor given it the name of the mapped file
        std::string commandLine = std::string("-tempFile ") + mTempFilePath;
        mEditorProcess = executeProcess(kEditorExecutableName, commandLine);

        assert(mEditorProcess);
        mEditorRunning = true;

        mpGraph->setOutput(mOutputString, pSample->getCurrentFbo()->getColorTexture(0));
    }
    
    if (mEditorProcess)
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

void RenderGraphViewer::fileWriteCallback(const std::string& fileName)
{
    std::ifstream inputStream(fileName);
    std::string script = std::string((std::istreambuf_iterator<char>(inputStream)), std::istreambuf_iterator<char>());
    RenderGraphLoader::runScript(script.data() + sizeof(size_t), *reinterpret_cast<const size_t*>(script.data()), *mpGraph);
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
            RenderGraphLoader::LoadAndRunScript(filePath, *mpGraph);
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

            openSharedFile(filePath, std::bind(&RenderGraphViewer::fileWriteCallback, this, std::placeholders::_1));
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
    mCaptureHdr = keyEvent.mods.isCtrlDown && (keyEvent.key == KeyboardEvent::Key::L);
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
