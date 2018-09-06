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

const std::string gkDefaultScene = "Arcade/Arcade.fscene";
const char* kEditorExecutableName = "RenderGraphEditor";
const char* kSaveFileFilter = "PNG(.png)\0*.png;\0BMP(.bmp)\0*.bmp;\
   \0JPG(.jpg)\0*.jpg;\0HDR(.hdr)\0*.hdr;\0TGA(.tga)\0*.tga;\0";

RenderGraphViewer::~RenderGraphViewer()
{
    closeSharedFile(mTempFilePath);

    if (mEditorProcess)
    {
        terminateProcess(mEditorProcess);
        mEditorProcess = 0;
    }
}

void RenderGraphViewer::resetGraphOutputs()
{
    // reset outputs to original state
    PerGraphInfo& graphInfo = mpRenderGraphs[mFocusedRenderGraphName];
    graphInfo.mCurrentOutputs = graphInfo.mpGraph->getAvailableOutputs();
    
    for (const auto& output : graphInfo.mCurrentOutputs)
    {
        auto outputIt = graphInfo.mOriginalOutputNames.find(output.first);
        if (output.second && outputIt == graphInfo.mOriginalOutputNames.end())
        {
            graphInfo.mpGraph->unmarkGraphOutput(output.first);
        }
        else if (!output.second && outputIt != graphInfo.mOriginalOutputNames.end())
        {
            graphInfo.mpGraph->markGraphOutput(output.first);
        }
    }
}

void RenderGraphViewer::onGuiRender(SampleCallbacks* pSample, Gui* pGui)
{
    PerGraphInfo& graphInfo = mpRenderGraphs[mFocusedRenderGraphName];
    RenderGraph::SharedPtr pGraph = graphInfo.mpGraph;

    if (pGui->addButton("Load Scene"))
    {
        std::string filename;
        if (openFileDialog(Scene::kFileFormatString, filename)) loadScene(filename, true, pSample);

        // TODO -- make this work ? pop up window?
        // if (pGui->addCheckBox("Depth Pass", graphInfo.mEnableDepthPrePass))
        // {
        //     createGraph(pSample);
        // }

    }

    if (pGui->addButton("Load Graph", true))
    {
        std::string filename;
        if (openFileDialog("", filename)) loadGraph(pSample, filename);
    }

    if (pGui->addDropdown("Focused Render Graph", mRenderGraphsList, mActiveGraphIndex))
    {
        mFocusedRenderGraphName  = std::string(mRenderGraphsList[mActiveGraphIndex].label);
    }

    if (!mEditorRunning && pGui->addButton("Open RenderGraph Editor"))
    {
        resetGraphOutputs();

        std::string renderGraphScript = RenderGraphLoader::saveRenderGraphAsScriptBuffer(*pGraph);
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
        mEditingRenderGraphName = mFocusedRenderGraphName;

        pGraph->setOutput(graphInfo.mOutputString, pSample->getCurrentFbo()->getColorTexture(0));
    }
    
    if (mEditorProcess && mEditorRunning)
    {
        if (!isProcessRunning(mEditorProcess))
        {
            terminateProcess(mEditorProcess);
            mEditorProcess = 0;
            mEditorRunning = false;
        }
    }

    // 
    if (pGraph)
    {
        bool displayGraphUI = pGui->beginGroup(mFocusedRenderGraphName.c_str());
        if (displayGraphUI)
        {
            pGui->addCheckBox("Show All Outputs", mShowAllOutputs);

            Gui::DropdownList renderGraphOutputs;
            if (mShowAllOutputs)
            {
                renderGraphOutputs = graphInfo.mOutputDropdown;
            }
            else
            {
                for (int32_t i = 0; i < static_cast<int32_t>(pGraph->getGraphOutputCount()); ++i)
                {
                    Gui::DropdownValue graphOutput;
                    graphOutput.label = pGraph->getGraphOutputName(i);
                    graphOutput.value = i;
                    renderGraphOutputs.push_back(graphOutput);
                }
            }

            // with switching between all outputs and only graph outputs
            if (graphInfo.mGraphOutputIndex > renderGraphOutputs.size())
            {
                graphInfo.mGraphOutputIndex = static_cast<uint32_t>(renderGraphOutputs.size()) - 1;
            }

            if (renderGraphOutputs.size() && pGui->addDropdown("Render Graph Output", renderGraphOutputs, graphInfo.mGraphOutputIndex))
            {
                pGraph->setOutput(graphInfo.mOutputString, nullptr);
                pGraph->unmarkGraphOutput(graphInfo.mOutputString);
                graphInfo.mOutputString = renderGraphOutputs[graphInfo.mGraphOutputIndex].label;
                pGraph->setOutput(graphInfo.mOutputString, pSample->getCurrentFbo()->getColorTexture(0));
            }

            if (renderGraphOutputs.size())
            {
                if (pGui->addButton("Open New Output Window"))
                {
                    size_t size = mDebugWindowInfos.size();
                    DebugWindowInfo debugWindowInfo;
                    debugWindowInfo.mGraphName = mFocusedRenderGraphName;
                    debugWindowInfo.mOutputName = renderGraphOutputs[0].label;
                    mDebugWindowInfos.insert(std::make_pair(std::string("Debug Window ") + std::to_string(size), debugWindowInfo));
                }
            }

            pGraph->renderUI(pGui, nullptr);
        }

        std::vector<std::string> windowsToRemove;
        for (auto& nameWindow : mDebugWindowInfos)
        {
            DebugWindowInfo& debugWindowInfo = nameWindow.second;
            RenderGraph::SharedPtr pPreviewGraph = mpRenderGraphs[debugWindowInfo.mGraphName].mpGraph;

            pGui->pushWindow((debugWindowInfo.mGraphName + " : " + nameWindow.first).c_str(), 330, 268);
        
            if (pGui->addDropdown("##Render Graph Outputs", mpRenderGraphs[debugWindowInfo.mGraphName].mOutputDropdown, debugWindowInfo.mNextOutputIndex))
            {
                debugWindowInfo.mOutputName = mpRenderGraphs[debugWindowInfo.mGraphName].mOutputDropdown[debugWindowInfo.mNextOutputIndex].label;
                debugWindowInfo.mRenderOutput = true;
            }

            if (pGui->addButton("Close"))
            {
                // mark to close after window updates
                windowsToRemove.push_back(nameWindow.first);
                debugWindowInfo.mRenderOutput = false;

                // unmark graph output checking the original graph state.
                if (graphInfo.mOriginalOutputNames.find(debugWindowInfo.mOutputName) == graphInfo.mOriginalOutputNames.end())
                {
                    pPreviewGraph->unmarkGraphOutput(debugWindowInfo.mOutputName);
                }
            }
            else
            {
                mActiveGraphNames.insert(debugWindowInfo.mGraphName);
            }

            if (debugWindowInfo.mRenderOutput)
            {
                // mark as graph output
                pPreviewGraph->markGraphOutput(debugWindowInfo.mOutputName);
                Texture::SharedPtr pPreviewTex = std::static_pointer_cast<Texture>(pPreviewGraph->getOutput(debugWindowInfo.mOutputName));

                if (pGui->addButton("Save to File", true))
                {
                    std::string filePath;

                    if (saveFileDialog(kSaveFileFilter, filePath))
                    {
                        size_t extensionPos = filePath.find_last_of('.', 0);
                        Bitmap::FileFormat fileFormat = Bitmap::FileFormat::PngFile;

                        if (extensionPos != std::string::npos)
                        {
                            std::string extensionString = filePath.substr(extensionPos, filePath.size() - extensionPos);

                            if (extensionString == "bmp")
                            {
                                fileFormat = Bitmap::FileFormat::BmpFile;
                            }
                            else if (extensionString == "hdr")
                            {
                                fileFormat = Bitmap::FileFormat::ExrFile;
                            }
                            else if (extensionString == "tga")
                            {
                                fileFormat = Bitmap::FileFormat::TgaFile;
                            }
                            else if (extensionString == "jpg" || extensionString == "jpeg")
                            {
                                fileFormat = Bitmap::FileFormat::JpegFile;
                            }
                        }

                        pPreviewTex->captureToFile(0, 0, filePath, fileFormat);
                    }
                }

                glm::vec2 imagePreviewSize = pGui->getCurrentWindowSize();
                float imageAspectRatio = static_cast<float>(pPreviewTex->getHeight()) / static_cast<float>(pPreviewTex->getWidth());
                // get size of window to scale image correctly
                imagePreviewSize.y = imagePreviewSize.x * imageAspectRatio;
                pGui->addImage(nameWindow.first.c_str(), pPreviewTex, imagePreviewSize);
            }

            pGui->popWindow();
        }

        for (const std::string& windowName : windowsToRemove)
        {
            mDebugWindowInfos.erase(windowName);
        }

        if (displayGraphUI) pGui->endGroup();
    }
}

RenderGraph::SharedPtr RenderGraphViewer::createGraph(SampleCallbacks* pSample)
{
    RenderGraph::SharedPtr pGraph = RenderGraph::create();

    auto pLightingPass = RenderPassLibrary::createRenderPass("SceneLightingPass");
    pGraph->addRenderPass(pLightingPass, "LightingPass");
    pGraph->addRenderPass(ToneMapping::createPass(), "ToneMapping");
    pGraph->addRenderPass(DepthPass::createPass(), "DepthPrePass");
    pGraph->addRenderPass(CascadedShadowMaps::createPass(), "ShadowPass");
    pGraph->addRenderPass(BlitPass::createPass(), "BlitPass");
    pGraph->addRenderPass(SSAO::createPass(), "SSAO");
    pGraph->addRenderPass(FXAA::createPass(), "FXAA");

    // Add the skybox
    Scene::UserVariable var = mpScene->getUserVariable("sky_box");
    assert(var.type == Scene::UserVariable::Type::String);
    std::string skyBox = getDirectoryFromFile(mSceneFilename) + '/' + var.str;
    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
    pGraph->addRenderPass(SkyBox::createFromTexture(skyBox, true, Sampler::create(samplerDesc)), "SkyBox");

    pGraph->addEdge("DepthPrePass.depth", "ShadowPass.depth");
    pGraph->addEdge("DepthPrePass.depth", "LightingPass.depth");
    pGraph->addEdge("DepthPrePass.depth", "SkyBox.depth");

    pGraph->addEdge("SkyBox.target", "LightingPass.color");
    pGraph->addEdge("ShadowPass.visibility", "LightingPass.visibilityBuffer");

    pGraph->addEdge("LightingPass.color", "ToneMapping.src");
    pGraph->addEdge("ToneMapping.dst", "SSAO.colorIn");
    pGraph->addEdge("LightingPass.normals", "SSAO.normals");
    pGraph->addEdge("LightingPass.depth", "SSAO.depth");
    
    pGraph->addEdge("SSAO.colorOut", "FXAA.src");
    pGraph->addEdge("FXAA.dst", "BlitPass.src");

    pGraph->setScene(mpScene);

    pGraph->markGraphOutput("BlitPass.dst");
    pGraph->onResizeSwapChain(pSample->getCurrentFbo().get());

    return pGraph;
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

    // set scene for all graphs
    for (auto& graphInfoPair : mpRenderGraphs)
    {
        RenderGraph::SharedPtr pGraph = graphInfoPair.second.mpGraph;
        if (pGraph) pGraph->setScene(mpScene);
    }
}

void RenderGraphViewer::fileWriteCallback(const std::string& fileName)
{
    std::ifstream inputStream(fileName);
    PerGraphInfo& graphInfo = mpRenderGraphs[mEditingRenderGraphName];
    std::string script = std::string((std::istreambuf_iterator<char>(inputStream)), std::istreambuf_iterator<char>());
    RenderGraphLoader::runScript(script.data() + sizeof(size_t), *reinterpret_cast<const size_t*>(script.data()), *graphInfo.mpGraphCpy);
    // Build backlog of changes until we can apply it resulting in a valid graph
    graphInfo.mScriptBacklog.push_back(script);

    // check valid
    std::string log;
    if (graphInfo.mpGraphCpy->isValid(log))
    {
        mApplyGraphChanges = true;
    }
    else
    {
        // TODO - display log in console in viewer.
    }
}

void RenderGraphViewer::loadGraph(SampleCallbacks* pSample, const std::string& filePath)
{
    if (!filePath.size())
    {
        msgBox("No path to graph file provided");
        return;
    }

    RenderGraph::SharedPtr pGraph = RenderGraph::create();
    RenderGraph::SharedPtr pGraphCpy = RenderGraph::create();
    RenderGraphLoader::LoadAndRunScript(filePath, *pGraph);
    RenderGraphLoader::LoadAndRunScript(filePath, *pGraphCpy);

    // test if scene was set from graph script
    Scene::SharedPtr pScene = pGraph->getScene();
    if (!pScene)
    {
        if (!mpScene) loadScene(gkDefaultScene, false, pSample);
        pGraph->setScene(mpScene);
    }
    pGraphCpy->setScene(mpScene);
    insertNewGraph(pGraph, pGraphCpy);

    pGraph->onResizeSwapChain(pSample->getCurrentFbo().get());
    pGraphCpy->onResizeSwapChain(pSample->getCurrentFbo().get());
}

void RenderGraphViewer::createDefaultGraph(SampleCallbacks* pSample)
{
    loadScene(gkDefaultScene, false, pSample);
    RenderGraph::SharedPtr pGraph = createGraph(pSample);
    RenderGraph::SharedPtr pGraphCpy = createGraph(pSample);
    insertNewGraph(pGraph, pGraphCpy);
}

void RenderGraphViewer::insertNewGraph(const RenderGraph::SharedPtr& pGraph, const RenderGraph::SharedPtr& pGraphCpy)
{
    // TODO -- possibly duplicate graph with deep copy instead of requiring another graph
    // TODO -- should be able to get render graph name from script?
    size_t renderGraphIndex = mRenderGraphsList.size();
    mFocusedRenderGraphName = "RenderGraph " + std::to_string(renderGraphIndex);

    Gui::DropdownValue value;
    value.value = static_cast<uint32_t>(mRenderGraphsList.size());
    value.label = mFocusedRenderGraphName;
    mRenderGraphsList.push_back(value);

    PerGraphInfo& graphInfo = mpRenderGraphs[mFocusedRenderGraphName];
    for (int32_t i = 0; i < static_cast<int32_t>(pGraph->getGraphOutputCount()); ++i)
    {
        graphInfo.mOriginalOutputNames.insert(pGraph->getGraphOutputName(i));
    }

    graphInfo.mpGraph = pGraph;
    graphInfo.mCurrentOutputs = pGraph->getAvailableOutputs();
    graphInfo.mpGraphCpy = pGraphCpy;
}

void RenderGraphViewer::updateOutputDropdown(const std::string& passName)
{
    PerGraphInfo& graphInfo = mpRenderGraphs[passName];
    int32_t i = 0;

    graphInfo.mOutputDropdown.clear();

    for (const auto& outputPair : graphInfo.mCurrentOutputs)
    {
        Gui::DropdownValue graphOutput;
        graphOutput.label = outputPair.first;
        if (outputPair.first == graphInfo.mOutputString)
        {
            graphInfo.mGraphOutputIndex = i;
        }
        graphOutput.value = i++;
        graphInfo.mOutputDropdown.push_back(graphOutput);
    }
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
        loadGraph(pSample, filePath);
        if (filePath.size()) openSharedFile(filePath, std::bind(&RenderGraphViewer::fileWriteCallback, this, std::placeholders::_1));
    }
    else
    {
        createDefaultGraph(pSample);
    }

    mpScene->getActiveCamera()->setAspectRatio((float)pSample->getCurrentFbo()->getWidth() / (float)pSample->getCurrentFbo()->getHeight());
    mCamControl.attachCamera(mpScene->getCamera(0));
}

void RenderGraphViewer::onFrameRender(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext, const Fbo::SharedPtr& pTargetFbo)
{
    const glm::vec4 clearColor(0.38f, 0.52f, 0.10f, 1);
    pRenderContext->clearFbo(pTargetFbo.get(), clearColor, 1.0f, 0, FboAttachmentType::All);

    if (mApplyGraphChanges)
    {
        PerGraphInfo& applyGraphInfo = mpRenderGraphs[mEditingRenderGraphName];
        for (const std::string& script : applyGraphInfo.mScriptBacklog)
        {
            RenderGraphLoader::runScript(script.data() + sizeof(size_t), *reinterpret_cast<const size_t*>(script.data()), *applyGraphInfo.mpGraph);
        }
        applyGraphInfo.mScriptBacklog.clear();
        applyGraphInfo.mCurrentOutputs = applyGraphInfo.mpGraph->getAvailableOutputs();
        mApplyGraphChanges = true;
    }

    mActiveGraphNames.insert(mFocusedRenderGraphName);

    for (const std::string& graphName : mActiveGraphNames)
    {
        PerGraphInfo& graphInfo = mpRenderGraphs[graphName];
        RenderGraph::SharedPtr pGraph = graphInfo.mpGraph;
        updateOutputDropdown(graphName);

        if (pGraph)
        {
            pGraph->setOutput(graphInfo.mOutputString, pSample->getCurrentFbo()->getColorTexture(0));
            pGraph->getScene()->update(pSample->getCurrentTime(), &mCamControl);
            pGraph->execute(pRenderContext.get());
        }
    }

    mActiveGraphNames.clear();
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
    // update swap chain data for all graphs
    for (auto& graphInfoPair : mpRenderGraphs)
    {
        RenderGraph::SharedPtr pGraph = graphInfoPair.second.mpGraph;
        RenderGraph::SharedPtr pGraphCpy = graphInfoPair.second.mpGraph;
        if (pGraph) pGraph->onResizeSwapChain(pSample->getCurrentFbo().get());
        if (pGraphCpy) pGraphCpy->onResizeSwapChain(pSample->getCurrentFbo().get());
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
