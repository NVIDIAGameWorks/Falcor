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
#include "Graphics/RenderGraph/RenderGraphScripting.h"

const std::string gkDefaultScene = "Arcade/Arcade.fscene";
const char* kEditorExecutableName = "RenderGraphEditor";
const char* kSaveFileFilter = "PNG(.png)\0*.png;\0BMP(.bmp)\0*.bmp;\
   \0JPG(.jpg)\0*.jpg;\0HDR(.hdr)\0*.hdr;\0TGA(.tga)\0*.tga;\0";
std::string ir;

RenderGraphViewer::~RenderGraphViewer()
{
    closeSharedFile(mTempFilePath);

    if (mEditorProcess)
    {
        terminateProcess(mEditorProcess);
        mEditorProcess = 0;
    }
}

void RenderGraphViewer::resetCurrentGraphOutputs()
{
    // reset outputs to original state
    GraphViewerInfo& graphInfo = mGraphInfos[mFocusedRenderGraphName];
    graphInfo.mCurrentOutputs = graphInfo.mpGraph->getAvailableOutputs();
    
    for (const auto& output : graphInfo.mCurrentOutputs)
    {
        auto outputIt = graphInfo.mOriginalOutputNames.find(output.outputName);
        if (output.isGraphOutput && outputIt == graphInfo.mOriginalOutputNames.end())
        {
            graphInfo.mpGraph->unmarkOutput(output.outputName);
        }
        else if (!output.isGraphOutput && outputIt != graphInfo.mOriginalOutputNames.end())
        {
            graphInfo.mpGraph->markOutput(output.outputName);
        }
    }
}

void RenderGraphViewer::onGuiRender(SampleCallbacks* pSample, Gui* pGui)
{
    GraphViewerInfo& graphInfo = mGraphInfos[mFocusedRenderGraphName];
    RenderGraph::SharedPtr pGraph = graphInfo.mpGraph;

    if (pGui->addButton("Load Scene"))
    {
        std::string filename;
        if (openFileDialog(Scene::kFileFormatString, filename)) loadScene(filename, true, pSample);
    }

    if (pGui->addButton("Load Graph"))
    {
        std::string filename;
        if (openFileDialog("", filename))
        {
            loadGraphFromFile(pSample, filename);
            mActiveGraphIndex = static_cast<uint32_t>(mGraphInfos.size() - 1);
            return;
        }
    }

    if (pGraph && pGui->addButton("Save Graph"))
    {
        std::string filename;
        if (saveFileDialog("", filename)) RenderGraphExporter::save(pGraph, mFocusedRenderGraphName, filename);
    }

    if (pGui->addDropdown("Focused Render Graph", mRenderGraphsList, mActiveGraphIndex))
    {
        mFocusedRenderGraphName  = std::string(mRenderGraphsList[mActiveGraphIndex].label);
    }

    if (!mEditorRunning && pGui->addButton("Open RenderGraph Editor"))
    {
        resetCurrentGraphOutputs();
    
        char* result = nullptr;
        mTempFilePath = std::tmpnam(result);

        RenderGraphExporter::save(pGraph, mFocusedRenderGraphName, mTempFilePath);

        graphInfo.mFileName = mTempFilePath;
        openSharedFile(mTempFilePath, std::bind(&RenderGraphViewer::fileWriteCallback, this, std::placeholders::_1));
    
        // load application for the editor given it the name of the mapped file
        std::string commandLine = std::string("-tempFile ") + mTempFilePath;
        mEditorProcess = executeProcess(kEditorExecutableName, commandLine);
    
        assert(mEditorProcess);
        mEditorRunning = true;
        mEditingRenderGraphName = mFocusedRenderGraphName;
        pGraph->markOutput(graphInfo.mOutputString);
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
                for (int32_t i = 0; i < static_cast<int32_t>(pGraph->getOutputCount()); ++i)
                {
                    Gui::DropdownValue graphOutput;
                    graphOutput.label = pGraph->getOutputName(i);
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
                pGraph->unmarkOutput(graphInfo.mOutputString);
                graphInfo.mOutputString = renderGraphOutputs[graphInfo.mGraphOutputIndex].label;
                pGraph->markOutput(graphInfo.mOutputString);
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

        renderGUIPreviewWindows(pGui);

        if (displayGraphUI) pGui->endGroup();
    }
}

void RenderGraphViewer::renderGUIPreviewWindows(Gui* pGui)
{
    std::vector<std::string> windowsToRemove;
    for (auto& nameWindow : mDebugWindowInfos)
    {
        const DebugWindowInfo& debugWindowInfo = nameWindow.second;
        RenderGraph::SharedPtr pPreviewGraph = mGraphInfos[debugWindowInfo.mGraphName].mpGraph;

        pGui->pushWindow((debugWindowInfo.mGraphName + " : " + nameWindow.first).c_str(), 330, 268);

        if (pGui->addDropdown("##Render Graph Outputs", mGraphInfos[debugWindowInfo.mGraphName].mOutputDropdown, nameWindow.second.mNextOutputIndex))
        {
            nameWindow.second.mOutputName = mGraphInfos[debugWindowInfo.mGraphName].mOutputDropdown[debugWindowInfo.mNextOutputIndex].label;
            nameWindow.second.mRenderOutput = true;
        }

        if (pGui->addButton("Close"))
        {
            // mark to close after window updates
            windowsToRemove.push_back(nameWindow.first);
            nameWindow.second.mRenderOutput = false;

            // unmark graph output checking the original graph state.
            const auto& graphInfo = mGraphInfos[mFocusedRenderGraphName];
            if (graphInfo.mOriginalOutputNames.find(debugWindowInfo.mOutputName) == graphInfo.mOriginalOutputNames.end())
            {
                pPreviewGraph->markOutput(debugWindowInfo.mOutputName);
            }
        }
        else
        {
            mActiveGraphNames.insert(debugWindowInfo.mGraphName);
        }

        if (debugWindowInfo.mRenderOutput)
        {
            // mark as graph output
            pPreviewGraph->markOutput(debugWindowInfo.mOutputName);
            Texture::SharedPtr pPreviewTex = std::static_pointer_cast<Texture>(pPreviewGraph->getOutput(debugWindowInfo.mOutputName));
            auto format = pPreviewTex->getFormat();

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

            // get size of window to scale image correctly
            glm::vec2 imagePreviewSize = pGui->getCurrentWindowSize();
            uint32_t texHeight = pPreviewTex->getHeight();
            uint32_t texWidth = pPreviewTex->getWidth();
            float imageAspectRatio = static_cast<float>(texHeight) / static_cast<float>(texWidth);
            imagePreviewSize.y = imagePreviewSize.x * imageAspectRatio;

            pGui->addImage(nameWindow.first.c_str(), pPreviewTex, imagePreviewSize);
        }

        pGui->popWindow();
    }

    for (const std::string& windowName : windowsToRemove)
    {
        mDebugWindowInfos.erase(windowName);
    }
}

RenderGraph::SharedPtr RenderGraphViewer::createGraph(SampleCallbacks* pSample)
{
    auto pScripter = RenderGraphScripting::create();
    pScripter->runScript(ir);
    RenderGraph::SharedPtr pGraph = pScripter->getGraph("forward_renderer");

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
    for (auto& graphInfoPair : mGraphInfos)
    {
        if (graphInfoPair.second.mpGraph) graphInfoPair.second.mpGraph->setScene(mpScene);
    }
}

void RenderGraphViewer::fileWriteCallback(const std::string& fileName)
{
    GraphViewerInfo& graphInfo = mGraphInfos[mEditingRenderGraphName];
    std::string fullScript, script;
    readFileToString(fileName, fullScript);
    if (!fullScript.size()) return;
    script.resize(*(size_t*)(fullScript.data()));
    script.insert(script.begin(), fullScript.data() + sizeof(size_t), fullScript.data() + script.size() + sizeof(size_t));
    graphInfo.mLastScript = script;
    mApplyGraphChanges = true;
}

void RenderGraphViewer::loadGraphFromFile(SampleCallbacks* pSample, const std::string& filename)
{
    const auto pGraphs = RenderGraphImporter::importAllGraphs(filename);
    if (pGraphs.size())
    {
        // for now create a parallel copy of the render graph
        const auto pGraph = pGraphs.front().pGraph;
        const std::string graphName = pGraphs.front().name;
        pGraph->onResizeSwapChain(pSample->getCurrentFbo().get());
        insertNewGraph(pGraph, filename, graphName);
        pGraph->setScene(mpScene);
    }
    else
    {
        logError("Can't find a graph in " + filename);
    }
}

RenderGraph::SharedPtr RenderGraphViewer::createDefaultGraph(SampleCallbacks* pSample)
{
    Falcor::RenderGraphIR::SharedPtr pIr = RenderGraphIR::create("forward_renderer");

    pIr->addPass("DepthPass", "DepthPrePass");
    pIr->addPass("ForwardLightingPass", "LightingPass");
    pIr->addPass("CascadedShadowMaps", "ShadowPass");
    pIr->addPass("BlitPass", "BlitPass");
    pIr->addPass("ToneMapping", "ToneMapping");
    pIr->addPass("SSAO", "SSAO");
    pIr->addPass("FXAA", "FXAA");
    pIr->addPass("SkyBox", "SkyBox");

    pIr->addEdge("DepthPrePass.depth", "SkyBox.depth");
    pIr->addEdge("SkyBox.target", "LightingPass.color");
    pIr->addEdge("DepthPrePass.depth", "ShadowPass.depth");
    pIr->addEdge("DepthPrePass.depth", "LightingPass.depth");
    pIr->addEdge("ShadowPass.visibility", "LightingPass.visibilityBuffer");
    pIr->addEdge("LightingPass.color", "ToneMapping.src");
    pIr->addEdge("ToneMapping.dst", "SSAO.colorIn");
    pIr->addEdge("LightingPass.normals", "SSAO.normals");
    pIr->addEdge("LightingPass.depth", "SSAO.depth");
    pIr->addEdge("SSAO.colorOut", "FXAA.src");
    pIr->addEdge("FXAA.dst", "BlitPass.src");

    pIr->markOutput("BlitPass.dst");

    ir = pIr->getIR();
    ir += "forward_renderer = render_graph_forward_renderer()";

    RenderGraph::SharedPtr pGraph = createGraph(pSample);
    return pGraph;
}

void RenderGraphViewer::insertNewGraph(const RenderGraph::SharedPtr& pGraph, 
    const std::string& fileName, const std::string& graphName)
{
    size_t renderGraphIndex = mRenderGraphsList.size();
    mFocusedRenderGraphName = graphName;

    Gui::DropdownValue value;
    value.value = static_cast<uint32_t>(mRenderGraphsList.size());
    value.label = graphName;
    mRenderGraphsList.push_back(value);

    GraphViewerInfo& graphInfo = mGraphInfos[mFocusedRenderGraphName];
    for (int32_t i = 0; i < static_cast<int32_t>(pGraph->getOutputCount()); ++i)
    {
        graphInfo.mOriginalOutputNames.insert(pGraph->getOutputName(i));
    }

    graphInfo.mpGraph = pGraph;
    graphInfo.mCurrentOutputs = pGraph->getAvailableOutputs();
    graphInfo.mFileName = fileName;
}

void RenderGraphViewer::updateOutputDropdown(const std::string& passName)
{
    GraphViewerInfo& graphInfo = mGraphInfos[passName];
    int32_t i = 0;

    graphInfo.mOutputDropdown.clear();

    for (const auto& outputPair : graphInfo.mCurrentOutputs)
    {
        Gui::DropdownValue graphOutput;
        graphOutput.label = outputPair.outputName;
        if (outputPair.outputName == graphInfo.mOutputString)
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
        loadGraphFromFile(pSample, filePath);
        if (filePath.size())
        {
            openSharedFile(filePath, std::bind(&RenderGraphViewer::fileWriteCallback, this, std::placeholders::_1));
            mEditingRenderGraphName = mFocusedRenderGraphName;
        }
        else
        {
            msgBox("No path to temporary file provided");
        }
    }
    else
    {
        RenderGraph::SharedPtr pGraph = createDefaultGraph(pSample);
        insertNewGraph(pGraph, "", "forward_renderer");
    }

    loadScene(gkDefaultScene, false, pSample);
    mpScene->getActiveCamera()->setAspectRatio((float)pSample->getCurrentFbo()->getWidth() / (float)pSample->getCurrentFbo()->getHeight());
    mCamControl.attachCamera(mpScene->getCamera(0));
}

void RenderGraphViewer::onFrameRender(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext, const Fbo::SharedPtr& pTargetFbo)
{
    const glm::vec4 clearColor(0.38f, 0.52f, 0.10f, 1);
    pRenderContext->clearFbo(pTargetFbo.get(), clearColor, 1.0f, 0, FboAttachmentType::All);

    if (mApplyGraphChanges)
    {
        GraphViewerInfo& applyGraphInfo = mGraphInfos[mEditingRenderGraphName];

        // apply all change for valid graph
        auto pScripting = RenderGraphScripting::create();
        pScripting->addGraph(mEditingRenderGraphName, applyGraphInfo.mpGraph);
        pScripting->runScript(applyGraphInfo.mLastScript);

        applyGraphInfo.mLastScript.clear();
        applyGraphInfo.mCurrentOutputs = applyGraphInfo.mpGraph->getAvailableOutputs();
        mApplyGraphChanges = false;
    }

    mActiveGraphNames.insert(mFocusedRenderGraphName);

    for (const std::string& graphName : mActiveGraphNames)
    {
        GraphViewerInfo& graphInfo = mGraphInfos[graphName];
        RenderGraph::SharedPtr pGraph = graphInfo.mpGraph;
        updateOutputDropdown(graphName);

        if (pGraph)
        {
            pGraph->execute(pRenderContext.get());
            Texture::SharedPtr pDisplayTex = std::static_pointer_cast<Texture>(pGraph->getOutput(graphInfo.mOutputString));
            pGraph->getScene()->update(pSample->getCurrentTime(), &mCamControl);
            // Texture may be null for one frame on switching between graphs with different outputs
            if (pDisplayTex && (graphName == mFocusedRenderGraphName))
            {
                pRenderContext->blit(pDisplayTex->getSRV(), pTargetFbo->getColorTexture(0)->getRTV());
            }
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

void RenderGraphViewer::onDataReload(SampleCallbacks* pSample)
{
    if (mEditorRunning)
    {
        logWarning("Warning: Updating graph while editor is open. Graphs may not match.");
    }

    // Reload all graphs, while maintaining state
    for (const auto& graphInfo : mGraphInfos)
    {
        RenderGraph::SharedPtr pGraph = graphInfo.second.mpGraph;
        RenderGraph::SharedPtr pNewGraph;
        if (!graphInfo.second.mFileName.size())
        {
            pNewGraph = createDefaultGraph(pSample);
        }
        else
        {

            pNewGraph = RenderGraphImporter::import(graphInfo.first, graphInfo.second.mFileName);
        }
        
        pGraph->update(pNewGraph);
    }
}

void RenderGraphViewer::onResizeSwapChain(SampleCallbacks* pSample, uint32_t width, uint32_t height)
{
    // update swap chain data for all graphs
    for (auto& graphInfoPair : mGraphInfos)
    {
        RenderGraph::SharedPtr pGraph = graphInfoPair.second.mpGraph;
        if (pGraph) pGraph->onResizeSwapChain(pSample->getCurrentFbo().get());
    }
}

void RenderGraphViewer::loadModel(SampleCallbacks* pSample, const std::string& filename, bool showProgressBar)
{
    Mesh::resetGlobalIdCounter();
 
    ProgressBar::SharedPtr pBar;
    if (showProgressBar)
    {
        pBar = ProgressBar::create("Loading Model");
    }

    Model::SharedPtr pModel = Model::createFromFile(filename.c_str());
    if (!pModel)
    {
        logError(std::string("Failed to load model: \"") + filename + "\"");
    }

    mGraphInfos[mFocusedRenderGraphName].mpGraph->getScene()->addModelInstance(pModel, "instance");
}

void RenderGraphViewer::onInitializeTesting(SampleCallbacks* pSample)
{
    auto args = pSample->getArgList();
    std::vector<ArgList::Arg> model = args.getValues("loadmodel");
    if (!model.empty())
    {
        loadModel(pSample, model[0].asString(), false);
    }

    std::vector<ArgList::Arg> scene = args.getValues("loadscene");
    if (!scene.empty())
    {
        loadScene(scene[0].asString(), false, pSample);
    }

    std::vector<ArgList::Arg> cameraPos = args.getValues("camerapos");
    if (!cameraPos.empty())
    {
        mGraphInfos[mFocusedRenderGraphName].mpGraph->getScene()->getActiveCamera()->setPosition(
            glm::vec3(cameraPos[0].asFloat(), cameraPos[1].asFloat(), cameraPos[2].asFloat()));
    }

    std::vector<ArgList::Arg> cameraTarget = args.getValues("cameratarget");
    if (!cameraTarget.empty())
    {
        mGraphInfos[mFocusedRenderGraphName].mpGraph->getScene()->getActiveCamera()->setTarget(
            glm::vec3(cameraTarget[0].asFloat(), cameraTarget[1].asFloat(), cameraTarget[2].asFloat()));
    }
}

void RenderGraphViewer::onBeginTestFrame(SampleTest* pSampleTest)
{
    //  Already existing. Is this a problem?
    auto nextTriggerType = pSampleTest->getNextTriggerType();
    if (nextTriggerType == SampleTest::TriggerType::None)
    {
        SampleTest::TaskType taskType = (nextTriggerType == SampleTest::TriggerType::Frame) ? pSampleTest->getNextFrameTaskType() : pSampleTest->getNextTimeTaskType();
        RenderPass::SharedPtr pShadowPass = mGraphInfos[mFocusedRenderGraphName].mpGraph->getPass("ShadowPass");
        if (pShadowPass != nullptr)
        {
            std::static_pointer_cast<CascadedShadowMaps>(pShadowPass)->setSdsmReadbackLatency(taskType == SampleTest::TaskType::ScreenCaptureTask ? 0 : 1);
        }
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
