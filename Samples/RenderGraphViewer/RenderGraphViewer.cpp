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

const std::string gkDefaultScene = "Arcade/Arcade.fscene";// ;"EmeraldSquare/EmeraldSquare_day.fscene";
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
    mCurrentOutputs = mpGraph->getAvailableOutputs();
    
    for (const auto& output : mCurrentOutputs)
    {
        auto outputIt = mOriginalOutputNames.find(output.first);
        if (output.second && outputIt == mOriginalOutputNames.end())
        {
            mpGraph->unmarkGraphOutput(output.first);
        }
        else if (!output.second && outputIt != mOriginalOutputNames.end())
        {
            mpGraph->markGraphOutput(output.first);
        }
    }
}

void RenderGraphViewer::copyGraph(const RenderGraph::SharedPtr& pSrc, RenderGraph::SharedPtr pDst)
{
    // TODO copy graph state over before attempting to compile changes from scripts or editor


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

    if (!mEditorRunning && pGui->addButton("Edit RenderGraph"))
    {
        resetGraphOutputs();

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
            int32_t i = 0;

            for (const auto& outputPair : mCurrentOutputs)
            {
                Gui::DropdownValue graphOutput;
                graphOutput.label = outputPair.first;
                if (outputPair.first == mOutputString)
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

        if (renderGraphOutputs.size())
        {
            if (pGui->addButton("Open Output Window"))
            {
                size_t size =  mDebugWindowInfos.size();
                DebugWindowInfo debugWindowInfo;
                debugWindowInfo.mOutputName = renderGraphOutputs[0].label;
                mDebugWindowInfos.insert(std::make_pair(std::string("Debug Window ") + std::to_string(size), debugWindowInfo));
            }
        }
        
        std::vector<std::string> windowsToRemove;

        for (auto& nameWindow : mDebugWindowInfos)
        {
            DebugWindowInfo& debugWindowInfo = nameWindow.second;

            pGui->pushWindow((std::string("mpGraphName : ") + nameWindow.first).c_str(), 330, 268);
        
            if (pGui->addDropdown("##Render Graph Outputs", renderGraphOutputs, debugWindowInfo.mNextOutputIndex))
            {
                debugWindowInfo.mOutputName = renderGraphOutputs[debugWindowInfo.mNextOutputIndex].label;
                debugWindowInfo.mRenderOutput = true;
            }

            if (pGui->addButton("Close"))
            {
                // mark to close after window updates
                windowsToRemove.push_back(nameWindow.first);
                debugWindowInfo.mRenderOutput = false;

                // unmark graph output checking the original graph state.
                if (mOriginalOutputNames.find(debugWindowInfo.mOutputName) == mOriginalOutputNames.end())
                {
                    mpGraph->unmarkGraphOutput(debugWindowInfo.mOutputName);
                }
            }

            if (debugWindowInfo.mRenderOutput)
            {
                // mark as graph output
                mpGraph->markGraphOutput(debugWindowInfo.mOutputName);
                Texture::SharedPtr pPreviewTex = std::static_pointer_cast<Texture>(mpGraph->getOutput(debugWindowInfo.mOutputName));

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

                glm::vec2 imagePreviewSize = pGui->getWindowSize();
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

    // check valid


    // rebuild data
    mCurrentOutputs = mpGraph->getAvailableOutputs();
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
        mOriginalOutputNames.insert(mpGraph->getGraphOutputName(i));
    }

    mCurrentOutputs = mpGraph->getAvailableOutputs();
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
