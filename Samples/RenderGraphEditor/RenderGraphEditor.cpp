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
#include <fstream>
#include "RenderGraphEditor.h"
#include "Externals/dear_imgui/imgui.h"
#include "Utils/RenderGraphLoader.h"
#include "ArgList.h"

const std::string gkDefaultScene = "SunTemple/SunTemple.fscene";

RenderGraphEditor::RenderGraphEditor()
    : mCurrentGraphIndex(0), mCreatingRenderGraph(false), mPreviewing(false)
{
    mNextGraphString.resize(255, 0);
    mCurrentGraphOutput = "BlitPass.dst";
    mGraphOutputEditString = mCurrentGraphOutput;
    mGraphOutputEditString.resize(255, 0);
}

// some of this will need to be moved to render graph ui
void RenderGraphEditor::onGuiRender(SampleCallbacks* pSample, Gui* pGui)
{
    uint32_t screenHeight = pSample->getWindow()->getClientAreaHeight();
    uint32_t screenWidth = pSample->getWindow()->getClientAreaWidth();

    if (pGui->beginMainMenuBar())
    {
        if (pGui->beginDropDownMenu("File"))
        {
            if (!mShowCreateGraphWindow && pGui->addMenuItem("Create New Graph"))
            {
                mShowCreateGraphWindow = true;
            }

            if (pGui->addMenuItem("Load Graph"))
            {
                std::string renderGraphFileName;
                if (openFileDialog("", renderGraphFileName))
                {
                    size_t nameOffset = renderGraphFileName.find_last_of('\\') + 1;
                    size_t fileExtOffset = renderGraphFileName.find_last_of('.');
                    if (fileExtOffset == std::string::npos)
                    {
                        fileExtOffset = renderGraphFileName.size();
                    }

                    createRenderGraph(renderGraphFileName.substr(nameOffset, fileExtOffset - nameOffset), renderGraphFileName);
                }
            }

            if (pGui->addMenuItem("Save Graph"))
            {
                std::string renderGraphFileName;
                if (saveFileDialog("", renderGraphFileName))
                {
                    serializeRenderGraph(renderGraphFileName);
                }
            }

            if (pGui->addMenuItem("RunScript"))
            {
                std::string renderGraphFileName;
                if (openFileDialog("", renderGraphFileName))
                {
                    deserializeRenderGraph(renderGraphFileName);
                }
            }

            pGui->endDropDownMenu();
        }

        pGui->endMainMenuBar();
    }

    // sub window for listing available window passes
    pGui->pushWindow("Render Passes", screenWidth * 7 / 8, screenHeight / 4, screenWidth / 8, screenHeight * 4 / 5);

    // for each dll that was found. (or manually imported????)
    for (auto& availableRenderPasses : RenderGraphLoader::sBaseRenderCreateFuncs)
    {
        ImVec2 nextDragRegionPos{ ImGui::GetCursorScreenPos().x + 64.0f, ImGui::GetCursorScreenPos().y + 32.0f };
        ImGui::GetWindowDrawList()->AddRect(ImGui::GetCursorScreenPos(), nextDragRegionPos, 0xFFFFFFFF);
        ImGui::Dummy({ 64.0f , 32.0f });

        std::string command = std::string("AddRenderPass ") + availableRenderPasses.first + " " + availableRenderPasses.first;
        pGui->dragDropSource(availableRenderPasses.first.c_str(), "RenderPassScript", command);

        ImGui::SameLine();
        pGui->addText(availableRenderPasses.first.c_str());

        ImGui::SetCursorScreenPos(nextDragRegionPos);
        ImGui::SameLine();
    }

    pGui->popWindow();

    // push a sub GUI window for the node editor
    pGui->pushWindow("Graph Editor", screenWidth * 7 / 8, screenHeight * 4 / 5, screenWidth / 8, 1);
    mRenderGraphUIs[mCurrentGraphIndex].renderUI(pGui);
    pGui->popWindow();

    pGui->pushWindow("Graph Editor Settings", screenWidth / 8, screenHeight / 2, 0, screenHeight / 2, false);

    uint32_t selection = static_cast<uint32_t>(mCurrentGraphIndex);
    if (mOpenGraphNames.size() && pGui->addDropdown("Open Graph", mOpenGraphNames, selection))
    {
        // Display graph
        mCurrentGraphIndex = selection;
        mRenderGraphUIs[mCurrentGraphIndex].reset();
    }

    if (pGui->addButton("Update Graph"))
    {
        mRenderGraphUIs[mCurrentGraphIndex].writeUpdateScriptToFile(mFilePath);   
    }

    if (pGui->addButton("Preview Graph"))
    {
        mPreviewing = true;
    }

    // Load scene for graph
    if (pGui->addButton("LoadScene"))
    {
        std::string filename;
        if (openFileDialog(Scene::kFileFormatString, filename))
        {
            loadScene(filename, true);
        }
    }

    // update the display if the render graph loader has set a new output
    if (RenderGraphLoader::sGraphOutputString[0] != '0' && mCurrentGraphOutput != RenderGraphLoader::sGraphOutputString)
    {
        // mpGraphs[mCurrentGraphIndex]->unmarkGraphOutput(mCurrentGraphOutput);
        mCurrentGraphOutput = (mGraphOutputEditString = RenderGraphLoader::sGraphOutputString);
        mpGraphs[mCurrentGraphIndex]->setOutput(mCurrentGraphOutput, pSample->getCurrentFbo()->getColorTexture(0));
    }

    std::vector<std::string> graphOutputString{mGraphOutputEditString};
    if (pGui->addMultiTextBox("Update", {"GraphOutput"}, graphOutputString)) // addButton("Update"))
    {
        if (mCurrentGraphOutput != mGraphOutputEditString)
        {
            mpGraphs[mCurrentGraphIndex]->unmarkGraphOutput(mCurrentGraphOutput);
            mCurrentGraphOutput = mGraphOutputEditString;
            mpGraphs[mCurrentGraphIndex]->markGraphOutput(mCurrentGraphOutput);
            mpGraphs[mCurrentGraphIndex]->setOutput(mCurrentGraphOutput, pSample->getCurrentFbo()->getColorTexture(0));
        }
    }

    pGui->popWindow();

    // pop up window for naming a new render graph
    if (mShowCreateGraphWindow)
    {
        pGui->pushWindow("CreateNewGraph", 256, 128, screenWidth / 2 - 128, screenHeight / 2 - 64);

        pGui->addTextBox("Graph Name", mNextGraphString);

        if (pGui->addButton("Create Graph") && mNextGraphString[0])
        {
            createRenderGraph(mNextGraphString, "");
            mNextGraphString.clear();
            mNextGraphString.resize(255, '0');
            mShowCreateGraphWindow = false;
        }

        if (pGui->addButton("Cancel", true))
        {
            mNextGraphString.clear();
            mNextGraphString.resize(255, '0');
            mShowCreateGraphWindow = false;
        }

        pGui->popWindow();
    }
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

    if (!pScene) { logWarning("Failed to load scene for current render graph"); }

    mpGraphs[mCurrentGraphIndex]->setScene(pScene);
    mCamControl.attachCamera(pScene->getCamera(0));
}

void RenderGraphEditor::serializeRenderGraph(const std::string& fileName)
{
    RenderGraphLoader::SaveRenderGraphAsScript(fileName, *mpGraphs[mCurrentGraphIndex]);
}

void RenderGraphEditor::deserializeRenderGraph(const std::string& fileName)
{
    RenderGraphLoader::LoadAndRunScript(fileName, *mpGraphs[mCurrentGraphIndex]);
    RenderGraphUI::sRebuildDisplayData = true;
}

void RenderGraphEditor::createForwardRendererGraph()
{
    mCreatingRenderGraph = true;

    Gui::DropdownValue nextGraphID;
    nextGraphID.value = static_cast<int32_t>(mOpenGraphNames.size());
    nextGraphID.label = "ForwardRenderer";
    mOpenGraphNames.push_back(nextGraphID);

    RenderGraph::SharedPtr newGraph;

    // test that this graph shows up in the editor correctly
    newGraph = RenderGraph::create();

    Scene::SharedPtr pScene = Scene::loadFromFile(gkDefaultScene);

    if (!pScene) { logWarning("Failed to load scene for current render graph"); }

    newGraph->setScene(pScene);
    mCamControl.attachCamera(pScene->getCamera(0));

    auto pLightingPass = SceneLightingPass::create();
    pLightingPass->setColorFormat(ResourceFormat::RGBA32Float).setMotionVecFormat(ResourceFormat::RG16Float).setNormalMapFormat(ResourceFormat::RGBA8Unorm).setSampleCount(1).usePreGeneratedDepthBuffer(true);
    newGraph->addRenderPass(pLightingPass, "LightingPass");

    newGraph->addRenderPass(DepthPass::create(), "DepthPrePass");
    newGraph->addRenderPass(CascadedShadowMaps::create(pScene->getLight(0)), "ShadowPass");
    newGraph->addRenderPass(BlitPass::create(), "BlitPass");
    newGraph->addRenderPass(ToneMapping::create(ToneMapping::Operator::Aces), "ToneMapping");
    newGraph->addRenderPass(SSAO::create(uvec2(1024)), "SSAO");
    newGraph->addRenderPass(FXAA::create(), "FXAA");

    // Add the skybox
    Scene::UserVariable var = pScene->getUserVariable("sky_box");
    assert(var.type == Scene::UserVariable::Type::String);
    std::string skyBox = getDirectoryFromFile(gkDefaultScene) + '/' + var.str;
    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
    newGraph->addRenderPass(SkyBox::createFromTexture(skyBox, true, Sampler::create(samplerDesc)), "SkyBox");

    newGraph->addEdge("DepthPrePass.depth", "ShadowPass.depth");
    newGraph->addEdge("DepthPrePass.depth", "LightingPass.depth");
    newGraph->addEdge("DepthPrePass.depth", "SkyBox.depth");

    newGraph->addEdge("SkyBox.target", "LightingPass.color");
    newGraph->addEdge("ShadowPass.visibility", "LightingPass.visibilityBuffer");

    newGraph->addEdge("LightingPass.color", "ToneMapping.src");
    newGraph->addEdge("ToneMapping.dst", "SSAO.colorIn");
    newGraph->addEdge("LightingPass.normals", "SSAO.normals");
    newGraph->addEdge("LightingPass.depth", "SSAO.depth");

    newGraph->addEdge("SSAO.colorOut", "FXAA.src");
    newGraph->addEdge("FXAA.dst", "BlitPass.src");

    newGraph->setScene(pScene);
    newGraph->onResizeSwapChain(mpLastSample->getCurrentFbo().get());
    mCurrentGraphIndex = mpGraphs.size();
    mpGraphs.push_back(newGraph);

    RenderGraphUI graphUI(*newGraph);
    mRenderGraphUIs.emplace_back(std::move(graphUI));

    mpGraphs[mCurrentGraphIndex]->setOutput(mCurrentGraphOutput, mpLastSample->getCurrentFbo()->getColorTexture(0));
    mpGraphs[mCurrentGraphIndex]->onResizeSwapChain(mpLastSample->getCurrentFbo().get());

    mCreatingRenderGraph = false;
    RenderGraphUI::sRebuildDisplayData = true;
}

void RenderGraphEditor::createRenderGraph(const std::string& renderGraphName, const std::string& renderGraphNameFileName)
{
    mCreatingRenderGraph = true;

    Gui::DropdownValue nextGraphID;
    nextGraphID.value = static_cast<int32_t>(mOpenGraphNames.size());
    nextGraphID.label = renderGraphName;
    mOpenGraphNames.push_back(nextGraphID);
    
    RenderGraph::SharedPtr newGraph;

    // test that this graph shows up in the editor correctly
    newGraph = RenderGraph::create();

    Scene::SharedPtr pScene = Scene::loadFromFile(gkDefaultScene);
    if (!pScene) { logWarning("Failed to load scene for current render graph"); }
    newGraph->setScene(pScene);

    newGraph->onResizeSwapChain(mpLastSample->getCurrentFbo().get());
    mCurrentGraphIndex = mpGraphs.size();
    mpGraphs.push_back(newGraph);

    RenderGraphUI graphUI(*newGraph);
    mRenderGraphUIs.emplace_back(std::move(graphUI));

    if (renderGraphNameFileName.size())
    {
        RenderGraphLoader::LoadAndRunScript(renderGraphNameFileName, *newGraph);
    }

    if (mCurrentGraphIndex >= 1)
    {
        mpGraphs[mCurrentGraphIndex]->setScene(mpGraphs[0]->getScene());
    }
    else
    {
        loadScene(gkDefaultScene, false);
    }
    
    mpGraphs[mCurrentGraphIndex]->setOutput(mCurrentGraphOutput, mpLastSample->getCurrentFbo()->getColorTexture(0));
    mpGraphs[mCurrentGraphIndex]->onResizeSwapChain(mpLastSample->getCurrentFbo().get());

    mCreatingRenderGraph = false;
    RenderGraphUI::sRebuildDisplayData = true;
}

void RenderGraphEditor::createAndAddConnection(const std::string& srcRenderPass, const std::string& dstRenderPass, const std::string& srcField, const std::string& dstField)
{
    // add information for GUI to avoid costly drawing in renderUI function for graph
    if (mpGraphs[mCurrentGraphIndex]->addEdge(srcRenderPass + std::string(".") + srcField, dstRenderPass + std::string(".") + dstField) == RenderGraph::kInvalidIndex)
    {
        logWarning(std::string("Failed to create edge between nodes: ").append(srcRenderPass)
            .append(" and ").append(dstRenderPass).append( " connecting fields ").append(srcField).append(" to ").append(dstField).append(".\n"));
    }
}

void RenderGraphEditor::createAndAddRenderPass(const std::string& renderPassType, const std::string& renderPassName)
{
    mpGraphs[mCurrentGraphIndex]->addRenderPass(RenderGraphLoader::sBaseRenderCreateFuncs[renderPassType](), renderPassName);
}

void RenderGraphEditor::onLoad(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext)
{
    mpLastSample = pSample;
    
#ifdef _WIN32
    // if editor opened from running render graph, get memory view for live update
    std::string commandLine(GetCommandLineA());
    size_t firstSpace = commandLine.find_first_of(' ') + 1;
    mFilePath = (commandLine.substr(firstSpace, commandLine.size() - firstSpace));
    msgBox(mFilePath);
#endif

    if (mFilePath.size())
    {
        createRenderGraph("Test", mFilePath);
        // loadGraphFromSharedMemory(filePath);
    }
    else
    {
        createRenderGraph("DefaultRenderGraph", "");
        //createForwardRendererGraph();
    }
}

void RenderGraphEditor::renderGraphEditorGUI(SampleCallbacks* pSample, Gui* pGui)
{
}

void RenderGraphEditor::onFrameRender(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext, const Fbo::SharedPtr& pTargetFbo)
{
    if (mPreviewing && pSample->isKeyPressed(KeyboardEvent::Key::E))
    {
        mPreviewing = false;
    }

    mpLastSample = pSample;

    // render the editor GUI graph
    const glm::vec4 clearColor(1, 1, 1 , 1);
    pRenderContext->clearFbo(pTargetFbo.get(), clearColor, 1.0f, 0, FboAttachmentType::All);

    if (!mPreviewing)
    {
        // draw node graph editor into specialized graph
        pSample->getRenderContext()->getGraphicsState()->setFbo(pTargetFbo);
    }
    else
    {
        mpGraphs[mCurrentGraphIndex]->getScene()->update(pSample->getCurrentTime(), &mCamControl);
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
    mpGraphs[mCurrentGraphIndex]->setOutput(mCurrentGraphOutput, pSample->getCurrentFbo()->getColorTexture(0));
    mpGraphs[mCurrentGraphIndex]->onResizeSwapChain(pSample->getCurrentFbo().get());

    mpGraphs[mCurrentGraphIndex]->getScene()->getActiveCamera()->setAspectRatio((float)width / (float)height);
}

void RenderGraphEditor::loadGraphFromSharedMemory(const std::string& renderGraphFilePath)
{
#ifdef _WIN32
    OFSTRUCT of;
    // HFILE fileHandle =  OpenFile(renderGraphFilePath.c_str(), &of, OF_READ);
    
    std::string wideFilePath;
    wideFilePath.resize(renderGraphFilePath.size() * 2 + 1);
    mbstowcs((wchar_t*)&wideFilePath.front(), renderGraphFilePath.c_str(), renderGraphFilePath.size());

    HFILE fileHandle = OpenFile(renderGraphFilePath.c_str(), &of, OF_READ);
    if (!fileHandle)
    {
        logError("Can't open the file");
        return;
    }

    HANDLE tempFileHndl = *reinterpret_cast<HANDLE*>(&fileHandle);
    //CreateFileMapping(tempFileHndl, NULL, PAGE_READWRITE, 0, 0, NULL); //=
    HANDLE tempFileMappingHndl =  OpenFileMapping(FILE_MAP_ALL_ACCESS, TRUE, (LPCWSTR)wideFilePath.c_str());
    if (!tempFileMappingHndl)
    {
        logError("Unable to map temporary file for graph editor");
        return;
    }

    // change this to a persistant mapping
    char* pData = (char*)MapViewOfFile(tempFileMappingHndl, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, 0);

    if (!pData)
    {
        logError("Unable to map temporary file for render graph data");
        if (tempFileMappingHndl) { CloseHandle(tempFileMappingHndl);  }
    }

    // copy over the memory for any change
    std::string sharedMemoryStage;
    sharedMemoryStage.resize((size_t)0x01400000);
    CopyMemory(&sharedMemoryStage.front(), pData, sharedMemoryStage.size());

    // read and execute the data
    RenderGraphLoader::runScript(sharedMemoryStage.data() + sizeof(size_t), *(size_t*)sharedMemoryStage.data(), *mpGraphs[mCurrentGraphIndex]);
#endif
}

#ifdef _WIN32
int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd)
#else
int main(int argc, char** argv)
#endif
{
    RenderGraphEditor::UniquePtr pEditor = std::make_unique<RenderGraphEditor>();
    SampleConfig config;
    config.windowDesc.title = "Render Graph Editor";
    config.windowDesc.resizableWindow = true;
    Sample::run(config, pEditor);
    return 0;
}
