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
#include <fstream>
#include "RenderGraphEditor.h"
#include "Utils/RenderGraphLoader.h"
#include "ArgList.h"

const std::string gkDefaultScene = "SunTemple/SunTemple.fscene";

RenderGraphEditor::RenderGraphEditor()
    : mCurrentGraphIndex(0), mCreatingRenderGraph(false), mPreviewing(false)
{
    mNextGraphString.resize(255, 0);
    mCurrentGraphOutput = "";
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
                std::string renderGraphFilePath;
                if (openFileDialog("", renderGraphFilePath))
                {
                    std::string renderGraphFileName = getFilenameFromPath(renderGraphFilePath);
                    createRenderGraph(renderGraphFileName, renderGraphFilePath);
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
    pGui->pushWindow("Render Passes", screenWidth * 2 / 3, screenHeight / 4 - 16, screenWidth / 3, screenHeight * 3 / 4 + 16);

    size_t numRenderPasses = RenderPassLibrary::getRenderPassCount();
    pGui->beginColumns(5);
    for (size_t i = 0; i < numRenderPasses; ++i)
    {
        std::string renderPassClassName = RenderPassLibrary::getRenderPassClassName(i);
        std::string command = std::string("AddRenderPass ") + renderPassClassName + " " + renderPassClassName;
        pGui->addRect((std::string("RenderPass##") + std::to_string(i)).c_str(), { 128.0f, 64.0f }, RenderGraphUI::pickNodeColor(renderPassClassName), false, true);
        pGui->dragDropSource(renderPassClassName.c_str(), "RenderPassScript", command);
        pGui->addText(RenderPassLibrary::getRenderPassClassName(i).c_str());
        pGui->addTooltip(RenderPassLibrary::getRenderPassDesc(i).c_str(), true);
        pGui->nextColumn();
    }

    pGui->popWindow();

    // push a sub GUI window for the node editor
    pGui->pushWindow("Graph Editor", screenWidth, screenHeight * 3 / 4, 0, 16);
    mRenderGraphUIs[mCurrentGraphIndex].renderUI(pGui);
    pGui->popWindow();

    pGui->pushWindow("Graph Editor Settings", screenWidth / 3, screenHeight / 4 - 16, 0, screenHeight * 3 / 4 + 16);

    uint32_t selection = static_cast<uint32_t>(mCurrentGraphIndex);
    if (mOpenGraphNames.size() && pGui->addDropdown("Open Graph", mOpenGraphNames, selection))
    {
        // Display graph
        mCurrentGraphIndex = selection;
        mRenderGraphUIs[mCurrentGraphIndex].reset();
    }

    if (mFilePath.size())
    {
        mRenderGraphUIs[mCurrentGraphIndex].writeUpdateScriptToFile(mFilePath, pSample->getLastFrameTime());
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
            RenderGraphLoader::ExecuteStatement(std::string("SetScene ") + filename, *mpGraphs[mCurrentGraphIndex]);
        }
    }

    // update the display if the render graph loader has set a new output
    if (RenderGraphLoader::sGraphOutputString[0] != '0' && mCurrentGraphOutput != RenderGraphLoader::sGraphOutputString)
    {
        mCurrentGraphOutput = (mGraphOutputEditString = RenderGraphLoader::sGraphOutputString);
    }

    std::vector<std::string> graphOutputString{mGraphOutputEditString};
    if (pGui->addMultiTextBox("Add Output", {"GraphOutput"}, graphOutputString))
    {
        if (mCurrentGraphOutput != mGraphOutputEditString)
        {
            if (mCurrentGraphOutput.size())
            {
                mpGraphs[mCurrentGraphIndex]->unmarkGraphOutput(mCurrentGraphOutput);
            }

            mCurrentGraphOutput = graphOutputString[0];
            mRenderGraphUIs[mCurrentGraphIndex].addOutput(mCurrentGraphOutput);
            mpGraphs[mCurrentGraphIndex]->setOutput(mCurrentGraphOutput, pSample->getCurrentFbo()->getColorTexture(0));
        }
    }
    mGraphOutputEditString = graphOutputString[0];

    if (!mRenderGraphLiveEditor.isOpen() && pGui->addButton("Open Graph Viewer"))
    {
        mRenderGraphLiveEditor.openViewer(*mpGraphs[mCurrentGraphIndex]);
        mFilePath = mRenderGraphLiveEditor.getTempFilePath();
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

    newGraph->onResizeSwapChain(mpLastSample->getCurrentFbo().get());
    mCurrentGraphIndex = mpGraphs.size();
    mpGraphs.push_back(newGraph);

    RenderGraphUI graphUI(*newGraph);
    mRenderGraphUIs.emplace_back(std::move(graphUI));

    if (renderGraphNameFileName.size())
    {
        RenderGraphLoader::LoadAndRunScript(renderGraphNameFileName, *newGraph);
    }

    // load the default scene if none was specified
    if (mpGraphs[mCurrentGraphIndex]->getScene() == nullptr)
    {
        loadScene(gkDefaultScene, false);
    }
    
    // update the display if the render graph loader has set a new output
    if (RenderGraphLoader::sGraphOutputString[0] != '0')
    {
        mCurrentGraphOutput = (mGraphOutputEditString = RenderGraphLoader::sGraphOutputString);
    }

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
    mpGraphs[mCurrentGraphIndex]->addRenderPass(RenderPassLibrary::createRenderPass(renderPassType.c_str()), renderPassName);
}

void RenderGraphEditor::onLoad(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext)
{
    mpLastSample = pSample;
    
#ifdef _WIN32
    // if editor opened from running render graph, get memory view for live update
    std::string commandLine(GetCommandLineA());
    size_t firstSpace = commandLine.find_first_of(' ') + 1;
    mFilePath = (commandLine.substr(firstSpace, commandLine.size() - firstSpace));
#endif

    if (mFilePath.size())
    {
        // TODO -- what do we actually want to name this graph?
        createRenderGraph("Test", mFilePath);
    }
    else
    {
        createRenderGraph("DefaultRenderGraph", "");
    }
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
    mpGraphs[mCurrentGraphIndex]->onResizeSwapChain(pSample->getCurrentFbo().get());
    mpGraphs[mCurrentGraphIndex]->getScene()->getActiveCamera()->setAspectRatio((float)width / (float)height);
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
