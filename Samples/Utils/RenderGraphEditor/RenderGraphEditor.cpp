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

const char* kViewerExecutableName = "RenderGraphViewer";

RenderGraphEditor::RenderGraphEditor()
    : mCurrentGraphIndex(0)
{
    mNextGraphString.resize(255, 0);
    mCurrentGraphOutput = "";
    mGraphOutputEditString = mCurrentGraphOutput;
    mGraphOutputEditString.resize(255, 0);
}

RenderGraphEditor::~RenderGraphEditor()
{
    if (mViewerProcess)
    {
        terminateProcess(mViewerProcess);
        mViewerProcess = 0;
    }
}

// some of this will need to be moved to render graph ui
void RenderGraphEditor::onGuiRender(SampleCallbacks* pSample, Gui* pGui)
{
    uint32_t screenHeight = mWindowSize.y;
    uint32_t screenWidth = mWindowSize.x;

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
                    mpGraphs[mCurrentGraphIndex]->onResizeSwapChain(pSample->getCurrentFbo().get());
                }
            }

            if (pGui->addMenuItem("Save Graph"))
            {
                bool bSaveGraph = true;
                std::string log;

                if (!mpGraphs[mCurrentGraphIndex]->isValid(log))
                {
                    MsgBoxButton msgBoxButton = msgBox("Attempting to save invalid graph.\nGraph may not execute correctly when loaded\nAre you sure you want to save the graph?"
                        , MsgBoxType::OkCancel);
                    bSaveGraph = !(msgBoxButton == MsgBoxButton::Cancel);
                }

                if (bSaveGraph)
                {
                    std::string renderGraphFileName;
                    if (saveFileDialog("", renderGraphFileName))
                    {
                        serializeRenderGraph(renderGraphFileName);
                    }
                }
            }

            if (pGui->addMenuItem("RunScript"))
            {
                std::string renderGraphFileName;
                if (openFileDialog("", renderGraphFileName))
                {
                    RenderGraphScripting::SharedPtr pScripting = RenderGraphScripting::create();
                    pScripting->addGraph(mRenderGraphUIs[mCurrentGraphIndex].getName(), mpGraphs[mCurrentGraphIndex]);
                    pScripting->runScript(readFile(renderGraphFileName));
                    mRenderGraphUIs[mCurrentGraphIndex].setToRebuild();
                }
            }

            pGui->endDropDownMenu();
        }

        pGui->endMainMenuBar();
    }

    // sub window for listing available window passes
    pGui->pushWindow("Render Passes", screenWidth / 2, screenHeight / 4 - 20, screenWidth / 4, screenHeight * 3 / 4 + 20, true);
    if (mResetGuiWindows)
    {
        pGui->setCurrentWindowSize(screenWidth / 2, screenHeight / 4 - 20);
        pGui->setCurrentWindowPos(screenWidth / 4, screenHeight * 3 / 4 + 20);
    }

    size_t numRenderPasses = RenderPassLibrary::getClassCount();
    pGui->beginColumns(5);
    for (size_t i = 0; i < numRenderPasses; ++i)
    {
        std::string renderPassClassName = RenderPassLibrary::getClassName(i);
        pGui->addRect({ 148.0f, 64.0f }, pGui->pickUniqueColor(renderPassClassName), false);
        pGui->addDummyItem((std::string("RenderPass##") + std::to_string(i)).c_str(), { 148.0f, 44.0f });
        pGui->dragDropSource(renderPassClassName.c_str(), "RenderPassType", renderPassClassName);
        pGui->addText(RenderPassLibrary::getClassName(i).c_str());
        pGui->addTooltip(RenderPassLibrary::getPassDesc(i).c_str(), false);
        pGui->nextColumn();
    }

    pGui->popWindow();

    // push a sub GUI window for the node editor
    pGui->pushWindow("Graph Editor", screenWidth, screenHeight * 3 / 4, 0, 20, false);
    if (mResetGuiWindows)
    {
        pGui->setCurrentWindowSize(screenWidth, screenHeight * 3 / 4);
        pGui->setCurrentWindowPos(0, 20);
    }
    mRenderGraphUIs[mCurrentGraphIndex].renderUI(pGui);
    pGui->popWindow();

    for (auto& renderGraphUI : mRenderGraphUIs)
    {
        mCurrentLog += renderGraphUI.getCurrentLog();
        renderGraphUI.clearCurrentLog();
    }
    
    pGui->pushWindow("Graph Editor Settings", screenWidth / 4, screenHeight / 4 - 20, 0, screenHeight * 3 / 4 + 20, true);
    if (mResetGuiWindows)
    {
        pGui->setCurrentWindowSize(screenWidth / 4, screenHeight / 4 - 20);
        pGui->setCurrentWindowPos(0, screenHeight * 3 / 4 + 20);
    }

    uint32_t selection = static_cast<uint32_t>(mCurrentGraphIndex);
    if (mOpenGraphNames.size() && pGui->addDropdown("Open Graph", mOpenGraphNames, selection))
    {
        // Display graph
        mCurrentGraphIndex = selection;
    }

    if (mFilePath.size())
    {
        mRenderGraphUIs[mCurrentGraphIndex].writeUpdateScriptToFile(mFilePath, pSample->getLastFrameTime());
    }

    if (mViewerRunning && mViewerProcess)
    {
        if (!isProcessRunning(mViewerProcess))
        {
            terminateProcess(mViewerProcess);
            mViewerProcess = 0;
            mViewerRunning = false;
            mFilePath.clear();
        }
    }
    
    // validate the graph and output the current status to the console
    if (pGui->addButton("Validate Graph"))
    {
        std::string currentLog;
        bool valid = mpGraphs[mCurrentGraphIndex]->isValid(currentLog);
        mCurrentLog += (valid ? "Graph is Valid\n" : "Graph is currently invalid.\n");
        mCurrentLog += currentLog;
    }

    if (pGui->addButton("Auto-Generate Edges"))
    {
        std::vector<uint32_t> executionOrder = mRenderGraphUIs[mCurrentGraphIndex].getExecutionOrder();
        mpGraphs[mCurrentGraphIndex]->autoGenEdges(executionOrder);
        mRenderGraphUIs[mCurrentGraphIndex].setToRebuild();
    }

    std::vector<std::string> graphOutputString{mGraphOutputEditString};
    if (pGui->addMultiTextBox("Add Output", {"GraphOutput"}, graphOutputString))
    {
        if (mCurrentGraphOutput != mGraphOutputEditString)
        {
            if (mCurrentGraphOutput.size())
            {
                mpGraphs[mCurrentGraphIndex]->unmarkOutput(mCurrentGraphOutput);
            }

            mCurrentGraphOutput = graphOutputString[0];
            mRenderGraphUIs[mCurrentGraphIndex].addOutput(mCurrentGraphOutput);
            mpGraphs[mCurrentGraphIndex]->setOutput(mCurrentGraphOutput, pSample->getCurrentFbo()->getColorTexture(0));
        }
    }
    mGraphOutputEditString = graphOutputString[0];

    if (!mViewerRunning && pGui->addButton("Open Graph Viewer"))
    {
        std::string log;
        bool openViewer = true;
        if (!mpGraphs[mCurrentGraphIndex]->isValid(log))
        {
            openViewer = msgBox("Graph is invalid :\n " + log + "\n Are you sure you want to attempt preview?", MsgBoxType::OkCancel) == MsgBoxButton::Ok;
        }

        // TODO -- get render graph viewer to open with live viewer properly with the editor
        if (openViewer)
        {
            // std::string renderGraphScript = ; // RenderGraphLoader::saveRenderGraphAsScriptBuffer(*mpGraphs[mCurrentGraphIndex]);
            // if (!renderGraphScript.size())
            // {
            //     logError("No graph data to display in editor.");
            // }
            // 
            // char* result = nullptr;
            // mFilePath = std::tmpnam(result);
            // std::ofstream updatesFileOut(mFilePath);
            // assert(updatesFileOut.is_open());
            // 
            // updatesFileOut.write(renderGraphScript.c_str(), renderGraphScript.size());
            // updatesFileOut.close();
            // 
            // // load application for the editor given it the name of the mapped file
            // std::string commandLine = std::string("-tempFile ") + mFilePath;
            // mViewerProcess = executeProcess(kViewerExecutableName, commandLine);
            // 
            // assert(mViewerProcess);
            // mViewerRunning = true;
        }
    }

    pGui->popWindow();

    pGui->pushWindow("output", screenWidth / 4, screenHeight / 4 - 20, screenWidth * 3 / 4, screenHeight * 3 / 4 + 20, true);
    if (mResetGuiWindows)
    {
        pGui->setCurrentWindowSize(screenWidth / 4, screenHeight / 4 - 20);
        pGui->setCurrentWindowPos(screenWidth * 3 / 4, screenHeight * 3 / 4 + 20);
    }
    renderLogWindow(pGui);
    pGui->popWindow();

    // pop up window for naming a new render graph
    if (mShowCreateGraphWindow)
    {
        pGui->pushWindow("CreateNewGraph", 256, 128, screenWidth / 2 - 128, screenHeight / 2 - 64);

        pGui->addTextBox("Graph Name", mNextGraphString);

        if (pGui->addButton("Create Graph") && mNextGraphString[0])
        {
            createRenderGraph(mNextGraphString, "");
            mpGraphs[mCurrentGraphIndex]->onResizeSwapChain(pSample->getCurrentFbo().get());
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

    mResetGuiWindows = false;
}

void RenderGraphEditor::renderLogWindow(Gui* pGui)
{
    // window for displaying log from render graph validation
    pGui->addText(mCurrentLog.c_str());
}

void RenderGraphEditor::serializeRenderGraph(const std::string& fileName)
{
    // TODO -- call exporter to save out graph
    // RenderGraphLoader::SaveRenderGraphAsScript(fileName, *mpGraphs[mCurrentGraphIndex]);
    
}

void RenderGraphEditor::deserializeRenderGraph(const std::string& fileName)
{
    mpGraphs[mCurrentGraphIndex] = RenderGraphImporter::import(fileName);
    if (mRenderGraphUIs.size() < mCurrentGraphIndex)
    {
        mRenderGraphUIs[mCurrentGraphIndex].setToRebuild();
    }
}

void RenderGraphEditor::createRenderGraph(const std::string& renderGraphName, const std::string& renderGraphFileName)
{
    std::string graphName = renderGraphName;
    size_t offset = graphName.find_first_of('.');
    RenderGraph::SharedPtr newGraph;

    if (offset != std::string::npos)
    {
        std::string tempGraphName = graphName.substr(0, offset);
        graphName = tempGraphName;
    }

    auto nameToIndexIt = mGraphNamesToIndex.find(graphName);
    if (renderGraphFileName.size())
    {
        newGraph = RenderGraphImporter::import(renderGraphFileName, graphName);

        // if graph already exists, just update that one
        if (nameToIndexIt != mGraphNamesToIndex.end())
        {
            // TODO display warning msgBox

            // mCurrentGraphIndex = nameToIndexIt->second;
            mCurrentGraphIndex = mpGraphs.size();
            mpGraphs[nameToIndexIt->second]->update(newGraph);
            mpGraphs.push_back(mpGraphs[nameToIndexIt->second]);
            // reset the render graph ui
            // mRenderGraphUIs[mCurrentGraphIndex] = (RenderGraphUI(pGraph, graphName));
            mRenderGraphUIs.push_back(RenderGraphUI(mpGraphs[nameToIndexIt->second], graphName));

            Gui::DropdownValue nextGraphID;
            mGraphNamesToIndex.insert(std::make_pair(graphName, static_cast<uint32_t>(mCurrentGraphIndex)));
            nextGraphID.value = static_cast<int32_t>(mOpenGraphNames.size());
            nextGraphID.label = graphName;
            mOpenGraphNames.push_back(nextGraphID);
            return;
        }
    }
    else
    {
        newGraph = RenderGraph::create();

        std::string tempGraphName = graphName;
        while (mGraphNamesToIndex.find(tempGraphName) != mGraphNamesToIndex.end())
        {
            tempGraphName.append("_");
        }
        graphName = tempGraphName;
    }
    
    mCurrentGraphIndex = mpGraphs.size();
    mpGraphs.push_back(newGraph);
    mRenderGraphUIs.push_back(RenderGraphUI(newGraph, graphName));

    Gui::DropdownValue nextGraphID;
    mGraphNamesToIndex.insert(std::make_pair(graphName, static_cast<uint32_t>(mCurrentGraphIndex) ));
    nextGraphID.value = static_cast<int32_t>(mOpenGraphNames.size());
    nextGraphID.label = graphName;
    mOpenGraphNames.push_back(nextGraphID);
}

void RenderGraphEditor::onLoad(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext)
{
    std::vector<ArgList::Arg> commandArgs = pSample->getArgList().getValues("tempFile");
    
    if (commandArgs.size())
    {
        mFilePath = commandArgs.front().asString();
    }

    pSample->toggleText(false);
    pSample->toggleGlobalUI(false);
    
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
    const glm::vec4 clearColor(0.25, 0.25, 0.25 , 1);
    pRenderContext->clearFbo(pTargetFbo.get(), clearColor, 1.0f, 0, FboAttachmentType::All);
    pSample->getRenderContext()->getGraphicsState()->setFbo(pTargetFbo);
    mRenderGraphUIs[mCurrentGraphIndex].updateGraph();
}

void RenderGraphEditor::onResizeSwapChain(SampleCallbacks* pSample, uint32_t width, uint32_t height)
{
    mpGraphs[mCurrentGraphIndex]->onResizeSwapChain(pSample->getCurrentFbo().get());
    mWindowSize = {width, height};
    mResetGuiWindows = true;
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
