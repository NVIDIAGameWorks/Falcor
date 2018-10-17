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
#include "RenderGraphEditor.h"
#include "Graphics/RenderGraph/RenderPassLibrary.h"
#include <fstream>
#include "RenderGraphEditor.h"

const char* kViewerExecutableName = "RenderGraphViewer";
const char* kGraphFileSwitch = "graphFile";
const char* kGraphNameSwitch = "graphname";
const char* kEditorSwitch = "editor";

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

void RenderGraphEditor::onLoad(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext)
{
    const auto& argList = pSample->getArgList();
    std::string filePath;
    if (argList.argExists(kGraphFileSwitch))
    {
        filePath = argList[kGraphFileSwitch].asString();
    }

    pSample->toggleText(false);
    pSample->toggleGlobalUI(false);

    if (filePath.size())
    {
        std::string graphName;
        if (argList.argExists(kGraphNameSwitch)) graphName = argList[kGraphNameSwitch].asString();

        mViewerRunning = true;
        loadGraphsFromFile(filePath, graphName);

        if (argList.argExists(kEditorSwitch)) mUpdateFilePath = filePath;
    }
    else
    {
        createNewGraph("DefaultRenderGraph");
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

            if (pGui->addMenuItem("Load File"))
            {
                std::string renderGraphFilePath;
                if (mViewerRunning)
                {
                    msgBox("Viewer is running. Please close the viewer before loading a graph file.", MsgBoxType::Ok);
                }
                else
                {
                    if (openFileDialog("", renderGraphFilePath))
                    {
                        loadGraphsFromFile(renderGraphFilePath);
                        mpGraphs[mCurrentGraphIndex]->onResize(pSample->getCurrentFbo().get());
                    }
                }
            }

            if (pGui->addMenuItem("Save To File"))
            {
                bool saveGraph = true;
                std::string log;

                if (!mpGraphs[mCurrentGraphIndex]->isValid(log))
                {
                    MsgBoxButton msgBoxButton = msgBox("Attempting to save invalid graph.\nGraph may not execute correctly when loaded\nAre you sure you want to save the graph?"
                        , MsgBoxType::OkCancel);
                    saveGraph = !(msgBoxButton == MsgBoxButton::Cancel);
                }

                if (saveGraph)
                {
                    std::string renderGraphFileName = mOpenGraphNames[mCurrentGraphIndex].label + ".py";
                    if (saveFileDialog("PY(.py)\0*.py", renderGraphFileName)) serializeRenderGraph(renderGraphFileName);
                }
            }

            if (pGui->addMenuItem("Load Pass Library"))
            {
                std::string passLib;
                if(openFileDialog("*.dll", passLib))
                {
                    RenderPassLibrary::instance().loadLibrary(passLib);
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

    pGui->beginColumns(5);
    auto renderPasses = RenderPassLibrary::instance().enumerateClasses();
    for (size_t i = 0 ; i < renderPasses.size() ; i++)
    {
        const auto& pass = renderPasses[i];
        pGui->addRect({ 148.0f, 64.0f }, pGui->pickUniqueColor(pass.className), false);
        pGui->addDummyItem((std::string("RenderPass##") + std::to_string(i)).c_str(), { 148.0f, 44.0f });
        pGui->dragDropSource(pass.className, "RenderPassType", pass.className);
        pGui->addText(pass.className);
        pGui->addTooltip(pass.desc, false);
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

    if (mUpdateFilePath.size())
    {
        mRenderGraphUIs[mCurrentGraphIndex].writeUpdateScriptToFile(mUpdateFilePath, pSample->getLastFrameTime());
    }

    if (mViewerRunning && mViewerProcess)
    {
        if (!isProcessRunning(mViewerProcess))
        {
            terminateProcess(mViewerProcess);
            mViewerProcess = 0;
            mViewerRunning = false;
            mUpdateFilePath.clear();
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
        std::vector<uint32_t> executionOrder = mRenderGraphUIs[mCurrentGraphIndex].getPassOrder();
        mpGraphs[mCurrentGraphIndex]->autoGenEdges(executionOrder);
        mRenderGraphUIs[mCurrentGraphIndex].setToRebuild();
    }

    if (pGui->addButton("Set Scene"))
    {
        // display warning when setting scene so that there is no confusion for overwriting default scene
        MsgBoxButton setSceneMsg = msgBox("Note: Setting scene in graph will overwrite default scene from viewer.");
        if (setSceneMsg == MsgBoxButton::Ok)
        {
            std::string filename;
            if (openFileDialog("*.fscene", filename))
            {
                filename = stripDataDirectories(filename);
                auto pDummyScene = Scene::create(filename);
                mpGraphs[mCurrentGraphIndex]->setScene(pDummyScene);
                mSceneSet = true;
            }
        }
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
        }
    }
    mGraphOutputEditString = graphOutputString[0];

    mRenderGraphUIs[mCurrentGraphIndex].setRecordUpdates(mViewerRunning);
    if (!mViewerRunning && pGui->addButton("Open Graph Viewer"))
    {
        std::string log;
        bool openViewer = true;
        if (!mpGraphs[mCurrentGraphIndex]->isValid(log))
        {
            openViewer = msgBox("Graph is invalid :\n " + log + "\n Are you sure you want to attempt preview?", MsgBoxType::OkCancel) == MsgBoxButton::Ok;
        }

        if (openViewer)
        {
            mUpdateFilePath = getTempFilename();
            RenderGraphExporter::save(mpGraphs[mCurrentGraphIndex], mRenderGraphUIs[mCurrentGraphIndex].getName(), mUpdateFilePath, {}, static_cast<RenderGraphExporter::ExportFlags>(mSceneSet));
            
            // load application for the editor given it the name of the mapped file
            std::string commandLineArgs = "-" + std::string(kEditorSwitch) + " -" + std::string(kGraphFileSwitch) + ' ' + mUpdateFilePath;
            commandLineArgs += " -" + std::string(kGraphNameSwitch) + ' ' + std::string(mOpenGraphNames[mCurrentGraphIndex].label);
            mViewerProcess = executeProcess(kViewerExecutableName, commandLineArgs);
            assert(mViewerProcess);
            mViewerRunning = true;
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
            createNewGraph(mNextGraphString);
            mpGraphs[mCurrentGraphIndex]->onResize(pSample->getCurrentFbo().get());
            mNextGraphString.clear();
            mNextGraphString.resize(255, 0);
            mShowCreateGraphWindow = false;
        }

        if (pGui->addButton("Cancel", true))
        {
            mNextGraphString.clear();
            mNextGraphString.resize(255, 0);
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
    RenderGraphExporter::save(mpGraphs[mCurrentGraphIndex], mRenderGraphUIs[mCurrentGraphIndex].getName(), fileName, {});
}

void RenderGraphEditor::deserializeRenderGraph(const std::string& fileName)
{
    mpGraphs[mCurrentGraphIndex] = RenderGraphImporter::import(fileName);
    if (mRenderGraphUIs.size() < mCurrentGraphIndex)
    {
        mRenderGraphUIs[mCurrentGraphIndex].setToRebuild();
    }
}

void RenderGraphEditor::loadGraphsFromFile(const std::string& fileName, const std::string& graphName)
{
    assert(fileName.size());

    // behavior is load each graph defined within the file as a separate editor ui
    std::vector <RenderGraphImporter::GraphData> newGraphs;
    if (graphName.size())
    {
        auto pGraph = RenderGraphImporter::import(graphName, fileName);
        if (pGraph) newGraphs.push_back({ graphName, pGraph});
    }
    else
    {
        newGraphs = RenderGraphImporter::importAllGraphs(fileName);
    }

    for (const auto& graphInfo : newGraphs)
    {
        const std::string& name = graphInfo.name;
        const RenderGraph::SharedPtr& newGraph = graphInfo.pGraph;

        auto nameToIndexIt = mGraphNamesToIndex.find(name);
        if (nameToIndexIt != mGraphNamesToIndex.end())
        {
             MsgBoxButton button = msgBox("Warning! Graph is already open. Update graph from file?", MsgBoxType::OkCancel);
            if (button == MsgBoxButton::Ok)
            {
                mCurrentGraphIndex = nameToIndexIt->second;
                mpGraphs[mCurrentGraphIndex]->update(newGraph);
                mRenderGraphUIs[mCurrentGraphIndex].reset();
                continue;
            }
        }
        else
        {
            mCurrentGraphIndex = mpGraphs.size();
            mpGraphs.push_back(newGraph);
            mRenderGraphUIs.push_back(RenderGraphUI(mpGraphs[mCurrentGraphIndex], name));

            Gui::DropdownValue nextGraphID;
            mGraphNamesToIndex.insert(std::make_pair(name, static_cast<uint32_t>(mCurrentGraphIndex)));
            nextGraphID.value = static_cast<int32_t>(mOpenGraphNames.size());
            nextGraphID.label = name;
            mOpenGraphNames.push_back(nextGraphID);
        }
    }
}

void RenderGraphEditor::createNewGraph(const std::string& renderGraphName)
{
    std::string graphName = renderGraphName;
    auto nameToIndexIt = mGraphNamesToIndex.find(graphName);
    RenderGraph::SharedPtr newGraph = RenderGraph::create();

    std::string tempGraphName = graphName;
    while (mGraphNamesToIndex.find(tempGraphName) != mGraphNamesToIndex.end())
    {
        tempGraphName.append("_");
    }
    // Matt TODO can we put the GUI dropdown code in a shared function shared with 'loadFromFile'?
    graphName = tempGraphName;
    mCurrentGraphIndex = mpGraphs.size();
    mpGraphs.push_back(newGraph);
    mRenderGraphUIs.push_back(RenderGraphUI(newGraph, graphName));

    Gui::DropdownValue nextGraphID;
    mGraphNamesToIndex.insert(std::make_pair(graphName, static_cast<uint32_t>(mCurrentGraphIndex) ));
    nextGraphID.value = static_cast<int32_t>(mOpenGraphNames.size());
    nextGraphID.label = graphName;
    mOpenGraphNames.push_back(nextGraphID);
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
    mpGraphs[mCurrentGraphIndex]->onResize(pSample->getCurrentFbo().get());
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
#ifndef _WIN32
    config.argv = argv;
    config.argc = (uint32_t)argc;
#endif
    config.windowDesc.title = "Render Graph Editor";
    config.windowDesc.resizableWindow = true;
    Sample::run(config, pEditor);
    return 0;
}
