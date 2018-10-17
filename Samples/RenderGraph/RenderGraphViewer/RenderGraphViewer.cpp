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

size_t RenderGraphViewer::DebugWindow::index = 0;

const std::string gkDefaultScene = "alphatest/alpha_test.fscene";
const char* kEditorExecutableName = "RenderGraphEditor";

const char* kSceneSwitch = "scene";
const char* kImageSwitch = "image";
const char* kGraphFileSwitch = "graphFile";
const char* kGraphNameSwitch = "graphname";
const char* kEditorSwitch = "editor";

void RenderGraphViewer::onShutdown(SampleCallbacks* pSample)
{
    resetEditor();
    mGraphs.clear();
}

void RenderGraphViewer::onLoad(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext)
{
    // if editor opened from running render graph, get the name of the file to read
    mDefaultSceneName = gkDefaultScene;
    parseArguments(pSample, pSample->getArgList());

    const auto& pFbo = pSample->getCurrentFbo();
    uint32_t w = pFbo->getWidth();
    uint32_t h = pFbo->getHeight();
    w = (uint32_t)(w * 0.25f);
    h = (uint32_t)(h * 0.6f);
    pSample->setDefaultGuiSize(w, h);
}

void RenderGraphViewer::parseArguments(SampleCallbacks* pSample, const ArgList& argList)
{
    if (argList.argExists(kSceneSwitch))
    {
        std::string filename = argList[kSceneSwitch].asString();
        if (filename.size()) { mDefaultSceneName = filename; }
        else msgBox("No path to default scene provided.");
    }
    if (argList.argExists(kImageSwitch))
    {
        std::string filename = argList[kImageSwitch].asString();
        if (filename.size()) { mDefaultImageName = filename; }
        else msgBox("No path to default image provided.");
    }
    if (argList.argExists(kGraphFileSwitch))
    {
        std::string filename = argList[kGraphFileSwitch].asString();
        if (filename.size())
        {
            if (argList.argExists(kGraphNameSwitch))
            {
                std::string graphName = argList[kGraphNameSwitch].asString();
                if (graphName.size())
                {
                    auto pGraph = RenderGraphImporter::import(graphName, filename);
                    mGraphs.push_back({});
                    initGraph(pGraph, graphName, filename, pSample, mGraphs.back());
                }
            }
            else { addGraphsFromFile(filename, pSample); }
        }
        else { msgBox("No file path provided for input graph file"); }

        if (argList.argExists(kEditorSwitch))
        {
            mEditorTempFile = filename;
            monitorFileUpdates(filename, std::bind(&RenderGraphViewer::editorFileChangeCB, this));
            mEditorProcess = kInvalidProcessId;
        }
    }
}

bool isInVector(const std::vector<std::string>& strVec, const std::string& str)
{
    return std::find(strVec.begin(), strVec.end(), str) != strVec.end();
}

Gui::DropdownList createDropdownFromVec(const std::vector<std::string>& strVec, const std::string& currentLabel)
{
    Gui::DropdownList dropdown;
    for (size_t i = 0; i < strVec.size(); i++) dropdown.push_back({ (int32_t)i, strVec[i] });    
    return dropdown;
}

void RenderGraphViewer::addDebugWindow()
{
    DebugWindow window;
    window.windowName = "Debug Window " + std::to_string(DebugWindow::index++);
    window.currentOutput = mGraphs[mActiveGraph].mainOutput;
    mGraphs[mActiveGraph].debugWindows.push_back(window);
}

void RenderGraphViewer::renderOutputUI(Gui* pGui, const Gui::DropdownList& dropdown, std::string& selectedOutput)
{
    uint32_t activeOut = -1;
    for(size_t i = 0 ; i < dropdown.size() ; i++) 
    {
        if (dropdown[i].label == selectedOutput)
        {
            activeOut = (uint32_t)i;
            break;
        }
    }

    // This can happen when `showAllOutputs` changes to false, and the chosen output is not an original output. We will force an output change
    bool forceOutputChange = activeOut == -1;
    if (forceOutputChange) activeOut = 0;

    if (pGui->addDropdown("Output", dropdown, activeOut) || forceOutputChange)
    {
        // If the previous output wasn't an original output, unmark it
        if (isInVector(mGraphs[mActiveGraph].originalOutputs, selectedOutput) == false) mGraphs[mActiveGraph].pGraph->unmarkOutput(selectedOutput);
        // If the new output isn't a graph output, mark it
        if (isInVector(mGraphs[mActiveGraph].originalOutputs, dropdown[activeOut].label) == false) mGraphs[mActiveGraph].pGraph->markOutput(dropdown[activeOut].label);
        selectedOutput = dropdown[activeOut].label;
    }
}

bool RenderGraphViewer::renderDebugWindow(Gui* pGui, const Gui::DropdownList& dropdown, DebugWindow& data, const uvec2& winSize)
{
    // Get the current output, in case `renderOutputUI()` unmarks it
    Texture::SharedPtr pTex = std::dynamic_pointer_cast<Texture>(mGraphs[mActiveGraph].pGraph->getOutput(data.currentOutput));
    std::string label = data.currentOutput + "##" + mGraphs[mActiveGraph].name;

    uvec2 debugSize = (uvec2)(vec2(winSize) * vec2(0.4f, 0.55f));
    uvec2 debugPos = winSize - debugSize;
    debugPos -= 10;

    // Display the dropdown
    pGui->pushWindow(data.windowName.c_str(), debugSize.x, debugSize.y, debugPos.x, debugPos.y);
    bool close = pGui->addButton("Close");
    if (pGui->addButton("Save To File", true)) Bitmap::saveImageDialog(pTex);
    renderOutputUI(pGui, dropdown, data.currentOutput);
    pGui->addSeparator();

    // Display the image
    pGui->addImage(label.c_str(), pTex);

    pGui->popWindow();

    return close;
}

void RenderGraphViewer::graphOutputsGui(Gui* pGui, SampleCallbacks* pSample)
{
    RenderGraph::SharedPtr pGraph = mGraphs[mActiveGraph].pGraph;
    pGui->addCheckBox("Show All Outputs", mGraphs[mActiveGraph].showAllOutputs);
    pGui->addTooltip("If there's a debug window open, you won't be able to uncheck this");
    if (mGraphs[mActiveGraph].debugWindows.size()) mGraphs[mActiveGraph].showAllOutputs = true;

    auto strVec = mGraphs[mActiveGraph].showAllOutputs ? pGraph->getAvailableOutputs() : mGraphs[mActiveGraph].originalOutputs;
    Gui::DropdownList graphOuts = createDropdownFromVec(strVec, mGraphs[mActiveGraph].mainOutput);

    if (graphOuts.size())
    {
        uvec2 dims(pSample->getCurrentFbo()->getWidth(), pSample->getCurrentFbo()->getHeight());

        renderOutputUI(pGui, graphOuts, mGraphs[mActiveGraph].mainOutput);
        for (size_t i = 0; i < mGraphs[mActiveGraph].debugWindows.size();)
        {
            if (renderDebugWindow(pGui, graphOuts, mGraphs[mActiveGraph].debugWindows[i], dims))
            {
                mGraphs[mActiveGraph].debugWindows.erase(mGraphs[mActiveGraph].debugWindows.begin() + i);
            }
            else i++;
        }

        // Render the debug windows *before* adding/removing debug windows
        if (pGui->addButton("Show In Debug Window")) addDebugWindow();
        if(mGraphs[mActiveGraph].debugWindows.size())
        {
            if (pGui->addButton("Close all debug windows")) mGraphs[mActiveGraph].debugWindows.clear();
        }
    }
}

void RenderGraphViewer::onGuiRender(SampleCallbacks* pSample, Gui* pGui)
{
    if (pGui->addButton("Add Graphs")) addGraphDialog(pSample);
    if (pGui->addButton("Load Scene", true)) loadScene(pSample);
    pGui->addSeparator();

    // Display a list with all the graphs
    if (mGraphs.size())
    {
        if(pGui->addButton("Reload libraries"))
        {
            RenderPassLibrary::instance().reloadLibraries();
        }

        Gui::DropdownList graphList;
        for (size_t i = 0; i < mGraphs.size(); i++) graphList.push_back({ (int32_t)i, mGraphs[i].pGraph->getName() });
        if(mEditorProcess == 0) 
        {
            pGui->addDropdown("Active Graph", graphList, mActiveGraph);
            if (pGui->addButton("Edit")) openEditor();
            if (pGui->addButton("Remove", true)) removeActiveGraph();
            pGui->addSeparator();
        }

        // Active graph output
        graphOutputsGui(pGui, pSample);

        // Graph UI
        pGui->addSeparator();
        mGraphs[mActiveGraph].pGraph->renderUI(pGui, mGraphs[mActiveGraph].pGraph->getName().c_str());
    }

}

void RenderGraphViewer::editorFileChangeCB()
{
    mEditorScript = readFile(mEditorTempFile);
}

void RenderGraphViewer::openEditor()
{
    bool unmarkOut = (isInVector(mGraphs[mActiveGraph].originalOutputs, mGraphs[mActiveGraph].mainOutput) == false);
    // If the current graph output is not an original output, unmark it
    if (unmarkOut) mGraphs[mActiveGraph].pGraph->unmarkOutput(mGraphs[mActiveGraph].mainOutput);

    mEditorTempFile = getTempFilename();

    // Save the graph
    RenderGraphExporter::save(mGraphs[mActiveGraph].pGraph, mGraphs[mActiveGraph].name, mEditorTempFile);

    // Register an update callback
    monitorFileUpdates(mEditorTempFile, std::bind(&RenderGraphViewer::editorFileChangeCB, this));

    // Run the process
    std::string commandLineArgs = '-' + std::string(kEditorSwitch) + " -" + std::string(kGraphFileSwitch);
    commandLineArgs += ' ' + mEditorTempFile + " -" + std::string(kGraphNameSwitch) + ' ' + mGraphs[mActiveGraph].name;
    mEditorProcess = executeProcess(kEditorExecutableName, commandLineArgs);

    // Mark the output if it's required
    if (unmarkOut) mGraphs[mActiveGraph].pGraph->markOutput(mGraphs[mActiveGraph].mainOutput);
}

void RenderGraphViewer::resetEditor()
{
    if(mEditorProcess)
    {
        closeSharedFile(mEditorTempFile);
        std::remove(mEditorTempFile.c_str());
        if(mEditorProcess != kInvalidProcessId)
        {
            terminateProcess(mEditorProcess);
            mEditorProcess = 0;
        }
    }
}

void RenderGraphViewer::removeActiveGraph()
{
    assert(mGraphs.size());
    mGraphs.erase(mGraphs.begin() + mActiveGraph);
    mActiveGraph = 0;
}

std::vector<std::string> RenderGraphViewer::getGraphOutputs(const RenderGraph::SharedPtr& pGraph)
{
    std::vector<std::string> outputs;
    for (size_t i = 0; i < pGraph->getOutputCount(); i++) outputs.push_back(pGraph->getOutputName(i));
    return outputs;
}

void RenderGraphViewer::initGraph(const RenderGraph::SharedPtr& pGraph, const std::string& name, const std::string& filename, SampleCallbacks* pCallbacks, GraphData& data)
{
    if (pGraph->getName().empty()) pGraph->setName(name);

    // Set input image if it exists
    if(mDefaultImageName.size())    (*pGraph->getPassesDictionary())[kImageSwitch] = mDefaultImageName;

    data.name = name;
    data.filename = filename;
    data.fileModifiedTime = getFileModifiedTime(filename);
    data.pGraph = pGraph;
    if(data.pGraph->getScene() == nullptr)
    {
        if (!mpDefaultScene) loadSceneFromFile(mDefaultSceneName, pCallbacks);
        data.pGraph->setScene(mpDefaultScene);
    }
    if (data.pGraph->getOutputCount() != 0) data.mainOutput = data.pGraph->getOutputName(0);

    // Store the original outputs
    data.originalOutputs = getGraphOutputs(pGraph);
}

void RenderGraphViewer::addGraphDialog(SampleCallbacks* pCallbacks)
{
    std::string filename;
    if (openFileDialog("*.py", filename)) addGraphsFromFile(filename, pCallbacks);
}

void RenderGraphViewer::addGraphsFromFile(const std::string& filename, SampleCallbacks* pCallbacks)
{
    const auto& pTargetFbo = pCallbacks->getCurrentFbo().get();
    auto graphs = RenderGraphImporter::importAllGraphs(filename, pTargetFbo);
    
    for (auto& newG : graphs)
    {
        bool found = false;
        // Check if the graph already exists. If it is, replace it
        for (auto& oldG : mGraphs)
        {
            if (oldG.name == newG.name)
            {
                found = true;
                logWarning("Graph `" + newG.name + "` already exists. Replacing it");
                initGraph(newG.pGraph, newG.name, filename, pCallbacks, oldG);
            }
        }

        if (!found)
        {
            mGraphs.push_back({});
            initGraph(newG.pGraph, newG.name, filename, pCallbacks, mGraphs.back());
        }
    }
}

void RenderGraphViewer::loadScene(SampleCallbacks* pCallbacks)
{
    std::string filename;
    if (openFileDialog(Scene::kFileFormatString, filename))
    {
        loadSceneFromFile(filename, pCallbacks);
    }
}

void RenderGraphViewer::loadSceneFromFile(const std::string& filename, SampleCallbacks* pCallbacks)
{
    mpDefaultScene = Scene::loadFromFile(filename);
    const auto& pFbo = pCallbacks->getCurrentFbo();
    float ratio = float(pFbo->getWidth()) / float(pFbo->getHeight());
    mpDefaultScene->setCamerasAspectRatio(ratio);
    for(auto& g : mGraphs) g.pGraph->setScene(mpDefaultScene);
}

void RenderGraphViewer::applyEditorChanges()
{
    if (!mEditorProcess) return;
    // If the editor was closed, reset the handles
    if ((mEditorProcess != kInvalidProcessId) && isProcessRunning(mEditorProcess) == false) resetEditor();

    if (mEditorScript.empty()) return;

    // Unmark the current output if it wasn't an original one
    bool unmarkOut = (isInVector(mGraphs[mActiveGraph].originalOutputs, mGraphs[mActiveGraph].mainOutput) == false);
    if (unmarkOut) mGraphs[mActiveGraph].pGraph->unmarkOutput(mGraphs[mActiveGraph].mainOutput);

    // Run the scripting
    auto pScripting = RenderGraphScripting::create();
    pScripting->addGraph(mGraphs[mActiveGraph].name, mGraphs[mActiveGraph].pGraph);
    pScripting->runScript(mEditorScript);

    // Update the original output list
    mGraphs[mActiveGraph].originalOutputs = getGraphOutputs(mGraphs[mActiveGraph].pGraph);

    // Mark the current output if it's required
    if (unmarkOut) mGraphs[mActiveGraph].pGraph->markOutput(mGraphs[mActiveGraph].mainOutput);

    mEditorScript.clear();
}

void RenderGraphViewer::onFrameRender(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext, const Fbo::SharedPtr& pTargetFbo)
{
    applyEditorChanges();

    // Render
    const glm::vec4 clearColor(0.38f, 0.52f, 0.10f, 1);
    pRenderContext->clearFbo(pTargetFbo.get(), clearColor, 1.0f, 0, FboAttachmentType::All);

    if (mGraphs.size())
    {
        auto& pGraph = mGraphs[mActiveGraph].pGraph;
        if (pGraph->getScene()) pGraph->getScene()->update(pSample->getCurrentTime(), &mCamController);

        pGraph->execute(pRenderContext.get());
        if(mGraphs[mActiveGraph].mainOutput.size())
        {
            Texture::SharedPtr pOutTex = std::dynamic_pointer_cast<Texture>(pGraph->getOutput(mGraphs[mActiveGraph].mainOutput));
            pRenderContext->blit(pOutTex->getSRV(), pTargetFbo->getRenderTargetView(0));
        }
    }
}

bool RenderGraphViewer::onMouseEvent(SampleCallbacks* pSample, const MouseEvent& mouseEvent)
{
    if (mGraphs.size()) mGraphs[mActiveGraph].pGraph->onMouseEvent(mouseEvent);
    return mCamController.onMouseEvent(mouseEvent);
}

bool RenderGraphViewer::onKeyEvent(SampleCallbacks* pSample, const KeyboardEvent& keyEvent)
{
    if (mGraphs.size()) mGraphs[mActiveGraph].pGraph->onKeyEvent(keyEvent);
    return mCamController.onKeyEvent(keyEvent);
}

void RenderGraphViewer::onResizeSwapChain(SampleCallbacks* pSample, uint32_t width, uint32_t height)
{
    for(auto& g : mGraphs) 
    {
        g.pGraph->onResize(pSample->getCurrentFbo().get());
        g.pGraph->getScene()->setCamerasAspectRatio((float)width / (float)height);
    }
    if (mpDefaultScene)  mpDefaultScene->setCamerasAspectRatio((float)width / (float)height);
}

void RenderGraphViewer::onInitializeTesting(SampleCallbacks* pSample)
{
    auto args = pSample->getArgList();
    std::vector<ArgList::Arg> scene = args.getValues("loadscene");
    if (!scene.empty())
    {
        loadSceneFromFile(scene[0].asString(), pSample);
    }

    std::vector<ArgList::Arg> cameraPos = args.getValues("camerapos");
    if (!cameraPos.empty())
    {
        mpDefaultScene->getActiveCamera()->setPosition(glm::vec3(cameraPos[0].asFloat(), cameraPos[1].asFloat(), cameraPos[2].asFloat()));
    }

    std::vector<ArgList::Arg> cameraTarget = args.getValues("cameratarget");
    if (!cameraTarget.empty())
    {
        mpDefaultScene->getActiveCamera()->setTarget(glm::vec3(cameraTarget[0].asFloat(), cameraTarget[1].asFloat(), cameraTarget[2].asFloat()));
    }
}

void RenderGraphViewer::onBeginTestFrame(SampleTest* pSampleTest)
{
    //  Already existing. Is this a problem?    
    auto nextTriggerType = pSampleTest->getNextTriggerType();
    if (nextTriggerType == SampleTest::TriggerType::None)
    {
        SampleTest::TaskType taskType = (nextTriggerType == SampleTest::TriggerType::Frame) ? pSampleTest->getNextFrameTaskType() : pSampleTest->getNextTimeTaskType();
        RenderPass::SharedPtr pShadowPass = mGraphs[mActiveGraph].pGraph->getPass("ShadowPass");
        if (pShadowPass != nullptr)
        {
            // Matt TODO this should be part of CascadedShadowMaps::Dictionary and store in the graph file
            std::static_pointer_cast<CascadedShadowMaps>(pShadowPass)->setSdsmReadbackLatency(taskType == SampleTest::TaskType::ScreenCaptureTask ? 0 : 1);
        }
    }
}

void RenderGraphViewer::onDataReload(SampleCallbacks* pSample)
{
    if (mEditorProcess)
    {
        logWarning("Warning: Updating graph while editor is open. Graphs will not be reloaded from file.");
        return;
    }

    // Reload all DLLs
    RenderPassLibrary::instance().reloadLibraries();

    // Reload all graphs, while maintaining state
    for (const auto& g : mGraphs)
    {
        if(g.fileModifiedTime != getFileModifiedTime(g.filename))
        {
            RenderGraph::SharedPtr pGraph = g.pGraph;
            RenderGraph::SharedPtr pNewGraph;
            pNewGraph = RenderGraphImporter::import(g.name, g.filename);
            pGraph->update(pNewGraph);
        }
    }
}

#ifdef _WIN32
int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd)
#else
int main(int argc, char** argv)
#endif
{
    RenderGraphViewer::UniquePtr pRenderer = std::make_unique<RenderGraphViewer>();
    SampleConfig config;
    config.windowDesc.title = "Render Graph Viewer";
    config.windowDesc.resizableWindow = true;
#ifdef FALCOR_DXR
    RtSample::run(config, pRenderer);
#else
#ifndef _WIN32
    config.argc = argc;
    config.argv = argv;
#endif
    Sample::run(config, pRenderer);
#endif
    return 0;
}
