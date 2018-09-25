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

const std::string gkDefaultScene = "Arcade/Arcade.fscene";
const char* kEditorExecutableName = "RenderGraphEditor";

void RenderGraphViewer::onShutdown(SampleCallbacks* pSample)
{
}

void RenderGraphViewer::onLoad(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext)
{

}

bool isInVector(const std::vector<std::string>& strVec, const std::string& str)
{
    return std::find(strVec.begin(), strVec.end(), str) != strVec.end();
}

Gui::DropdownList createDropdownFromVec(const std::vector<std::string>& strVec, uint32_t& activeIndex, const std::string& currentLabel)
{
    Gui::DropdownList dropdown;

    for (size_t i = 0; i < strVec.size(); i++)
    {
        dropdown.push_back({ (int32_t)i, strVec[i] });
        if (strVec[i] == currentLabel) activeIndex = (uint32_t)i;
    }

    return dropdown;
}

void RenderGraphViewer::graphOutputsGui(Gui* pGui)
{
    RenderGraph::SharedPtr pGraph = mGraphs[mActiveGraph].pGraph;
    pGui->addCheckBox("Show All Outputs", mGraphs[mActiveGraph].showAllOutputs);

    auto& strVec = mGraphs[mActiveGraph].showAllOutputs ? pGraph->getAvailableOutputs() : mGraphs[mActiveGraph].originalOutputs;
    uint32_t activeOut = -1;
    Gui::DropdownList graphOuts = createDropdownFromVec(strVec, activeOut, mGraphs[mActiveGraph].mainOutput);

    // This can happen when `showAllOutputs` changes to false, and the chosen output is not an original output. We will force an ouptut change
    bool forceOutputChange = activeOut == -1;
    if (forceOutputChange) activeOut = 0;

    if (graphOuts.size())
    {
        if (pGui->addDropdown("Main Output", graphOuts, activeOut) || forceOutputChange)
        {
            // If the previous output wasn't an original output, unmark it
            if (isInVector(mGraphs[mActiveGraph].originalOutputs, mGraphs[mActiveGraph].mainOutput) == false) pGraph->unmarkOutput(mGraphs[mActiveGraph].mainOutput);
            // If the new output isn't a graph output, mark it
            if (isInVector(mGraphs[mActiveGraph].originalOutputs, graphOuts[activeOut].label) == false) pGraph->markOutput(graphOuts[activeOut].label);
            mGraphs[mActiveGraph].mainOutput = graphOuts[activeOut].label;
        }
    }
}

void RenderGraphViewer::onGuiRender(SampleCallbacks* pSample, Gui* pGui)
{
    if (pGui->addButton("Load Scene")) loadScene();
    if (pGui->addButton("Add Graph")) addGraph(pSample->getCurrentFbo().get());
    pGui->addSeparator();

    // Display a list with all the graphs
    if (mGraphs.size())
    {
        Gui::DropdownList graphList;
        for (size_t i = 0; i < mGraphs.size(); i++) graphList.push_back({ (int32_t)i, mGraphs[i].name });
        pGui->addDropdown("Active Graph", graphList, mActiveGraph);

        if (pGui->addButton("Remove Active Graph")) removeActiveGraph();

        // Active graph output
        pGui->addSeparator();
        graphOutputsGui(pGui);

        // Graph UI
        pGui->addSeparator();
        mGraphs[mActiveGraph].pGraph->renderUI(pGui, mGraphs[mActiveGraph].name.c_str());
    }

}

void RenderGraphViewer::removeActiveGraph()
{
    assert(mGraphs.size());
    mGraphs.erase(mGraphs.begin() + mActiveGraph);
    mActiveGraph = 0;
}

void RenderGraphViewer::initGraph(const RenderGraph::SharedPtr& pGraph, const std::string& name, GraphData& data)
{
    data.name = name;
    data.pGraph = pGraph;
    data.pGraph->setScene(mpScene);
    if (data.pGraph->getOutputCount() != 0) data.mainOutput = data.pGraph->getOutputName(0);

    // Store the original outputs
    for (size_t i = 0; i < data.pGraph->getOutputCount(); i++) data.originalOutputs.push_back(data.pGraph->getOutputName(i));
}

void RenderGraphViewer::addGraph(const Fbo* pTargetFbo)
{
    std::string filename;
    if (openFileDialog("py", filename))
    {
        auto graphs = RenderGraphImporter::importAllGraphs(filename, pTargetFbo);
        if(graphs.size() && !mpScene) loadSceneFromFile(gkDefaultScene);

        for(auto& newG : graphs)
        {
            bool found = false;
            // Check if the graph already exists. If it is, replace it
            for (auto& oldG : mGraphs)
            {
                if (oldG.name == newG.name)
                {
                    found = true;
                    logWarning("Graph `" + newG.name + "` already exists. Replacing it");
                    initGraph(newG.pGraph, newG.name, oldG);
                 
                }
            }

            if(!found)
            {
                mGraphs.push_back({});
                initGraph(newG.pGraph, newG.name, mGraphs.back());
            }
        }
    }
}

void RenderGraphViewer::loadScene()
{
    std::string filename;
    if (openFileDialog(Scene::kFileFormatString, filename))
    {
        loadSceneFromFile(filename);
    }
}

void RenderGraphViewer::loadSceneFromFile(const std::string& filename)
{
    mpScene = Scene::loadFromFile(filename);
    for(auto& g : mGraphs) g.pGraph->setScene(mpScene);
}

void RenderGraphViewer::onFrameRender(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext, const Fbo::SharedPtr& pTargetFbo)
{
    const glm::vec4 clearColor(0.38f, 0.52f, 0.10f, 1);
    pRenderContext->clearFbo(pTargetFbo.get(), clearColor, 1.0f, 0, FboAttachmentType::All);

    if (mGraphs.size())
    {
        auto& pGraph = mGraphs[mActiveGraph].pGraph;
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
    for(auto& g : mGraphs) g.pGraph->onResize(pSample->getCurrentFbo().get());
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
    Sample::run(config, pRenderer);
    return 0;
}
