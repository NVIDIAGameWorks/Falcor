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

void RenderGraphViewer::onGuiRender(SampleCallbacks* pSample, Gui* pGui)
{
    if (pGui->addButton("Load Scene")) loadScene();
    if (pGui->addButton("Add Graph")) addGraph(pSample->getCurrentFbo().get());

    // Display a list with all the graphs
    if (mGraphs.size() > 1)
    {
        Gui::DropdownList graphList;
        for (size_t i = 0; i < mGraphs.size(); i++) graphList.push_back({ (int32_t)i, mGraphs[i].name });
        pGui->addDropdown("Active Graph", graphList, mActiveGraph);
    }

    pGui->addSeparator();
    if (mGraphs.size()) mGraphs[mActiveGraph].pGraph->renderUI(pGui, mGraphs[mActiveGraph].name.c_str());
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
                    oldG.pGraph = newG.pGraph;
                }
            }
            if (!found) mGraphs.push_back({ newG.name, newG.pGraph });
            newG.pGraph->setScene(mpScene);
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
        Texture::SharedPtr pOutTex = std::dynamic_pointer_cast<Texture>(pGraph->getOutput("BlitPass.dst"));
        pRenderContext->blit(pOutTex->getSRV(), pTargetFbo->getRenderTargetView(0));
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
