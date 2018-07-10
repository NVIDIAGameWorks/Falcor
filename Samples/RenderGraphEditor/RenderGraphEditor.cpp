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
#include "RenderGraphEditor.h"

#include <fstream>

#include "Externals/RapidJson/include/rapidjson/rapidjson.h"
#include "Externals/RapidJson/include/rapidjson/document.h"
#include "Externals/RapidJson/include/rapidjson/istreamwrapper.h"
#include "Externals/RapidJson/include/rapidjson/ostreamwrapper.h"
#include "Externals/RapidJson/include/rapidjson/prettywriter.h"

#include "Externals/dear_imgui/imgui.h"
#include "Externals/dear_imgui/imgui_internal.h"

const std::string gkDefaultScene = "Arcade/Arcade.fscene";

std::unordered_map<std::string, std::function<RenderPass::SharedPtr()>> RenderGraphEditor::sBaseRenderCreateFuncs;
std::unordered_map<std::string, std::function<RenderPass::PassData(RenderPass::SharedPtr)>> RenderGraphEditor::sGetRenderPassData;

RenderGraphEditor::RenderGraphEditor()
    : mCurrentGraphIndex(0), mCreatingRenderGraph(false), mPreviewing(false)
{
    Gui::DropdownValue dropdownValue;

#define register_render_pass(renderPassType) \
    sBaseRenderCreateFuncs.insert(std::make_pair(#renderPassType, std::function<RenderPass::SharedPtr()> ( \
        []() { return renderPassType::create(); }) )\
    ); \
    sGetRenderPassData.insert(std::make_pair(#renderPassType, std::function<RenderPass::PassData(RenderPass::SharedPtr renderPass)> ( \
        [](RenderPass::SharedPtr renderPass) { auto toReturn = dynamic_cast<renderPassType*>( renderPass.get() ); assert(toReturn); return toReturn->getRenderPassData(); }) )\
    ); \
    dropdownValue.label = #renderPassType; dropdownValue.value = static_cast<int32_t>(mRenderPassTypes.size()); \
    mRenderPassTypes.push_back(dropdownValue)


    register_render_pass(SceneRenderPass);
    register_render_pass(BlitPass);
    register_render_pass(DepthPass);
    register_render_pass(ShadowPass);
    register_render_pass(NodeGraphGuiPass);
    register_render_pass(GraphEditorGuiPass);

#undef register_render_pass

#define register_resource_type() 

    mNextGraphString.resize(255, 0);
    mNodeString.resize(255, 0);
    mCurrentGraphOutput = "BlitPass.dst";
    mGraphOutputEditString = mCurrentGraphOutput;
    mGraphOutputEditString.resize(255, 0);
}

void RenderGraphEditor::onGuiRender(SampleCallbacks* pSample, Gui* pGui)
{
    mWindowSize = pGui->getCurrentWindowSize();
    mWindowPos = pGui->getCurrentWindowPosition();

    // please remove this
    static bool firstFrame = true;

    // we should move everything below here into the render graph ui struct

    // sub window for listing available window passes
    pGui->pushWindow("Render Passes");

    // Title menu bar for adding render pass dll s
    if (ImGui::BeginMenuBar())
    {
        if (ImGui::BeginMenu("File"))
        {

            ImGui::EndMenu();
        }

        ImGui::EndMenuBar();
    }

    // for each dll that was found. (or manually imported????)
    for (auto& availableRenderPasses : sBaseRenderCreateFuncs)
    {
        // ImGui::BeginChildFrame();
        
        // REMOVE THIS
        if (availableRenderPasses.first[0] == 'N' || availableRenderPasses.first[0] == 'G')
        {
            continue;
        }

        ImVec2 nextDragRegionPos{ ImGui::GetCursorScreenPos().x + 320.0f, ImGui::GetCursorScreenPos().y + 180.0f };
        ImGui::GetCurrentWindow()->DrawList->AddRect(ImGui::GetCursorScreenPos(), nextDragRegionPos, 0xFFFFFFFF);
        ImGui::Dummy({ 320.0f , 180.0f });

        static bool payLoadSet = false;

        if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID))
        {
            if (!payLoadSet)
            {
                payLoadSet = true;
                std::string addCommand = std::string("AddRenderPass ") + availableRenderPasses.first + " " + availableRenderPasses.first;
                ImGui::SetDragDropPayload("RenderPassScript", addCommand.c_str(), addCommand.size(), ImGuiCond_Once);
            }
            
            ImGui::EndDragDropSource();
        }
        else
        {
            payLoadSet = false;
        }

        ImGui::SameLine();
        pGui->addText(availableRenderPasses.first.c_str());

        ImGui::SetCursorScreenPos(nextDragRegionPos);
        ImGui::SameLine();
    }

    pGui->popWindow();



    mRenderGraphUIs[mCurrentGraphIndex].renderUI(pGui);

    if (!firstFrame)
    {
        auto& nodeEditorFBO = static_cast<const NodeGraphGuiPass*>(mpEditorGraph->getRenderPass("NodeGraphPass").get())->getFbo();
        pGui->addImageForContext("RenderGraphContext", nodeEditorFBO->getColorTexture(0)); 

    }
    
    firstFrame = false;

    uint32_t screenHeight = pSample->getWindow()->getClientAreaHeight();
    uint32_t screenWidth = pSample->getWindow()->getClientAreaWidth();
    
    pGui->pushWindow("Graph Editor Settings", screenWidth / 8, screenHeight - 1, screenWidth - screenWidth / 8, 0, false);

    // DO you want to keep these ?? -- possible custom contexts outside of what is rendered?  would that even be useful
    pGui->setContextSize(mWindowSize);
    
    pGui->addTextBox("Graph Name", mNextGraphString);
    if (mCreatingRenderGraph)
    {
        pGui->addText("Creating Graph ...");
    }
    else
    {
        if (mNextGraphString[0] && pGui->addButton("Create New Graph"))
        {
            createRenderGraph(mNextGraphString, "");
        }
    }

    if (pGui->addButton("Load Graph"))
    {
        std::string renderGraphFileName;
        if (openFileDialog("", renderGraphFileName))
        {
            createRenderGraph(mNextGraphString, renderGraphFileName);
        }
        
    }

    if (pGui->addButton("Save Graph"))
    {
        std::string renderGraphFileName;
        if (saveFileDialog("", renderGraphFileName))
        {
            serializeRenderGraph(renderGraphFileName);
        }
    }

    if (pGui->addButton("RunScript"))
    {
        std::string renderGraphFileName;
        if (openFileDialog("", renderGraphFileName))
        {
            mRenderGraphLoader.LoadAndRunScript(renderGraphFileName, *mpGraphs[mCurrentGraphIndex]);
        }
    }

    uint32_t selection = static_cast<uint32_t>(mCurrentGraphIndex);
    if (mOpenGraphNames.size() && pGui->addDropdown("Open Graph", mOpenGraphNames, selection))
    {
        // Display graph
        mCurrentGraphIndex = selection;
    }

    pGui->addDropdown("RenderPassType", mRenderPassTypes, mTypeSelection);

    pGui->addTextBox("Render Pass Name", mNodeString);
    if (pGui->addButton("Create Node"))
    {
        createAndAddRenderPass(mRenderPassTypes[mTypeSelection].label, mNodeString);
    }

    if (pGui->addButton("Preview Graph"))
    {
        // recompile the graph before a preview
        updateAndCompileGraph();
        mPreviewing = true;
    }

    
    pGui->addTextBox("GraphOutput", mGraphOutputEditString);

    if (pGui->addButton("Update"))
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
    mpGraphs[mCurrentGraphIndex]->setScene(pScene);
    mCamControl.attachCamera(pScene->getCamera(0));
}

void RenderGraphEditor::serializeRenderGraph(const std::string& fileName)
{
    RenderGraphLoader::SaveRenderGraphAsScript(fileName, *mpGraphs[mCurrentGraphIndex]);
}

void RenderGraphEditor::deserializeRenderGraph(const std::string& fileName)
{
}

void RenderGraphEditor::updateAndCompileGraph()
{
    // force a resize event to configure the newly loaded render graph
    Fbo::SharedPtr pBackBufferFBO = gpDevice->getSwapChainFbo();
    auto width = pBackBufferFBO->getWidth();
    auto height = pBackBufferFBO->getHeight();
    onResizeSwapChain(mpLastSample, width, height);
}

void RenderGraphEditor::createRenderGraph(const std::string& renderGraphName, const std::string& renderGraphNameFileName)
{
    mCreatingRenderGraph = true;

    Gui::DropdownValue nextGraphID;
    nextGraphID.value = static_cast<int32_t>(mOpenGraphNames.size());
    nextGraphID.label = renderGraphName;
    mOpenGraphNames.push_back(nextGraphID);
    
    RenderGraph::SharedPtr newGraph = RenderGraph::create();
    mCurrentGraphIndex = mpGraphs.size();
    mpGraphs.push_back(newGraph);

    RenderGraphUI graphUI(*newGraph);
    mRenderGraphUIs.emplace_back(std::move(graphUI));

    if (renderGraphNameFileName.size())
    {
        mRenderGraphLoader.LoadAndRunScript(renderGraphNameFileName, *newGraph);
    }

    // only load the scene for the first graph for now
    if (mCurrentGraphIndex >= 1)
    {
        mpGraphs[mCurrentGraphIndex]->setScene(mpGraphs[0]->getScene());
        updateAndCompileGraph();
    }
    else
    {
        loadScene(gkDefaultScene, false);
    }
    
    mCreatingRenderGraph = false;
}

void RenderGraphEditor::createAndAddConnection(const std::string& srcRenderPass, const std::string& dstRenderPass, const std::string& srcField, const std::string& dstField)
{
    // add information for GUI to avoid costly drawing in renderUI function for graph
    if (!mpGraphs[mCurrentGraphIndex]->addEdge(srcRenderPass + std::string(".") + srcField, dstRenderPass + std::string(".") + dstField))
    {
        logWarning(std::string("Failed to create edge between nodes: ").append(srcRenderPass)
            .append(" and ").append(dstRenderPass).append( " connecting fields ").append(srcField).append(" to ").append(dstField).append(".\n"));
    }
}

void RenderGraphEditor::createAndAddRenderPass(const std::string& renderPassType, const std::string& renderPassName)
{
    mpGraphs[mCurrentGraphIndex]->addRenderPass(sBaseRenderCreateFuncs[renderPassType](), renderPassName);
}

void RenderGraphEditor::onLoad(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext)
{
    mpEditorGraph = RenderGraph::create();
    // Maybe add another specialized node here for initial GUI drawing
    mpEditorGraph->addRenderPass(sBaseRenderCreateFuncs["NodeGraphGuiPass"](), "NodeGraphPass");
    mpEditorGraph->addRenderPass(sBaseRenderCreateFuncs["BlitPass"](), "BlitPass");
    // mpEditorGraph->addRenderPass(sBaseRenderTypes["GraphEditorGuiPass"](), "BlitPass");

    mpEditorGraph->addEdge("NodeGraphPass.color", "BlitPass.src");
    
    createRenderGraph("DefaultRenderGraph", "" /*"DefaultRenderGraph.json"*/);
}

void RenderGraphEditor::renderGraphEditorGUI(SampleCallbacks* pSample, Gui* pGui)
{
    // TODO Make an abstraction for this
    pGui->pushContext("RenderGraphContext");

    pGui->beginFrame();

    pGui->setContextPosition(mWindowPos);

    if (mPreviewing)
    {
        return;
    }

    // mpGraphs[mCurrentGraphIndex]->renderUI(pGui);

    pGui->renderBeforeEndOfFrame(pSample->getRenderContext().get(), pSample->getLastFrameTime());

    pGui->popContext();
}

void RenderGraphEditor::onFrameRender(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext, const Fbo::SharedPtr& pTargetFbo)
{
    if (mPreviewing && pSample->isKeyPressed(KeyboardEvent::Key::E))
    {
        mPreviewing = false;
    }

    mpLastSample = pSample;

    // render the editor GUI graph
    const glm::vec4 clearColor(0.38f, 0.52f, 0.10f, 1);
    pRenderContext->clearFbo(pTargetFbo.get(), clearColor, 1.0f, 0, FboAttachmentType::All);

    static bool firstFrame = true;// TEMPERARY

    if (!mPreviewing)
    {
        // draw node graph editor into specialized graph
        if (!firstFrame)
        {
            auto& nodeEditorFBO = static_cast<const NodeGraphGuiPass*>(mpEditorGraph->getRenderPass("NodeGraphPass").get())->getFbo();
            pRenderContext->clearFbo(nodeEditorFBO.get(), vec4(1), 1, 0);
            // nodeEditorFBO->set
            pSample->getRenderContext()->getGraphicsState()->setFbo(nodeEditorFBO); // TODO put this in the nodegraphguipass node please
            mpEditorGraph->execute(&*pRenderContext);
            renderGraphEditorGUI(pSample, pSample->getGui());
        }

        firstFrame = false;
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
    mpGraphs[mCurrentGraphIndex]->onResizeSwapChain(pSample, width, height);

    mpEditorGraph->setOutput(mCurrentGraphOutput, pSample->getCurrentFbo()->getColorTexture(0));
    mpEditorGraph->onResizeSwapChain(pSample, width, height);
}

int WINAPI WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd)
{
    RenderGraphEditor::UniquePtr pEditor = std::make_unique<RenderGraphEditor>();
    SampleConfig config;
    config.windowDesc.title = "Render Graph Editor";
    config.windowDesc.resizableWindow = true;
    Sample::run(config, pEditor);
    return 0;
}
