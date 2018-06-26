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

#include "Externals/imgui-node-editor/NodeEditor/Include/NodeEditor.h"

#include "Externals/dear_imgui/imgui.h"
#include "Externals/dear_imgui/imgui_internal.h"

const std::string gkDefaultScene = "Arcade/Arcade.fscene";

std::unordered_map<std::string, std::function<RenderPass::SharedPtr()>> RenderGraphEditor::sBaseRenderTypes;

RenderGraphEditor::RenderGraphEditor()
    : mCurrentGraphIndex(0), mCreatingRenderGraph(false), mPreviewing(false)
{
    Gui::DropdownValue dropdownValue;

#define register_render_pass(renderPassType) \
    sBaseRenderTypes.insert(std::make_pair(#renderPassType, std::function<RenderPass::SharedPtr()> ( \
        []() { return renderPassType ::create(); }) )\
    ); \
    dropdownValue.label = #renderPassType; dropdownValue.value = static_cast<int32_t>(mRenderPassTypes.size()); \
    mRenderPassTypes.push_back(dropdownValue)


    register_render_pass(SceneRenderPass);
    register_render_pass(BlitPass);
    register_render_pass(NodeGraphGuiPass);
    register_render_pass(GraphEditorGuiPass);

#undef register_render_pass

    mNextGraphString.resize(255, 0);
    mNodeString.resize(255, 0);
}

void RenderGraphEditor::onGuiRender(SampleCallbacks* pSample, Gui* pGui)
{
    mWindowSize = pGui->getCurrentWindowSize();
    mWindowPos = pGui->getCurrentWindowPosition();

    // please remove this
    static bool firstFrame = true;

    static ax::NodeEditor::EditorContext* spContext = nullptr;
    if (!spContext)
    {
        spContext = ax::NodeEditor::CreateEditor();
    }

    ax::NodeEditor::SetCurrentEditor(spContext);

    // Test node editor
    ax::NodeEditor::Begin("Editor");

    mpGraphs[mCurrentGraphIndex]->renderUI(pGui);

    ax::NodeEditor::End();


    if (!firstFrame)
    {
        auto& nodeEditorFBO = static_cast<const NodeGraphGuiPass*>(mpEditorGraph->getRenderPass("NodeGraphPass").get())->getFbo();
        pGui->addImageForContext("RenderGraphContext", nodeEditorFBO->getColorTexture(0)); 

    }
    
    firstFrame = false;

    uint32_t screenHeight = pSample->getWindow()->getClientAreaHeight();
    uint32_t screenWidth = pSample->getWindow()->getClientAreaWidth();
    
    pGui->pushWindow("Graph Editor Settings", screenWidth / 8, screenHeight - 1, screenWidth - screenWidth / 8, 0, false);

    // DO you want to keep these ?? -- posible custom contexts outside of what is rendered?  would that even be useful
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
            createRenderGraph(mNextGraphString, "DefaultRenderGraph.json");
        }
    }

    if (pGui->addButton("Load Graph"))
    {
        createRenderGraph(mNextGraphString, "DefaultRenderGraph.json");
    }

    if (pGui->addButton("Save Graph"))
    {
        serializeRenderGraph(mNextGraphString);
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
        mPreviewing = true;
    }

    pGui->popWindow();

    // update the viewport of the editor render graph from first draw - blit
    mpEditorGraph->setEdgeViewport("NodeGraphPass.color", "BlitPass.src", { pGui->getCurrentWindowSize(), 1.0f });
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
    
    std::string filePath(getExecutableDirectory() + "/Data/");
    filePath.append(fileName);
    if (doesFileExist(filePath))
    {
        // Are you sure you want to overwrite this file
        logWarning(std::string("Overwriting render graph file ").append(fileName).append("\n"));
    }

    std::ofstream outStream(filePath);
    rapidjson::OStreamWrapper oStream(outStream);
    rapidjson::Writer<rapidjson::OStreamWrapper> writer(oStream);
    
    writer.StartObject();
    mpGraphs[mCurrentGraphIndex]->serializeJson(&writer);
    writer.EndObject();
    
    outStream.close();
}

void RenderGraphEditor::deserializeRenderGraph(const std::string& fileName)
{
    std::string filePath;
    
    assert (findFileInDataDirectories(fileName, filePath) );
    
    std::ifstream instream(filePath);
    rapidjson::IStreamWrapper istream(instream);

    rapidjson::Document document;
    document.ParseStream(istream);

    // make sure nodes are created before we connect the edges
    auto nodesArray = (document.FindMember("RenderPassNodes")->value).GetArray();
    assert(!nodesArray.Empty());

    for (const auto& node : nodesArray)
    {
        std::string renderPassName;
        std::string renderPassType;

        // first create the graph
        renderPassType = node.FindMember("RenderPassType")->value.GetString();
        renderPassName = node.FindMember("RenderPassName")->value.GetString();

        createAndAddRenderPass(renderPassType, renderPassName);
    }

    // add all edges
    auto edgesArray = (document.FindMember("Edges")->value).GetArray();
    assert (!edgesArray.Empty());

    for(const auto& edge : edgesArray)
    {
        createAndAddConnection(edge.FindMember("SrcRenderPassName")->value.GetString(), edge.FindMember("DstRenderPassName")->value.GetString(),
            edge.FindMember("SrcField")->value.GetString(), edge.FindMember("DstField")->value.GetString());
    }
    
    instream.close();
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

    if (renderGraphNameFileName.size())
    {
        deserializeRenderGraph(renderGraphNameFileName);
    }

    // only load the scene for the first graph for now
    if (mCurrentGraphIndex >= 1)
    {
        mpGraphs[mCurrentGraphIndex]->setScene(mpGraphs[0]->getScene());

        // force a resize event to configure the newly loaded render graph
        Fbo::SharedPtr pBackBufferFBO = gpDevice->getSwapChainFbo();
        auto width = pBackBufferFBO->getWidth();
        auto height = pBackBufferFBO->getHeight();
        onResizeSwapChain(mpLastSample, width, height);

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
    mpGraphs[mCurrentGraphIndex]->addRenderPass(sBaseRenderTypes[renderPassType](), renderPassName);
}

void RenderGraphEditor::onLoad(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext)
{
    mpEditorGraph = RenderGraph::create();
    // Maybe add another specialized node here for initial GUI drawing
    mpEditorGraph->addRenderPass(sBaseRenderTypes["NodeGraphGuiPass"](), "NodeGraphPass");
    mpEditorGraph->addRenderPass(sBaseRenderTypes["BlitPass"](), "BlitPass");
    // mpEditorGraph->addRenderPass(sBaseRenderTypes["GraphEditorGuiPass"](), "BlitPass");

    mpEditorGraph->addEdge("NodeGraphPass.color", "BlitPass.src");

    createRenderGraph("DefaultRenderGraph", "DefaultRenderGraph.json");
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

    pGui->renderBeforeEndOfFrame(&*(pSample->getRenderContext()), pSample->getLastFrameTime());

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
    mpGraphs[mCurrentGraphIndex]->setOutput("BlitPass.dst", pSample->getCurrentFbo()->getColorTexture(0));
    mpGraphs[mCurrentGraphIndex]->onResizeSwapChain(pSample, width, height);

    mpEditorGraph->setOutput("BlitPass.dst", pSample->getCurrentFbo()->getColorTexture(0));
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
