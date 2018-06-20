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

#undef register_render_pass

    mNextGraphString.resize(255, 0);
    mNodeString.resize(255, 0);
}

void RenderGraphEditor::onGuiRender(SampleCallbacks* pSample, Gui* pGui)
{
    if (mPreviewing)
    {
        return;
    }

    if (pGui->addButton("Load Scene"))
    {
        std::string filename;
        if (openFileDialog(Scene::kFileFormatString, filename)) loadScene(filename, true);
    }

    uint32_t screenHeight = pSample->getWindow()->getClientAreaHeight();
    uint32_t screenWidth  =  pSample->getWindow()->getClientAreaWidth();

    pGui->pushWindow("Graph Editor Settings", screenWidth / 8, screenHeight - 1, screenWidth - screenWidth / 8, 0, false);

    pGui->addTextBox("Graph Name", mNextGraphString);
    if (mCreatingRenderGraph)
    {
        pGui->addText("Creating Graph ...");
    }
    else
    {
        if (mNextGraphString[0] && pGui->addButton("Create New Graph"))
        {
            createRenderGraph(mNextGraphString, "DefaultRenderGraph.json"); // TODO - create it as blank plz
        }
    }

    if (pGui->addButton("Load Graph"))
    {
        createRenderGraph(mNextGraphString, "DefaultRenderGraph.json");
    }
    
    if (pGui->addButton("Save Graph"))
    {
        serializeRenderGraph (mNextGraphString);
    }

    uint32_t selection = false;
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

    mpGraphs[mCurrentGraphIndex]->renderUI(pGui);
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
    std::string filePath(fileName);

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
    mpGraphs.push_back(newGraph);

    mCurrentGraphIndex = mOpenGraphNames.size() - static_cast<size_t>(1);

    deserializeRenderGraph(renderGraphNameFileName);

    // only load the scene for the first graph for now
    if (mCurrentGraphIndex >= 1)
    {
        mpGraphs[mCurrentGraphIndex]->setScene(mpGraphs[0]->getScene());
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
    mpGraphs[mCurrentGraphIndex]->addEdge(srcRenderPass + std::string(".") + srcField, dstRenderPass + std::string(".") + dstField);
}

void RenderGraphEditor::createAndAddRenderPass(const std::string& renderPassType, const std::string& renderPassName)
{
    mpGraphs[mCurrentGraphIndex]->addRenderPass(sBaseRenderTypes[renderPassType](), renderPassName);
}

void RenderGraphEditor::onLoad(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext)
{
    mpEditorGraph = RenderGraph::create();
    createRenderGraph("DefaultRenderGraph", "DefaultRenderGraph.json");
}

void RenderGraphEditor::onFrameRender(SampleCallbacks* pSample, const RenderContext::SharedPtr& pRenderContext, const Fbo::SharedPtr& pTargetFbo)
{
    if (mPreviewing && pSample->isKeyPressed(KeyboardEvent::Key::E))
    {
        mPreviewing = false;
    }

    if (!mPreviewing)
    {
        // render the editor GUI graph
        const glm::vec4 clearColor(0.38f, 0.52f, 0.10f, 1);
        pRenderContext->clearFbo(pTargetFbo.get(), clearColor, 1.0f, 0, FboAttachmentType::All);

    }
    else
    {
        mpGraphs[mCurrentGraphIndex]->getScene()->update(pSample->getCurrentTime(), &mCamControl);

        // render the editor GUI graph
        const glm::vec4 clearColor(0.38f, 0.52f, 0.10f, 1);
        pRenderContext->clearFbo(pTargetFbo.get(), clearColor, 1.0f, 0, FboAttachmentType::All);

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
    auto pColor = Texture::create2D(width, height, pSample->getCurrentFbo()->getColorTexture(0)->getFormat(), 1, 1, nullptr, Resource::BindFlags::RenderTarget | Resource::BindFlags::ShaderResource);
    auto pDepth = Texture::create2D(width, height, ResourceFormat::D32Float, 1, 1, nullptr, Resource::BindFlags::DepthStencil);
    mpGraphs[mCurrentGraphIndex]->setOutput("BlitPass.dst", pSample->getCurrentFbo()->getColorTexture(0));
    mpGraphs[mCurrentGraphIndex]->onResizeSwapChain(pSample, width, height);
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
