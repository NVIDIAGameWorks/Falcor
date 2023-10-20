/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
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
 **************************************************************************/
#include "RenderGraph.h"
#include "RenderGraphIR.h"
#include "RenderGraphImportExport.h"
#include "RenderGraphCompiler.h"
#include "GlobalState.h"
#include "Core/ObjectPython.h"
#include "Core/API/Device.h"
#include "Utils/Algorithm/DirectedGraphTraversal.h"
#include "Utils/Scripting/Scripting.h"
#include "Utils/Scripting/ScriptBindings.h"

namespace Falcor
{
const FileDialogFilterVec RenderGraph::kFileExtensionFilters = {{"py", "Render Graph Files"}};

RenderGraph::RenderGraph(ref<Device> pDevice, const std::string& name) : mpDevice(pDevice), mName(name)
{
    mpGraph = std::make_unique<DirectedGraph>();
}

RenderGraph::~RenderGraph() {}

ref<RenderGraph> RenderGraph::create(ref<Device> pDevice, const std::string& name)
{
    return ref<RenderGraph>(new RenderGraph(pDevice, name));
}

ref<RenderGraph> RenderGraph::createFromFile(ref<Device> pDevice, const std::filesystem::path& path)
{
    using namespace pybind11::literals;

    ref<RenderGraph> pGraph;

    // Setup a temporary scripting context that defines a local variable 'm' that
    // has a 'addGraph' function exposed. This mimmicks the old Mogwai Python API
    // allowing to load python based render graph scripts.
    auto addGraph = pybind11::cpp_function([&pGraph](ref<RenderGraph> graph) { pGraph = graph; });

    auto SimpleNamespace = pybind11::module_::import("types").attr("SimpleNamespace");
    pybind11::object m = SimpleNamespace("addGraph"_a = addGraph);

    Scripting::Context ctx;
    ctx.setObject("m", m);

    ref<Device> pPrevDevice = getActivePythonRenderGraphDevice();
    setActivePythonRenderGraphDevice(pDevice);
    Scripting::runScriptFromFile(path, ctx);
    setActivePythonRenderGraphDevice(pPrevDevice);

    return pGraph;
}

uint32_t RenderGraph::getPassIndex(const std::string& name) const
{
    auto it = mNameToIndex.find(name);
    return (it == mNameToIndex.end()) ? kInvalidIndex : it->second;
}

void RenderGraph::setScene(const ref<Scene>& pScene)
{
    if (mpScene == pScene)
        return;

    // @skallweit: check that scene resides on the same GPU device

    mpScene = pScene;
    for (auto& it : mNodeData)
    {
        it.second.pPass->setScene(mpDevice->getRenderContext(), pScene);
    }
    mRecompile = true;
}

ref<RenderPass> RenderGraph::createPass(const std::string& passName, const std::string& passType, const Properties& props)
{
    ref<RenderPass> pPass = RenderPass::create(passType, mpDevice, props);
    if (pPass)
        addPass(pPass, passName);
    return pPass;
}

uint32_t RenderGraph::addPass(const ref<RenderPass>& pPass, const std::string& passName)
{
    FALCOR_CHECK(pPass != nullptr, "Added pass must not be null.");
    FALCOR_CHECK(getPassIndex(passName) == kInvalidIndex, "Pass name '{}' already exists.", passName);

    uint32_t passIndex = mpGraph->addNode();
    mNameToIndex[passName] = passIndex;

    pPass->mPassChangedCB = [this]() { mRecompile = true; };
    pPass->mName = passName;

    if (mpScene)
        pPass->setScene(mpDevice->getRenderContext(), mpScene);
    mNodeData[passIndex] = {passName, pPass};
    mRecompile = true;
    return passIndex;
}

void RenderGraph::removePass(const std::string& name)
{
    uint32_t index = getPassIndex(name);
    FALCOR_CHECK(index != kInvalidIndex, "Can't remove render pass '{}'. Pass doesn't exist.", name);

    // Unmark graph outputs that belong to this pass.
    // Because the way std::vector works, we can't call unmarkOutput() immediately, so we store the outputs in a vector
    std::vector<std::string> outputsToDelete;
    const std::string& outputPrefix = name + '.';
    for (auto& o : mOutputs)
    {
        if (o.nodeId == index)
            outputsToDelete.push_back(outputPrefix + o.field);
    }

    // Remove all the edges, indices and pass-data associated with this pass
    for (const auto& outputName : outputsToDelete)
        unmarkOutput(outputName);
    mNameToIndex.erase(name);
    mNodeData.erase(index);
    const auto& removedEdges = mpGraph->removeNode(index);
    for (const auto& e : removedEdges)
        mEdgeData.erase(e);
    mRecompile = true;
}

void RenderGraph::updatePass(const std::string& passName, const Properties& props)
{
    uint32_t index = getPassIndex(passName);
    const auto pPassIt = mNodeData.find(index);

    FALCOR_CHECK(pPassIt != mNodeData.end(), "Can't update render pass '{}'. Pass doesn't exist.", passName);

    // Recreate pass without changing graph using new dictionary
    auto pOldPass = pPassIt->second.pPass;
    std::string passTypeName = pOldPass->getType();
    auto pPass = RenderPass::create(passTypeName, mpDevice, props);
    pPassIt->second.pPass = pPass;
    pPass->mPassChangedCB = [this]() { mRecompile = true; };
    pPass->mName = pOldPass->getName();

    if (mpScene)
        pPass->setScene(mpDevice->getRenderContext(), mpScene);
    mRecompile = true;
}

const ref<RenderPass>& RenderGraph::getPass(const std::string& name) const
{
    uint32_t index = getPassIndex(name);

    FALCOR_CHECK(index != kInvalidIndex, "Can't find render pass '{}'.", name);

    return mNodeData.at(index).pPass;
}

using str_pair = std::pair<std::string, std::string>;

static bool checkRenderPassIoExist(RenderPass* pPass, const std::string& name, const bool input, const RenderPass::CompileData& compileData)
{
    FALCOR_ASSERT(pPass);
    RenderPassReflection reflect = pPass->reflect(compileData);
    for (size_t i = 0; i < reflect.getFieldCount(); i++)
    {
        const auto& f = *reflect.getField(i);
        if (f.getName() == name)
        {
            return input ? is_set(f.getVisibility(), RenderPassReflection::Field::Visibility::Input)
                         : is_set(f.getVisibility(), RenderPassReflection::Field::Visibility::Output);
        }
    }

    return false;
}

static str_pair parseFieldName(const std::string& fullname)
{
    str_pair strPair;
    if (std::count(fullname.begin(), fullname.end(), '.') == 0)
    {
        // No field name
        strPair.first = fullname;
    }
    else
    {
        size_t dot = fullname.find_last_of('.');
        strPair.first = fullname.substr(0, dot);
        strPair.second = fullname.substr(dot + 1);
    }
    return strPair;
}

RenderPass* RenderGraph::getRenderPassAndNamePair(
    const bool input,
    const std::string& fullname,
    std::pair<std::string, std::string>& nameAndField
) const
{
    nameAndField = parseFieldName(fullname);

    RenderPass* pPass = getPass(nameAndField.first).get();
    FALCOR_CHECK(pPass, "Can't find render pass '{}'.", nameAndField.first);

    RenderPass::CompileData compileData;
    compileData.defaultTexDims = mCompilerDeps.defaultResourceProps.dims;
    compileData.defaultTexFormat = mCompilerDeps.defaultResourceProps.format;

    if (nameAndField.second.size() && checkRenderPassIoExist(pPass, nameAndField.second, input, compileData) == false)
        FALCOR_THROW("Can't find field named '{}' in render pass '{}'.", nameAndField.second, nameAndField.first);

    return pPass;
}

static bool checkMatchingEdgeTypes(const std::string& srcField, const std::string& dstField)
{
    if (srcField.empty() && dstField.empty())
        return true;
    if (dstField.size() && dstField.size())
        return true;
    return false;
}

uint32_t RenderGraph::addEdge(const std::string& src, const std::string& dst)
{
    EdgeData newEdge;
    str_pair srcPair, dstPair;
    getRenderPassAndNamePair(false, src, srcPair);
    getRenderPassAndNamePair(true, dst, dstPair);
    newEdge.srcField = srcPair.second;
    newEdge.dstField = dstPair.second;

    if (checkMatchingEdgeTypes(newEdge.srcField, newEdge.dstField) == false)
        FALCOR_THROW(
            "Can't add from '{}' to '{}'. One of the nodes is a resource while the other is a pass. Can't tell if you want a "
            "data-dependency or an execution-dependency",
            src,
            dst
        );

    uint32_t srcIndex = mNameToIndex[srcPair.first];
    uint32_t dstIndex = mNameToIndex[dstPair.first];

    // If this is a data edge, check that the dst field is not already initialized
    if (newEdge.dstField.size())
    {
        const DirectedGraph::Node* pNode = mpGraph->getNode(dstIndex);

        for (uint32_t e = 0; e < pNode->getIncomingEdgeCount(); e++)
        {
            uint32_t incomingEdgeId = pNode->getIncomingEdge(e);
            const auto& edgeData = mEdgeData[incomingEdgeId];

            if (edgeData.dstField == newEdge.dstField)
            {
                FALCOR_THROW(
                    "Edge destination '{}' is already initialized. Please remove the existing connection before trying to add a new edge.",
                    dst
                );
            }
        }
    }

    // Make sure that this doesn't create a cycle
    if (DirectedGraphPathDetector::hasPath(*mpGraph, dstIndex, srcIndex))
        FALCOR_THROW("Can't add the edge from '{}' to '{}'. The edge will create a cycle in the graph which is not allowed.", src, dst);

    uint32_t e = mpGraph->addEdge(srcIndex, dstIndex);
    mEdgeData[e] = newEdge;
    mRecompile = true;
    return e;
}

void RenderGraph::removeEdge(const std::string& src, const std::string& dst)
{
    str_pair srcPair, dstPair;
    const RenderPass* pSrc = getRenderPassAndNamePair(false, src, srcPair);
    const RenderPass* pDst = getRenderPassAndNamePair(true, dst, dstPair);

    uint32_t srcIndex = mNameToIndex[srcPair.first];

    const DirectedGraph::Node* pSrcNode = mpGraph->getNode(srcIndex);

    for (uint32_t i = 0; i < pSrcNode->getOutgoingEdgeCount(); ++i)
    {
        uint32_t edgeID = pSrcNode->getOutgoingEdge(i);
        if (mEdgeData[edgeID].srcField == srcPair.second)
        {
            if (mEdgeData[edgeID].dstField == dstPair.second)
            {
                removeEdge(edgeID);
                return;
            }
        }
    }
}

void RenderGraph::removeEdge(uint32_t edgeID)
{
    FALCOR_CHECK(mEdgeData.find(edgeID) != mEdgeData.end(), "Can't remove edge with index {}. The edge doesn't exist.", edgeID);

    mEdgeData.erase(edgeID);
    mpGraph->removeEdge(edgeID);
    mRecompile = true;
}

uint32_t RenderGraph::getEdge(const std::string& src, const std::string& dst)
{
    str_pair srcPair = parseFieldName(src);
    str_pair dstPair = parseFieldName(dst);

    for (uint32_t i = 0; i < mpGraph->getCurrentEdgeId(); ++i)
    {
        if (!mpGraph->doesEdgeExist(i))
        {
            continue;
        }

        const DirectedGraph::Edge* pEdge = mpGraph->getEdge(i);
        if (dstPair.first == mNodeData[pEdge->getDestNode()].name && srcPair.first == mNodeData[pEdge->getSourceNode()].name)
        {
            if (mEdgeData[i].dstField == dstPair.second && mEdgeData[i].srcField == srcPair.second)
                return i;
        }
    }

    return static_cast<uint32_t>(-1);
}

bool RenderGraph::isGraphOutput(const GraphOut& graphOut) const
{
    for (const GraphOut& currentOut : mOutputs)
    {
        if (graphOut == currentOut)
            return true;
    }

    return false;
}

std::vector<std::string> RenderGraph::getAvailableOutputs() const
{
    std::vector<std::string> outputs;

    RenderPass::CompileData compileData;
    compileData.defaultTexDims = mCompilerDeps.defaultResourceProps.dims;
    compileData.defaultTexFormat = mCompilerDeps.defaultResourceProps.format;

    for (const auto& node : mNodeData)
    {
        RenderPassReflection reflection = node.second.pPass->reflect(compileData);
        for (size_t i = 0; i < reflection.getFieldCount(); i++)
        {
            const auto& f = *reflection.getField(i);
            if (is_set(f.getVisibility(), RenderPassReflection::Field::Visibility::Output))
                outputs.push_back(node.second.name + "." + f.getName());
        }
    }
    return outputs;
}

std::vector<std::string> RenderGraph::getUnmarkedOutputs() const
{
    std::vector<std::string> outputs;

    for (const auto& output : getAvailableOutputs())
    {
        if (!isGraphOutput(output))
            outputs.push_back(output);
    }

    return outputs;
}

bool RenderGraph::compile(RenderContext* pRenderContext, std::string& log)
{
    if (!mRecompile)
        return true;
    mpExe = nullptr;

    try
    {
        mpExe = RenderGraphCompiler::compile(*this, pRenderContext, mCompilerDeps);
        mRecompile = false;
        return true;
    }
    catch (const std::exception& e)
    {
        log = e.what();
        return false;
    }
}

void RenderGraph::execute(RenderContext* pRenderContext)
{
    std::string log;
    if (!compile(pRenderContext, log))
        FALCOR_THROW("Failed to compile render graph:\n{}", log);

    FALCOR_ASSERT(mpExe);
    RenderGraphExe::Context c{
        pRenderContext, mPassesDictionary, mCompilerDeps.defaultResourceProps.dims, mCompilerDeps.defaultResourceProps.format};
    mpExe->execute(c);
}

void RenderGraph::update(const ref<RenderGraph>& pGraph)
{
    // Fill in missing passes from referenced graph.
    for (const auto& nameIndexPair : pGraph->mNameToIndex)
    {
        ref<RenderPass> pRenderPass = pGraph->mNodeData[nameIndexPair.second].pPass;
        if (!doesPassExist(nameIndexPair.first))
            addPass(pRenderPass, nameIndexPair.first);
    }

    // Remove nodes that should no longer be within the graph.
    std::vector<std::string> passesToRemove;

    for (const auto& nameIndexPair : mNameToIndex)
    {
        if (!pGraph->doesPassExist(nameIndexPair.first))
        {
            passesToRemove.push_back(nameIndexPair.first);
        }
    }

    for (const std::string& passName : passesToRemove)
    {
        removePass(passName);
    }

    // Remove all edges from this graph.
    for (uint32_t i = 0; i < mpGraph->getCurrentEdgeId(); ++i)
    {
        if (!mpGraph->doesEdgeExist(i))
        {
            continue;
        }

        mpGraph->removeEdge(i);
    }
    mEdgeData.clear();

    // Add all edges from the other graph.
    for (uint32_t i = 0; i < pGraph->mpGraph->getCurrentEdgeId(); ++i)
    {
        if (!pGraph->mpGraph->doesEdgeExist(i))
        {
            continue;
        }

        const DirectedGraph::Edge* pEdge = pGraph->mpGraph->getEdge(i);
        std::string dst = pGraph->mNodeData.find(pEdge->getDestNode())->second.name;
        std::string src = pGraph->mNodeData.find(pEdge->getSourceNode())->second.name;

        if ((mNameToIndex.find(src) != mNameToIndex.end()) && (mNameToIndex.find(dst) != mNameToIndex.end()))
        {
            if (pGraph->mEdgeData[i].dstField.size())
                dst += std::string(".") + pGraph->mEdgeData[i].dstField;
            if (pGraph->mEdgeData[i].srcField.size())
                src += std::string(".") + pGraph->mEdgeData[i].srcField;
            addEdge(src, dst);
        }
    }

    // Mark all unmarked outputs from referenced graph.
    for (uint32_t i = 0; i < pGraph->getOutputCount(); ++i)
    {
        auto name = pGraph->getOutputName(i);
        for (auto mask : pGraph->getOutputMasks(i))
        {
            markOutput(name, mask);
        }
    }
}

void RenderGraph::setInput(const std::string& name, const ref<Resource>& pResource)
{
    str_pair strPair;
    RenderPass* pPass = getRenderPassAndNamePair(true, name, strPair);

    if (pResource)
    {
        mCompilerDeps.externalResources[name] = pResource;
    }
    else
    {
        if (mCompilerDeps.externalResources.find(name) == mCompilerDeps.externalResources.end())
        {
            FALCOR_THROW("Trying to remove an external resource named '{}' but the resource wasn't registered before.", name);
        }
        mCompilerDeps.externalResources.erase(name);
    }

    if (mpExe)
        mpExe->setInput(name, pResource);
}

void RenderGraph::markOutput(const std::string& name, TextureChannelFlags mask)
{
    FALCOR_CHECK(mask != TextureChannelFlags::None, "Mask must be non-empty");

    // Recursive call to handle '*' wildcard.
    if (name == "*")
    {
        auto outputs = getAvailableOutputs();
        for (const auto& o : outputs)
            markOutput(o, mask);
        return;
    }

    str_pair strPair;
    getRenderPassAndNamePair(false, name, strPair);

    GraphOut newOut;
    newOut.field = strPair.second;
    newOut.nodeId = mNameToIndex[strPair.first];

    // Check if output is already marked.
    // If it is, add the mask to its set of generated masks.
    auto it = std::find(mOutputs.begin(), mOutputs.end(), newOut);
    if (it != mOutputs.end())
    {
        it->masks.insert(mask);
        // No recompile necessary as output is already generated.
    }
    else
    {
        newOut.masks.insert(mask);
        mOutputs.push_back(newOut);
        mRecompile = true;
    }
}

void RenderGraph::unmarkOutput(const std::string& name)
{
    str_pair strPair;
    getRenderPassAndNamePair(false, name, strPair);

    GraphOut removeMe;
    removeMe.field = strPair.second;
    removeMe.nodeId = mNameToIndex[strPair.first];

    auto it = std::find(mOutputs.begin(), mOutputs.end(), removeMe);
    if (it != mOutputs.end())
    {
        mOutputs.erase(it);
        mRecompile = true;
    }
}

bool RenderGraph::isGraphOutput(const std::string& name) const
{
    str_pair strPair;
    getRenderPassAndNamePair(false, name, strPair);

    uint32_t passIndex = getPassIndex(strPair.first);
    GraphOut thisOutput = {passIndex, strPair.second};
    return isGraphOutput(thisOutput);
}

ref<Resource> RenderGraph::getOutput(const std::string& name)
{
    if (mRecompile)
        FALCOR_THROW("Can't fetch the output '{}'. The graph wasn't successfuly compiled yet.", name);

    str_pair strPair;
    getRenderPassAndNamePair(false, name, strPair);

    uint32_t passIndex = getPassIndex(strPair.first);
    GraphOut thisOutput = {passIndex, strPair.second};
    bool isOutput = isGraphOutput(thisOutput);
    if (!isOutput)
        FALCOR_THROW("Can't fetch the output '{}'. The resource is wasn't marked as an output.", name);

    return mpExe->getResource(name);
}

ref<Resource> RenderGraph::getOutput(uint32_t index)
{
    auto name = getOutputName(index);
    return getOutput(name);
}

std::string RenderGraph::getOutputName(size_t index) const
{
    FALCOR_ASSERT(index < mOutputs.size());
    const GraphOut& graphOut = mOutputs[index];
    return mNodeData.find(graphOut.nodeId)->second.name + "." + graphOut.field;
}

std::unordered_set<TextureChannelFlags> RenderGraph::getOutputMasks(size_t index) const
{
    FALCOR_ASSERT(index < mOutputs.size());
    return mOutputs[index].masks;
}

void RenderGraph::onResize(const Fbo* pTargetFbo)
{
    // Store the back-buffer values
    const Texture* pColor = pTargetFbo ? pTargetFbo->getColorTexture(0).get() : nullptr;
    if (pColor == nullptr)
        FALCOR_THROW("Can't resize render graph without a frame buffer.");

    // Store the values
    mCompilerDeps.defaultResourceProps.format = pColor->getFormat();
    mCompilerDeps.defaultResourceProps.dims = {pTargetFbo->getWidth(), pTargetFbo->getHeight()};

    // Invalidate the graph. Render passes might change their reflection based on the resize information
    mRecompile = true;
}

bool canFieldsConnect(const RenderPassReflection::Field& src, const RenderPassReflection::Field& dst)
{
    FALCOR_ASSERT(
        is_set(src.getVisibility(), RenderPassReflection::Field::Visibility::Output) &&
        is_set(dst.getVisibility(), RenderPassReflection::Field::Visibility::Input)
    );

    return src.getName() == dst.getName() && (dst.getWidth() == 0 || src.getWidth() == dst.getWidth()) &&
           (dst.getHeight() == 0 || src.getHeight() == dst.getHeight()) && (dst.getDepth() == 0 || src.getDepth() == dst.getDepth()) &&
           (dst.getFormat() == ResourceFormat::Unknown || src.getFormat() == dst.getFormat()) &&
           src.getSampleCount() == dst.getSampleCount() && // TODO: allow dst sample count to be 1 when auto MSAA resolve is implemented in
                                                           // graph compilation
           src.getType() == dst.getType() && src.getSampleCount() == dst.getSampleCount();
}

void RenderGraph::renderUI(RenderContext* pRenderContext, Gui::Widgets& widget)
{
    if (mpExe)
        mpExe->renderUI(pRenderContext, widget);
}

void RenderGraph::onSceneUpdates(RenderContext* pRenderContext, Scene::UpdateFlags sceneUpdates)
{
    // Notify all passes in graph about scene updates.
    // Note we don't rely on `mpExe` here because it is not created until the graph is compiled in `execute()`.
    for (auto& it : mNodeData)
    {
        it.second.pPass->onSceneUpdates(pRenderContext, sceneUpdates);
    }
}

bool RenderGraph::onMouseEvent(const MouseEvent& mouseEvent)
{
    return mpExe ? mpExe->onMouseEvent(mouseEvent) : false;
}

bool RenderGraph::onKeyEvent(const KeyboardEvent& keyEvent)
{
    return mpExe ? mpExe->onKeyEvent(keyEvent) : false;
}

void RenderGraph::onHotReload(HotReloadFlags reloaded)
{
    if (mpExe)
        mpExe->onHotReload(reloaded);
}

FALCOR_SCRIPT_BINDING(RenderGraph)
{
    using namespace pybind11::literals;

    FALCOR_SCRIPT_BINDING_DEPENDENCY(Formats)
    FALCOR_SCRIPT_BINDING_DEPENDENCY(Resource)

    // RenderPass
    pybind11::class_<RenderPass, ref<RenderPass>> renderPass(m, "RenderPass");
    renderPass.def_property_readonly("name", &RenderPass::getName);
    renderPass.def_property_readonly("type", &RenderPass::getType);
    renderPass.def_property_readonly("desc", &RenderPass::getDesc);
    renderPass.def_property_readonly("properties", [](RenderPass& self) { return self.getProperties().toPython(); });
    renderPass.def("set_properties", [](RenderPass& self, pybind11::dict dict) { self.setProperties(Properties(dict)); });

    // PYTHONDEPRECATED BEGIN
    renderPass.def("getDictionary", [](RenderPass& pass) { return pass.getProperties().toPython(); });
    // PYTHONDEPRECATED END

    // RenderGraph
    pybind11::class_<RenderGraph, ref<RenderGraph>> renderGraph(m, "RenderGraph");
    renderGraph.def_property("name", &RenderGraph::getName, &RenderGraph::setName);

    renderGraph.def(
        "create_pass",
        [](RenderGraph& graph, const std::string& pass_name, const std::string& pass_type, pybind11::dict dict = {})
        { return graph.createPass(pass_name, pass_type, Properties(dict)); },
        "pass_name"_a,
        "pass_type"_a,
        "dict"_a = pybind11::dict()
    );
    renderGraph.def("remove_pass", &RenderGraph::removePass, "name"_a);
    renderGraph.def(
        "update_pass",
        [](RenderGraph& graph, const std::string& name, pybind11::dict dict) { graph.updatePass(name, Properties(dict)); },
        "name"_a,
        "dict"_a
    );
    renderGraph.def("add_edge", &RenderGraph::addEdge, "src"_a, "dst"_a);
    renderGraph.def(
        "remove_edge", pybind11::overload_cast<const std::string&, const std::string&>(&RenderGraph::removeEdge), "src"_a, "dst"_a
    );
    renderGraph.def("mark_output", &RenderGraph::markOutput, "name"_a, "mask"_a = TextureChannelFlags::RGB);
    renderGraph.def("unmark_output", &RenderGraph::unmarkOutput, "name"_a);
    renderGraph.def("get_pass", &RenderGraph::getPass, "name"_a);
    renderGraph.def("__getitem__", [](RenderGraph& self, const std::string& name) { return self.getPass(name); });
    renderGraph.def("get_output", pybind11::overload_cast<const std::string&>(&RenderGraph::getOutput), "name"_a);

    // PYTHONDEPRECATED BEGIN
    renderGraph.def(
        pybind11::init([](const std::string& name) { return RenderGraph::create(accessActivePythonRenderGraphDevice(), name); }), "name"_a
    );
    renderGraph.def_static(
        "createFromFile",
        [](const std::filesystem::path& path) { return RenderGraph::createFromFile(accessActivePythonRenderGraphDevice(), path); },
        "path"_a
    );

    renderGraph.def("print", [](ref<RenderGraph> graph) { pybind11::print(RenderGraphExporter::getIR(graph)); });

    renderGraph.def(
        "createPass",
        [](RenderGraph& graph, const std::string& passName, const std::string& passType, pybind11::dict dict = {})
        { return graph.createPass(passName, passType, Properties(dict)); },
        "passName"_a,
        "passType"_a,
        "dict"_a = pybind11::dict()
    );
    renderGraph.def("addPass", &RenderGraph::addPass, "pass_"_a, "name"_a);
    renderGraph.def("removePass", &RenderGraph::removePass, "name"_a);
    renderGraph.def(
        "updatePass",
        [](RenderGraph& graph, const std::string& passName, pybind11::dict d) { graph.updatePass(passName, Properties(d)); },
        "name"_a,
        "dict"_a
    );
    renderGraph.def("addEdge", &RenderGraph::addEdge, "src"_a, "dst"_a);
    renderGraph.def(
        "removeEdge", pybind11::overload_cast<const std::string&, const std::string&>(&RenderGraph::removeEdge), "src"_a, "dst"_a
    );
    renderGraph.def("markOutput", &RenderGraph::markOutput, "name"_a, "mask"_a = TextureChannelFlags::RGB);
    renderGraph.def("unmarkOutput", &RenderGraph::unmarkOutput, "name"_a);
    renderGraph.def("getPass", &RenderGraph::getPass, "name"_a);
    renderGraph.def("getOutput", pybind11::overload_cast<const std::string&>(&RenderGraph::getOutput), "name"_a);
    // PYTHONDEPRECATED END

    // RenderPassLibrary
    const auto& globalCreateRenderPass = [](const std::string& type, pybind11::dict d = {})
    {
        auto pPass = RenderPass::create(type, accessActivePythonRenderGraphDevice(), Properties(d));
        if (!pPass)
            FALCOR_THROW("Can't create a render pass of type '{}'. Make sure the required plugin library was loaded.", type);
        return pPass;
    };
    m.def("createPass", globalCreateRenderPass, "type"_a, "dict"_a = pybind11::dict()); // PYTHONDEPRECATED
}
} // namespace Falcor
