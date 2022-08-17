/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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
#include "RenderPassLibrary.h"
#include "RenderGraphCompiler.h"
#include "Core/Renderer.h"
#include "Utils/Algorithm/DirectedGraphTraversal.h"
#include "Utils/Scripting/ScriptBindings.h"

namespace Falcor
{
    std::vector<RenderGraph*> gRenderGraphs;
    const FileDialogFilterVec RenderGraph::kFileExtensionFilters = { { "py", "Render Graph Files"} };

    RenderGraph::SharedPtr RenderGraph::create(const std::string& name)
    {
        return SharedPtr(new RenderGraph(name));
    }

    RenderGraph::RenderGraph(const std::string& name)
        : mName(name)
    {
        if (gpFramework == nullptr) throw RuntimeError("Can't construct RenderGraph - framework is not initialized");
        mpGraph = DirectedGraph::create();
        mpPassDictionary = InternalDictionary::create();
        gRenderGraphs.push_back(this);
        onResize(gpFramework->getTargetFbo().get());
    }

    RenderGraph::~RenderGraph()
    {
        auto it = std::find(gRenderGraphs.begin(), gRenderGraphs.end(), this);
        FALCOR_ASSERT(it != gRenderGraphs.end());
        gRenderGraphs.erase(it);
    }

    uint32_t RenderGraph::getPassIndex(const std::string& name) const
    {
        auto it = mNameToIndex.find(name);
        return (it == mNameToIndex.end()) ? kInvalidIndex : it->second;
    }

    void RenderGraph::setScene(const Scene::SharedPtr& pScene)
    {
        if (mpScene == pScene) return;

        mpScene = pScene;
        for (auto& it : mNodeData)
        {
            it.second.pPass->setScene(gpDevice->getRenderContext(), pScene);
        }
        mRecompile = true;
    }

    uint32_t RenderGraph::addPass(const RenderPass::SharedPtr& pPass, const std::string& passName)
    {
        FALCOR_ASSERT(pPass);
        uint32_t passIndex = getPassIndex(passName);
        if (passIndex != kInvalidIndex)
        {
            reportError("Pass named '" + passName + "' already exists. Ignoring call");
            return kInvalidIndex;
        }
        else
        {
            passIndex = mpGraph->addNode();
            mNameToIndex[passName] = passIndex;
        }

        pPass->mPassChangedCB = [this]() { mRecompile = true; };
        pPass->mName = passName;

        if (mpScene) pPass->setScene(gpDevice->getRenderContext(), mpScene);
        mNodeData[passIndex] = { passName, pPass };
        mRecompile = true;
        return passIndex;
    }

    void RenderGraph::removePass(const std::string& name)
    {
        uint32_t index = getPassIndex(name);
        if (index == kInvalidIndex)
        {
            logWarning("Can't remove pass '{}'. Pass doesn't exist.", name);
            return;
        }

        // Unmark graph outputs that belong to this pass.
        // Because the way std::vector works, we can't call unmarkOutput() immediately, so we store the outputs in a vector
        std::vector<std::string> outputsToDelete;
        const std::string& outputPrefix = name + '.';
        for (auto& o : mOutputs)
        {
            if (o.nodeId == index) outputsToDelete.push_back(outputPrefix + o.field);
        }

        // Remove all the edges, indices and pass-data associated with this pass
        for (const auto& name : outputsToDelete) unmarkOutput(name);
        mNameToIndex.erase(name);
        mNodeData.erase(index);
        const auto& removedEdges = mpGraph->removeNode(index);
        for (const auto& e : removedEdges) mEdgeData.erase(e);
        mRecompile = true;
    }

    void RenderGraph::applyPassSettings(const std::string& passName, const Dictionary& dict)
    {
        uint32_t index = getPassIndex(passName);
        const auto pPassIt = mNodeData.find(index);

        if (pPassIt == mNodeData.end())
        {
            logError("Error in RenderGraph::updatePass(). Unable to find pass " + passName);
            return;
        }
        auto pPass = pPassIt->second.pPass;

        pPass->applySettings(dict);
    }

    void RenderGraph::updatePass(RenderContext* pRenderContext, const std::string& passName, const Dictionary& dict)
    {
        uint32_t index = getPassIndex(passName);
        const auto pPassIt = mNodeData.find(index);

        if (pPassIt == mNodeData.end())
        {
            reportError("Error in RenderGraph::updatePass(). Unable to find pass " + passName);
            return;
        }

        // Recreate pass without changing graph using new dictionary
        auto pOldPass = pPassIt->second.pPass;
        std::string passTypeName = pOldPass->getType();
        auto pPass = RenderPassLibrary::instance().createPass(pRenderContext, passTypeName.c_str(), dict);
        pPassIt->second.pPass = pPass;
        pPass->mPassChangedCB = [this]() { mRecompile = true; };
        pPass->mName = pOldPass->getName();

        if (mpScene) pPass->setScene(gpDevice->getRenderContext(), mpScene);
        mRecompile = true;
    }

    const RenderPass::SharedPtr& RenderGraph::getPass(const std::string& name) const
    {
        uint32_t index = getPassIndex(name);
        if (index == kInvalidIndex)
        {
            static RenderPass::SharedPtr pNull;
            reportError("RenderGraph::getRenderPass() - can't find a pass named '" + name + "'");
            return pNull;
        }
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
                return input ? is_set(f.getVisibility(), RenderPassReflection::Field::Visibility::Input) : is_set(f.getVisibility(), RenderPassReflection::Field::Visibility::Output);
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

    RenderPass* RenderGraph::getRenderPassAndNamePair(const bool input, const std::string& fullname, const std::string& errorPrefix, std::pair<std::string, std::string>& nameAndField) const
    {
        nameAndField = parseFieldName(fullname);

        RenderPass* pPass = getPass(nameAndField.first).get();
        if (!pPass)
        {
            reportError(errorPrefix + " - can't find render pass named '" + nameAndField.first + "'");
            return nullptr;
        }

        RenderPass::CompileData compileData;
        compileData.defaultTexDims = mCompilerDeps.defaultResourceProps.dims;
        compileData.defaultTexFormat = mCompilerDeps.defaultResourceProps.format;

        if (nameAndField.second.size() && checkRenderPassIoExist(pPass, nameAndField.second, input, compileData) == false)
        {
            reportError(errorPrefix + "- can't find field named '" + nameAndField.second + "' in render pass '" + nameAndField.first + "'");
            return nullptr;
        }
        return pPass;
    }

    static bool checkMatchingEdgeTypes(const std::string& srcField, const std::string& dstField)
    {
        if (srcField.empty() && dstField.empty()) return true;
        if (dstField.size() && dstField.size()) return true;
        return false;
    }

    uint32_t RenderGraph::addEdge(const std::string& src, const std::string& dst)
    {
        EdgeData newEdge;
        str_pair srcPair, dstPair;
        const auto& pSrc = getRenderPassAndNamePair(false, src, "Invalid src string in RenderGraph::addEdge()", srcPair);
        const auto& pDst = getRenderPassAndNamePair(true, dst, "Invalid dst string in RenderGraph::addEdge()", dstPair);
        newEdge.srcField = srcPair.second;
        newEdge.dstField = dstPair.second;

        if (pSrc == nullptr || pDst == nullptr) return kInvalidIndex;
        if (checkMatchingEdgeTypes(newEdge.srcField, newEdge.dstField) == false)
        {
            reportError("RenderGraph::addEdge() - can't add the edge [" + src + ", " + dst + "]. One of the nodes is a resource while the other is a pass. Can't tell if you want a data-dependency or an execution-dependency");
            return kInvalidIndex;
        }

        uint32_t srcIndex = mNameToIndex[srcPair.first];
        uint32_t dstIndex = mNameToIndex[dstPair.first];

        // If this is a data edge, check that the dst field is not already initialized
        if(newEdge.dstField.size())
        {
            const DirectedGraph::Node* pNode = mpGraph->getNode(dstIndex);

            for (uint32_t e = 0; e < pNode->getIncomingEdgeCount(); e++)
            {
                uint32_t incomingEdgeId = pNode->getIncomingEdge(e);
                const auto& edgeData = mEdgeData[incomingEdgeId];

                if (edgeData.dstField == newEdge.dstField)
                {
                    reportError("RenderGraph::addEdge() - destination '" + dst + "' is already initialized. Please remove the existing connection before trying to add an edge");
                    return kInvalidIndex;
                }
            }
        }

        // Make sure that this doesn't create a cycle
        if (DirectedGraphPathDetector::hasPath(mpGraph, dstIndex, srcIndex))
        {
            reportError("RenderGraph::addEdge() - can't add the edge [" + src + ", " + dst + "]. The edge will create a cycle in the graph which is not allowed");
            return kInvalidIndex;
        }

        uint32_t e = mpGraph->addEdge(srcIndex, dstIndex);
        mEdgeData[e] = newEdge;
        mRecompile = true;
        return e;
    }

    void RenderGraph::removeEdge(const std::string& src, const std::string& dst)
    {
        str_pair srcPair, dstPair;
        const auto& pSrc = getRenderPassAndNamePair(false, src, "Invalid src string in RenderGraph::addEdge()", srcPair);
        const auto& pDst = getRenderPassAndNamePair(true, dst, "Invalid dst string in RenderGraph::addEdge()", dstPair);

        if (pSrc == nullptr || pDst == nullptr)
        {
            reportError("Unable to remove edge. Input or output node not found.");
            return;
        }

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
        if (mEdgeData.find(edgeID) == mEdgeData.end())
        {
            reportError("Can't remove edge with index " + std::to_string(edgeID) + ". The edge doesn't exist");
            return;
        }
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
            if (!mpGraph->doesEdgeExist(i)) { continue; }

            const DirectedGraph::Edge* pEdge = mpGraph->getEdge(i);
            if (dstPair.first == mNodeData[pEdge->getDestNode()].name &&
                srcPair.first == mNodeData[pEdge->getSourceNode()].name)
            {
                if (mEdgeData[i].dstField == dstPair.second && mEdgeData[i].srcField == srcPair.second) return i;
            }
        }

        return static_cast<uint32_t>(-1);
    }

    bool RenderGraph::isGraphOutput(const GraphOut& graphOut) const
    {
        for (const GraphOut& currentOut : mOutputs)
        {
            if (graphOut == currentOut) return true;
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
                if(is_set(f.getVisibility(), RenderPassReflection::Field::Visibility::Output)) outputs.push_back(node.second.name + "." + f.getName());
            }
        }
        return outputs;
    }

    std::vector<std::string> RenderGraph::getUnmarkedOutputs() const
    {
        std::vector<std::string> outputs;

        for (const auto& output : getAvailableOutputs())
        {
            if (!isGraphOutput(output)) outputs.push_back(output);
        }

        return outputs;
    }

    bool RenderGraph::compile(RenderContext* pRenderContext, std::string& log)
    {
        if (!mRecompile) return true;
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
        {
            reportError("Failed to compile RenderGraph\n" + log + "Ignoring RenderGraph::execute() call");
            return;
        }

        FALCOR_ASSERT(mpExe);
        RenderGraphExe::Context c;
        c.pGraphDictionary = mpPassDictionary;
        c.pRenderContext = pRenderContext;
        c.defaultTexDims = mCompilerDeps.defaultResourceProps.dims;
        c.defaultTexFormat = mCompilerDeps.defaultResourceProps.format;
        mpExe->execute(c);
    }

    void RenderGraph::update(const SharedPtr& pGraph)
    {
        // Fill in missing passes from referenced graph.
        for (const auto& nameIndexPair : pGraph->mNameToIndex)
        {
            RenderPass::SharedPtr pRenderPass = pGraph->mNodeData[nameIndexPair.second].pPass;
            if (!doesPassExist(nameIndexPair.first)) addPass(pRenderPass, nameIndexPair.first);
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
            if (!mpGraph->doesEdgeExist(i)) { continue; }

            mpGraph->removeEdge(i);
        }
        mEdgeData.clear();

        // Add all edges from the other graph.
        for (uint32_t i = 0; i < pGraph->mpGraph->getCurrentEdgeId(); ++i)
        {
            if (!pGraph->mpGraph->doesEdgeExist(i)) { continue; }

            const DirectedGraph::Edge* pEdge = pGraph->mpGraph->getEdge(i);
            std::string dst = pGraph->mNodeData.find(pEdge->getDestNode())->second.name;
            std::string src = pGraph->mNodeData.find(pEdge->getSourceNode())->second.name;

            if ((mNameToIndex.find(src) != mNameToIndex.end()) && (mNameToIndex.find(dst) != mNameToIndex.end()))
            {

                if (pGraph->mEdgeData[i].dstField.size()) dst += std::string(".") + pGraph->mEdgeData[i].dstField;
                if (pGraph->mEdgeData[i].srcField.size()) src += std::string(".") + pGraph->mEdgeData[i].srcField;
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

    void RenderGraph::setInput(const std::string& name, const Resource::SharedPtr& pResource)
    {
        str_pair strPair;
        RenderPass* pPass = getRenderPassAndNamePair(true, name, "RenderGraph::setInput()", strPair);
        if (pPass == nullptr) return;

        if (pResource)
        {
            mCompilerDeps.externalResources[name] = pResource;
        }
        else
        {
            if (mCompilerDeps.externalResources.find(name) == mCompilerDeps.externalResources.end())
            {
                logWarning("RenderGraph::setInput() - Trying to remove an external resource named '{}' but the resource wasn't registered before. Ignoring call.", name);
                return;
            }
            mCompilerDeps.externalResources.erase(name);
        }

        if (mpExe) mpExe->setInput(name, pResource);
    }

    void RenderGraph::markOutput(const std::string& name, TextureChannelFlags mask)
    {
        if (mask == TextureChannelFlags::None) throw RuntimeError("RenderGraph::markOutput() mask must be non-empty");

        // Recursive call to handle '*' wildcard.
        if (name == "*")
        {
            auto outputs = getAvailableOutputs();
            for (const auto& o : outputs) markOutput(o, mask);
            return;
        }

        str_pair strPair;
        const auto& pPass = getRenderPassAndNamePair(false, name, "RenderGraph::markOutput()", strPair);
        if (pPass == nullptr) return;

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
        const auto& pPass = getRenderPassAndNamePair(false, name, "RenderGraph::unmarkOutput()", strPair);
        if (pPass == nullptr) return;

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
        const auto& pPass = getRenderPassAndNamePair(false, name, "RenderGraph::isGraphOutput()", strPair);
        if (pPass == nullptr) return false;
        uint32_t passIndex = getPassIndex(strPair.first);
        GraphOut thisOutput = { passIndex, strPair.second };
        return isGraphOutput(thisOutput);
    }

    Resource::SharedPtr RenderGraph::getOutput(const std::string& name)
    {
        if (mRecompile)
        {
            reportError("RenderGraph::getOutput() - can't fetch an output resource because the graph wasn't successfuly compiled yet");
            return nullptr;
        }

        str_pair strPair;
        RenderPass* pPass = getRenderPassAndNamePair(false, name, "RenderGraph::getOutput()", strPair);
        if (!pPass) return nullptr;

        uint32_t passIndex = getPassIndex(strPair.first);
        GraphOut thisOutput = { passIndex, strPair.second };
        bool isOutput = isGraphOutput(thisOutput);
        if (!isOutput)
        {
            reportError("RenderGraph::getOutput() - can't fetch the output '" + name + "'. The resource is wasn't marked as an output");
            return nullptr;
        }

        return mpExe->getResource(name);
    }

    Resource::SharedPtr RenderGraph::getOutput(uint32_t index)
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
        if (pColor == nullptr) throw RuntimeError("Can't resize render graph without a frame buffer.");

        // Store the values
        mCompilerDeps.defaultResourceProps.format = pColor->getFormat();
        mCompilerDeps.defaultResourceProps.dims = { pTargetFbo->getWidth(), pTargetFbo->getHeight() };

        // Invalidate the graph. Render passes might change their reflection based on the resize information
        mRecompile = true;
    }

    bool canFieldsConnect(const RenderPassReflection::Field& src, const RenderPassReflection::Field& dst)
    {
        FALCOR_ASSERT(is_set(src.getVisibility(), RenderPassReflection::Field::Visibility::Output) && is_set(dst.getVisibility(), RenderPassReflection::Field::Visibility::Input));

        return src.getName() == dst.getName() &&
            (dst.getWidth() == 0 || src.getWidth() == dst.getWidth()) &&
            (dst.getHeight() == 0 || src.getHeight() == dst.getHeight()) &&
            (dst.getDepth() == 0 || src.getDepth() == dst.getDepth()) &&
            (dst.getFormat() == ResourceFormat::Unknown || src.getFormat() == dst.getFormat()) &&
            src.getSampleCount() == dst.getSampleCount() && // TODO: allow dst sample count to be 1 when auto MSAA resolve is implemented in graph compilation
            src.getType() == dst.getType() &&
            src.getSampleCount() == dst.getSampleCount();
    }

    void RenderGraph::renderUI(Gui::Widgets& widget)
    {
        if (mpExe) mpExe->renderUI(widget);
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
        if (mpExe) mpExe->onHotReload(reloaded);
    }

    FALCOR_SCRIPT_BINDING(RenderGraph)
    {
        using namespace pybind11::literals;

        FALCOR_SCRIPT_BINDING_DEPENDENCY(Formats);

        pybind11::class_<RenderGraph, RenderGraph::SharedPtr> renderGraph(m, "RenderGraph");
        renderGraph.def(pybind11::init(&RenderGraph::create));
        renderGraph.def_property("name", &RenderGraph::getName, &RenderGraph::setName);
        renderGraph.def(RenderGraphIR::kAddPass, &RenderGraph::addPass, "pass"_a, "name"_a);
        renderGraph.def(RenderGraphIR::kRemovePass, &RenderGraph::removePass, "name"_a);
        renderGraph.def(RenderGraphIR::kAddEdge, &RenderGraph::addEdge, "src"_a, "dst"_a);
        renderGraph.def(RenderGraphIR::kRemoveEdge, pybind11::overload_cast<const std::string&, const std::string&>(&RenderGraph::removeEdge), "src"_a, "src"_a);
        renderGraph.def(RenderGraphIR::kMarkOutput, &RenderGraph::markOutput, "name"_a, "mask"_a = TextureChannelFlags::RGB);
        renderGraph.def(RenderGraphIR::kUnmarkOutput, &RenderGraph::unmarkOutput, "name"_a);
        renderGraph.def("getPass", &RenderGraph::getPass, "name"_a);
        renderGraph.def("getOutput", pybind11::overload_cast<const std::string&>(&RenderGraph::getOutput), "name"_a);
        auto printGraph = [](RenderGraph::SharedPtr pGraph) { pybind11::print(RenderGraphExporter::getIR(pGraph)); };
        renderGraph.def("print", printGraph);

        // RenderPass
        pybind11::class_<RenderPass, RenderPass::SharedPtr> renderPass(m, "RenderPass");
        renderPass.def_property_readonly("name", &RenderPass::getName);
        renderPass.def_property_readonly("type", &RenderPass::getType);
        renderPass.def_property_readonly("desc", &RenderPass::getDesc);
        auto getDictionary = [](RenderPass::SharedPtr pPass) { return pPass->getScriptingDictionary().toPython(); };
        renderPass.def("getDictionary", getDictionary);

        // RenderPassLibrary
        const auto& createRenderPass = [](const std::string& passName, pybind11::dict d = {})
        {
            auto pPass = RenderPassLibrary::instance().createPass(gpDevice->getRenderContext(), passName, Dictionary(d));
            if (!pPass) throw RuntimeError("Can't create a render pass named '{}'. Make sure the required DLL was loaded.", passName);
            return pPass;
        };
        m.def("createPass", createRenderPass, "name"_a, "dict"_a = pybind11::dict());

        const auto& loadPassLibrary = [](const std::string& library)
        {
            return RenderPassLibrary::instance().loadLibrary(library);
        };
        m.def(RenderGraphIR::kLoadPassLibrary, loadPassLibrary, "name"_a);

        const auto& updateRenderPass = [](const RenderGraph::SharedPtr& pGraph, const std::string& passName, pybind11::dict d)
        {
            pGraph->updatePass(gpDevice->getRenderContext(), passName, Dictionary(d));
        };
        renderGraph.def(RenderGraphIR::kUpdatePass, updateRenderPass, "name"_a, "dict"_a);
    }
}
