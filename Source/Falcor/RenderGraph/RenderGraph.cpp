/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include "stdafx.h"
#include "RenderGraph.h"
#include "RenderPassLibrary.h"
#include "Utils/Algorithm/DirectedGraphTraversal.h"
#include "RenderGraphCompiler.h"

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
        if (gpFramework == nullptr) throw std::exception("Can't construct RenderGraph - framework is not initialized");
        mpGraph = DirectedGraph::create();
        mpPassDictionary = InternalDictionary::create();
        gRenderGraphs.push_back(this);
        onResize(gpFramework->getTargetFbo().get());
    }

    RenderGraph::~RenderGraph()
    {
        auto it = std::find(gRenderGraphs.begin(), gRenderGraphs.end(), this);
        assert(it != gRenderGraphs.end());
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
        assert(pPass);
        uint32_t passIndex = getPassIndex(passName);
        if (passIndex != kInvalidIndex)
        {
            logError("Pass named '" + passName + "' already exists. Ignoring call");
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
            logWarning("Can't remove pass '" + name + "'. Pass doesn't exist");
            return;
        }

        // Unmark graph outputs that belong to this pass
        // Because the way std::vector works, we can't call unmarkOutput() immediately, so we store the outputs in a vector
        std::vector<std::string> outputsToDelete;
        const std::string& outputPrefix = name + '.';
        for(auto& o : mOutputs)
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

    void RenderGraph::updatePass(RenderContext* pRenderContext, const std::string& passName, const Dictionary& dict)
    {
        uint32_t index = getPassIndex(passName);
        const auto pPassIt = mNodeData.find(index);

        if (pPassIt == mNodeData.end())
        {
            logError("Error in RenderGraph::updatePass(). Unable to find pass " + passName);
            return;
        }

        // Recreate pass without changing graph using new dictionary
        auto pOldPass = pPassIt->second.pPass;
        std::string passTypeName = getClassTypeName(pOldPass.get());
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
            logError("RenderGraph::getRenderPass() - can't find a pass named '" + name + "'");
            return pNull;
        }
        return mNodeData.at(index).pPass;
    }

    using str_pair = std::pair<std::string, std::string>;

    template<bool input>
    static bool checkRenderPassIoExist(RenderPass* pPass, const std::string& name)
    {
        RenderPassReflection reflect = pPass->reflect({});
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

    template<bool input>
    static RenderPass* getRenderPassAndNamePair(const RenderGraph* pGraph, const std::string& fullname, const std::string& errorPrefix, str_pair& nameAndField)
    {
        nameAndField = parseFieldName(fullname);

        RenderPass* pPass = pGraph->getPass(nameAndField.first).get();
        if (!pPass)
        {
            logError(errorPrefix + " - can't find render-pass named '" + nameAndField.first + "'");
            return nullptr;
        }

        if (nameAndField.second.size() && checkRenderPassIoExist<input>(pPass, nameAndField.second) == false)
        {
            logError(errorPrefix + "- can't find field named '" + nameAndField.second + "' in render-pass '" + nameAndField.first + "'");
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
        const auto& pSrc = getRenderPassAndNamePair<false>(this, src, "Invalid src string in RenderGraph::addEdge()", srcPair);
        const auto& pDst = getRenderPassAndNamePair<true>(this, dst, "Invalid dst string in RenderGraph::addEdge()", dstPair);
        newEdge.srcField = srcPair.second;
        newEdge.dstField = dstPair.second;

        if (pSrc == nullptr || pDst == nullptr) return kInvalidIndex;
        if (checkMatchingEdgeTypes(newEdge.srcField, newEdge.dstField) == false)
        {
            logError("RenderGraph::addEdge() - can't add the edge [" + src + ", " + dst + "]. One of the nodes is a resource while the other is a pass. Can't tell if you want a data-dependency or an execution-dependency");
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
                    if (edgeData.autoGenerated)
                    {
                        removeEdge(incomingEdgeId);
                        break;
                    }
                    else
                    {
                        logError("RenderGraph::addEdge() - destination '" + dst + "' is already initialized. Please remove the existing connection before trying to add an edge");
                        return kInvalidIndex;
                    }
                }
            }
        }

        // Make sure that this doesn't create a cycle
        if (DirectedGraphPathDetector::hasPath(mpGraph, dstIndex, srcIndex))
        {
            logError("RenderGraph::addEdge() - can't add the edge [" + src + ", " + dst + "]. The edge will create a cycle in the graph which is not allowed");
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
        const auto& pSrc = getRenderPassAndNamePair<false>(this, src, "Invalid src string in RenderGraph::addEdge()", srcPair);
        const auto& pDst = getRenderPassAndNamePair<true>(this, dst, "Invalid dst string in RenderGraph::addEdge()", dstPair);

        if (pSrc == nullptr || pDst == nullptr)
        {
            logError("Unable to remove edge. Input or output node not found.");
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
            logError("Can't remove edge with index " + std::to_string(edgeID) + ". The edge doesn't exist");
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

        for (const auto& node : mNodeData)
        {
            RenderPassReflection reflection = node.second.pPass->reflect({});
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

    bool RenderGraph::compile(RenderContext* pContext, std::string& log)
    {
        if (!mRecompile) return true;
        mpExe = nullptr;

        try
        {
            mpExe = RenderGraphCompiler::compile(*this, pContext, mCompilerDeps);
            mRecompile = false;
            return true;
        }
        catch (const std::exception& e)
        {
            log = e.what();
            return false;
        }
    }

    void RenderGraph::execute(RenderContext* pContext)
    {
        std::string log;
        if (!compile(pContext, log))
        {
            logError("Failed to compile RenderGraph\n" + log + "Ignoring RenderGraph::execute() call");
            return;
        }

        assert(mpExe);
        RenderGraphExe::Context c;
        c.pGraphDictionary = mpPassDictionary;
        c.pRenderContext = pContext;
        c.defaultTexDims = mCompilerDeps.defaultResourceProps.dims;
        c.defaultTexFormat = mCompilerDeps.defaultResourceProps.format;
        mpExe->execute(c);
    }

    void RenderGraph::update(const SharedPtr& pGraph)
    {
        // fill in missing passes from referenced graph
        for (const auto& nameIndexPair : pGraph->mNameToIndex)
        {
            // if same name and type
            RenderPass::SharedPtr pRenderPass = pGraph->mNodeData[nameIndexPair.second].pPass;
            if (!doesPassExist(nameIndexPair.first)) addPass(pRenderPass, nameIndexPair.first);
        }

        std::vector<std::string> passesToRemove;

        // remove nodes that should no longer be within the graph
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

        // remove all edges from this graph
        for (uint32_t i = 0; i < mpGraph->getCurrentEdgeId(); ++i)
        {
            if (!mpGraph->doesEdgeExist(i)) { continue; }

            mpGraph->removeEdge(i);
        }
        mEdgeData.clear();

        // add all edges from the other graph
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

        // mark all unmarked outputs from referenced graph
        for (uint32_t i = 0; i < pGraph->getOutputCount(); ++i) { markOutput(pGraph->getOutputName(i)); }
    }

    void RenderGraph::setInput(const std::string& name, const Resource::SharedPtr& pResource)
    {
        str_pair strPair;
        RenderPass* pPass = getRenderPassAndNamePair<true>(this, name, "RenderGraph::setInput()", strPair);
        if (pPass == nullptr) return;

        if (pResource) mCompilerDeps.externalResources[name] = pResource;
        else
        {
            if (mCompilerDeps.externalResources.find(name) == mCompilerDeps.externalResources.end())
            {
                logWarning("RenderGraph::setInput() - Trying to remove an external resource named '" + name + "' but the resource wasn't registered before. Ignoring call");
                return;
            }
            mCompilerDeps.externalResources.erase(name);
        }

        if (mpExe) mpExe->setInput(name, pResource);
    }

    void RenderGraph::markOutput(const std::string& name)
    {
        if (name == "*")
        {
            auto outputs = getAvailableOutputs();
            for (const auto& o : outputs) markOutput(o);
            return;
        }

        str_pair strPair;
        const auto& pPass = getRenderPassAndNamePair<false>(this, name, "RenderGraph::markGraphOutput()", strPair);
        if (pPass == nullptr) return;

        GraphOut newOut;
        newOut.field = strPair.second;
        newOut.nodeId = mNameToIndex[strPair.first];

        // Check that this is not already marked
        for (const auto& o : mOutputs)
        {
            if (newOut.nodeId == o.nodeId && newOut.field == o.field) return;
        }

        mOutputs.push_back(newOut);
        mRecompile = true;
    }

    void RenderGraph::unmarkOutput(const std::string& name)
    {
        str_pair strPair;
        const auto& pPass = getRenderPassAndNamePair<false>(this, name, "RenderGraph::unmarkGraphOutput()", strPair);
        if (pPass == nullptr) return;

        GraphOut removeMe;
        removeMe.field = strPair.second;
        removeMe.nodeId = mNameToIndex[strPair.first];

        for (size_t i = 0; i < mOutputs.size(); i++)
        {
            if (mOutputs[i].nodeId == removeMe.nodeId && mOutputs[i].field == removeMe.field)
            {
                mOutputs.erase(mOutputs.begin() + i);
                mRecompile = true;
                return;
            }
        }
    }

    bool RenderGraph::isGraphOutput(const std::string& name) const
    {
        str_pair strPair;
        const auto& pPass = getRenderPassAndNamePair<false>(this, name, "RenderGraph::unmarkGraphOutput()", strPair);
        if (pPass == nullptr) return false;
        uint32_t passIndex = getPassIndex(strPair.first);
        GraphOut thisOutput = { passIndex, strPair.second };
        return isGraphOutput(thisOutput);
    }

    Resource::SharedPtr RenderGraph::getOutput(const std::string& name)
    {
        if (mRecompile)
        {
            logError("RenderGraph::getOutput() - can't fetch an output resource because the graph wasn't successfuly compiled yet");
            return nullptr;
        }

        str_pair strPair;
        RenderPass* pPass = getRenderPassAndNamePair<false>(this, name, "RenderGraph::getOutput()", strPair);
        if (!pPass) return nullptr;

        uint32_t passIndex = getPassIndex(strPair.first);
        GraphOut thisOutput = { passIndex, strPair.second };
        bool isOutput = isGraphOutput(thisOutput);
        if (!isOutput)
        {
            logError("RenderGraph::getOutput() - can't fetch the output '" + name + "'. The resource is wasn't marked as an output");
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
        assert(index < mOutputs.size());
        const GraphOut& graphOut = mOutputs[index];
        return mNodeData.find(graphOut.nodeId)->second.name + "." + graphOut.field;
    }

    void RenderGraph::onResize(const Fbo* pTargetFbo)
    {
        // Store the back-buffer values
        const Texture* pColor = pTargetFbo ? pTargetFbo->getColorTexture(0).get() : nullptr;
        if (pColor == nullptr) throw std::exception("Can't resize render graph without a frame buffer.");

        // Store the values
        mCompilerDeps.defaultResourceProps.format = pColor->getFormat();
        mCompilerDeps.defaultResourceProps.dims = { pTargetFbo->getWidth(), pTargetFbo->getHeight() };

        // Invalidate the graph. Render-passes might change their reflection based on the resize information
        mRecompile = true;
    }

    bool canFieldsConnect(const RenderPassReflection::Field& src, const RenderPassReflection::Field& dst)
    {
        assert(is_set(src.getVisibility(), RenderPassReflection::Field::Visibility::Output) && is_set(dst.getVisibility(), RenderPassReflection::Field::Visibility::Input));

        return src.getName() == dst.getName() &&
            (dst.getWidth() == 0 || src.getWidth() == dst.getWidth()) &&
            (dst.getHeight() == 0 || src.getHeight() == dst.getHeight()) &&
            (dst.getDepth() == 0 || src.getDepth() == dst.getDepth()) &&
            (dst.getFormat() == ResourceFormat::Unknown || src.getFormat() == dst.getFormat()) &&
            src.getSampleCount() == dst.getSampleCount() && // TODO: allow dst sample count to be 1 when auto MSAA resolve is implemented in graph compilation
            src.getType() == dst.getType() &&
            src.getSampleCount() == dst.getSampleCount();
    }

    // Given a pair of src and dst RenderPass data, check if any src outputs can fulfill unsatisfied dst inputs
    void RenderGraph::autoConnectPasses(const NodeData* pSrcNode, const RenderPassReflection& srcReflection, const NodeData* pdstNode, std::vector<RenderPassReflection::Field>& unsatisfiedInputs)
    {
        // For every unsatisfied input in dst pass
        auto dstFieldIt = unsatisfiedInputs.begin();
        while (dstFieldIt != unsatisfiedInputs.end())
        {
            bool inputSatisfied = false;

            // For every output in src pass
            for (uint32_t i = 0; i < srcReflection.getFieldCount(); i++)
            {
                const RenderPassReflection::Field& srcField = *srcReflection.getField(i);
                if (is_set(srcField.getVisibility(), RenderPassReflection::Field::Visibility::Output) && canFieldsConnect(srcField, *dstFieldIt))
                {
                    // Add Edge
                    uint32_t srcIndex = mNameToIndex[pSrcNode->name];
                    uint32_t dstIndex = mNameToIndex[pdstNode->name];

                    uint32_t e = mpGraph->addEdge(srcIndex, dstIndex);
                    mEdgeData[e] = { true, srcField.getName(), dstFieldIt->getName() };
                    mRecompile = true;
                    inputSatisfied = true; // If connection was found, continue to next unsatisfied input
                    break;
                }
            }

            // If input was satisfied, remove from unsatisfied list, else increment iterator
            dstFieldIt = inputSatisfied ? unsatisfiedInputs.erase(dstFieldIt) : dstFieldIt + 1;
        }
    }

    void RenderGraph::getUnsatisfiedInputs(const NodeData* pNodeData, const RenderPassReflection& passReflection, std::vector<RenderPassReflection::Field>& outList) const
    {
        assert(mNameToIndex.count(pNodeData->name) > 0);

        // Get names of connected input edges
        std::vector<std::string> satisfiedFields;
        const DirectedGraph::Node* pNode = mpGraph->getNode(mNameToIndex.at(pNodeData->name));
        for (uint32_t i = 0; i < pNode->getIncomingEdgeCount(); i++)
        {
            const auto& edgeData = mEdgeData.at(pNode->getIncomingEdge(i));
            satisfiedFields.push_back(edgeData.dstField);
        }

        // Build list of unsatisfied fields by comparing names with which edges/fields are connected
        for (uint32_t i = 0; i < passReflection.getFieldCount(); i++)
        {
            const RenderPassReflection::Field& field = *passReflection.getField(i);

            bool isUnsatisfied = std::find(satisfiedFields.begin(), satisfiedFields.end(), field.getName()) == satisfiedFields.end();
            if (is_set(field.getVisibility(), RenderPassReflection::Field::Visibility::Input) && isUnsatisfied)
            {
                outList.push_back(*passReflection.getField(i));
            }
        }
    }

    void RenderGraph::autoGenEdges(const std::vector<uint32_t>& executionOrder)
    {
        // Remove all previously auto-generated edges
        auto it = mEdgeData.begin();
        while (it != mEdgeData.end())
        {
            if (it->second.autoGenerated)
            {
                removeEdge(it->first);
            }
            else it++;
        }

        // Gather list of passes by order they were added
        std::vector<NodeData*> nodeVec;
        std::unordered_map<RenderPass*, RenderPassReflection> passReflectionMap;

        if (executionOrder.size() > 0)
        {
            assert(executionOrder.size() == mNodeData.size());
            for (const uint32_t& nodeId : executionOrder)
            {
                nodeVec.push_back(&mNodeData[nodeId]);
            }
        }
        else
        {
            for (uint32_t i = 0; i < mpGraph->getCurrentNodeId(); i++)
            {
                if (mpGraph->doesNodeExist(i))
                {
                    nodeVec.push_back(&mNodeData[i]);
                }
            }
        }

        // For all nodes, starting at end, iterate until index 1 of vector
        for (size_t dst = nodeVec.size() - 1; dst > 0; dst--)
        {
            std::vector<RenderPassReflection::Field> unsatisfiedInputs;
            getUnsatisfiedInputs(nodeVec[dst], passReflectionMap[nodeVec[dst]->pPass.get()], unsatisfiedInputs);

            // Find outputs to connect.
            // Start one before i, iterate until the beginning of vector
            for (size_t src = dst - 1; src != size_t(-1) && unsatisfiedInputs.size() > 0; src--)
            {
                // While there are unsatisfied inputs, keep searching for passes with outputs that can connect
                autoConnectPasses(nodeVec[src], passReflectionMap[nodeVec[src]->pPass.get()], nodeVec[dst], unsatisfiedInputs);
            }
        }
    }

    void RenderGraph::renderUI(Gui::Widgets& widget)
    {
        if (mpScene)
        {
            if (auto sceneGroup = widget.group("Scene Settings"))
            {
                mpScene->renderUI(sceneGroup);
            }

            widget.separator();
        }

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

    SCRIPT_BINDING(RenderGraph)
    {
        pybind11::class_<RenderGraph, RenderGraph::SharedPtr> renderGraph(m, "RenderGraph");
        renderGraph.def(pybind11::init(&RenderGraph::create));
        renderGraph.def_property("name", &RenderGraph::getName, &RenderGraph::setName);
        renderGraph.def(RenderGraphIR::kAddPass, &RenderGraph::addPass, "pass"_a, "name"_a);
        renderGraph.def(RenderGraphIR::kRemovePass, &RenderGraph::removePass, "name"_a);
        renderGraph.def(RenderGraphIR::kAddEdge, &RenderGraph::addEdge, "src"_a, "dst"_a);
        renderGraph.def(RenderGraphIR::kRemoveEdge, pybind11::overload_cast<const std::string&, const std::string&>(&RenderGraph::removeEdge), "src"_a, "src"_a);
        renderGraph.def(RenderGraphIR::kMarkOutput, &RenderGraph::markOutput, "name"_a);
        renderGraph.def(RenderGraphIR::kUnmarkOutput, &RenderGraph::unmarkOutput, "name"_a);
        renderGraph.def(RenderGraphIR::kAutoGenEdges, &RenderGraph::autoGenEdges, "executionOrder"_a);
        renderGraph.def("getPass", &RenderGraph::getPass, "name"_a);
        renderGraph.def("getOutput", pybind11::overload_cast<const std::string&>(&RenderGraph::getOutput), "name"_a);
        auto printGraph = [](RenderGraph::SharedPtr pGraph) { pybind11::print(RenderGraphExporter::getIR(pGraph)); };
        renderGraph.def("print", printGraph);

        // RenderPass
        pybind11::class_<RenderPass, RenderPass::SharedPtr> renderPass(m, "RenderPass");

        // RenderPassLibrary
        const auto& createRenderPass = [](const std::string& passName, pybind11::dict d = {})
        {
            auto pPass = RenderPassLibrary::instance().createPass(gpDevice->getRenderContext(), passName.c_str(), Dictionary(d));
            if (!pPass) throw std::exception(("Can't create a render pass named '" + passName + "'. Make sure the required DLL was loaded.").c_str());
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
