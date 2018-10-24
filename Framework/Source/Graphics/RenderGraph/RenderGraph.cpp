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
#include "Framework.h"
#include "RenderGraph.h"
#include "API/FBO.h"
#include "Utils/DirectedGraphTraversal.h"
#include "Utils/Gui.h"
#include "Graphics/RenderGraph/RenderPassLibrary.h"
#include "RenderPasses/ResolvePass.h"

namespace Falcor
{
    RenderGraph::SharedPtr RenderGraph::create(const std::string& name)
    {
        try
        {
            return SharedPtr(new RenderGraph(name));
        }
        catch (const std::exception&)
        {
            return nullptr;
        }
    }

    RenderGraph::RenderGraph(const std::string& name) : mName(name)
    {
        mpGraph = DirectedGraph::create();
        mpResourcesCache = ResourceCache::create();
        mpPassDictionary = Dictionary::create();
        gRenderGraphs.push_back(this);
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

    void RenderGraph::setScene(const std::shared_ptr<Scene>& pScene)
    {
        mpScene = pScene;
        for (auto& it : mNodeData)
        {
            it.second.pPass->setScene(pScene);
        }
    }

    uint32_t RenderGraph::addPass(const RenderPass::SharedPtr& pPass, const std::string& passName)
    {
        assert(pPass);
        uint32_t passIndex = getPassIndex(passName);
        if (passIndex != kInvalidIndex)
        {
            logWarning("Pass named `" + passName + "' already exists. Replacing existing pass");
        }
        else
        {
            passIndex = mpGraph->addNode();
            mNameToIndex[passName] = passIndex;
        }

        auto passChangedCB = [this]() {mRecompile = true; };
        pPass->setPassChangedCB(passChangedCB);
        pPass->setScene(mpScene);
        mNodeData[passIndex] = { passName, pPass };
        mRecompile = true;
        return passIndex;
    }

    void RenderGraph::removePass(const std::string& name)
    {
        uint32_t index = getPassIndex(name);
        if (index == kInvalidIndex)
        {
            logWarning("Can't remove pass `" + name + "`. Pass doesn't exist");
            return;
        }

        // Unmark graph outputs that belong to this pass
        // Because the way std::vector works, we can't call umarkOutput() immediatly, so we store the outputs in a vector
        std::vector<std::string> outputsToDelete;
        const std::string& outputPrefix = name + '.';
        for(auto& o : mOutputs)
        {
            if (o.nodeId == index) outputsToDelete.push_back(outputPrefix + o.field);
        }
        
        for (const auto& name : outputsToDelete)
        {
            unmarkOutput(name);
        }

        // Update the indices
        mNameToIndex.erase(name);

        // remove pass data
        mNodeData.erase(index);
        
        // Remove all the edges associated with this pass
        const auto& removedEdges = mpGraph->removeNode(index);
        for (const auto& e : removedEdges) mEdgeData.erase(e);

        mRecompile = true;
    }

    void RenderGraph::updatePass(const std::string& passName, const Dictionary& dict)
    {
        uint32_t index = getPassIndex(passName);
        const auto pPassIt = mNodeData.find(index);

        if (pPassIt == mNodeData.end())
        {
            logWarning("Unable to find pass " + passName + ".");
        }

        // recreate pass without changing graph using new dictionary
        auto pOldPass = pPassIt->second.pPass;
        std::string passTypeName = pOldPass->getName();
        auto pPass = RenderPassLibrary::instance().createPass(passTypeName.c_str(), dict);
        pPassIt->second.pPass = pPass;

        auto passChangedCB = [this]() {mRecompile = true; };
        pPass->setPassChangedCB(passChangedCB);

        pPass->setScene(mpScene);
        mRecompile = true;
    }

    const RenderPass::SharedPtr& RenderGraph::getPass(const std::string& name) const
    {
        uint32_t index = getPassIndex(name);
        if (index == kInvalidIndex)
        {
            static RenderPass::SharedPtr pNull;
            logWarning("RenderGraph::getRenderPass() - can't find a pass named `" + name + "`");
            return pNull;
        }
        return mNodeData.at(index).pPass;
    }

    using str_pair = std::pair<std::string, std::string>;

    template<bool input>
    static bool checkRenderPassIoExist(const RenderPass* pPass, const std::string& name)
    {
        RenderPassReflection reflect = pPass->reflect();
        for (size_t i = 0; i < reflect.getFieldCount(); i++)
        {
            const auto& f = reflect.getField(i);
            if (f.getName() == name)
            {
                return input ? is_set(f.getType(), RenderPassReflection::Field::Type::Input) : is_set(f.getType(), RenderPassReflection::Field::Type::Output);
            }
        }

        return false;
    }

    static bool parseFieldName(const std::string& fullname, str_pair& strPair)
    {
        if (std::count(fullname.begin(), fullname.end(), '.') == 0)
        {
            logError("RenderGraph node field string is incorrect. Must be in the form of `PassName.FieldName` but got `" + fullname + "`", false);
            return false;
        }

        size_t dot = fullname.find_last_of('.');
        strPair.first = fullname.substr(0, dot);
        strPair.second = fullname.substr(dot + 1);
        return true;
    }

    template<bool input>
    static RenderPass* getRenderPassAndNamePair(const RenderGraph* pGraph, const std::string& fullname, const std::string& errorPrefix, str_pair& nameAndField)
    {
        if (parseFieldName(fullname, nameAndField) == false) return nullptr;

        RenderPass* pPass = pGraph->getPass(nameAndField.first).get();
        if (!pPass)
        {
            logError(errorPrefix + " - can't find render-pass named '" + nameAndField.first + "'");
            return nullptr;
        }

        if (checkRenderPassIoExist<input>(pPass, nameAndField.second) == false)
        {
            logError(errorPrefix + "- can't find field named `" + nameAndField.second + "` in render-pass `" + nameAndField.first + "`");
            return nullptr;
        }
        return pPass;
    }

    uint32_t RenderGraph::addEdge(const std::string& src, const std::string& dst)
    {
        EdgeData newEdge;
        str_pair srcPair, dstPair;
        const auto& pSrc = getRenderPassAndNamePair<false>(this, src, "Invalid src string in RenderGraph::addEdge()", srcPair);
        const auto& pDst = getRenderPassAndNamePair<true>(this, dst, "Invalid dst string in RenderGraph::addEdge()", dstPair);
        newEdge.srcField = srcPair.second;
        newEdge.dstField = dstPair.second;

        if (pSrc == nullptr || pDst == nullptr) return false;
        uint32_t srcIndex = mNameToIndex[srcPair.first];
        uint32_t dstIndex = mNameToIndex[dstPair.first];

        // Check that the dst field is not already initialized
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
                    logError("RenderGraph::addEdge() - destination `" + dst + "` is already initialized. Please remove the existing connection before trying to add an edge");
                    return kInvalidIndex;
                }
            }
        }

        // Make sure that this doesn't create a cycle
        if (DirectedGraphPathDetector::hasPath(mpGraph, dstIndex, srcIndex))
        {
            logError("RenderGraph::addEdge() - can't add the edge [" + src + ", " + dst + "]. This will create a cycle in the graph which is not allowed");
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
            logWarning("Unable to remove edge. Input or output node not found.");
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
        mEdgeData.erase(edgeID);
        mpGraph->removeEdge(edgeID);
        mRecompile = true;
    }

    uint32_t RenderGraph::getEdge(const std::string& src, const std::string& dst)
    {
        str_pair srcPair, dstPair;
        parseFieldName(src, srcPair);
        parseFieldName(dst, dstPair);

        for (uint32_t i = 0; i < mpGraph->getCurrentEdgeId(); ++i)
        {
            if (!mpGraph->doesEdgeExist(i)) { continue; }

            const DirectedGraph::Edge* pEdge = mpGraph->getEdge(i);
            if (dstPair.first == mNodeData[pEdge->getDestNode()].nodeName &&
                srcPair.first == mNodeData[pEdge->getSourceNode()].nodeName)
            {
                if (mEdgeData[i].dstField == dstPair.second
                    && mEdgeData[i].srcField == srcPair.second)
                {
                    return i;
                }
            }
        }

        return static_cast<uint32_t>(-1);
    }

    // MATT TODO
    bool RenderGraph::isValid(std::string& log) const
    {
        std::vector<const NodeData*> nodeVec;
        
        // If there are no marked graph outputs, is not valid
        if (!mOutputs.size()) return false;

        for (uint32_t i = 0; i < mpGraph->getCurrentNodeId(); i++)
        {
            if (mpGraph->doesNodeExist(i))
            {
                const auto& nodeIt = mNodeData.find(i);
                nodeVec.push_back(&nodeIt->second);
            }
        }
        
        for (const NodeData* pNodeData : nodeVec)
        {
            RenderPassReflection passReflection = pNodeData->pPass->reflect();

            if (is_set(passReflection.getFlags(), RenderPassReflection::Flags::ForceExecution)) continue;

            uint32_t numRequiredInputs = 0;
            bool hasGraphOutput = false;
            const DirectedGraph::Node* pNode = mpGraph->getNode(mNameToIndex.at(pNodeData->nodeName));
            
            // get input count
            for (uint32_t i = 0; i < passReflection.getFieldCount(); ++i)
            {
                const RenderPassReflection::Field& field = passReflection.getField(i);

                if (is_set(field.getType(), RenderPassReflection::Field::Type::Input))
                {
                    if (!is_set(field.getFlags(), RenderPassReflection::Field::Flags::Optional)) numRequiredInputs++;
                }

                GraphOut graphOut;
                graphOut.field = field.getName();
                graphOut.nodeId = getPassIndex(pNodeData->nodeName);
                hasGraphOutput |= isGraphOutput(graphOut);
            }

            // check if node has no inputs, and has connected outgoing edges
            bool hasOutputs = (pNode->getOutgoingEdgeCount() || hasGraphOutput);
            if (hasOutputs && (numRequiredInputs > pNode->getIncomingEdgeCount()) )
            {
                return false;
            }
        }

        return true;
    }

    bool RenderGraph::resolveExecutionOrder()
    {
        mExecutionList.clear();

        // Find all passes that affect the outputs
        std::unordered_set<uint32_t> participatingPasses;
        for (auto& o : mOutputs)
        {
            uint32_t nodeId = o.nodeId;
            auto dfs = DirectedGraphDfsTraversal(mpGraph, nodeId, DirectedGraphDfsTraversal::Flags::IgnoreVisited | DirectedGraphDfsTraversal::Flags::Reverse);
            while (nodeId != DirectedGraph::kInvalidID)
            {
                participatingPasses.insert(nodeId);
                nodeId = dfs.traverse();
            }
        }

        // Run topological sort
        auto topologicalSort = DirectedGraphTopologicalSort::sort(mpGraph.get());

        // For each object in the vector, if it's being used in the execution, put it in the list
        for (auto& node : topologicalSort)
        {
            if (participatingPasses.find(node) != participatingPasses.end())
            {
                mExecutionList.push_back(node);
            }
        }

        return true;
    }

    bool RenderGraph::insertAutoPasses()
    {
        bool addedPasses = false;
        for (size_t i = 0; i < mExecutionList.size(); i++)
        {
            uint32_t nodeIndex = mExecutionList[i];
            const DirectedGraph::Node* pNode = mpGraph->getNode(nodeIndex);
            assert(pNode);
            RenderPass* pSrcPass = mNodeData[nodeIndex].pPass.get();
            RenderPassReflection passReflection = pSrcPass->reflect();

            // Check for opportunities to automatically resolve MSAA
            // - Only take explicitly specified MS output
            for (uint32_t f = 0; f < passReflection.getFieldCount(); f++)
            {
                // Iterate over output fields
                auto& srcField = passReflection.getField(f);
                if (is_set(srcField.getType(), RenderPassReflection::Field::Type::Output) == false) continue;

                const std::string& srcPassName = mNodeData[nodeIndex].nodeName;
                // Gather src field name, and every input it is connected to
                std::string srcFieldName = srcPassName + '.' + srcField.getName();
                std::vector<std::string> dstFieldNames;

                for (uint32_t e = 0; e < pNode->getOutgoingEdgeCount(); e++)
                {
                    uint32_t edgeIndex = pNode->getOutgoingEdge(e);
                    const auto& edgeData = mEdgeData[edgeIndex];

                    // For every output field, iterate over all edges extending from that field
                    if (srcField.getName() == edgeData.srcField)
                    {
                        const auto& pEdge = mpGraph->getEdge(edgeIndex);
                        const std::string& dstPassName = mNodeData[pEdge->getDestNode()].nodeName;

                        // If edge is connected to something that isn't executed, ignore
                        if (std::find(mExecutionList.begin(), mExecutionList.end(), pEdge->getDestNode()) == mExecutionList.end()) continue;

                        RenderPassReflection dstReflection = mNodeData[pEdge->getDestNode()].pPass->reflect();
                        const auto& dstField = dstReflection.getField(edgeData.dstField, RenderPassReflection::Field::Type::Input);

                        assert(srcField.isValid() && dstField.isValid());
                        if (canAutoResolve(srcField, dstField))
                        {
                            std::string dstFieldName = dstPassName + '.' + dstField.getName();
                            dstFieldNames.push_back(dstFieldName);
                        }
                    }
                }

                // If there are connections to add MSAA Resolve
                if (dstFieldNames.size() > 0)
                {
                    // One resolve pass is made for every output that requires it
                    auto pResolvePass = std::static_pointer_cast<ResolvePass>(RenderPassLibrary::instance().createPass("ResolvePass"));
                    pResolvePass->setFormat(srcField.getFormat()); // Match input texture format

                    // Create pass and attach src to it
                    std::string resolvePassName = srcFieldName + "-ResolvePass";
                    addPass(pResolvePass, resolvePassName);
                    addEdge(srcFieldName, resolvePassName + ".src");

                    // For every input the src field is connected to, connect the resolve pass output to the input
                    for (const auto& dstFieldName : dstFieldNames)
                    {
                        // Remove original edge
                        removeEdge(srcFieldName, dstFieldName);
                        // Replace with edge coming from resolve output
                        addEdge(resolvePassName + ".dst", dstFieldName);

                        // Log changes made to user's graph by compilation process
                        mCompilationChanges.removedEdges.emplace_back(srcFieldName, dstFieldName);
                    }

                    mCompilationChanges.generatedPasses.push_back(resolvePassName);
                    addedPasses = true;
                }
            }
        }

        return addedPasses;
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
            RenderPassReflection reflection = node.second.pPass->reflect();
            for (size_t i = 0; i < reflection.getFieldCount(); i++)
            {
                const auto& f = reflection.getField(i);
                if(is_set(f.getType(), RenderPassReflection::Field::Type::Output)) outputs.push_back(node.second.nodeName + "." + f.getName());
            }
        }
        return outputs;
    }

    bool RenderGraph::resolveResourceTypes()
    {
        // Build list to look up execution order index from the pass
        std::unordered_map<RenderPass*, uint32_t> passToIndex;
        for (size_t i = 0; i < mExecutionList.size(); i++)
        {
            passToIndex.emplace(mNodeData[mExecutionList[i]].pPass.get(), uint32_t(i));
        }

        for (size_t i = 0; i < mExecutionList.size(); i++)
        {
            uint32_t nodeIndex = mExecutionList[i];
            const DirectedGraph::Node* pNode = mpGraph->getNode(nodeIndex);
            assert(pNode);
            RenderPass* pCurrPass = mNodeData[nodeIndex].pPass.get();
            RenderPassReflection passReflection = pCurrPass->reflect();

            const auto isGraphOutput = [=](uint32_t nodeId, const std::string& field)
            {
                for (const auto& out : mOutputs)
                {
                    if (out.nodeId == nodeId && out.field == field) return true;
                }
                return false;
            };

            // Register all pass outputs
            for (size_t f = 0; f < passReflection.getFieldCount(); f++)
            {
                const auto& field = passReflection.getField(f);
                std::string fullFieldName = mNodeData[nodeIndex].nodeName + '.' + field.getName();

                // Skip input resources, we never allocate them
                if (is_set(field.getType(), RenderPassReflection::Field::Type::Output | RenderPassReflection::Field::Type::Internal))
                {
                    mpResourcesCache->registerField(fullFieldName, field, uint32_t(i));
                    
                    // Resource lifetime for graph outputs must extend to end of graph execution
                    if(isGraphOutput(nodeIndex, field.getName())) mpResourcesCache->registerField(fullFieldName, field, uint32_t(-1));
                }
            }

            // Go over the pass inputs, add them as aliases to the outputs that connect to them (which should be already registered above)
            for (uint32_t e = 0; e < pNode->getIncomingEdgeCount(); e++)
            {
                uint32_t edgeIndex = pNode->getIncomingEdge(e);
                const auto& pEdge = mpGraph->getEdge(edgeIndex);
                const auto& edgeData = mEdgeData[edgeIndex];

                const auto& dstField = passReflection.getField(edgeData.dstField, RenderPassReflection::Field::Type::Input);
                assert(dstField.isValid());

                // Merge dst/input field into same resource data
                std::string srcFieldName = mNodeData[pEdge->getSourceNode()].nodeName + '.' + edgeData.srcField;
                std::string dstFieldName = mNodeData[nodeIndex].nodeName + '.' + dstField.getName();

                const auto& pSrcPass = mNodeData[pEdge->getSourceNode()].pPass;
                auto srcReflection = pSrcPass->reflect();
                const RenderPassReflection::Field& srcField = srcReflection.getField(edgeData.srcField);

                assert(passToIndex.count(pSrcPass.get()) > 0);
                mpResourcesCache->registerField(dstFieldName, srcField, passToIndex[pSrcPass.get()], srcFieldName);
            }
        }

        mpResourcesCache->allocateResources(mSwapChainData);
        return true;
    }

    bool RenderGraph::compile(std::string& log)
    {
        if (mRecompile)
        {
            mpResourcesCache->reset();
            restoreCompilationChanges();

            if (resolveExecutionOrder() == false) return false;
            // If passes were added, resolve execution order again
            if (insertAutoPasses()) if (resolveExecutionOrder() == false) return false;
            if (resolveResourceTypes() == false) return false;
            if (isValid(log) == false) return false;
        }
        mRecompile = false;
        return true;
    }

    void RenderGraph::execute(RenderContext* pContext)
    {
        bool profile = mProfileGraph && gProfileEnabled;

        if (profile) Profiler::startEvent("RenderGraph::execute()");

        std::string log;
        if (!compile(log))
        {
            logWarning("Failed to compile RenderGraph\n" + log + "Ignoring RenderGraph::execute() call");
            return;
        }

        for (const auto& node : mExecutionList)
        {
            if (profile) Profiler::startEvent(mNodeData[node].nodeName);
            RenderData renderData(mNodeData[node].nodeName, mpResourcesCache, mpPassDictionary);
            mNodeData[node].pPass->execute(pContext, &renderData);
            if (profile) Profiler::endEvent(mNodeData[node].nodeName);
        }

        if (profile) Profiler::endEvent("RenderGraph::execute()");
    }

    void RenderGraph::update(const SharedPtr& pGraph)
    {
        // fill in missing passes from referenced graph
        for (const auto& nameIndexPair : pGraph->mNameToIndex)
        {
            // if same name and type
            RenderPass::SharedPtr pRenderPass = pGraph->mNodeData[nameIndexPair.second].pPass;
            std::string passTypeName = pRenderPass->getName();

            if (!doesPassExist(nameIndexPair.first))
            { 
                addPass(pRenderPass, nameIndexPair.first);
            }
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
            std::string dst = pGraph->mNodeData.find(pEdge->getDestNode())->second.nodeName;
            std::string src = pGraph->mNodeData.find(pEdge->getSourceNode())->second.nodeName;

            if ((mNameToIndex.find(src) != mNameToIndex.end()) && (mNameToIndex.find(dst) != mNameToIndex.end()))
            {
                dst += std::string(".") + pGraph->mEdgeData[i].dstField;
                src += std::string(".") + pGraph->mEdgeData[i].srcField;
                addEdge(src, dst);
            }
        }
    }

    bool RenderGraph::setInput(const std::string& name, const std::shared_ptr<Resource>& pResource)
    {
        str_pair strPair;
        RenderPass* pPass = getRenderPassAndNamePair<true>(this, name, "RenderGraph::setInput()", strPair);
        if (pPass == nullptr) return false;
        mpResourcesCache->registerExternalInput(name, pResource);
        return true;
    }

    void RenderGraph::markOutput(const std::string& name)
    {
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

    const Resource::SharedPtr RenderGraph::getOutput(const std::string& name)
    {
        static const Resource::SharedPtr pNull;
        str_pair strPair;
        RenderPass* pPass = getRenderPassAndNamePair<false>(this, name, "RenderGraph::getOutput()", strPair);
        uint32_t passIndex = getPassIndex(strPair.first);
        GraphOut thisOutput = { passIndex, strPair.second };
        bool isOuput = isGraphOutput(thisOutput);

        return (pPass && isOuput) ? mpResourcesCache->getResource(name) : pNull;
    }

    std::string RenderGraph::getOutputName(size_t index) const
    {
        assert(index < mOutputs.size());
        const GraphOut& graphOut = mOutputs[index];
        return mNodeData.find(graphOut.nodeId)->second.nodeName + "." + graphOut.field;
    }

    void RenderGraph::onResize(const Fbo* pTargetFbo)
    {
        // Store the back-buffer values
        const Texture* pColor = pTargetFbo->getColorTexture(0).get();
        const Texture* pDepth = pTargetFbo->getDepthStencilTexture().get();
        assert(pColor && pDepth);

        // If the back-buffer values changed, recompile
        mRecompile = mRecompile || (mSwapChainData.format != pColor->getFormat());
        mRecompile = mRecompile || (mSwapChainData.width != pTargetFbo->getWidth());
        mRecompile = mRecompile || (mSwapChainData.height != pTargetFbo->getHeight());

        // Store the values
        mSwapChainData.format = pColor->getFormat();
        mSwapChainData.width = pTargetFbo->getWidth();
        mSwapChainData.height = pTargetFbo->getHeight();

        // Invoke the passes' callback
        for (const auto& it : mNodeData)
        {
            it.second.pPass->onResize(mSwapChainData.width, mSwapChainData.height);
        }

        // Invalidate the graph. Render-passes might change their reflection based on the resize information
        mRecompile = true;
    }

    bool canFieldsConnect(const RenderPassReflection::Field& src, const RenderPassReflection::Field& dst)
    {
        assert(is_set(src.getType(), RenderPassReflection::Field::Type::Output) && is_set(dst.getType(), RenderPassReflection::Field::Type::Input));
        
        return src.getName() == dst.getName() &&
            (dst.getWidth() == 0 || src.getWidth() == dst.getWidth()) &&
            (dst.getHeight() == 0 || src.getHeight() == dst.getHeight()) &&
            (dst.getDepth() == 0 || src.getDepth() == dst.getDepth()) &&
            (dst.getFormat() == ResourceFormat::Unknown || src.getFormat() == dst.getFormat()) &&
            src.getSampleCount() == dst.getSampleCount() && // TODO: allow dst sample count to be 1 when auto MSAA resolve is implemented in graph compilation
            src.getResourceType() == dst.getResourceType() &&
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
                const RenderPassReflection::Field& srcField = srcReflection.getField(i);
                if (is_set(srcField.getType(), RenderPassReflection::Field::Type::Output) && canFieldsConnect(srcField, *dstFieldIt))
                {
                    // Add Edge
                    uint32_t srcIndex = mNameToIndex[pSrcNode->nodeName];
                    uint32_t dstIndex = mNameToIndex[pdstNode->nodeName];

                    uint32_t e = mpGraph->addEdge(srcIndex, dstIndex);
                    mEdgeData[e] = { true, srcField.getName(), dstFieldIt->getName() };
                    mRecompile = true;

                    // If connection was found, continue to next unsatisfied input
                    inputSatisfied = true;
                    break;
                }
            }

            // If input was satisfied, remove from unsatisfied list, else increment iterator
            dstFieldIt = inputSatisfied ? unsatisfiedInputs.erase(dstFieldIt) : dstFieldIt + 1;
        }
    }

    bool RenderGraph::canAutoResolve(const RenderPassReflection::Field& src, const RenderPassReflection::Field& dst)
    {
        return src.getSampleCount() > 1 && dst.getSampleCount() == 1;
    }

    void RenderGraph::restoreCompilationChanges()
    {
        for (const auto& name : mCompilationChanges.generatedPasses) removePass(name);
        for (const auto& e : mCompilationChanges.removedEdges) addEdge(e.first, e.second);

        mCompilationChanges.generatedPasses.clear();
        mCompilationChanges.removedEdges.clear();
    }

    void RenderGraph::getUnsatisfiedInputs(const NodeData* pNodeData, const RenderPassReflection& passReflection, std::vector<RenderPassReflection::Field>& outList) const
    {
        assert(mNameToIndex.count(pNodeData->nodeName) > 0);

        // Get names of connected input edges
        std::vector<std::string> satisfiedFields;
        const DirectedGraph::Node* pNode = mpGraph->getNode(mNameToIndex.at(pNodeData->nodeName));
        for (uint32_t i = 0; i < pNode->getIncomingEdgeCount(); i++)
        {
            const auto& edgeData = mEdgeData.at(pNode->getIncomingEdge(i));
            satisfiedFields.push_back(edgeData.dstField);
        }

        // Build list of unsatisfied fields by comparing names with which edges/fields are connected
        for (uint32_t i = 0; i < passReflection.getFieldCount(); i++)
        {
            const RenderPassReflection::Field& field = passReflection.getField(i);

            bool isUnsatisfied = std::find(satisfiedFields.begin(), satisfiedFields.end(), field.getName()) == satisfiedFields.end();
            if (is_set(field.getType(), RenderPassReflection::Field::Type::Input) && isUnsatisfied)
            {
                outList.push_back(passReflection.getField(i));
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

    void RenderGraph::renderUI(Gui* pGui, const char* uiGroup)
    {
        if (!uiGroup || pGui->beginGroup(uiGroup, true))
        {
            pGui->addCheckBox("Profile Passes", mProfileGraph);
            pGui->addTooltip("Profile the render-passes. The results will be shown in the profiler window. If you can't see it, click 'P'");

            for (const auto& passId : mExecutionList)
            {
                const auto& pass = mNodeData[passId];

                // If you are thinking about displaying the profiler results next to the group label, it won't work. Since the times change every frame, IMGUI thinks it's a different group and will not expand it
                if (pGui->beginGroup(pass.nodeName))
                {
                    uint32_t w = (uint32_t)(mSwapChainData.width * 0.25f);
                    uint32_t h = (uint32_t)(mSwapChainData.height * 0.4f);
                    uint32_t y = 20;
                    uint32_t x = mSwapChainData.width - w - 20;

                    pGui->pushWindow(pass.nodeName.c_str(), w, h, x, y);

                    pass.pPass->renderUI(pGui, nullptr);
                    pGui->popWindow();
                    pGui->endGroup();
                }
            }

            if (uiGroup) pGui->endGroup();
        }
    }

    bool RenderGraph::onMouseEvent(const MouseEvent& mouseEvent)
    {
        bool b = false;
        for (const auto& passId : mExecutionList)
        {
            const auto& pPass = mNodeData[passId].pPass;
            b = b || pPass->onMouseEvent(mouseEvent);
        }
        return b;
    }

    bool RenderGraph::onKeyEvent(const KeyboardEvent& keyEvent)
    {
        bool b = false;
        for (const auto& passId : mExecutionList)
        {
            const auto& pPass = mNodeData[passId].pPass;
            b = b || pPass->onKeyEvent(keyEvent);
        }
        return b;
    }
}
