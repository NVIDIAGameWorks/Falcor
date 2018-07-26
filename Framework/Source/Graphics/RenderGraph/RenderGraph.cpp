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

namespace Falcor
{
    RenderGraph::SharedPtr RenderGraph::create()
    {
        try
        {
            return SharedPtr(new RenderGraph);
        }
        catch (const std::exception&)
        {
            return nullptr;
        }
    }

    RenderGraph::RenderGraph()
    {
        mpGraph = DirectedGraph::create();
        mpResourcesCache = ResourceCache::create();
    }

    uint32_t RenderGraph::getPassIndex(const std::string& name) const
    {
        auto& it = mNameToIndex.find(name);
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

    uint32_t RenderGraph::addRenderPass(const RenderPass::SharedPtr& pPass, const std::string& passName)
    {
        assert(pPass);
        if (getPassIndex(passName) != kInvalidIndex)
        {
            logWarning("Pass named `" + passName + "' already exists. Pass names must be unique");
            return false;
        }

        auto passChangedCB = [this]() {mRecompile = true; };
        pPass->setPassChangedCB(passChangedCB);

        pPass->setScene(mpScene);
        uint32_t node = mpGraph->addNode();
        mNameToIndex[passName] = node;
        mNodeData[node] = { passName, pPass };
        mRecompile = true;
        return true;
    }

    void RenderGraph::removeRenderPass(const std::string& name)
    {
        uint32_t index = getPassIndex(name);
        if (index == kInvalidIndex)
        {
            logWarning("Can't remove pass `" + name + "`. Pass doesn't exist");
            return;
        }

        // Update the indices
        mNameToIndex.erase(name);

        // Remove all the edges associated with this pass
        const auto& removedEdges = mpGraph->removeNode(index);
        for (const auto& e : removedEdges) mEdgeData.erase(e);

        mRecompile = true;
    }

    const RenderPass::SharedPtr& RenderGraph::getRenderPass(const std::string& name) const
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
        RenderPassReflection reflect;
        pPass->reflect(reflect);
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
        if (std::count(fullname.begin(), fullname.end(), '.') != 1)
        {
            logError("RenderGraph node field string is incorrect. Must be in the form of `PassName.FieldName` but got `" + fullname + "`", false);
            return false;
        }

        size_t dot = fullname.find_first_of('.');
        strPair.first = fullname.substr(0, dot);
        strPair.second = fullname.substr(dot + 1);
        return true;
    }

    template<bool input>
    static RenderPass* getRenderPassAndNamePair(const RenderGraph* pGraph, const std::string& fullname, const std::string& errorPrefix, str_pair& nameAndField)
    {
        if (parseFieldName(fullname, nameAndField) == false) return false;

        RenderPass* pPass = pGraph->getRenderPass(nameAndField.first).get();
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

    bool RenderGraph::isValid(std::string& log) const
    {
        return true;
    }

    Texture::SharedPtr RenderGraph::createTextureForPass(const RenderPassReflection::Field& field)
    {
        uint32_t width = field.getWidth() ? field.getWidth() : mSwapChainData.width;
        uint32_t height = field.getHeight() ? field.getHeight() : mSwapChainData.height;
        uint32_t depth = field.getDepth() ? field.getDepth() : 1;
        uint32_t sampleCount = field.getSampleCount() ? field.getSampleCount() : 1;
        ResourceFormat format = field.getFormat() == ResourceFormat::Unknown ? mSwapChainData.colorFormat : field.getFormat();
        Texture::SharedPtr pTexture;

        if (depth > 1)
        {
            assert(sampleCount == 1);
            pTexture = Texture::create3D(width, height, depth, format, 1, nullptr, field.getBindFlags() | Resource::BindFlags::ShaderResource);
        }
        else if (height > 1 || sampleCount > 1)
        {
            if (sampleCount > 1)
            {
                pTexture = Texture::create2DMS(width, height, format, sampleCount, 1, field.getBindFlags() | Resource::BindFlags::ShaderResource);
            }
            else
            {
                pTexture = Texture::create2D(width, height, format, 1, 1, nullptr, field.getBindFlags() | Resource::BindFlags::ShaderResource);
            }
        }
        else
        {
            pTexture = Texture::create1D(width, format, 1, 1, nullptr, field.getBindFlags() | Resource::BindFlags::ShaderResource);
        }

        return pTexture;
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

    bool RenderGraph::allocateResources()
    {
        for (const auto& nodeIndex : mExecutionList)
        {
            const DirectedGraph::Node* pNode = mpGraph->getNode(nodeIndex);
            assert(pNode);
            RenderPass* pSrcPass = mNodeData[nodeIndex].pPass.get();
            RenderPassReflection passReflection;
            pSrcPass->reflect(passReflection);

            const auto isGraphOutput = [=](uint32_t nodeId, const std::string& field)
            {
                for (const auto& out : mOutputs)
                {
                    if (out.nodeId == nodeId && out.field == field) return true;
                }
                return false;
            };

            // Set all the pass' outputs to either null or allocate a resource if it is required
            for (size_t i = 0; i < passReflection.getFieldCount(); i++)
            {
                const auto& field = passReflection.getField(i);
                if (is_set(field.getType(), RenderPassReflection::Field::Type::Input) == false)
                {
                    if (isGraphOutput(nodeIndex, field.getName()) == false)
                    {
                        bool allocate = is_set(field.getFlags(), RenderPassReflection::Field::Flags::Optional) == false;
                        Texture::SharedPtr pTex = allocate ? createTextureForPass(field) : nullptr;
                        mpResourcesCache->addResource(mNodeData[nodeIndex].nodeName + '.' + field.getName(), pTex);
                    }
                }
            }

            // Go over the edges, allocate the required resources and attach them to the input pass
            for (uint32_t e = 0; e < pNode->getOutgoingEdgeCount(); e++)
            {
                uint32_t edgeIndex = pNode->getOutgoingEdge(e);
                const auto& pEdge = mpGraph->getEdge(edgeIndex);
                const auto& edgeData = mEdgeData[edgeIndex];

                // Find the input
                for (size_t i = 0; i < passReflection.getFieldCount(); i++)
                {
                    const auto& field = passReflection.getField(i);

                    // Skip the field if it's not an output field
                    if (is_set(field.getType(), RenderPassReflection::Field::Type::Output) == false) continue;

                    if (field.getName() == edgeData.srcField)
                    {
                        std::string srcResourceName = mNodeData[nodeIndex].nodeName + '.' + field.getName();

                        Texture::SharedPtr pTexture;
                        pTexture = std::dynamic_pointer_cast<Texture>(mpResourcesCache->getResource(srcResourceName));

                        if (pTexture == nullptr)
                        {
                            pTexture = createTextureForPass(field);
                            mpResourcesCache->addResource(srcResourceName, pTexture);
                        }

                        // Connect it to the dst pass
                        const auto& dstPass = mNodeData[pEdge->getDestNode()].nodeName;
                        mpResourcesCache->addResource(dstPass + '.' + edgeData.dstField, pTexture);
                        break;
                    }
                }
            }
        }
        return true;
    }

    bool RenderGraph::compile(std::string& log)
    {
        if (mRecompile)
        {
            if (resolveExecutionOrder() == false) return false;
            if (allocateResources() == false) return false;
            if (isValid(log) == false) return false;
        }
        mRecompile = false;
        return true;
    }

    void RenderGraph::execute(RenderContext* pContext)
    {
        if (mProfileGraph) Profiler::startEvent("RenderGraph::execute()");

        std::string log;
        if (!compile(log))
        {
            logWarning("Failed to compile RenderGraph\n" + log + "Ignoreing RenderGraph::execute() call");
            return;
        }

        for (const auto& node : mExecutionList)
        {
            if (mProfileGraph) Profiler::startEvent(mNodeData[node].nodeName);
            RenderData renderData(mNodeData[node].nodeName, nullptr, mpResourcesCache);
            mNodeData[node].pPass->execute(pContext, &renderData);
            if (mProfileGraph) Profiler::endEvent(mNodeData[node].nodeName);
        }

        if (mProfileGraph) Profiler::endEvent("RenderGraph::execute()");
    }

    bool RenderGraph::setInput(const std::string& name, const std::shared_ptr<Resource>& pResource)
    {
        str_pair strPair;
        RenderPass* pPass = getRenderPassAndNamePair<true>(this, name, "RenderGraph::setInput()", strPair);
        if (pPass == nullptr) return false;
        mpResourcesCache->addResource(name, pResource);
        return true;
    }

    bool RenderGraph::setOutput(const std::string& name, const std::shared_ptr<Resource>& pResource)
    {
        str_pair strPair;
        RenderPass* pPass = getRenderPassAndNamePair<false>(this, name, "RenderGraph::setOutput()", strPair);
        if (pPass == nullptr) return false;
        mpResourcesCache->addResource(name, pResource);
        markGraphOutput(name);
        if (!pResource) mRecompile = true;
        return true;
    }

    void RenderGraph::markGraphOutput(const std::string& name)
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

    void RenderGraph::unmarkGraphOutput(const std::string& name)
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
                mpResourcesCache->removeResource(name);
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

        return pPass ? mpResourcesCache->getResource(name) : pNull;
    }

    std::string RenderGraph::getGraphOutputName(size_t index) const
    {
        assert(index < mOutputs.size());
        const GraphOut& graphOut = mOutputs[index];
        return mNodeData.find(graphOut.nodeId)->second.nodeName + "." + graphOut.field;
    }

    void RenderGraph::onResizeSwapChain(const Fbo* pTargetFbo)
    {
        // Store the back-buffer values
        const Texture* pColor = pTargetFbo->getColorTexture(0).get();
        const Texture* pDepth = pTargetFbo->getDepthStencilTexture().get();
        assert(pColor && pDepth);

        // If the back-buffer values changed, recompile
        mRecompile = mRecompile || (mSwapChainData.colorFormat != pColor->getFormat());
        mRecompile = mRecompile || (mSwapChainData.depthFormat != pDepth->getFormat());
        mRecompile = mRecompile || (mSwapChainData.width != pTargetFbo->getWidth());
        mRecompile = mRecompile || (mSwapChainData.height != pTargetFbo->getHeight());

        // Store the values
        mSwapChainData.colorFormat = pColor->getFormat();
        mSwapChainData.depthFormat = pDepth->getFormat();
        mSwapChainData.width = pTargetFbo->getWidth();
        mSwapChainData.height = pTargetFbo->getHeight();

        // Invoke the passes' callback
        for (const auto& it : mNodeData)
        {
            it.second.pPass->onResize(mSwapChainData.width, mSwapChainData.height);
        }
    }

    bool canFieldsConnect(const RenderPassReflection::Field& src, const RenderPassReflection::Field& dst)
    {
        assert(is_set(src.getType(), RenderPassReflection::Field::Type::Output) && is_set(dst.getType(), RenderPassReflection::Field::Type::Input));
        
        return (src.getFormat() == dst.getFormat() || dst.getFormat() == ResourceFormat::Unknown) &&
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

    std::vector<RenderPassReflection::Field> RenderGraph::getUnsatisfiedInputs(const NodeData* pNodeData, const RenderPassReflection& passReflection)
    {
        assert(mNameToIndex.count(pNodeData->nodeName) > 0);

        // Get names of connected input edges
        std::vector<std::string> satisfiedFields;
        const DirectedGraph::Node* pNode = mpGraph->getNode(mNameToIndex[pNodeData->nodeName]);
        for (uint32_t i = 0; i < pNode->getIncomingEdgeCount(); i++)
        {
            const auto& edgeData = mEdgeData[pNode->getIncomingEdge(i)];
            satisfiedFields.push_back(edgeData.dstField);
        }

        std::vector<RenderPassReflection::Field> unsatisfiedInputs;

        // Build list of unsatisfied fields by comparing names with which edges/fields are connected
        for (uint32_t i = 0; i < passReflection.getFieldCount(); i++)
        {
            const RenderPassReflection::Field& field = passReflection.getField(i);

            bool isUnsatisfied = std::find(satisfiedFields.begin(), satisfiedFields.end(), field.getName()) == satisfiedFields.end();
            if (is_set(field.getType(), RenderPassReflection::Field::Type::Input) && isUnsatisfied)
            {
                unsatisfiedInputs.push_back(passReflection.getField(i));
            }
        }

        return unsatisfiedInputs;
    }

    void RenderGraph::autoGenerateEdges()
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
        for (uint32_t i = 0; i < mpGraph->getCurrentNodeId(); i++)
        {
            if (mpGraph->doesNodeExist(i))
            {
                nodeVec.push_back(&mNodeData[i]);

                RenderPassReflection r;
                mNodeData[i].pPass->reflect(r);
                passReflectionMap[mNodeData[i].pPass.get()] = std::move(r);
            }
        }

        // For all nodes, starting at end, iterate until index 1 of vector
        for (size_t dst = nodeVec.size() - 1; dst > 0; dst--)
        {
            auto unsatisfiedInputs = getUnsatisfiedInputs(nodeVec[dst], passReflectionMap[nodeVec[dst]->pPass.get()]);

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
        if (!uiGroup || pGui->beginGroup(uiGroup))
        {
            pGui->addCheckBox("Profile Passes", mProfileGraph);
            pGui->addTooltip("Profile the render-passes. The results will be shown in the profiler window. If you can't see it, click 'P'");

            for (const auto& passId : mExecutionList)
            {
                const auto& pass = mNodeData[passId];

                // If you are thinking about displaying the profiler results next to the group label, it won't work. Since the times change every frame, IMGUI thinks it's a different group and will not expand it
                if (pGui->beginGroup(pass.nodeName))
                {
                    pass.pPass->renderUI(pGui, nullptr);
                    pGui->endGroup();
                }
            }
        }
    }
}
