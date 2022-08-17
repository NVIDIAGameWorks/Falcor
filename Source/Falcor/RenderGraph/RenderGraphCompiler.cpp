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
#include "RenderGraphCompiler.h"
#include "RenderGraph.h"
#include "RenderPasses/ResolvePass.h"
#include "Utils/Algorithm/DirectedGraphTraversal.h"
#include "Utils/StringUtils.h"

namespace Falcor
{
    namespace
    {
        bool canAutoResolve(const RenderPassReflection::Field& src, const RenderPassReflection::Field& dst)
        {
            return src.getSampleCount() > 1 && dst.getSampleCount() == 1;
        }
    }

    RenderGraphCompiler::RenderGraphCompiler(RenderGraph& graph, const Dependencies& dependencies) : mGraph(graph), mDependencies(dependencies) {}

    RenderGraphExe::SharedPtr RenderGraphCompiler::compile(RenderGraph& graph, RenderContext* pRenderContext, const Dependencies& dependencies)
    {
        RenderGraphCompiler c = RenderGraphCompiler(graph, dependencies);

        // Register the external resources
        auto pResourcesCache = ResourceCache::create();
        for (const auto&[name, pRes] : dependencies.externalResources) pResourcesCache->registerExternalResource(name, pRes);

        c.resolveExecutionOrder();
        c.compilePasses(pRenderContext);
        if (c.insertAutoPasses()) c.resolveExecutionOrder();
        c.validateGraph();
        c.allocateResources(pResourcesCache.get());

        auto pExe = RenderGraphExe::create();
        pExe->mExecutionList.reserve(c.mExecutionList.size());

        for (auto e : c.mExecutionList)
        {
            pExe->insertPass(e.name, e.pPass);
        }
        c.restoreCompilationChanges();
        pExe->mpResourceCache = pResourcesCache;
        return pExe;
    }

    void RenderGraphCompiler::validateGraph() const
    {
        std::string err;

        for (const auto& p : mExecutionList)
        {
            // Make sure all the inputs are satisfied
            for (uint32_t i = 0; i < p.reflector.getFieldCount(); i++)
            {
                const auto& f = *p.reflector.getField(i);
                if (!is_set(f.getVisibility(), RenderPassReflection::Field::Visibility::Input)) continue;
                if (is_set(f.getFlags(), RenderPassReflection::Field::Flags::Optional)) continue;

                const DirectedGraph::Node* pGraphNode = mGraph.mpGraph->getNode(p.index);
                const std::string& name = f.getName();
                bool found = false;
                for (uint32_t e = 0; e < pGraphNode->getIncomingEdgeCount(); e++)
                {
                    const auto& edgeData = mGraph.mEdgeData.at(pGraphNode->getIncomingEdge(e));
                    found = (edgeData.dstField == name);
                    if (found) break;
                }
                std::string resName = p.name + '.' + name;
                bool hasExternal = mDependencies.externalResources.find(resName) != mDependencies.externalResources.end();
                if (hasExternal && found)  err += "Input field '" + resName + "' has an incoming edge and an external resource bound. This is illegal";
                if (!hasExternal && !found) err += "Input field '" + resName + "' is required but not satisfied\n";
            }
        }

        if (mGraph.getOutputCount() == 0) err += "Graph must have at least one output.\n";

        if (err.size()) throw RuntimeError(err);
    }

    void RenderGraphCompiler::resolveExecutionOrder()
    {
        mExecutionList.clear();

        // Find out which passes are mandatory
        std::unordered_set<uint32_t> mandatoryPasses;
        for (auto& o : mGraph.mOutputs) mandatoryPasses.insert(o.nodeId); // Add direct-graph outputs

        for (auto& e : mGraph.mEdgeData) // Add all the passes which have an execution-edge connected to them
        {
            if (e.second.dstField.empty())
            {
                FALCOR_ASSERT(e.second.srcField.empty());
                const auto& edge = mGraph.mpGraph->getEdge(e.first);
                mandatoryPasses.insert(edge->getDestNode());
                mandatoryPasses.insert(edge->getSourceNode());
            }
        }

        // Find all passes that affect the outputs
        std::unordered_set<uint32_t> participatingPasses;
        for (auto& o : mandatoryPasses)
        {
            uint32_t nodeId = o;
            auto dfs = DirectedGraphDfsTraversal(mGraph.mpGraph, nodeId, DirectedGraphDfsTraversal::Flags::IgnoreVisited | DirectedGraphDfsTraversal::Flags::Reverse);
            while (nodeId != DirectedGraph::kInvalidID)
            {
                participatingPasses.insert(nodeId);
                nodeId = dfs.traverse();
            }
        }

        // Run topological sort
        auto topologicalSort = DirectedGraphTopologicalSort::sort(mGraph.mpGraph.get());

        RenderPass::CompileData compileData;
        compileData.defaultTexDims = mDependencies.defaultResourceProps.dims;
        compileData.defaultTexFormat = mDependencies.defaultResourceProps.format;

        // For each object in the vector, if it's being used in the execution, put it in the list
        for (auto& node : topologicalSort)
        {
            if (participatingPasses.find(node) != participatingPasses.end())
            {
                const auto pData = mGraph.mNodeData[node];
                mExecutionList.push_back({ node, pData.pPass, pData.name, pData.pPass->reflect(compileData) });
            }
        }
    }

    bool RenderGraphCompiler::insertAutoPasses()
    {
        bool addedPasses = false;
        for (size_t i = 0; i < mExecutionList.size(); i++)
        {
            const RenderPassReflection& passReflection = mExecutionList[i].reflector;

            // Check for opportunities to automatically resolve MSAA
            // - Only take explicitly specified MS output
            for (uint32_t f = 0; f < passReflection.getFieldCount(); f++)
            {
                // Iterate over output fields
                auto& srcField = *passReflection.getField(f);
                if (is_set(srcField.getVisibility(), RenderPassReflection::Field::Visibility::Output) == false) continue;

                const std::string& srcPassName = mExecutionList[i].name;
                // Gather src field name, and every input it is connected to
                std::string srcFieldName = srcPassName + '.' + srcField.getName();
                std::vector<std::string> dstFieldNames;

                const DirectedGraph::Node* pNode = mGraph.mpGraph->getNode(mExecutionList[i].index);
                for (uint32_t e = 0; e < pNode->getOutgoingEdgeCount(); e++)
                {
                    uint32_t edgeIndex = pNode->getOutgoingEdge(e);
                    const auto& edgeData = mGraph.mEdgeData.at(edgeIndex);

                    // For every output field, iterate over all edges extending from that field
                    if (srcField.getName() == edgeData.srcField)
                    {
                        const auto& pEdge = mGraph.mpGraph->getEdge(edgeIndex);
                        const std::string& dstPassName = mGraph.mNodeData.at(pEdge->getDestNode()).name;

                        // If edge is connected to something that isn't executed, ignore
                        auto getPassReflection = [&](uint32_t index) -> std::optional<RenderPassReflection>
                        {
                            for (const auto& e : mExecutionList) if (e.index == index) return e.reflector;
                            return std::nullopt;
                        };

                        const auto& dstReflection = getPassReflection(pEdge->getDestNode());
                        if (!dstReflection) continue;
                        const auto& dstField = *dstReflection->getField(edgeData.dstField);

                        FALCOR_ASSERT(srcField.isValid() && dstField.isValid());
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
                    auto pResolvePass = ResolvePass::create();
                    pResolvePass->setFormat(srcField.getFormat()); // Match input texture format

                    // Create pass and attach src to it
                    std::string resolvePassName = srcFieldName + "-ResolvePass";
                    mGraph.addPass(pResolvePass, resolvePassName);
                    mGraph.addEdge(srcFieldName, resolvePassName + ".src");

                    // For every input the src field is connected to, connect the resolve pass output to the input
                    for (const auto& dstFieldName : dstFieldNames)
                    {
                        // Remove original edge
                        mGraph.removeEdge(srcFieldName, dstFieldName);
                        // Replace with edge coming from resolve output
                        mGraph.addEdge(resolvePassName + ".dst", dstFieldName);

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

    void RenderGraphCompiler::allocateResources(ResourceCache* pResourceCache)
    {
        // Build list to look up execution order index from the pass
        std::unordered_map<RenderPass*, uint32_t> passToIndex;
        for (size_t i = 0; i < mExecutionList.size(); i++)
        {
            passToIndex.emplace(mExecutionList[i].pPass.get(), uint32_t(i));
        }

        for (size_t i = 0; i < mExecutionList.size(); i++)
        {
            uint32_t nodeIndex = mExecutionList[i].index;

            const DirectedGraph::Node* pNode = mGraph.mpGraph->getNode(nodeIndex);
            FALCOR_ASSERT(pNode);
            RenderPass* pCurrPass = mGraph.mNodeData[nodeIndex].pPass.get();
            const auto& passReflection = mExecutionList[i].reflector;

            auto isResourceUsed = [&](auto field)
            {
                if (!is_set(field.getFlags(), RenderPassReflection::Field::Flags::Optional)) return true;
                if (mGraph.isGraphOutput({ nodeIndex, field.getName() })) return true;
                for (uint32_t e = 0; e < pNode->getOutgoingEdgeCount(); e++)
                {
                    const auto& edgeData = mGraph.mEdgeData[pNode->getOutgoingEdge(e)];
                    if (edgeData.srcField == field.getName()) return true;
                }
                return false;
            };

            // Register all pass outputs
            for (size_t f = 0; f < passReflection.getFieldCount(); f++)
            {
                auto field = *passReflection.getField(f);
                std::string fullFieldName = mGraph.mNodeData[nodeIndex].name + '.' + field.getName();

                // Skip input resources, we never allocate them
                if (!is_set(field.getVisibility(), RenderPassReflection::Field::Visibility::Input))
                {
                    if (isResourceUsed(field) == false) continue;

                    // Resource lifetime for graph outputs must extend to end of graph execution
                    bool graphOutput = mGraph.isGraphOutput({ nodeIndex, field.getName() });
                    uint32_t lifetime = graphOutput ? uint32_t(-1) : uint32_t(i);
                    if (graphOutput && field.getBindFlags() != ResourceBindFlags::None) field.bindFlags(field.getBindFlags() | ResourceBindFlags::ShaderResource); // Adding ShaderResource for graph outputs
                    pResourceCache->registerField(fullFieldName, field, lifetime);
                }
            }

            // Go over the pass inputs, add them as aliases to the outputs that connect to them (which should be already registered above)
            for (uint32_t e = 0; e < pNode->getIncomingEdgeCount(); e++)
            {
                uint32_t edgeIndex = pNode->getIncomingEdge(e);
                const auto& pEdge = mGraph.mpGraph->getEdge(edgeIndex);
                const auto& edgeData = mGraph.mEdgeData[edgeIndex];

                // Skip execution-edges
                if (edgeData.dstField.empty())
                {
                    FALCOR_ASSERT(edgeData.srcField.empty());
                    continue;
                }

                const auto& dstField = *passReflection.getField(edgeData.dstField);
                FALCOR_ASSERT(dstField.isValid() && is_set(dstField.getVisibility(), RenderPassReflection::Field::Visibility::Input));

                // Merge dst/input field into same resource data
                std::string srcFieldName = mGraph.mNodeData[pEdge->getSourceNode()].name + '.' + edgeData.srcField;
                std::string dstFieldName = mGraph.mNodeData[nodeIndex].name + '.' + dstField.getName();

                const auto& pSrcPass = mGraph.mNodeData[pEdge->getSourceNode()].pPass.get();
                const auto& srcReflection = mExecutionList[passToIndex.at(pSrcPass)].reflector;
                pResourceCache->registerField(dstFieldName, dstField, passToIndex[pSrcPass], srcFieldName);
            }
        }

        pResourceCache->allocateResources(mDependencies.defaultResourceProps);
    }


    void RenderGraphCompiler::restoreCompilationChanges()
    {
        for (const auto& name : mCompilationChanges.generatedPasses) mGraph.removePass(name);
        for (const auto& e : mCompilationChanges.removedEdges) mGraph.addEdge(e.first, e.second);

        mCompilationChanges.generatedPasses.clear();
        mCompilationChanges.removedEdges.clear();
    }

    RenderPass::CompileData RenderGraphCompiler::prepPassCompilationData(const PassData& passData)
    {
        RenderPass::CompileData compileData;
        compileData.defaultTexDims = mDependencies.defaultResourceProps.dims;
        compileData.defaultTexFormat = mDependencies.defaultResourceProps.format;

        auto isExecutionEdge = [this](uint32_t edgeId)
        {
            return mGraph.mEdgeData[edgeId].srcField.empty();
        };

        // Get the list of input resources
        const auto pNode = mGraph.mpGraph->getNode(passData.index);
        for (uint32_t i = 0; i < pNode->getIncomingEdgeCount(); i++)
        {
            uint32_t e = pNode->getIncomingEdge(i);
            if (isExecutionEdge(e)) continue;

            uint32_t incomingPass = mGraph.mpGraph->getEdge(e)->getSourceNode();
            for (const auto& otherPass : mExecutionList)
            {
                if (otherPass.index == incomingPass)
                {
                    auto f = *otherPass.reflector.getField(mGraph.mEdgeData[e].srcField);
                    const auto& fIn = *passData.reflector.getField(mGraph.mEdgeData[e].dstField);
                    f.name(fIn.getName()).visibility(fIn.getVisibility()).desc(fIn.getDesc());
                    compileData.connectedResources.addField(f);
                    break;
                }
                else if (otherPass.index == passData.index) break;
            }
        }

        // Add the external resources
        for (auto& [name, pRes] : mDependencies.externalResources)
        {
            if (hasPrefix(name, passData.name + "."))
            {
                auto pTex = pRes->asTexture();
                std::string resName = name.substr((passData.name + ".").size());
                compileData.connectedResources.addInput(resName, "External input resource").format(pTex->getFormat()).resourceType(resourceTypeToFieldType(pTex->getType()), pTex->getWidth(), pTex->getHeight(), pTex->getDepth(), pTex->getSampleCount(), pTex->getMipCount(), pTex->getArraySize());
            }
        }

        // Get a list of output resources. It's slightly different then the inputs, because we can have multiple edges for each output resource
        for (uint32_t i = 0; i < pNode->getOutgoingEdgeCount(); i++)
        {
            uint32_t e = pNode->getOutgoingEdge(i);
            if (isExecutionEdge(e)) continue;

            uint32_t outgoingPass = mGraph.mpGraph->getEdge(e)->getDestNode();
            for (const auto& otherPass : mExecutionList)
            {
                if (otherPass.index == outgoingPass)
                {
                    auto f = *otherPass.reflector.getField(mGraph.mEdgeData[e].dstField);
                    auto pField = compileData.connectedResources.getField(mGraph.mEdgeData[e].srcField);
                    if (pField)
                    {
                        const_cast<RenderPassReflection::Field*>(pField)->merge(f);
                    }
                    else
                    {
                        const auto& fOut = *passData.reflector.getField(mGraph.mEdgeData[e].srcField);
                        f.name(fOut.getName()).visibility(fOut.getVisibility()).desc(fOut.getDesc());
                        compileData.connectedResources.addField(f);
                    }
                }
            }
        }

        return compileData;
    }

    void RenderGraphCompiler::compilePasses(RenderContext* pRenderContext)
    {
        while(1)
        {
            std::string log;
            bool success = true;
            for (auto& p : mExecutionList)
            {
                try
                {
                    p.pPass->compile(pRenderContext, prepPassCompilationData(p));
                }
                catch (const std::exception& e)
                {
                    log += std::string(e.what()) + "\n";
                    success = false;
                }
            }

            if (success) return;

            // Retry
            bool changed = false;
            for (auto& p : mExecutionList)
            {
                auto newR = p.pPass->reflect(prepPassCompilationData(p));
                if (newR != p.reflector)
                {
                    p.reflector = newR;
                    changed = true;
                }
            }

            if (!changed)
            {
                reportError("Graph compilation failed.\n" + log);
                return;
            }
        }
    }
}
