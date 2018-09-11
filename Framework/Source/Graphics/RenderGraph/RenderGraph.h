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
#pragma once
#include "RenderPass.h"
#include "Utils/DirectedGraph.h"
#include "ResourceCache.h"

namespace Falcor
{
    class Scene;
    class Texture;
    class Fbo;
    class RenderGraphExporter;

    class RenderGraph
    {
    public:
        using SharedPtr = std::shared_ptr<RenderGraph>;

        static const uint32_t kInvalidIndex = -1;

        /** Create a new object
        */
        static SharedPtr create();

        /** Set a scene
        */
        void setScene(const std::shared_ptr<Scene>& pScene);

        /** Add a render-pass. The name has to be unique, otherwise the call will be ignored
        */
        uint32_t addPass(const RenderPass::SharedPtr& pPass, const std::string& passName);

        /** Get a render-pass
        */
        const RenderPass::SharedPtr& getPass(const std::string& name) const;

        /** Remove a render-pass. You need to make sure the edges are still valid after the node was removed
        */
        void removePass(const std::string& name);

        /** Insert an edge from a render-pass' output into a different render-pass input.
            The render passes must be different, the graph must be a DAG.
            The src/dst strings have the format `renderPassName.resourceName`, where the `renderPassName` is the name used in `setRenderPass()` and the `resourceName` is the resource-name as described by the render-pass object
        */
        uint32_t addEdge(const std::string& src, const std::string& dst);

        /** Remove an edge
         */
        void removeEdge(const std::string& src, const std::string& dst);

        /** Remove an edge
         */
        void removeEdge(uint32_t edgeID);

        /** Check if the graph is ready for execution (all passes inputs/outputs have been initialized correctly, no loops in the graph)
        */
        bool isValid(std::string& log) const;

        /** Execute the graph
        */
        void execute(RenderContext* pContext);

        /** Set an input resource. The name has the format `renderPassName.resourceName`.
            This is an alias for `getRenderPass(renderPassName)->setInput(resourceName, pResource)`
        */
        bool setInput(const std::string& name, const std::shared_ptr<Resource>& pResource);

        /** Set an output resource. The name has the format `renderPassName.resourceName`.
            This is an alias for `getRenderPass(renderPassName)->setOutput(resourceName, pResource)`
            Calling this function will automatically mark the output as one of the graph's outputs (even if called with nullptr)
        */
        bool setOutput(const std::string& name, const std::shared_ptr<Resource>& pResource);
        
        /** Returns true if a render pass exists by this name in the graph.
         */
        bool doesPassExist(const std::string& name) const { return (mNameToIndex.find(name) != mNameToIndex.end()); }

        /** Get an output resource. The name has the format `renderPassName.resourceName`.
        This is an alias for `getRenderPass(renderPassName)->getOutput(resourceName)`
        */
        const std::shared_ptr<Resource> getOutput(const std::string& name);

        /** Mark a render-pass output as the graph's output. If the graph has no outputs it is invalid.
            The name has the format `renderPassName.resourceName`. You can also use `renderPassName` which will allocate all the render-pass outputs.
            If the user didn't set the output resource using `setOutput()`, the graph will automatically allocate it
        */
        void markOutput(const std::string& name);

        /** Unmark a graph output
            The name has the format `renderPassName.resourceName`. You can also use `renderPassName` which will allocate all the render-pass outputs
        */
        void unmarkOutput(const std::string& name);

        /** Call this when the swap-chain was resized
        */
        void onResizeSwapChain(const Fbo* pTargetFbo);

        /** Get the attached scene
        */
        const std::shared_ptr<Scene>& getScene() const { return mpScene; }

        /** Get an graph output name from the graph outputs
        */
        std::string getOutputName(size_t index) const;

        /** Get the num of outputs from this graph
        */
        size_t getOutputCount() const { return mOutputs.size(); }

        /** Get all output names for the render graph
        */
        std::vector<std::string> getAllOutputs() const;

        friend class RenderGraphUI;

        /** Attempts to auto generate edges for render passes.
            \param[in] executionOrder Optional. Ordered list of node ID's as an override of pass search order to use when generating edges.
        */
        void autoGenEdges(const std::vector<uint32_t>& executionOrder = std::vector<uint32_t>());

        /** Render the UI
        */
        void renderUI(Gui* pGui, const char* uiGroup);

        /** Enable/disable pass profiling
        */
        void profileGraph(bool enabled) { mProfileGraph = enabled; }

    private:
        friend class RenderGraphExporter;

        RenderGraph();
        uint32_t getPassIndex(const std::string& name) const;
        bool compile(std::string& log);
        bool resolveExecutionOrder();
        bool allocateResources();

        struct EdgeData
        {
            enum class Flags
            {
                None = 0,
                AutoResolve
            };

            bool autoGenerated = false;
            Flags flags = Flags::None;
            std::string srcField;
            std::string dstField;
        };

        struct NodeData
        {
            std::string nodeName;
            RenderPass::SharedPtr pPass;
        };

        void getUnsatisfiedInputs(const NodeData* pNodeData, const RenderPassReflection& passReflection, std::vector<RenderPassReflection::Field>& outList);
        void autoConnectPasses(const NodeData* pSrcNode, const RenderPassReflection& srcReflection, const NodeData* pDestNode, std::vector<RenderPassReflection::Field>& unsatisfiedInputs);

        bool mRecompile = true;
        std::shared_ptr<Scene> mpScene;

        std::unordered_map<std::string, uint32_t> mNameToIndex;

        DirectedGraph::SharedPtr mpGraph;
        std::unordered_map<uint32_t, EdgeData> mEdgeData;
        std::unordered_map<uint32_t, NodeData> mNodeData;

        struct GraphOut
        {
            uint32_t nodeId;
            std::string field;
        };

        std::vector<GraphOut> mOutputs; // GRAPH_TODO should this be an unordered set?

        std::shared_ptr<Texture> createTextureForPass(const RenderPassReflection::Field& field);

        struct  
        {
            uint32_t width = 0;
            uint32_t height = 0;
            ResourceFormat colorFormat = ResourceFormat::Unknown;
            ResourceFormat depthFormat = ResourceFormat::Unknown;
        } mSwapChainData;

        std::vector<uint32_t> mExecutionList;
        ResourceCache::SharedPtr mpResourcesCache;
        bool mProfileGraph = true;
    };
}