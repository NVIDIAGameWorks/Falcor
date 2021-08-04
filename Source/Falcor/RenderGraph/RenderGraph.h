/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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
#pragma once
#include "RenderPassReflection.h"
#include "ResourceCache.h"
#include "Scene/Scene.h"
#include "RenderPass.h"
#include "Utils/Algorithm/DirectedGraph.h"
#include "RenderGraphExe.h"
#include "RenderGraphCompiler.h"

namespace Falcor
{
    class dlldecl RenderGraph : public std::enable_shared_from_this<RenderGraph>
    {
    public:
        using SharedPtr = std::shared_ptr<RenderGraph>;
        static const FileDialogFilterVec kFileExtensionFilters;

        static const uint32_t kInvalidIndex = -1;

        ~RenderGraph();

        /** Create a new render graph.
            \param[in] name Name of the render graph.
            \return New object, or throws an exception if creation failed.
        */
        static SharedPtr create(const std::string& name = "");

        /** Set a scene
        */
        void setScene(const Scene::SharedPtr& pScene);

        /** Add a render-pass. The name has to be unique, otherwise the call will be ignored
        */
        uint32_t addPass(const RenderPass::SharedPtr& pPass, const std::string& passName);

        /** Get a render-pass
        */
        const RenderPass::SharedPtr& getPass(const std::string& name) const;

        /** Remove a render-pass. You need to make sure the edges are still valid after the node was removed
        */
        void removePass(const std::string& name);

        /** Update dictionary for specified render pass.
        */
        void updatePass(RenderContext* pRenderContext, const std::string& passName, const Dictionary& dict);

        /** Insert an edge from a render-pass' output into a different render-pass input.
            The render passes must be different, the graph must be a DAG.
            There are 2 types of edges:
            - Data dependency edge - Connecting as pass` output resource to another pass` input resource.
                                     The src/dst strings have the format `renderPassName.resourceName`, where the `renderPassName` is the name used in `addPass()` and the `resourceName` is the resource-name as described by the render-pass object
            - Execution dependency edge - As the name implies, it creates an execution dependency between 2 passes, even if there's no data dependency. You can use it to control the execution order of the graph, or to force execution of passes which have no inputs/outputs.
                                          The src/dst string are `srcPass` and `dstPass` as used in `addPass()`

            Note that data-dependency edges may be optimized out of the execution, if they are determined not to influence the requested graph-output. Execution-dependency edges are never optimized and will always execute
        */
        uint32_t addEdge(const std::string& src, const std::string& dst);

        /** Remove an edge
         */
        void removeEdge(const std::string& src, const std::string& dst);

        /** Remove an edge
         */
        void removeEdge(uint32_t edgeID);

        /** Execute the graph
        */
        void execute(RenderContext* pContext);

        /** Update graph based on another graph's topology
        */
        void update(const SharedPtr& pGraph);

        /** Set an external input resource
            \param[in] name Input name. Has the format `renderPassName.resourceName`
            \param[in] pResource The resource to bind. If this is nullptr, will unregister the resource
        */
        void setInput(const std::string& name, const Resource::SharedPtr& pResource);

        /** Returns true if a render pass exists by this name in the graph.
         */
        bool doesPassExist(const std::string& name) const { return (mNameToIndex.find(name) != mNameToIndex.end()); }

        /** Return the index of a pass from a name, or kInvalidIndex if the pass doesn't exists
        */
        uint32_t getPassIndex(const std::string& name) const;

        /** Get an output resource. The name has the format `renderPassName.resourceName`.
        This is an alias for `getRenderPass(renderPassName)->getOutput(resourceName)`
        */
        Resource::SharedPtr getOutput(const std::string& name);

        /** Get an output resource. The index corresponds to getOutputCount().
            If markOutput() or unmarkOutput() was called, the indices might change so don't cache them
        */
        Resource::SharedPtr getOutput(uint32_t index);

        /** Mark a render-pass output as the graph's output. If the graph has no outputs it is invalid.
            The name has the format `renderPassName.resourceName`. You can also use `renderPassName` which will allocate all the render-pass outputs.
            The graph will automatically allocate the output resource
            If name is '*', it will mark all available outputs
        */
        void markOutput(const std::string& name);

        /** Unmark a graph output
            The name has the format `renderPassName.resourceName`. You can also use `renderPassName` which will allocate all the render-pass outputs
        */
        void unmarkOutput(const std::string& name);

        /** Check if a name is marked as output
        */
        bool isGraphOutput(const std::string& name) const;

        /** Call this when the swap-chain was resized
        */
        void onResize(const Fbo* pTargetFbo);

        /** Get the attached scene
        */
        const Scene::SharedPtr& getScene() const { return mpScene; }

        /** Get an graph output name from the graph outputs
        */
        std::string getOutputName(size_t index) const;

        /** Get the num of outputs from this graph
        */
        size_t getOutputCount() const { return mOutputs.size(); }

        /** Get all output names for the render graph
        */
        std::vector<std::string> getAvailableOutputs() const;

        /** Get all output names for the render graph that are currently unmarked.
        */
        std::vector<std::string> getUnmarkedOutputs() const;

        /** Render the UI
        */
        void renderUI(Gui::Widgets& widget);

        /** Mouse event handler.
            Returns true if the event was handled by the object, false otherwise
        */
        bool onMouseEvent(const MouseEvent& mouseEvent);

        /** Keyboard event handler
        Returns true if the event was handled by the object, false otherwise
        */
        bool onKeyEvent(const KeyboardEvent& keyEvent);

        /** Called upon hot reload (by pressing F5).
            \param[in] reloaded Resources that have been reloaded.
        */
        void onHotReload(HotReloadFlags reloaded);

        /** Get the dictionary objects used to communicate app data to the render-passes
        */
        const InternalDictionary::SharedPtr& getPassesDictionary() const { return mpPassDictionary; }

        /** Get the name
        */
        const std::string& getName() const { return mName; }

        /** Get the name
        */
        void setName(const std::string& name) { mName = name; }

        /** Compile the graph
        */
        bool compile(RenderContext* pContext, std::string& log);
        bool compile(RenderContext* pContext) { std::string s; return compile(pContext, s); }

    private:
        friend class RenderGraphUI;
        friend class RenderGraphExporter;
        friend class RenderPassLibrary;
        friend class RenderGraphCompiler;

        RenderGraph(const std::string& name);
        std::string mName;

        struct EdgeData
        {
            std::string srcField;
            std::string dstField;
        };

        struct NodeData
        {
            std::string name;
            RenderPass::SharedPtr pPass;
        };

        uint32_t getEdge(const std::string& src, const std::string& dst);

        Scene::SharedPtr mpScene;
        std::unordered_map<std::string, uint32_t> mNameToIndex;
        DirectedGraph::SharedPtr mpGraph;
        std::unordered_map<uint32_t, EdgeData> mEdgeData;
        std::unordered_map<uint32_t, NodeData> mNodeData;

        struct GraphOut
        {
            uint32_t nodeId;
            std::string field;

            bool operator==(const GraphOut& other) const
            {
                if (nodeId != other.nodeId) return false;
                if (field != other.field) return false;
                return true;
            }

            bool operator!=(const GraphOut& other) const { return !(*this == other); }
        };

        std::vector<GraphOut> mOutputs; // GRAPH_TODO should this be an unordered set?
        bool isGraphOutput(const GraphOut& graphOut) const;

        InternalDictionary::SharedPtr mpPassDictionary;
        RenderGraphExe::SharedPtr mpExe;
        bool mRecompile = false;
        RenderGraphCompiler::Dependencies mCompilerDeps;
    };
}
