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
#pragma once
#include "RenderPass.h"
#include "RenderGraphExe.h"
#include "RenderGraphCompiler.h"
#include "Core/Macros.h"
#include "Core/API/Formats.h"
#include "Utils/UI/Gui.h"
#include "Utils/Algorithm/DirectedGraph.h"
#include "Scene/Scene.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Falcor
{
    /** Represents a render graph.
        The render graph is a direct acyclic graph (DAG) of render passes.
    */
    class FALCOR_API RenderGraph
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

        /** Set a scene.
            \param[in] pScene New scene. This may be nullptr to unset the scene.
        */
        void setScene(const Scene::SharedPtr& pScene);

        /** Add a render pass. The name has to be unique, otherwise the call will be ignored.
            \return Render pass ID, or kInvalidIndex upon failure.
        */
        uint32_t addPass(const RenderPass::SharedPtr& pPass, const std::string& passName);

        /** Get a render pass by name.
            \return Ptr to render pass, or nullptr if pass wasn't found.
        */
        const RenderPass::SharedPtr& getPass(const std::string& name) const;

        /** Remove a render pass and all its edges. You need to make sure the graph is still valid after the pass was removed.
        */
        void removePass(const std::string& name);

        /** Update render pass using the specified dictionary. This function recreates the pass in place.
        */
        void updatePass(RenderContext* pRenderContext, const std::string& passName, const Dictionary& dict);

        /** Update render pass using the specified dictionary. This function calls the pass' applySettings method.
        */
        void applyPassSettings(const std::string& passName, const Dictionary& dict);

        /** Insert an edge from a render pass' output to a different render pass input.
            The render passes must be different, the graph must be a DAG.
            There are 2 types of edges:
            - Data dependency edge - Connecting as pass output resource to another pass input resource.
                                     The src/dst strings have the format `renderPassName.resourceName`, where the `renderPassName` is the name used in `addPass()` and the `resourceName` is the resource name as described by the render pass `reflect()` call.
            - Execution dependency edge - As the name implies, it creates an execution dependency between 2 passes, even if there's no data dependency. You can use it to control the execution order of the graph, or to force execution of passes which have no inputs/outputs.
                                          The src/dst string are `srcPass` and `dstPass` as used in `addPass()`.

            Note that data-dependency edges may be optimized out of the execution, if they are determined not to influence the requested graph output. Execution-dependency edges are never optimized and will always execute.
            \return Edge ID, or kInvalidIndex upon failure.
        */
        uint32_t addEdge(const std::string& src, const std::string& dst);

        /** Remove an edge by name. See naming conventions in addEdge().
        */
        void removeEdge(const std::string& src, const std::string& dst);

        /** Remove an edge by ID.
        */
        void removeEdge(uint32_t edgeID);

        /** Execute the graph.
        */
        void execute(RenderContext* pRenderContext);

        /** Update graph based on another graph's topology.
        */
        void update(const SharedPtr& pGraph);

        /** Set an external input resource.
            \param[in] name Input name. Has the format `renderPassName.resourceName`.
            \param[in] pResource The resource to bind. If this is nullptr, will unregister the resource.
        */
        void setInput(const std::string& name, const Resource::SharedPtr& pResource);

        /** Returns true if a render pass exists by this name in the graph.
        */
        bool doesPassExist(const std::string& name) const { return (mNameToIndex.find(name) != mNameToIndex.end()); }

        /** Return the index of a pass from a name, or kInvalidIndex if the pass doesn't exists.
        */
        uint32_t getPassIndex(const std::string& name) const;

        /** Get an output resource by name.
            This is an alias for `getRenderPass(renderPassName)->getOutput(resourceName)`.
            \param[in] name Output name. Has format `renderPassName.resourceName`.
            \return Resource object, or nullptr if output not available.
        */
        Resource::SharedPtr getOutput(const std::string& name);

        /** Get an output resource by index.
            If markOutput() or unmarkOutput() was called, the indices might change so don't cache them.
            \param[in] index Output index. The index corresponds to getOutputCount().
            \return Resource object, or nullptr if output not available.
        */
        Resource::SharedPtr getOutput(uint32_t index);

        /** Mark a render pass output as the graph's output.
            The name has the format `renderPassName.resourceName`. You can also use `renderPassName` which will allocate all the render pass outputs.
            The graph will automatically allocate the output resource. If the graph has no outputs it is invalid.
            If name is '*', it will mark all available outputs.
            \param[in] name Render pass output.
            \param[in] mask Mask of color channels. The default is RGB.
        */
        void markOutput(const std::string& name, const TextureChannelFlags mask = TextureChannelFlags::RGB);

        /** Unmark a graph output.
            The name has the format `renderPassName.resourceName`. You can also use `renderPassName` which will unmark all the render pass outputs.
            \param[in] name Render pass output.
        */
        void unmarkOutput(const std::string& name);

        /** Check if a name is marked as output.
            \param[in] name Render pass output.
        */
        bool isGraphOutput(const std::string& name) const;

        /** Call this when the swap chain was resized.
        */
        void onResize(const Fbo* pTargetFbo);

        /** Get the attached scene.
        */
        const Scene::SharedPtr& getScene() const { return mpScene; }

        /** Get the number of outputs from this graph.
        */
        size_t getOutputCount() const { return mOutputs.size(); }

        /** Get an graph output name from the graph output index.
            \param[in] index Output index. The index corresponds to getOutputCount().
            \return Output name.
        */
        std::string getOutputName(size_t index) const;

        /** Get the set of output masks from the graph output index.
            \param[in] index Output index. The index corresponds to getOutputCount().
            \return Set of color channel masks.
        */
        std::unordered_set<TextureChannelFlags> getOutputMasks(size_t index) const;

        /** Get all output names for the render graph.
        */
        std::vector<std::string> getAvailableOutputs() const;

        /** Get all output names for the render graph that are currently unmarked.
        */
        std::vector<std::string> getUnmarkedOutputs() const;

        /** Render the graph UI.
        */
        void renderUI(Gui::Widgets& widget);

        /** Mouse event handler.
            \return True if the event was handled by the object, false otherwise.
        */
        bool onMouseEvent(const MouseEvent& mouseEvent);

        /** Keyboard event handler.
            \return True if the event was handled by the object, false otherwise.
        */
        bool onKeyEvent(const KeyboardEvent& keyEvent);

        /** Called upon hot reload (by pressing F5).
            \param[in] reloaded Resources that have been reloaded.
        */
        void onHotReload(HotReloadFlags reloaded);

        /** Get the dictionary objects used to communicate app data to the render passes.
        */
        const InternalDictionary::SharedPtr& getPassesDictionary() const { return mpPassDictionary; }

        /** Get the graph name.
        */
        const std::string& getName() const { return mName; }

        /** Set the graph name.
        */
        void setName(const std::string& name) { mName = name; }

        /** Compile the graph.
        */
        bool compile(RenderContext* pRenderContext, std::string& log);
        bool compile(RenderContext* pRenderContext) { std::string s; return compile(pRenderContext, s); }

    private:
        RenderGraph(const std::string& name);

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

        struct GraphOut
        {
            uint32_t nodeId = kInvalidIndex;                ///< Render pass node ID.
            std::string field;                              ///< Name of the output field.
            std::unordered_set<TextureChannelFlags> masks;  ///< Set of output color channel masks.

            bool operator==(const GraphOut& other) const
            {
                if (nodeId != other.nodeId) return false;
                if (field != other.field) return false;
                // Do not compare masks, the other fields uniquely identify the output.
                return true;
            }

            bool operator!=(const GraphOut& other) const { return !(*this == other); }
        };

        RenderPass* getRenderPassAndNamePair(const bool input, const std::string& fullname, const std::string& errorPrefix, std::pair<std::string, std::string>& nameAndField) const;
        uint32_t getEdge(const std::string& src, const std::string& dst);
        void getUnsatisfiedInputs(const NodeData* pNodeData, const RenderPassReflection& passReflection, std::vector<RenderPassReflection::Field>& outList) const;
        void autoConnectPasses(const NodeData* pSrcNode, const RenderPassReflection& srcReflection, const NodeData* pDestNode, std::vector<RenderPassReflection::Field>& unsatisfiedInputs);
        bool isGraphOutput(const GraphOut& graphOut) const;

        std::string mName;                                          ///< Name of render graph.
        Scene::SharedPtr mpScene;                                   ///< Current scene. This may be nullptr.

        DirectedGraph::SharedPtr mpGraph;                           ///< DAG of render passes. Only IDs are stored, not the actual passes.
        std::unordered_map<std::string, uint32_t> mNameToIndex;     ///< Map from render pass name to node ID in graph.
        std::unordered_map<uint32_t, EdgeData> mEdgeData;
        std::unordered_map<uint32_t, NodeData> mNodeData;           ///< Map from node ID to render pass name and ptr.
        std::vector<GraphOut> mOutputs;                             ///< Array of all outputs marked as graph outputs. GRAPH_TODO should this be an unordered set?

        InternalDictionary::SharedPtr mpPassDictionary;             ///< Dictionary used to communicate between passes.
        RenderGraphExe::SharedPtr mpExe;                            ///< Helper for allocating resources and executing the graph.
        RenderGraphCompiler::Dependencies mCompilerDeps;            ///< Data needed by the graph compiler.
        bool mRecompile = false;                                    ///< Set to true to trigger a recompilation after any graph changes (topology/scene/size/passes/etc.)

        friend class RenderGraphUI;
        friend class RenderGraphExporter;
        friend class RenderPassLibrary;
        friend class RenderGraphCompiler;
    };
}
