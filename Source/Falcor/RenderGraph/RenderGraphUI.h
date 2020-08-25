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
#pragma once
#include "RenderPassReflection.h"
#include "RenderGraph.h"
#include "RenderGraphIR.h"

namespace Falcor
{
    class NodeGraphEditorGui;

    class dlldecl RenderPassUI
    {
    public:

        // wrapper around inserting new pin for a given pass
        void addUIPin(const std::string& fieldName, uint32_t guiPinID, bool isInput, const std::string& connectedPinName = "", const std::string& connectedNodeName = "", bool isGraphOutput = false);
        void renderPinUI(const std::string& passName, RenderGraphUI* pGraphUI, uint32_t index = 0, bool input = false);

        friend class RenderGraphUI;

    private:
        class PinUI
        {
        public:

            std::string mPinName;
            uint32_t mGuiPinID;
            std::string mConnectedPinName;
            std::string mConnectedNodeName;
            bool mIsGraphOutput;

            static void renderFieldInfo(const RenderPassReflection::Field& field, RenderGraphUI* pGraphUI, const std::string& passName, const std::string& fieldName);
            void renderUI(const RenderPassReflection::Field& field, RenderGraphUI* graphUI, const std::string& passName);
        };

        std::vector<PinUI> mInputPins;
        std::unordered_map<std::string, uint32_t> mNameToIndexInput;

        std::vector<PinUI> mOutputPins;
        std::unordered_map<std::string, uint32_t> mNameToIndexOutput;

        uint32_t mGuiNodeID;
        RenderPassReflection mReflection;
    };

    class dlldecl RenderGraphUI
    {
    public:
        RenderGraphUI();

        RenderGraphUI(const RenderGraph::SharedPtr& pGraph, const std::string& graphName);

        ~RenderGraphUI();

        /** Display enter graph in GUI.
        */
        void renderUI(RenderContext* pContext, Gui* pGui);

        /** Clear graph ui for rebuilding node graph
        */
        void reset();

        /** Set ui to rebuild all display data before next render ui
        */
        void setToRebuild() { mRebuildDisplayData = true; }

        /** Writes out all the changes made to the graph
        */
        void writeUpdateScriptToFile(RenderContext* pContext, const std::string& filePath, float lastFrameTimes);

        /** function used to add an edge for the internally referenced render graph and update ui data
         */
        bool addLink(const std::string& srcPass, const std::string& dstPass, const std::string& srcField, const std::string& dstField, uint32_t& color);

        /** function used to remove edge referenced graph and update ui data
        */
        void removeEdge(const std::string& srcPass, const std::string& dstPass, const std::string& srcField, const std::string& dstField);

        /** function used to remove pass on referenced graph and update ui data
        */
        void removeRenderPass(const std::string& name);

        /** function used to add a graph output on referenced graph from one string
        */
        void addOutput(const std::string& outputParam);

        /** function used to add a graph output on referenced graph and update ui data
        */
        void addOutput(const std::string& outputPass, const std::string& outputField);

        /** function used to remove graph output on referenced graph and update ui data
        */
        void removeOutput(const std::string& outputPass, const std::string& outputField);

        /** function used to add a new node for a render pass referenced graph and update ui data
        */
        void addRenderPass(const std::string& name, const std::string& nodeTypeName);

        /** Get an execution order for graph based on the visual layout of the graph
        */
        std::vector<uint32_t> getPassOrder();

        /** Returns the current log from the events in the editor
        */
        std::string getCurrentLog() const { return mLogString; }

        /** Toggle building up delta changes for live preview
        */
        void setRecordUpdates(bool recordUpdates);

        /** Clears the current log
        */
        void clearCurrentLog() { mLogString.clear();  }

        /** Update change for the graph based on script
        */
        void updateGraph(RenderContext* pContext);

        /** Get name of reference graph
        */
        std::string getName() { return mRenderGraphName; }

    private:
        // forward declaration. private to avoid initialization outside of implementation file
        class NodeGraphEditorGui;
        class RenderGraphNode;

        /** Updates structure for drawing the GUI graph
        */
        void updateDisplayData(RenderContext* pContext);

        /** Updates information about pin connections and graph output.
        */
        void updatePins(bool addLinks = true);

        /** Helper function to calculate position of the next node in execution order
        */
        float2 getNextNodePosition(uint32_t nodeID);

        /** Renders specialized pop up menu.
        */
        void renderPopupMenu();

        /** Displays pop-up message if can auto resolve on an edge
        */
        bool autoResolveWarning(const std::string& srcString, const std::string& dstString);

        /** String containing the most recent log results from and isValid render graph call
        */
        std::string mLogString;

        // start with reference of render graph
        RenderGraph::SharedPtr mpRenderGraph;

        RenderGraphIR::SharedPtr mpIr;

        float2 mNewNodeStartPosition{ -40.0f, 100.0f };
        float mMaxNodePositionX = 0.0f;

        std::unordered_set<std::string> mAllNodeTypeStrings;
        std::vector<const char*> mAllNodeTypes;

        std::unordered_map <std::string, RenderPassUI> mRenderPassUI;

        std::unordered_map <std::string, uint32_t> mInputPinStringToLinkID;

        // maps output pin name to input pin ids. Pair first is pin id, second is node id
        std::unordered_map <std::string, std::vector< std::pair<uint32_t, uint32_t > > > mOutputToInputPins;

        // if in external editing mode, building list of commands for changes to send to the other process
        std::string mUpdateCommands;
        std::string mLastCommand;
        bool mRecordUpdates = false;

        // to avoid attempting to write changes every frame.
        float mTimeSinceLastUpdate = 0.0f;
        bool mDisplayDragAndDropPopup = false;
        bool mAddedFromDragAndDrop = false;
        std::string  mNextPassName = "";
        std::string mRenderGraphName;
        bool mDisplayAutoResolvePopup = true;

        // internal node GUi structure
        std::shared_ptr<NodeGraphEditorGui> mpNodeGraphEditor;

        // Flag to re-traverse the graph and build on of the intermediate data again.
        bool mRebuildDisplayData = true;
        bool mShouldUpdate = false;
    };
}
