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

#include "RenderGraphUI.h"

#include "Utils/Gui.h"

// TODO get rid of this too

#include "Externals/dear_imgui_addons/imguinodegrapheditor/imguinodegrapheditor.h"

#include "RenderGraphEditor.h"
#include "RenderGraphLoader.h"

// TODO Don't do this
#include "Externals/dear_imgui/imgui.h"
#include "Externals/dear_imgui/imgui_internal.h"

namespace Falcor
{
    std::string gName;
    std::string gOutputsString;
    std::string gInputsString;
    Gui* gpGui;
    uint32_t gGuiNodeID;
    RenderPass* gpCurrentRenderPass; // This is for renderUI callback

    const float kPinRadius = 6.0f;

    static std::unordered_map<uint32_t, ImGui::Node*> spIDToNode;
    static RenderGraphUI* spCurrentGraphUI = nullptr;

    class NodeGraphEditorGui : public ImGui::NodeGraphEditor
    {
    public:
        ImGui::Node * pGraphOutputNode = nullptr;

        void reset()
        {
            inited = false;
        }

        glm::vec2 getOffsetPos()
        {
            return { offset.x, offset.y };
        }
    };

    static NodeGraphEditorGui sNodeGraphEditor;

    class RenderGraphNode : public ImGui::Node
    {
    public:
        bool mDisplayProperties;
        bool mOutputPinConnected[IMGUINODE_MAX_OUTPUT_SLOTS];
        bool mInputPinConnected[IMGUINODE_MAX_INPUT_SLOTS];
        RenderPass* mpRenderPass;

        bool pinIsConnected(uint32_t id, bool isInput)
        {
            assert(isInput ? id < IMGUINODE_MAX_INPUT_SLOTS : id < IMGUINODE_MAX_OUTPUT_SLOTS);
            return isInput ? mInputPinConnected[id] : mOutputPinConnected[id];
        }

        std::string getInputName(uint32_t index)
        {
            return InputNames[index];
        }

        std::string getOutputName(uint32_t index)
        {
            return OutputNames[index];
        }

        ImGui::FieldInfoVector& getFields()
        {
            return fields;
        }

        // Allow the rendergraphui to set the position of each node
        void setPos(const glm::vec2& pos)
        {
            Pos.x = pos.x;
            Pos.y = pos.y;
        }

        glm::vec2 getPos()
        {
            return {Pos.x, Pos.y};
        }

        // render Gui within the nodes
        static bool renderUI(ImGui::FieldInfo& field)
        {
            if (sNodeGraphEditor.isInited())
            {
                return false;
            }

            std::string dummyText; dummyText.resize(64, ' ');
            gpGui->addText(dummyText.c_str());
            gpGui->addText(dummyText.c_str());
            gpGui->addText(dummyText.c_str());

            // static_cast<RenderPass*>(field.userData)->onGuiRender(nullptr, gpGui);

            // with this library there is no way of modifying the positioning of the labels on the node
            // manually making labels to align correctly from within the node

            // grab all of the fields again
            RenderGraphNode* pCurrentNode = static_cast<RenderGraphNode*>(field.userData);
            RenderPass* pRenderPass = static_cast<RenderPass*>(pCurrentNode->mpRenderPass);

            ImVec2 oldScreenPos = ImGui::GetCursorScreenPos();
            ImVec2 currentScreenPos{ sNodeGraphEditor.offset.x  + pCurrentNode->Pos.x * ImGui::GetCurrentWindow()->FontWindowScale, 
                sNodeGraphEditor.offset.y + pCurrentNode->Pos.y * ImGui::GetCurrentWindow()->FontWindowScale };
            ImVec2 pinRectBoundsOffsetx{ -kPinRadius * 2.0f, kPinRadius * 4.0f };

            // TODO the pin colors need to be taken from the global style
            ImU32 pinColor = 0xFFFFFFFF;
            
            float slotNum = 1.0f;
            float pinOffsetx = kPinRadius * 2.0f;
            uint32_t pinCount = static_cast<uint32_t>(pCurrentNode->InputsCount);
            bool isInputs = true;

            for (uint32_t i = 0; i < 2; ++i)
            {
                
                for (uint32_t i = 0; i < pinCount; ++i)
                {
                    // custom pins as an extension of the built ones
                    ImVec2 inputPos = currentScreenPos;
                    inputPos.y += pCurrentNode->Size.y * ((i + 1) / static_cast<float>(pinCount + 1));

                    // fill in circle for the pin if connected to a link
                    if (pCurrentNode->pinIsConnected(i, isInputs))
                    {
                        ImGui::GetOverlayDrawList()->AddCircleFilled(ImVec2(inputPos.x, inputPos.y), kPinRadius,pinColor);
                    }

                    if (ImGui::IsMouseHoveringRect(ImVec2(inputPos.x + pinRectBoundsOffsetx.x, inputPos.y - kPinRadius), ImVec2(inputPos.x + pinRectBoundsOffsetx.y, inputPos.y + kPinRadius)))
                    {
                        ImGui::GetOverlayDrawList()->AddCircleFilled(ImVec2(inputPos.x, inputPos.y), kPinRadius, ImGui::GetColorU32(ImGui::NodeGraphEditor::GetStyle().color_node_title));
                    }
                    else
                    {
                        ImGui::GetOverlayDrawList()->AddCircle(ImVec2(inputPos.x, inputPos.y), kPinRadius, pinColor);
                    }
                    ImGui::SetCursorScreenPos({ inputPos.x + pinOffsetx - ((pinOffsetx < 0.0f) ? ImGui::CalcTextSize(isInputs ? pCurrentNode->InputNames[i] : pCurrentNode->OutputNames[i]).x : 0.0f), inputPos.y - kPinRadius });

                    slotNum++;
                    gpGui->addText(isInputs ? pCurrentNode->InputNames[i] : pCurrentNode->OutputNames[i]);
                }

                // reset and set up offsets for the output pins
                slotNum = 1.0f;
                currentScreenPos.x += pCurrentNode->Size.x;
                pinOffsetx *= -1.0f;
                pinRectBoundsOffsetx.y = kPinRadius;
                pinCount = static_cast<uint32_t>(pCurrentNode->OutputsCount);
                isInputs = false;
            }

            ImGui::SetCursorScreenPos(oldScreenPos);
            
            if (!pRenderPass)
            {
                // special formatting for render graph output nodes
                
            }
            
            
            gpGui->addText(dummyText.c_str());

            return false;
        }

        static RenderGraphNode* create(const ImVec2& pos)
        {
            RenderGraphNode* node = (RenderGraphNode*)ImGui::MemAlloc(sizeof(RenderGraphNode));
            IM_PLACEMENT_NEW(node) RenderGraphNode();

            node->init(gName.c_str(), pos, gInputsString.c_str(), gOutputsString.c_str(), gGuiNodeID);

            node->overrideTitleBgColor = ImGui::GetColorU32(ImGuiCol_Header);
            node->mpRenderPass = gpCurrentRenderPass;

            node->fields.addFieldCustom(static_cast<ImGui::FieldInfo::RenderFieldDelegate>(renderUI), nullptr, node);
            
            return node;
        }
    private:
    };


    static ImGui::Node* createNode(int, const ImVec2& pos, const ImGui::NodeGraphEditor&)
    {
        return RenderGraphNode::create(pos);
    }

    bool RenderGraphUI::addLink(const std::string& srcPass, const std::string& dstPass, const std::string& srcField, const std::string& dstField)
    {
        // outputs warning if edge could not be created 
        bool createdEdge = spCurrentGraphUI->mRenderGraphRef.addEdge(srcPass + "." + srcField, dstPass + "." + dstField);

        // update the ui to reflect the connections. This data is used for removal
        if (createdEdge)
        {
            spCurrentGraphUI->mRenderPassUI[srcPass].mPins[srcField].mConnectedPinName = dstField;
            spCurrentGraphUI->mRenderPassUI[srcPass].mPins[srcField].mConnectedNodeName = dstPass;
            spCurrentGraphUI->mRenderPassUI[dstPass].mPins[dstField].mConnectedPinName = srcField;
            spCurrentGraphUI->mRenderPassUI[dstPass].mPins[dstField].mConnectedNodeName = srcPass;

            sRebuildDisplayData = true;
        }

        return createdEdge;
    }

    void RenderGraphUI::removeRenderPass(const std::string& name)
    {
        spCurrentGraphUI->sRebuildDisplayData = true;
        spCurrentGraphUI->mRenderGraphRef.removeRenderPass(name);
    }

    void RenderGraphUI::removeLink()
    {
    }

    RenderGraphUI::RenderGraphUI(RenderGraph& renderGraphRef)
        : mRenderGraphRef(renderGraphRef), mNewNodeStartPosition(-40.0f, 100.0f)
    {
        sNodeGraphEditor.clear();
    }

    static void setNode(ImGui::Node*& node, ImGui::NodeGraphEditor::NodeState state, ImGui::NodeGraphEditor& editor)
    {
        if (!editor.isInited() && state == ImGui::NodeGraphEditor::NodeState::NS_DELETED)
        {
            static_cast<RenderGraphNode*>(node)->getFields().clear();
            spCurrentGraphUI->removeRenderPass(node->getName());
        }
    }

    static void setLink(const ImGui::NodeLink& link, ImGui::NodeGraphEditor::LinkState state, ImGui::NodeGraphEditor& editor)
    {
        if (state == ImGui::NodeGraphEditor::LinkState::LS_ADDED)
        {
            
            RenderGraphNode* inputNode = static_cast<RenderGraphNode*>(link.InputNode), 
                           * outputNode = static_cast<RenderGraphNode*>(link.OutputNode);

            if (outputNode == sNodeGraphEditor.pGraphOutputNode)
            {
                return;
            }

            bool addEdgeStatus = spCurrentGraphUI->addLink(inputNode->getName(), outputNode->getName(), inputNode->getOutputName(link.InputSlot), outputNode->getInputName(link.OutputSlot));

            // immediately remove link if it is not a legal edge in the render graph
            if (!editor.isInited() && !addEdgeStatus) //  only call after graph is setup
            {
                // does not call link callback surprisingly enough
                editor.removeLink(link.InputNode, link.InputSlot, link.OutputNode, link.OutputSlot);
                return;
            }
        }
    }

    void RenderPassUI::addUIPin(const std::string& fieldName, uint32_t guiPinID, bool isInput, const std::string& connectedPinName, const std::string& connectedNodeName, bool isGraphOutput)
    {
        PinUIData pinUIData;
        pinUIData.mGuiPinID = guiPinID;
        pinUIData.mIsInput = isInput;
        pinUIData.mConnectedPinName = connectedPinName;
        pinUIData.mConnectedNodeName = connectedNodeName;
        pinUIData.mIsGraphOutput = isGraphOutput;
        mPins.insert(std::make_pair(fieldName, pinUIData));
    }

    void RenderPassUI::renderUI(Gui* pGui)
    {

    }

    bool RenderGraphUI::sRebuildDisplayData = true;

    void RenderGraphUI::renderUI(Gui* pGui)
    {
        gpGui = pGui;


        ImGui::GetIO().FontAllowUserScaling = true;

        sNodeGraphEditor.show_top_pane = false;
        sNodeGraphEditor.show_node_copy_paste_buttons = false;
        sNodeGraphEditor.show_connection_names = false;
        sNodeGraphEditor.show_left_pane = false;

        sNodeGraphEditor.setLinkCallback(setLink);
        sNodeGraphEditor.setNodeCallback(setNode);
        
        const ImVec2& mousePos = ImGui::GetMousePos();
        
        // update the deleted links from the GUI since the library doesn't call its own callback
        
        if (sRebuildDisplayData)
        {
            sRebuildDisplayData = false;
            sNodeGraphEditor.setNodeCallback(nullptr);
            sNodeGraphEditor.reset();
        }
        else
        {
            drawPins(false);

            if (sRebuildDisplayData)
            {
                sRebuildDisplayData = false;
                sNodeGraphEditor.setNodeCallback(nullptr);
                sNodeGraphEditor.reset();
            }
        }

        if (!sNodeGraphEditor.isInited())
        {
            sNodeGraphEditor.render();

            if (ImGui::BeginDragDropTarget())
            {
                // Accept and run script from drag and drop
                auto dragDropPayload = ImGui::AcceptDragDropPayload("RenderPassScript");

                if (dragDropPayload)
                {
                    RenderGraphLoader::ExecuteStatement(std::string(static_cast<const char*>(dragDropPayload->Data), dragDropPayload->DataSize), mRenderGraphRef);
                    mNewNodeStartPosition = { -sNodeGraphEditor.offset.x + mousePos.x, -sNodeGraphEditor.offset.y + mousePos.y };
                    mNewNodeStartPosition /= ImGui::GetCurrentWindow()->FontWindowScale;
                }
                else
                {
                    mNewNodeStartPosition = { -40.0f, 100.0f };
                }

                ImGui::EndDragDropTarget();
            }

            return;
        }

        updateDisplayData();


        mAllNodeTypes.clear();

        // reset internal data of node if all nodes deleted
        if (!mRenderPassUI.size())
        {
            sNodeGraphEditor.clear();
            sNodeGraphEditor.render();
            return;
        }

        mAllNodeTypes.push_back("GraphOutputNode");

        for (auto& nodeTypeString : mAllNodeTypeStrings)
        {
            mAllNodeTypes.push_back(nodeTypeString.c_str());
        }

        sNodeGraphEditor.registerNodeTypes(mAllNodeTypes.data(), static_cast<uint32_t>(mAllNodeTypes.size()), createNode);

        spCurrentGraphUI = this;

        // create graph output node first
        if (!sNodeGraphEditor.pGraphOutputNode)
        {
            gOutputsString.clear();
            gInputsString.clear();

            gName = "GraphOutput";
            gGuiNodeID = 0;
            gpCurrentRenderPass = nullptr;

            for (const auto& graphOutput : mRenderGraphRef.mOutputs)
            {
                gInputsString += gInputsString.size() ? (";" + graphOutput.field) : graphOutput.field;
            }

            sNodeGraphEditor.pGraphOutputNode = sNodeGraphEditor.addNode(0, { 0, 0 });
        }
        else
        {
            std::string inputsString;

            for (const auto& graphOutput : mRenderGraphRef.mOutputs)
            {
                inputsString += inputsString.size() ? (";" + graphOutput.field) : graphOutput.field;
            }

            sNodeGraphEditor.removeAnyLinkFromNode(sNodeGraphEditor.pGraphOutputNode);
            // can't rely on this func to delete links cause its wrong and crashes
            sNodeGraphEditor.overrideNodeInputSlots(sNodeGraphEditor.pGraphOutputNode, inputsString.c_str());
        }
        

        for (auto& currentPass : mRenderPassUI)
        {
            auto& currentPassUI = currentPass.second;

            gOutputsString.clear();
            gInputsString.clear();

            for (const auto& currentPin : currentPassUI.mPins)
            {
                // Connect the graph nodes for each of the edges
                // need to iterate in here in order to use the right indices
                const RenderPassUI::PinUIData& currentPinUI = currentPin.second;
                const std::string& currentPinName = currentPin.first;
                bool isInput = currentPinUI.mIsInput;

                // draw label for input pin
                if (isInput)
                {
                    gInputsString += gInputsString.size() ? (";" + currentPinName) : currentPinName;
                }
                else
                {
                    gOutputsString += gOutputsString.size() ? (";" + currentPinName) : currentPinName;
                }
            }

            gName = currentPass.first;
            gGuiNodeID = currentPassUI.mGuiNodeID;
            gpCurrentRenderPass = mRenderGraphRef.getRenderPass(currentPass.first).get();

            if (!sNodeGraphEditor.getAllNodesOfType(currentPassUI.mGuiNodeID, nullptr, false))
            {
                glm::vec2 nextPosition = getNextNodePosition(mRenderGraphRef.getPassIndex(gName));
                spIDToNode[gGuiNodeID] = sNodeGraphEditor.addNode(gGuiNodeID, ImVec2(nextPosition.x, nextPosition.y));
                
            }
        }

        drawPins();

        sNodeGraphEditor.render();
    }

    void RenderGraphUI::reset()
    {
        sNodeGraphEditor.reset();
        sNodeGraphEditor.clear();
        sRebuildDisplayData = true;
    }

    void RenderGraphUI::drawPins(bool addLinks)
    {
        //  Draw pin connections. All the nodes have to be added to the GUI before the connections can be drawn
        for (auto& currentPass : mRenderPassUI)
        {
            auto& currentPassUI = currentPass.second;

            for (const auto& currentPin : currentPassUI.mPins)
            {
                const RenderPassUI::PinUIData& currentPinUI = currentPin.second;
                const std::string& currentPinName = currentPin.first;
                bool isInput = currentPinUI.mIsInput;

                // draw label for input pin
                if (!isInput)
                {
                    if (addLinks)
                    {
                        const auto& inputPins = mOutputToInputPins.find(currentPinName);
                        if (inputPins != mOutputToInputPins.end())
                        {
                            for (const auto& connectedPin : (inputPins->second))
                            {
                                if (!sNodeGraphEditor.isLinkPresent(spIDToNode[currentPassUI.mGuiNodeID], currentPinUI.mGuiPinID,
                                    spIDToNode[connectedPin.second], connectedPin.first))
                                {
                                    sNodeGraphEditor.addLink(spIDToNode[currentPassUI.mGuiNodeID], currentPinUI.mGuiPinID,
                                        spIDToNode[connectedPin.second], connectedPin.first);

                                    static_cast<RenderGraphNode*>(spIDToNode[connectedPin.second])->mInputPinConnected[connectedPin.first] = true;
                                    static_cast<RenderGraphNode*>(spIDToNode[currentPassUI.mGuiNodeID])->mOutputPinConnected[currentPinUI.mGuiPinID] = true;
                                }
                            }
                        }
                    }
                    else
                    {
                        if (currentPinUI.mIsGraphOutput)
                        {
                            // get the input pin for the graph output node
                            uint32_t graphOutPinID = 0;

                            for (const auto& output : mRenderGraphRef.mOutputs)
                            {
                                if (output.field == currentPinName)
                                {
                                    break;
                                }

                                graphOutPinID++;
                            }

                            if (!sNodeGraphEditor.isLinkPresent(spIDToNode[currentPassUI.mGuiNodeID], currentPinUI.mGuiPinID,
                                sNodeGraphEditor.pGraphOutputNode, graphOutPinID))
                            {
                                sNodeGraphEditor.addLink(spIDToNode[currentPassUI.mGuiNodeID], currentPinUI.mGuiPinID,
                                    sNodeGraphEditor.pGraphOutputNode, graphOutPinID, false, ImGui::GetColorU32({ 0.0f, 1.0f, 0.0f, 0.71f }));
                            }
                        }
                    }
                }
                else if(isInput && !addLinks)
                {
                    if (!currentPinUI.mConnectedNodeName.size())
                    {
                        return;
                    }

                    std::pair<uint32_t, uint32_t> inputIDs{ currentPinUI.mGuiPinID, currentPassUI.mGuiNodeID };
                    const auto& connectedNodeUI = mRenderPassUI[currentPinUI.mConnectedNodeName];
                    uint32_t inputPinID = connectedNodeUI.mPins.find(currentPinUI.mConnectedPinName)->second.mGuiPinID;

                    if (!sNodeGraphEditor.isLinkPresent(spIDToNode[connectedNodeUI.mGuiNodeID], inputPinID,
                        spIDToNode[inputIDs.second],inputIDs.first ))
                    {
                        auto edgeID = mInputPinStringToLinkID.find(currentPass.first + "." + currentPinName);
                        assert(edgeID != mInputPinStringToLinkID.end());
                        mRenderGraphRef.removeEdge(edgeID->second);

                        static_cast<RenderGraphNode*>(spIDToNode[inputIDs.second])->mInputPinConnected[inputIDs.first] = false;
                        static_cast<RenderGraphNode*>(spIDToNode[connectedNodeUI.mGuiNodeID])->mOutputPinConnected[inputPinID] = false;
                        
                        sRebuildDisplayData = true;

                        return;
                    }
                }
            }
        }
    }

    glm::vec2 RenderGraphUI::getNextNodePosition(uint32_t nodeID)
    {
        glm::vec2 newNodePosition = mNewNodeStartPosition;

        for (const auto& passID : mRenderGraphRef.mExecutionList)
        {
            newNodePosition.x += 512.0f;

            if (passID == nodeID)
            {
                const DirectedGraph::Node* pNode = mRenderGraphRef.mpGraph->getNode(nodeID);
                for (uint32_t i = 0; i < pNode->getIncomingEdgeCount(); ++i)
                {
                    uint32_t outgoingEdgeCount = mRenderGraphRef.mpGraph->getNode(mRenderGraphRef.mpGraph->getEdge(pNode->getIncomingEdge(i))->getSourceNode())->getOutgoingEdgeCount();
                    
                    if (outgoingEdgeCount > pNode->getIncomingEdgeCount())
                    {
                        // move down by index in 
                        newNodePosition.y += 256.0f * (outgoingEdgeCount - pNode->getIncomingEdgeCount() );
                    }
                }

                break;
            }
        }

        RenderGraphNode* pGraphOutputNode = static_cast<RenderGraphNode*>(sNodeGraphEditor.pGraphOutputNode);
        glm::vec2 graphOutputNodePos = pGraphOutputNode->getPos();
        if (graphOutputNodePos.x < newNodePosition.x)
        {
            pGraphOutputNode->setPos({ newNodePosition.x + 512.0f, newNodePosition.y });
        }

        return newNodePosition;
    }

    void RenderGraphUI::updateDisplayData()
    {
        uint32_t nodeIndex = 1;

        mOutputToInputPins.clear();

        // set of field names that have a connection and are represented in the graph
        std::unordered_set<std::string> nodeConnected;
        std::unordered_map<std::string, uint32_t> previousGuiNodeIDs;
        std::unordered_set<uint32_t> existingIDs;

        for (const auto& currentRenderPassUI : mRenderPassUI)
        {
            existingIDs.insert(currentRenderPassUI.second.mGuiNodeID);
            previousGuiNodeIDs.insert(std::make_pair(currentRenderPassUI.first, currentRenderPassUI.second.mGuiNodeID));
        }

        // std::string log;
        mRenderGraphRef.resolveExecutionOrder();

        mRenderPassUI.clear();
        mInputPinStringToLinkID.clear();

        // build information for displaying graph
        for (const auto& nameToIndex : mRenderGraphRef.mNameToIndex)
        {
            uint32_t inputPinIndex = 0;
            uint32_t outputPinIndex = 0;
            auto pCurrentPass = mRenderGraphRef.mpGraph->getNode(nameToIndex.second);
            RenderPassUI renderPassUI;

            mAllNodeTypeStrings.insert(nameToIndex.first);

            while (existingIDs.find(nodeIndex) != existingIDs.end())
            {
                nodeIndex++;
            }

            // keep the GUI id from the previous frame
            auto pPreviousID = previousGuiNodeIDs.find(nameToIndex.first);
            if (pPreviousID != previousGuiNodeIDs.end())
            {
                renderPassUI.mGuiNodeID = pPreviousID->second;
            }
            else
            {
                renderPassUI.mGuiNodeID = nodeIndex;
                nodeIndex++;
            }

            // clear and rebuild reflection for each pass. 
            renderPassUI.mReflection = RenderPassReflection();
            mRenderGraphRef.mNodeData[nameToIndex.second].pPass->reflect(renderPassUI.mReflection);

            // test to see if we have hit a graph output
            std::unordered_set<std::string> passGraphOutputs;

            for (const auto& output : mRenderGraphRef.mOutputs)
            {
                if (output.nodeId == nameToIndex.second)
                {
                    passGraphOutputs.insert(output.field);
                }
            }

            // add all of the incoming connections
            for (uint32_t i = 0; i < pCurrentPass->getIncomingEdgeCount(); ++i)
            {
                uint32_t edgeID = pCurrentPass->getIncomingEdge(i);
                auto currentEdge = mRenderGraphRef.mEdgeData[edgeID];
                uint32_t pinIndex = 0;

                while (renderPassUI.mReflection.getField(inputPinIndex).getName() != currentEdge.dstField)
                {
                    inputPinIndex++;
                }

                auto pSourceNode = mRenderGraphRef.mNodeData.find( mRenderGraphRef.mpGraph->getEdge(edgeID)->getSourceNode());
                assert(pSourceNode != mRenderGraphRef.mNodeData.end());

                
                renderPassUI.addUIPin(currentEdge.dstField, pinIndex, true, currentEdge.srcField, pSourceNode->second.nodeName);

                mOutputToInputPins[currentEdge.srcField].push_back(std::make_pair(pinIndex, renderPassUI.mGuiNodeID));
                std::string pinString = nameToIndex.first + "." + currentEdge.dstField;
                nodeConnected.insert(pinString);
                mInputPinStringToLinkID.insert(std::make_pair(pinString, edgeID));
            }

            // add all of the outgoing connections
            for (uint32_t i = 0; i < pCurrentPass->getOutgoingEdgeCount(); ++i)
            {
                uint32_t edgeID = pCurrentPass->getOutgoingEdge(i);
                auto currentEdge = mRenderGraphRef.mEdgeData[edgeID];
                bool isGraphOutput = passGraphOutputs.find(currentEdge.srcField) != passGraphOutputs.end();
                uint32_t pinIndex = 0;

                while (renderPassUI.mReflection.getField(outputPinIndex).getName() != currentEdge.srcField)
                {
                    outputPinIndex++;
                }
                
                auto pDestNode = mRenderGraphRef.mNodeData.find(mRenderGraphRef.mpGraph->getEdge(edgeID)->getSourceNode());
                assert(pDestNode != mRenderGraphRef.mNodeData.end());

                renderPassUI.addUIPin(currentEdge.srcField, pinIndex, false, currentEdge.dstField, pDestNode->second.nodeName, isGraphOutput);
                nodeConnected.insert(nameToIndex.first + "." + currentEdge.srcField);
            }

            // Now we know which nodes are connected within the graph and not

            for (uint32_t i = 0; i < renderPassUI.mReflection.getFieldCount(); ++i)
            {
                const auto& currentField = renderPassUI.mReflection.getField(i);

                if (currentField.getType() == RenderPassReflection::Field::Type::Input)
                {
                    if (nodeConnected.find(nameToIndex.first + "." + currentField.getName()) == nodeConnected.end())
                    {
                        renderPassUI.addUIPin(currentField.getName(), inputPinIndex++, true, "");
                    }
                }
                else
                {
                    if (nodeConnected.find(nameToIndex.first + "." + currentField.getName()) == nodeConnected.end())
                    {
                        bool isGraphOutput = passGraphOutputs.find(currentField.getName()) != passGraphOutputs.end();
                        renderPassUI.addUIPin(currentField.getName(), outputPinIndex++, false, "", "", isGraphOutput);
                    }
                }
                
            }

            mRenderPassUI.emplace(std::make_pair(nameToIndex.first, std::move(renderPassUI)));
        }
    }
}