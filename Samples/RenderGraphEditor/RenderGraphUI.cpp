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
    static ImGui::NodeGraphEditor sNodeGraphEditor;

    class RenderGraphNode : public ImGui::Node
    {
    public:
        bool mDisplayProperties;
        RenderPass* mpRenderPass;

        std::string getInputName(uint32_t index)
        {
            return InputNames[index];
        }

        std::string getOutputName(uint32_t index)
        {
            return OutputNames[index];
        }

        // Allow the rendergraphui to set the position of each node
        void SetPos(const glm::vec2& pos)
        {
            Pos.x = pos.x;
            Pos.y = pos.y;
        }

        // render Gui within the nodes
        static bool renderUI(ImGui::FieldInfo& field)
        {
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
            
            if (pRenderPass == nullptr)
            {
                logWarning("Custom field has no valid render pass information.");
                return false;
            }

            auto passData = RenderGraphEditor::sGetRenderPassData[pRenderPass->getTypeName()](pRenderPass);
            std::vector<RenderPass::PassData::Field>& fields = passData.inputs;
            
            ImVec2 oldScreenPos = ImGui::GetCursorScreenPos();
            ImVec2 currentScreenPos{ sNodeGraphEditor.offset.x  + pCurrentNode->Pos.x * ImGui::GetCurrentWindow()->FontWindowScale, 
                sNodeGraphEditor.offset.y + pCurrentNode->Pos.y * ImGui::GetCurrentWindow()->FontWindowScale };
            ImVec2 pinRectBoundsOffsetx{ kPinRadius, kPinRadius * 4.0f };

            // TODO the pin colors need to be taken from the global style
            ImU32 pinColor = 0xFFFFFFFF;
            
            float slotNum = 1.0f;
            float pinOffsetx = kPinRadius * 2.0f;
            

            for (uint32_t i = 0; i < 2; ++i)
            {
                for (const auto& field : fields)
                {
                    // custom pins as an extension of the built ones
                    ImVec2 inputPos = currentScreenPos;
                    inputPos.y += pCurrentNode->Size.y * (slotNum / static_cast<float>(fields.size() + 1));

                    if (ImGui::IsMouseHoveringRect(ImVec2(inputPos.x + pinRectBoundsOffsetx.x, inputPos.y - kPinRadius), ImVec2(inputPos.x + pinRectBoundsOffsetx.y, inputPos.y + kPinRadius)))
                    {
                        ImGui::GetOverlayDrawList()->AddCircleFilled(ImVec2(inputPos.x, inputPos.y), kPinRadius, pinColor);
                    }
                    else
                    {
                        ImGui::GetOverlayDrawList()->AddCircle(ImVec2(inputPos.x, inputPos.y), kPinRadius, pinColor);
                    }
                    ImGui::SetCursorScreenPos({ inputPos.x + pinOffsetx - ((pinOffsetx < 0.0f) ? ImGui::CalcTextSize(field.name.c_str()).x : 0.0f), inputPos.y - kPinRadius });

                    slotNum++;
                    gpGui->addText(field.name.c_str());
                }

                // reset and set up offsets for the output pins
                slotNum = 1.0f;
                currentScreenPos.x += pCurrentNode->Size.x;
                pinOffsetx *= -1.0f;
                pinRectBoundsOffsetx.x = -kPinRadius * 2.0f;
                pinRectBoundsOffsetx.y = kPinRadius;
                fields = passData.outputs;
            }

            ImGui::SetCursorScreenPos(oldScreenPos);
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

            // add dummy field for window spacing
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
        return spCurrentGraphUI->mRenderGraphRef.addEdge(srcPass + "." + srcField, dstPass + "." + dstField);
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
    }

    static void setNode(ImGui::Node*& node, ImGui::NodeGraphEditor::NodeState state, ImGui::NodeGraphEditor& editor)
    {
        if (!editor.isInited() && state == ImGui::NodeGraphEditor::NodeState::NS_DELETED)
        {
            spCurrentGraphUI->removeRenderPass(node->getName());
        }
    }

    static void setLink(const ImGui::NodeLink& link, ImGui::NodeGraphEditor::LinkState state, ImGui::NodeGraphEditor& editor)
    {
        // if (ImGui::IsMouseDown(0))
        // {
        //     return;
        // }

        if (state == ImGui::NodeGraphEditor::LinkState::LS_ADDED)
        {
            RenderGraphNode* inputNode = static_cast<RenderGraphNode*>(link.InputNode), 
                           * outputNode = static_cast<RenderGraphNode*>(link.OutputNode);

            bool addEdgeStatus = spCurrentGraphUI->addLink(inputNode->getName(), outputNode->getName(), inputNode->getOutputName(link.InputSlot), outputNode->getInputName(link.OutputSlot));

            // immediately remove link if it is not a legal edge in the render graph
            if (!editor.isInited() && !addEdgeStatus) //  only call after graph is setup
            {
                // does not call link callback surprisingly enough
                editor.removeLink(link.InputNode, link.InputSlot, link.OutputNode, link.OutputSlot);
            }
        }
    }

    void RenderPassUI::addUIPin(const std::string& fieldName, uint32_t guiPinID, bool isInput)
    {
        PinUIData pinUIData;
        pinUIData.mGuiPinID = guiPinID;
        pinUIData.mIsInput = isInput;
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

        uint32_t debugPinIndex = 0;
        const ImVec2& mousePos = ImGui::GetMousePos();

        // move this out of a button next
        //mRebuildDisplayData |= pGui->addButton("refresh");


        if (sRebuildDisplayData)
        {
            sRebuildDisplayData = false;
            sNodeGraphEditor.setNodeCallback(nullptr);
            sNodeGraphEditor.inited = false;
        }

        sNodeGraphEditor.setLinkCallback(setLink);
        sNodeGraphEditor.setNodeCallback(setNode);
        
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
                }

                ImGui::EndDragDropTarget();
            }
            
            // ? if (!sRebuildDisplayData)
            {
                return;
            }
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

        for (auto& nodeTypeString : mAllNodeTypeStrings)
        {
            mAllNodeTypes.push_back(nodeTypeString.c_str());
        }

        if (mAllNodeTypes.size())
        {
            sNodeGraphEditor.registerNodeTypes(mAllNodeTypes.data(), static_cast<uint32_t>(mAllNodeTypes.size()), createNode);
        }

        spCurrentGraphUI = this;
        spIDToNode.clear();

        for (auto& currentPass : mRenderPassUI)
        {
            auto& currentPassUI = currentPass.second;

            gOutputsString.clear();
            gInputsString.clear();

            // display name for the render pass
            pGui->addText(currentPass.first.c_str()); // might have to do this within the callback


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
                    pGui->addText(currentPinName.c_str(), true);
                    gInputsString += gInputsString.size() ? (";" + currentPinName) : currentPinName;
                }
                else
                {
                    gOutputsString += gOutputsString.size() ? (";" + currentPinName) : currentPinName;
                }

                debugPinIndex++;
            }

            gName = currentPass.first;
            gGuiNodeID = currentPassUI.mGuiNodeID;
            gpCurrentRenderPass = mRenderGraphRef.getRenderPass(currentPass.first).get();

            if (!sNodeGraphEditor.getAllNodesOfType(currentPassUI.mGuiNodeID, nullptr, false))
            {
                glm::vec2 nextPosition = getNextNodePosition(gGuiNodeID);
                spIDToNode[gGuiNodeID] = sNodeGraphEditor.addNode(gGuiNodeID, ImVec2(nextPosition.x, nextPosition.y));
            }
        }
        
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
                    const auto& inputPins = mOutputToInputPins.find(currentPinName);
                    if (inputPins != mOutputToInputPins.end())
                    {
                        for (const auto& connectedPin : (inputPins->second))
                        {
                            if(!sNodeGraphEditor.isLinkPresent(spIDToNode[currentPassUI.mGuiNodeID], currentPinUI.mGuiPinID,
                                    spIDToNode[connectedPin.second], connectedPin.first) )
                            {
                                sNodeGraphEditor.addLink(spIDToNode[currentPassUI.mGuiNodeID], currentPinUI.mGuiPinID,
                                    spIDToNode[connectedPin.second], connectedPin.first);
                            }
                        }
                    }
                }
            }
        }


        sNodeGraphEditor.render();
    }

    glm::vec2 RenderGraphUI::getNextNodePosition(uint32_t nodeID)
    {
        mNewNodeStartPosition = glm::vec2(-40.0f, 100.0f);
        
        for (const auto& passID : mRenderGraphRef.mExecutionList)
        {
            mNewNodeStartPosition.x += 256.0f;
            if (passID == nodeID)
            {
                break;
            }
        }

        return mNewNodeStartPosition;
    }

    void RenderGraphUI::updateDisplayData()
    {
        uint32_t nodeIndex = 0;

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

        std::string log;
        mRenderGraphRef.compile(log);

        mRenderPassUI.clear();

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

            // add all of the incoming connections
            for (uint32_t i = 0; i < pCurrentPass->getIncomingEdgeCount(); ++i)
            {
                auto currentEdge = mRenderGraphRef.mEdgeData[pCurrentPass->getIncomingEdge(i)];
                mOutputToInputPins[currentEdge.srcField].push_back(std::make_pair(inputPinIndex, renderPassUI.mGuiNodeID));
                renderPassUI.addUIPin(currentEdge.dstField, inputPinIndex++, true);
                nodeConnected.insert(nameToIndex.first + currentEdge.dstField);
            }

            // add all of the outgoing connections
            for (uint32_t i = 0; i < pCurrentPass->getOutgoingEdgeCount(); ++i)
            {
                auto currentEdge = mRenderGraphRef.mEdgeData[pCurrentPass->getOutgoingEdge(i)];
                renderPassUI.addUIPin(currentEdge.srcField, outputPinIndex++, false);
                nodeConnected.insert(nameToIndex.first + currentEdge.srcField);
            }

            // Now we know which nodes are connected within the graph and not

            auto passData = 
                RenderGraphEditor::sGetRenderPassData[mRenderGraphRef.mNodeData[nameToIndex.second]->getTypeName()]
                    (mRenderGraphRef.mNodeData[nameToIndex.second].get());

            for (const auto& inputNode : passData.inputs)
            {
                if (nodeConnected.find(nameToIndex.first + inputNode.name) == nodeConnected.end())
                {
                    renderPassUI.addUIPin(inputNode.name, inputPinIndex++, true);
                }

                // add the details description for each pin

            }

            for (const auto& outputNode : passData.outputs)
            {
                if (nodeConnected.find(nameToIndex.first + outputNode.name) == nodeConnected.end())
                {
                    renderPassUI.addUIPin(outputNode.name, outputPinIndex++, false);
                }

                // add the details description for each pin
            }

            mRenderPassUI.emplace(std::make_pair(nameToIndex.first, std::move(renderPassUI)));
        }
    }
}