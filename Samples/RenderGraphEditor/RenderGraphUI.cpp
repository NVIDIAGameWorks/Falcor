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

namespace Falcor
{
    std::string gName;
    std::string gOutputsString;
    std::string gInputsString;
    Gui* gpGui;
    uint32_t gGuiNodeID;
    RenderPass* gpCurrentRenderPass; // This is for renderUI callback

    static std::unordered_map<uint32_t, ImGui::Node*> spIDToNode;
    static RenderGraphUI* spCurrentGraphUI = nullptr;

    class RenderGraphNode : public ImGui::Node
    {
    public:
        bool mDisplayProperties;

        std::string getInputName(uint32_t index)
        {
            return InputNames[index];
        }

        std::string getOutputName(uint32_t index)
        {
            return OutputNames[index];
        }

        // render Gui within the nodes
        static bool renderUI(ImGui::FieldInfo& field)
        {
            std::string dummyText; dummyText.resize(64, ' ');
            gpGui->addText(dummyText.c_str());
            static_cast<RenderPass*>(field.userData)->onGuiRender(nullptr, gpGui);
            return false;
        }

        static RenderGraphNode* create(const ImVec2& pos)
        {
            RenderGraphNode* node = (RenderGraphNode*)ImGui::MemAlloc(sizeof(RenderGraphNode));
            IM_PLACEMENT_NEW(node) RenderGraphNode();

            node->init(gName.c_str(), pos, gInputsString.c_str(), gOutputsString.c_str(), gGuiNodeID);

            node->overrideTitleBgColor = ImGui::GetColorU32(ImGuiCol_Header);

            // add dummy field for window spacing
            node->fields.addFieldCustom(static_cast<ImGui::FieldInfo::RenderFieldDelegate>(renderUI), nullptr, gpCurrentRenderPass);
            

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
        : mRenderGraphRef(renderGraphRef), mNewNodeStartPosition(40.0f, 100.0f)
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
        if (state == ImGui::NodeGraphEditor::LinkState::LS_ADDED)
        {
            RenderGraphNode* inputNode = static_cast<RenderGraphNode*>(link.InputNode), 
                           * outputNode = static_cast<RenderGraphNode*>(link.OutputNode);

            bool addEdgeStatus = spCurrentGraphUI->addLink(inputNode->getName(), outputNode->getName(), inputNode->getOutputName(link.OutputSlot), outputNode->getInputName(link.InputSlot));

            // immediately remove link if it is not a legal edge in the render graph
            if (!editor.isInited() && !addEdgeStatus) //  only call after graph is setup
            {
                // does not call link callback surprisingly enough
                editor.removeLink(link.InputNode, link.InputSlot, link.OutputNode, link.OutputSlot);
            }
        }
        
        // TODO -- remove link ?? (Possibly remove connected nodes and read them??)
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
        static ImGui::NodeGraphEditor nodeGraphEditor;

        gpGui = pGui;

        nodeGraphEditor.show_top_pane = false;

        uint32_t debugPinIndex = 0;
        const ImVec2& mousePos = ImGui::GetMousePos();

        // handle deleting nodes and connections?
        std::unordered_set<uint32_t> nodeIDsToDelete;
        std::vector<std::string> nodeNamesToDelete;

        // move this out of a button next
        //mRebuildDisplayData |= pGui->addButton("refresh");
        
        if (sRebuildDisplayData)
        {
            sRebuildDisplayData = false;
            nodeGraphEditor.setNodeCallback(nullptr);
            nodeGraphEditor.inited = false;
        }

        nodeGraphEditor.setLinkCallback(setLink);
        nodeGraphEditor.setNodeCallback(setNode);

        if (!nodeGraphEditor.isInited())
        {
            nodeGraphEditor.render();

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
            nodeGraphEditor.clear();
            nodeGraphEditor.render();
            return;
        }

        for (auto& nodeTypeString : mAllNodeTypeStrings)
        {
            mAllNodeTypes.push_back(nodeTypeString.c_str());
        }

        if (mAllNodeTypes.size())
        {
            nodeGraphEditor.registerNodeTypes(mAllNodeTypes.data(), static_cast<uint32_t>(mAllNodeTypes.size()), createNode);
        }

        spCurrentGraphUI = this;
        spIDToNode.clear();

        for (auto& currentPass : mRenderPassUI)
        {
            auto& currentPassUI = currentPass.second;

            gOutputsString.clear();
            gInputsString.clear();

            // only worry about the GUI for the node if no deletion
            if (nodeIDsToDelete.find(currentPassUI.mGuiNodeID) != nodeIDsToDelete.end())
            {
                nodeNamesToDelete.push_back(currentPass.first);
            }

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

            if (!nodeGraphEditor.getAllNodesOfType(currentPassUI.mGuiNodeID, nullptr, false))
            {
                // TODO -- set up new positioning for adding nodes
                spIDToNode[gGuiNodeID] = nodeGraphEditor.addNode(gGuiNodeID, ImVec2(mNewNodeStartPosition.x, mNewNodeStartPosition.y));
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
                            if(!nodeGraphEditor.isLinkPresent(spIDToNode[currentPassUI.mGuiNodeID], currentPinUI.mGuiPinID,
                                    spIDToNode[connectedPin.second], connectedPin.first) )
                            {
                                nodeGraphEditor.addLink(spIDToNode[currentPassUI.mGuiNodeID], currentPinUI.mGuiPinID,
                                    spIDToNode[connectedPin.second], connectedPin.first);
                            }
                        }
                    }
                }
            }
        }

        nodeGraphEditor.render();

        // get rid of nodes for next frame
        for (const std::string& passName : nodeNamesToDelete)
        {
            mRenderGraphRef.removeRenderPass(passName);
        }
    }

    void RenderGraphUI::updateDisplayData()
    {
        uint32_t nodeIndex = 0;

        mOutputToInputPins.clear();

        // set of field names that have a connection and are represented in the graph
        std::unordered_set<std::string> nodeConnected;
        std::unordered_map<std::string, uint32_t> previousGuiNodeIDs;

        for (const auto& currentRenderPassUI : mRenderPassUI)
        {
            previousGuiNodeIDs.insert(std::make_pair(currentRenderPassUI.first, currentRenderPassUI.second.mGuiNodeID));
        }

        mRenderPassUI.clear();

        // build information for displaying graph
        for (const auto& nameToIndex : mRenderGraphRef.mNameToIndex)
        {
            uint32_t inputPinIndex = 0;
            uint32_t outputPinIndex = 0;
            auto pCurrentPass = mRenderGraphRef.mpGraph->getNode(nameToIndex.second);
            RenderPassUI renderPassUI;

            mAllNodeTypeStrings.insert(mRenderGraphRef.mNodeData[nameToIndex.second]->getTypeName());

            // keep the GUI id from the previous frame
            auto pPreviousID = previousGuiNodeIDs.find(nameToIndex.first);
            if (pPreviousID != previousGuiNodeIDs.end())
            {
        
                if (nodeIndex + 1 != pPreviousID->second)
                {
                    nodeIndex++;
                }

                renderPassUI.mGuiNodeID = pPreviousID->second;
            }
            else
            {
                renderPassUI.mGuiNodeID = nodeIndex++;
            }

            // add all of the incoming connections
            for (uint32_t i = 0; i < pCurrentPass->getIncomingEdgeCount(); ++i)
            {
                auto currentEdge = mRenderGraphRef.mEdgeData[pCurrentPass->getIncomingEdge(i)];
                mOutputToInputPins[currentEdge.srcField].push_back(std::make_pair(inputPinIndex, renderPassUI.mGuiNodeID));
                renderPassUI.addUIPin(currentEdge.dstField, inputPinIndex++, true);
                nodeConnected.insert(currentEdge.dstField);
            }

            // add all of the outgoing connections
            for (uint32_t i = 0; i < pCurrentPass->getOutgoingEdgeCount(); ++i)
            {
                auto currentEdge = mRenderGraphRef.mEdgeData[pCurrentPass->getOutgoingEdge(i)];
                renderPassUI.addUIPin(currentEdge.srcField, outputPinIndex++, false);
                nodeConnected.insert( currentEdge.srcField);
            }

            // Now we know which nodes are connected within the graph and not

            auto passData = 
                RenderGraphEditor::sGetRenderPassData[mRenderGraphRef.mNodeData[nameToIndex.second]->getTypeName()]
                    (mRenderGraphRef.mNodeData[nameToIndex.second]);

            for (const auto& inputNode : passData.inputs)
            {
                if (nodeConnected.find(inputNode.name) == nodeConnected.end())
                {
                    renderPassUI.addUIPin(inputNode.name, inputPinIndex++, true);
                }

                // add the details description for each pin

            }

            for (const auto& outputNode : passData.outputs)
            {
                if (nodeConnected.find(outputNode.name) == nodeConnected.end())
                {
                    renderPassUI.addUIPin(outputNode.name, outputPinIndex++, false);
                }

                // add the details description for each pin
            }

            mRenderPassUI.emplace(std::make_pair(nameToIndex.first, std::move(renderPassUI)));
        }
    }
}