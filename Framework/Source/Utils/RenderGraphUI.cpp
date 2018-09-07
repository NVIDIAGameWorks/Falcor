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
#include "RenderGraphUI.h"
#include "Utils/Gui.h"
#include "Utils/RenderGraphLoader.h"
#   define IMGUINODE_MAX_SLOT_NAME_LENGTH 255
#include "../Samples/Utils/RenderGraphEditor/dear_imgui_addons/imguinodegrapheditor/imguinodegrapheditor.h"
#include "Externals/dear_imgui/imgui.h"
// TODO Don't do this
#include "Externals/dear_imgui/imgui_internal.h"
#include <experimental/filesystem>
#include <fstream>

namespace Falcor
{
    const float kUpdateTimeInterval = 2.0f;
    const float kPinRadius = 6.0f;

    static std::unordered_map<uint32_t, ImGui::Node*> spIDToNode;
    static RenderGraphUI* spCurrentGraphUI = nullptr;

    class NodeGraphEditorGui : public ImGui::NodeGraphEditor
    {
    public:
        ImGui::Node* pGraphOutputNode = nullptr;

        void reset()
        {
            inited = false;
        }

        glm::vec2 getOffsetPos()
        {
            return { offset.x, offset.y };
        }

        ImGui::NodeLink& getLink(int32_t index)
        {
            return links[index];
        }
    };

    static NodeGraphEditorGui sNodeGraphEditor;

    class RenderGraphNode : public ImGui::Node
    {
    public:
        // Data for callback initialization from gui graph library
        class NodeInitData
        {
        public:
            std::string mName;
            std::string mOutputsString;
            std::string mInputsString;
            uint32_t mGuiNodeID;
            RenderPass* mpCurrentRenderPass;
        };

        static Gui* spGui;
        static NodeInitData sInitData;
        // set if node is added from graph data and not the ui
        static bool sAddedFromGraphData;

        static RenderGraphNode* spNodeToRenderPin;
        static bool sPinIsInput;
        static uint32_t sPinIndexToDisplay;
        
        bool mDisplayProperties;
        bool mOutputPinConnected[IMGUINODE_MAX_OUTPUT_SLOTS];
        bool mInputPinConnected[IMGUINODE_MAX_INPUT_SLOTS];
        RenderPass* mpRenderPass;

        static void setInitData(const std::string& name, const std::string& outputsString, const std::string& inputsString, uint32_t guiNodeID, RenderPass* pCurrentRenderPass)
        {
            sInitData.mName = name;
            sInitData.mOutputsString = outputsString;
            sInitData.mInputsString = inputsString;
            sInitData.mGuiNodeID = guiNodeID;
            sInitData.mpCurrentRenderPass = pCurrentRenderPass;
        }

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

            std::string dummyText; dummyText.resize(32, ' ');
            spGui->addText(dummyText.c_str());
            spGui->addText(dummyText.c_str());
            spGui->addText(dummyText.c_str());

            // with this library there is no way of modifying the positioning of the labels on the node
            // manually making labels to align correctly from within the node

            // grab all of the fields again
            RenderGraphNode* pCurrentNode = static_cast<RenderGraphNode*>(field.userData);
            RenderPass* pRenderPass = static_cast<RenderPass*>(pCurrentNode->mpRenderPass);
            int32_t paddingSpace = glm::max(pCurrentNode->OutputsCount, pCurrentNode->InputsCount) / 2 + 1;
            
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

            for (int32_t i = 0; i < paddingSpace; ++i)
            {
                spGui->addText(dummyText.c_str());
            }

            for (uint32_t j = 0; j < 2; ++j)
            {
                for (uint32_t i = 0; i < pinCount; ++i)
                {
                    // custom pins as an extension of the built ones
                    ImVec2 inputPos = currentScreenPos;
                    inputPos.y += pCurrentNode->Size.y * ((i + 1) / static_cast<float>(pinCount + 1));

                    // fill in circle for the pin if connected to a link
                    if (pCurrentNode->pinIsConnected(i, isInputs))
                    {
                        ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(inputPos.x, inputPos.y), kPinRadius,pinColor);
                    }

                    if (ImGui::IsMouseHoveringRect(ImVec2(inputPos.x + pinRectBoundsOffsetx.x, inputPos.y - kPinRadius), ImVec2(inputPos.x + pinRectBoundsOffsetx.y, inputPos.y + kPinRadius)))
                    {
                        ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(inputPos.x, inputPos.y), kPinRadius, ImGui::GetColorU32(ImGui::NodeGraphEditor::GetStyle().color_node_title));

                        if (pRenderPass && ImGui::GetIO().KeyCtrl && ImGui::IsMouseClicked(0))
                        {
                            RenderGraphNode::spNodeToRenderPin = pCurrentNode;
                            RenderGraphNode::sPinIndexToDisplay = i;
                            RenderGraphNode::sPinIsInput = !static_cast<bool>(j);
                        }
                    }
                    else
                    {
                        ImGui::GetWindowDrawList()->AddCircle(ImVec2(inputPos.x, inputPos.y), kPinRadius, pinColor);
                    }
                    ImGui::SetCursorScreenPos({ inputPos.x + pinOffsetx - ((pinOffsetx < 0.0f) ? ImGui::CalcTextSize(isInputs ? pCurrentNode->InputNames[i] : pCurrentNode->OutputNames[i]).x : 0.0f), inputPos.y - kPinRadius });

                    slotNum++;
                    spGui->addText(isInputs ? pCurrentNode->InputNames[i] : pCurrentNode->OutputNames[i]);
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
            
            for (int32_t i = 0; i < paddingSpace; ++i)
            {
                spGui->addText(dummyText.c_str());
            }

            spGui->addText(dummyText.c_str());

            return false;
        }

        static RenderGraphNode* create(const ImVec2& pos)
        {
            RenderGraphNode* node = (RenderGraphNode*)ImGui::MemAlloc(sizeof(RenderGraphNode));
            IM_PLACEMENT_NEW(node) RenderGraphNode();

            node->init(sInitData.mName.c_str(), pos, sInitData.mInputsString.c_str(), sInitData.mOutputsString.c_str(), sInitData.mGuiNodeID);

            if (sInitData.mpCurrentRenderPass)
            {
                node->mpRenderPass = sInitData.mpCurrentRenderPass;
                const glm::vec4 nodeColor = Gui::pickUniqueColor(node->mpRenderPass->getName());
                node->overrideTitleBgColor = ImGui::GetColorU32({ nodeColor.x, nodeColor.y, nodeColor.z, nodeColor.w });
            }
            
            node->fields.addFieldCustom(static_cast<ImGui::FieldInfo::RenderFieldDelegate>(renderUI), nullptr, node);
            
            return node;
        }
    private:
    };

    bool RenderGraphNode::sAddedFromGraphData = false;
    RenderGraphNode::NodeInitData RenderGraphNode::sInitData;
    Gui* RenderGraphNode::spGui = nullptr;

    RenderGraphNode* RenderGraphNode::spNodeToRenderPin = nullptr;
    bool RenderGraphNode::sPinIsInput = false;
    uint32_t RenderGraphNode::sPinIndexToDisplay = 0;

    bool RenderGraphUI::sRebuildDisplayData = true;
    std::string RenderGraphUI::sLogString;

    static ImGui::Node* createNode(int, const ImVec2& pos, const ImGui::NodeGraphEditor&)
    {
        return RenderGraphNode::create(pos);
    }

    bool RenderGraphUI::pushUpdateCommand(const std::string& commandString)
    {
        // make sure the graph is compiled
        mRenderGraphRef.resolveExecutionOrder();

        // only send updates that we know are valid.
        if (mRenderGraphRef.isValid(sLogString))
        {
            sLogString += commandString + " successful\n";
            mCommandStrings.push_back(commandString);
            return true;
        }
        else
        {
        }

        return false;
    }

    void RenderGraphUI::addRenderPass(const std::string& name, const std::string& nodeTypeName)
    {
        pushUpdateCommand(std::string("AddRenderPass ") + name + " " + nodeTypeName);
    }

    void RenderGraphUI::addOutput(const std::string& outputParam)
    {
        size_t offset = outputParam.find('.');
        std::string outputPass = outputParam.substr(0, offset);
        std::string outputField = outputParam.substr(offset + 1, outputParam.size());

        const auto passUIIt = mRenderPassUI.find(outputPass);
        if (passUIIt == mRenderPassUI.end())
        {
            msgBox("Error setting graph output. Can't find node name.");
            return;
        }
        auto& passUI = passUIIt->second;
        const auto outputIt = passUI.mNameToIndexOutput.find(outputField);
        if (outputIt == passUI.mNameToIndexOutput.end())
        {
            msgBox("Error setting graph output. Can't find output name.");
            return;
        }
        passUI.mOutputPins[outputIt->second].mIsGraphOutput = true;

        mRenderGraphRef.markGraphOutput(outputParam);
        pushUpdateCommand(std::string("AddGraphOutput ") + outputParam);
        sRebuildDisplayData = true;
    }

    void RenderGraphUI::addOutput(const std::string& outputPass, const std::string& outputField)
    {
        std::string outputParam = outputPass + "." + outputField;
        mRenderGraphRef.markGraphOutput(outputParam);
        pushUpdateCommand(std::string("AddGraphOutput ") + outputParam);
        auto& passUI = mRenderPassUI[outputPass];
        passUI.mOutputPins[passUI.mNameToIndexOutput[outputField]].mIsGraphOutput = true;
        sRebuildDisplayData = true;
    }


    void RenderGraphUI::removeOutput(const std::string& outputPass, const std::string& outputField)
    {
        std::string outputParam = outputPass + "." + outputField;
        mRenderGraphRef.unmarkGraphOutput(outputParam);
        pushUpdateCommand("RemoveGraphOutput " + outputParam);
        auto& passUI = mRenderPassUI[outputPass];
        passUI.mOutputPins[passUI.mNameToIndexOutput[outputField]].mIsGraphOutput = false;
    }

    bool RenderGraphUI::addLink(const std::string& srcPass, const std::string& dstPass, const std::string& srcField, const std::string& dstField)
    {
        // outputs warning if edge could not be created 
        std::string srcString = srcPass + "." + srcField, dstString = dstPass + "." + dstField;
        bool createdEdge = (spCurrentGraphUI->mRenderGraphRef.addEdge(srcString, dstString) != ((uint32_t)-1));

        pushUpdateCommand(std::string("AddEdge ") + srcString + " " + dstString);

        // update the ui to reflect the connections. This data is used for removal
        if (createdEdge)
        {
            RenderPassUI& srcRenderGraphUI = spCurrentGraphUI->mRenderPassUI[srcPass];
            RenderPassUI& dstRenderGraphUI = spCurrentGraphUI->mRenderPassUI[dstPass];

            uint32_t srcPinIndex = srcRenderGraphUI.mNameToIndexOutput[srcField];
            uint32_t dstPinIndex = dstRenderGraphUI.mNameToIndexInput[dstField];

            srcRenderGraphUI.mOutputPins[srcPinIndex].mConnectedPinName = dstField;
            srcRenderGraphUI.mOutputPins[srcPinIndex].mConnectedNodeName = dstPass;
            dstRenderGraphUI.mInputPins[dstPinIndex].mConnectedPinName = srcField;
            dstRenderGraphUI.mInputPins[dstPinIndex].mConnectedNodeName = srcPass;

            sRebuildDisplayData = true;
        }

        return createdEdge;
    }

    void RenderGraphUI::removeEdge(const std::string& srcPass, const std::string& dstPass, const std::string& srcField, const std::string& dstField)
    {
        std::string command("RemoveEdge ");
        command += srcPass + "." + srcField + " " + dstPass + "." + dstField;
        pushUpdateCommand(command);
    }

    void RenderGraphUI::removeRenderPass(const std::string& name)
    {
        spCurrentGraphUI->sRebuildDisplayData = true;
        spCurrentGraphUI->mRenderGraphRef.removeRenderPass(name);
        pushUpdateCommand(std::string("RemoveRenderPass ") + name);
    }

    void RenderGraphUI::writeUpdateScriptToFile(const std::string& filePath, float lastFrameTime)
    {
        if ((mTimeSinceLastUpdate += lastFrameTime) < kUpdateTimeInterval) return;
        mTimeSinceLastUpdate = 0.0f;
        if (!mCommandStrings.size()) return;
        static std::ofstream ofstream(filePath, std::ios_base::out);
        size_t totalSize = 0;
        
        ofstream.write(reinterpret_cast<const char*>(&totalSize), sizeof(size_t));
        
        for (std::string& statement : mCommandStrings)
        {
            statement.push_back('\n');
            totalSize += statement.size();
            ofstream.write(statement.c_str(), statement.size());
        }

        mCommandStrings.clear();
        
        // rewind and write the size of the script changes for the viewer to execute
        ofstream.seekp(0, std::ios::beg);
        ofstream.write(reinterpret_cast<const char*>(&totalSize), sizeof(size_t));
        ofstream.seekp(0, std::ios::beg);

        std::experimental::filesystem::last_write_time(filePath, std::chrono::system_clock::now());
    }

    RenderGraphUI::RenderGraphUI(RenderGraph& renderGraphRef)
        : mRenderGraphRef(renderGraphRef), mNewNodeStartPosition(-40.0f, 100.0f)
    {
        sNodeGraphEditor.clear();
    }

    RenderGraphUI::~RenderGraphUI()
    {
        sNodeGraphEditor.setNodeCallback(nullptr);
        sNodeGraphEditor.setLinkCallback(nullptr);
    }

    static void setNode(ImGui::Node*& node, ImGui::NodeGraphEditor::NodeState state, ImGui::NodeGraphEditor& editor)
    {
        if (!editor.isInited())
        {
            if (state == ImGui::NodeGraphEditor::NodeState::NS_DELETED)
            {
                static_cast<RenderGraphNode*>(node)->getFields().clear();
                spCurrentGraphUI->removeRenderPass(node->getName());
            }
        }
        if (state == ImGui::NodeGraphEditor::NodeState::NS_ADDED)
        {
            // always call the callback
            spCurrentGraphUI->addRenderPass(node->getName(), RenderGraphNode::sInitData.mpCurrentRenderPass->getName());
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
                if (!RenderGraphNode::sAddedFromGraphData)
                {
                    editor.removeLink(link.InputNode, link.InputSlot, link.OutputNode, link.OutputSlot);
                    spCurrentGraphUI->addOutput(inputNode->getName(), inputNode->getOutputName(link.InputSlot));
                }
                
                RenderGraphNode::sAddedFromGraphData = false;

                return;
            }

            bool addStatus = false;
            if (!RenderGraphNode::sAddedFromGraphData)
            {
                addStatus = spCurrentGraphUI->addLink(inputNode->getName(), outputNode->getName(), inputNode->getOutputName(link.InputSlot), outputNode->getInputName(link.OutputSlot));
            }
            RenderGraphNode::sAddedFromGraphData = false;

            // immediately remove link if it is not a legal edge in the render graph
            if (!addStatus && !editor.isInited()) //  only call after graph is setup
            {
                // does not call link callback surprisingly enough
                editor.removeLink(link.InputNode, link.InputSlot, link.OutputNode, link.OutputSlot);
                return;
            }
        }
    }

    void RenderPassUI::addUIPin(const std::string& fieldName, uint32_t guiPinID, bool isInput, const std::string& connectedPinName, const std::string& connectedNodeName, bool isGraphOutput)
    {
        auto& pinsRef = isInput ? mInputPins : mOutputPins;
        auto& nameToIndexMapRef = isInput ? mNameToIndexInput : mNameToIndexOutput;

        if (pinsRef.size() <= guiPinID)
        {
            pinsRef.resize(guiPinID + 1);
        }

        PinUIData& pinUIData = pinsRef[guiPinID];
        pinUIData.mPinName = fieldName;
        pinUIData.mGuiPinID = guiPinID;
        pinUIData.mIsInput = isInput;
        pinUIData.mConnectedPinName = connectedPinName;
        pinUIData.mConnectedNodeName = connectedNodeName;
        pinUIData.mIsGraphOutput = isGraphOutput;

        nameToIndexMapRef.insert(std::make_pair(fieldName, static_cast<uint32_t>(guiPinID) ));
    }

    void RenderPassUI::renderPinUI(Gui* pGui, uint32_t pinIndex, bool isInput)
    {
        const PinUIData& pinUI = isInput ? mInputPins[pinIndex] : mOutputPins[pinIndex];
        std::string pinName = pinUI.mPinName;

        size_t index = 0;
        for(size_t i  = 0; i < mReflection.getFieldCount(); ++i)
        {
            if (mReflection.getField(i).getName() == pinName)
            {
                index = i;
                break;
            }
        }

        const RenderPassReflection::Field& field = mReflection.getField(index);
        RenderPassReflection::Field::Type type = field.getType();

        pGui->addText(pinName.c_str());
        ImGui::Separator();

        pGui->addText("ResourceFlags : ");

        if (static_cast<bool>(type & RenderPassReflection::Field::Type::Input) &&
            static_cast<bool>(type & RenderPassReflection::Field::Type::Output))
        {
            pGui->addText("InputOutput", true);
        }
        else if (static_cast<bool>(type & RenderPassReflection::Field::Type::Input))
        {
            pGui->addText("Input", true);
        }
        else if (static_cast<bool>(type & RenderPassReflection::Field::Type::Output))
        {
            pGui->addText("Output", true);
        }

        pGui->addText("ResourceType : ");
        pGui->addText(to_string(field.getResourceType()->getType()).c_str(), true);

        pGui->addText("Width: ");
        pGui->addText(std::to_string(field.getWidth()).c_str(), true);

        pGui->addText("Height: ");
        pGui->addText(std::to_string(field.getHeight()).c_str(), true);

        pGui->addText("Depth: ");
        pGui->addText(std::to_string(field.getDepth()).c_str(), true);

        pGui->addText("Sample Count: ");
        pGui->addText(std::to_string(field.getSampleCount()).c_str(), true);

        pGui->addText("ResourceFormat: ");
        pGui->addText(to_string(field.getFormat()).c_str(), true);

        pGui->addText("BindFlags: ");
        pGui->addText(to_string(field.getBindFlags()).c_str(), true);

        pGui->addText("Flags: ");
        switch (field.getFlags())
        {
        case RenderPassReflection::Field::Flags::None:
            pGui->addText("None", true);
            break;
        case RenderPassReflection::Field::Flags::Optional:
            pGui->addText("Optional", true);
            break;
        case RenderPassReflection::Field::Flags::Persistent:
            pGui->addText("Persistent", true);
            break;
        default:
            should_not_get_here();
        }
        
        ImGui::Separator();
    }

    void RenderGraphUI::renderPopupMenu(Gui* pGui)
    {
        bool checkInWindow = false;

        if (ImGui::IsPopupOpen(ImGui::GetCurrentWindow()->GetID("PinMenu")))
        {
            checkInWindow = true;
        }
        else
        {
            ImGui::OpenPopup("PinMenu");
        }
        
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 8));
        if (ImGui::BeginPopup("PinMenu"))
        {
            if (checkInWindow && !ImGui::IsWindowHovered() && ImGui::IsMouseClicked(0))
            {
                RenderGraphNode::spNodeToRenderPin = nullptr;
                sNodeGraphEditor.selectedLink = -1;
                ImGui::EndPopup();
                ImGui::PopStyleVar();
                return;
            }

            if (RenderGraphNode::spNodeToRenderPin)
            {
                RenderPassUI& renderPassUI = mRenderPassUI[RenderGraphNode::spNodeToRenderPin->getName()];
                renderPassUI.renderPinUI(pGui, RenderGraphNode::sPinIndexToDisplay, RenderGraphNode::sPinIsInput);
                ImGui::Separator();

            }

            if (sNodeGraphEditor.selectedLink != -1)
            {
                ImGui::Text("Edge:");

                ImGui::NodeLink& selectedLink = sNodeGraphEditor.getLink(sNodeGraphEditor.selectedLink);
                auto edgeIt = mRenderGraphRef.mEdgeData.find(sNodeGraphEditor.selectedLink);
                
                // link exists, but is not an edge (such as graph output edge)
                if (edgeIt != mRenderGraphRef.mEdgeData.end())
                {
                    RenderGraph::EdgeData& edgeData = edgeIt->second;
                    int32_t autoResolve = static_cast<int32_t>(edgeData.flags);

                    pGui->addText("Dst Field : ");
                    pGui->addText(edgeData.dstField.c_str(), true);

                    pGui->addText("Src Field : ");
                    pGui->addText(edgeData.srcField.c_str(), true);

                    pGui->addText("Auto-Generated : ");
                    pGui->addText(edgeData.autoGenerated ? "true" : "false", true);

                    if (pGui->addCheckBox("Auto-Resolve", autoResolve))
                    {
                        mRenderGraphRef.mEdgeData[sNodeGraphEditor.selectedLink].flags = static_cast<RenderGraph::EdgeData::Flags>(autoResolve);
                        selectedLink.LinkColor = autoResolve ? mAutoResolveEdgesColor : mEdgesColor;
                    }
                }
            }

            ImGui::EndPopup();
        }
        ImGui::PopStyleVar();
    }

    void RenderGraphUI::renderUI(Gui* pGui)
    {
        RenderGraphNode::spGui = pGui;
        ImGui::GetIO().FontAllowUserScaling = true;

        sNodeGraphEditor.show_top_pane = false;
        sNodeGraphEditor.show_node_copy_paste_buttons = false;
        sNodeGraphEditor.show_connection_names = false;
        sNodeGraphEditor.show_left_pane = false;

        sNodeGraphEditor.setLinkCallback(setLink);
        sNodeGraphEditor.setNodeCallback(setNode);
        
        const ImVec2& mousePos = ImGui::GetMousePos();
        bool bFromDragAndDrop = true;

        // update the deleted links from the GUI since the library doesn't call its own callback
        
        if (sRebuildDisplayData)
        {
            sRebuildDisplayData = false;
            sNodeGraphEditor.setNodeCallback(nullptr);
            sNodeGraphEditor.reset();
        }
        else
        {
            updatePins(false);
        }

        if (!sNodeGraphEditor.isInited())
        {
            sNodeGraphEditor.render();
            if (RenderGraphNode::spNodeToRenderPin || (sNodeGraphEditor.selectedLink != -1))
            {
                renderPopupMenu(pGui);
            }
            else
            {
                if (ImGui::IsPopupOpen(ImGui::GetCurrentWindow()->GetID("PinMenu")))
                {
                    ImGui::CloseCurrentPopup();
                }
            }

            std::string statement;
            if (pGui->dragDropDest("RenderPassScript", statement))
            {
                // TODO Matt
//                 RenderGraphLoader::ExecuteStatement(statement, mRenderGraphRef);
//                 mNewNodeStartPosition = { -sNodeGraphEditor.offset.x + mousePos.x, -sNodeGraphEditor.offset.y + mousePos.y };
//                 mNewNodeStartPosition /= ImGui::GetCurrentWindow()->FontWindowScale;
//                 bFromDragAndDrop = true;
//                 sRebuildDisplayData = true;
//                 if (mMaxNodePositionX < mNewNodeStartPosition.x) mMaxNodePositionX = mNewNodeStartPosition.x;
            }
            else
            {
                mNewNodeStartPosition = { -40.0f, 100.0f };
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
        
        sNodeGraphEditor.registerNodeTypes(mAllNodeTypes.data(), static_cast<uint32_t>(mAllNodeTypes.size()), createNode, 0, -1, 0, 1);
        
        spCurrentGraphUI = this;
        
        // create graph output node first
        if (!sNodeGraphEditor.pGraphOutputNode)
        {
            RenderGraphNode::setInitData("GraphOutput", "", "inputs", 0, nullptr);
            sNodeGraphEditor.pGraphOutputNode = sNodeGraphEditor.addNode(0, { mMaxNodePositionX + 384.0f, mNewNodeStartPosition.y });
        }
        else
        {
            RenderGraphNode* pGraphOutputNode = static_cast<RenderGraphNode*>(sNodeGraphEditor.pGraphOutputNode);
            pGraphOutputNode->setPos({ mMaxNodePositionX + 384.0f, pGraphOutputNode->getPos().y});
        }
        
        for (auto& currentPass : mRenderPassUI)
        {
            auto& currentPassUI = currentPass.second;
            std::string inputsString;
            std::string outputsString;
            std::string nameString;
        
            for (const auto& currentPinUI : currentPassUI.mInputPins)
            {
                // Connect the graph nodes for each of the edges
                // need to iterate in here in order to use the right indices
                const std::string& currentPinName = currentPinUI.mPinName;
                inputsString += inputsString.size() ? (";" + currentPinName) : currentPinName;
            }
        
            for (const auto& currentPinUI : currentPassUI.mOutputPins)
            {
                const std::string& currentPinName = currentPinUI.mPinName;
                outputsString += outputsString.size() ? (";" + currentPinName) : currentPinName;
            }
        
            uint32_t guiNodeID = currentPassUI.mGuiNodeID;
            RenderPass* pNodeRenderPass = mRenderGraphRef.getRenderPass(currentPass.first).get();
            nameString = currentPass.first;
        
            if (!sNodeGraphEditor.getAllNodesOfType(currentPassUI.mGuiNodeID, nullptr, false))
            {
                glm::vec2 nextPosition = getNextNodePosition(mRenderGraphRef.getPassIndex(nameString));
        
                RenderGraphNode::setInitData(nameString, outputsString, inputsString, guiNodeID, pNodeRenderPass);
                spIDToNode[guiNodeID] = sNodeGraphEditor.addNode(guiNodeID, ImVec2(nextPosition.x, nextPosition.y));
                if (bFromDragAndDrop) addRenderPass(nameString, pNodeRenderPass->getName()); 
                bFromDragAndDrop = false;
            }
        }
        
        updatePins();
        
        sNodeGraphEditor.render();
    }

    void RenderGraphUI::reset()
    {
        sNodeGraphEditor.reset();
        sNodeGraphEditor.clear();
        sRebuildDisplayData = true;
    }

    std::vector<uint32_t> RenderGraphUI::getExecutionOrder()
    {
        std::vector<uint32_t> executionOrder;
        std::map<float, uint32_t> posToIndex;

        for ( uint32_t i = 0; i < static_cast<uint32_t>(sNodeGraphEditor.getNumNodes()); ++i)
        {
            RenderGraphNode* pCurrentGraphNode = static_cast<RenderGraphNode*>(sNodeGraphEditor.getNode(i));
            std::string currentNodeName = pCurrentGraphNode->getName();
            if (currentNodeName == "GraphOutput") continue;
            float position = pCurrentGraphNode->getPos().x;
            posToIndex[position] = (mRenderGraphRef.getPassIndex(currentNodeName));
        }

        for (const auto& posIndexPair : posToIndex)
        {
            executionOrder.push_back(posIndexPair.second);
        }

        return executionOrder;
    }

    void RenderGraphUI::updatePins(bool addLinks)
    {
        //  Draw pin connections. All the nodes have to be added to the GUI before the connections can be drawn
        for (auto& currentPass : mRenderPassUI)
        {
            auto& currentPassUI = currentPass.second;

            for (const auto& currentPinUI : currentPassUI.mOutputPins)
            {
                const std::string& currentPinName = currentPinUI.mPinName;

                if (addLinks)
                {
                    const auto& inputPins = mOutputToInputPins.find(currentPass.first + "." + currentPinName);
                    if (inputPins != mOutputToInputPins.end())
                    {
                        for (const auto& connectedPin : (inputPins->second))
                        {
                            if (!sNodeGraphEditor.isLinkPresent(spIDToNode[currentPassUI.mGuiNodeID], currentPinUI.mGuiPinID,
                                spIDToNode[connectedPin.second], connectedPin.first))
                            {
                                RenderGraphNode::sAddedFromGraphData = true;
                                
                                // get edge data from the referenced graph
                                uint32_t edgeId = mInputPinStringToLinkID[currentPass.first + "." + spIDToNode[connectedPin.second]->getName()];

                                // set color if autogenerated
                                uint32_t currentEdgeColor = mEdgesColor;
                                if (mRenderGraphRef.mEdgeData[edgeId].autoGenerated) currentEdgeColor = mAutoGenEdgesColor;

                                sNodeGraphEditor.addLink(spIDToNode[currentPassUI.mGuiNodeID], currentPinUI.mGuiPinID,
                                    spIDToNode[connectedPin.second], connectedPin.first, false, currentEdgeColor);

                                static_cast<RenderGraphNode*>(spIDToNode[connectedPin.second])->mInputPinConnected[connectedPin.first] = true;
                                static_cast<RenderGraphNode*>(spIDToNode[currentPassUI.mGuiNodeID])->mOutputPinConnected[currentPinUI.mGuiPinID] = true;
                            }
                        }
                    }

                    // mark graph outputs to graph output node
                    if (currentPinUI.mIsGraphOutput)
                    {
                        if (!sNodeGraphEditor.isLinkPresent(spIDToNode[currentPassUI.mGuiNodeID], currentPinUI.mGuiPinID,
                            sNodeGraphEditor.pGraphOutputNode, 0))
                        {
                            RenderGraphNode::sAddedFromGraphData = true;
                            ImU32 linkColor = ImGui::GetColorU32({ mGraphOutputsColor.x, mGraphOutputsColor.y, mGraphOutputsColor.z, mGraphOutputsColor.w });
                            sNodeGraphEditor.addLink(spIDToNode[currentPassUI.mGuiNodeID], currentPinUI.mGuiPinID,
                                sNodeGraphEditor.pGraphOutputNode, 0, false, linkColor);
                        }
                    }
                }
                else
                {
                    // remove graph output links
                    if (currentPinUI.mIsGraphOutput)
                    {
                        if (!sNodeGraphEditor.isLinkPresent(spIDToNode[currentPassUI.mGuiNodeID], currentPinUI.mGuiPinID,
                            sNodeGraphEditor.pGraphOutputNode, 0))
                        {
                            removeOutput(currentPass.first, currentPinUI.mPinName);
                            sNodeGraphEditor.removeLink(spIDToNode[currentPassUI.mGuiNodeID], currentPinUI.mGuiPinID,
                                sNodeGraphEditor.pGraphOutputNode, 0);
                        }
                    }
                }
            }

            if (!addLinks)
            {
                for (auto& currentPinUI : currentPassUI.mInputPins)
                {
                    const std::string& currentPinName = currentPinUI.mPinName;
                    bool isInput = currentPinUI.mIsInput;

                    // draw label for input pin
                
                    if (!currentPinUI.mConnectedNodeName.size())
                    {
                        continue;
                    }

                    std::pair<uint32_t, uint32_t> inputIDs{ currentPinUI.mGuiPinID, currentPassUI.mGuiNodeID };
                    const auto& connectedNodeUI = mRenderPassUI[currentPinUI.mConnectedNodeName];
                    uint32_t inputPinID = connectedNodeUI.mNameToIndexOutput.find(currentPinUI.mConnectedPinName)->second;

                    if (!sNodeGraphEditor.isLinkPresent(spIDToNode[connectedNodeUI.mGuiNodeID], inputPinID,
                        spIDToNode[inputIDs.second],inputIDs.first ))
                    {
                        auto edgeIt = mInputPinStringToLinkID.find(currentPass.first + "." + currentPinName);
                        assert(edgeIt != mInputPinStringToLinkID.end()); 
                        uint32_t edgeID = edgeIt->second;
                        
                        removeEdge(spIDToNode[connectedNodeUI.mGuiNodeID]->getName(), spIDToNode[inputIDs.second]->getName(), mRenderGraphRef.mEdgeData[edgeID].srcField, mRenderGraphRef.mEdgeData[edgeID].dstField);
                        mRenderGraphRef.removeEdge(edgeID);

                        currentPinUI.mConnectedNodeName = "";

                        static_cast<RenderGraphNode*>(spIDToNode[inputIDs.second])->mInputPinConnected[inputIDs.first] = false;
                        static_cast<RenderGraphNode*>(spIDToNode[connectedNodeUI.mGuiNodeID])->mOutputPinConnected[inputPinID] = false;

                        continue;
                    }
                }
            }
        }
    }

    glm::vec2 RenderGraphUI::getNextNodePosition(uint32_t nodeID)
    {
        const float offsetX = 384.0f;
        const float offsetY = 128.0f;
        glm::vec2 newNodePosition = mNewNodeStartPosition;

        if (std::find(mRenderGraphRef.mExecutionList.begin(), mRenderGraphRef.mExecutionList.end(), nodeID) == mRenderGraphRef.mExecutionList.end())
        {
            return newNodePosition;
        }

        for (const auto& passID : mRenderGraphRef.mExecutionList)
        {
            newNodePosition.x += offsetX;

            if (passID == nodeID)
            {
                const DirectedGraph::Node* pNode = mRenderGraphRef.mpGraph->getNode(nodeID);
                for (uint32_t i = 0; i < pNode->getIncomingEdgeCount(); ++i)
                {
                    uint32_t outgoingEdgeCount = mRenderGraphRef.mpGraph->getNode(mRenderGraphRef.mpGraph->getEdge(pNode->getIncomingEdge(i))->getSourceNode())->getOutgoingEdgeCount();
                    if (outgoingEdgeCount > pNode->getIncomingEdgeCount())
                    {
                        // move down by index in 
                        newNodePosition.y += offsetY * (outgoingEdgeCount - pNode->getIncomingEdgeCount() );
                        break;
                    }
                }

                break;
            }
        }

        RenderGraphNode* pGraphOutputNode = static_cast<RenderGraphNode*>(sNodeGraphEditor.pGraphOutputNode);
        glm::vec2 graphOutputNodePos = pGraphOutputNode->getPos();
        if (graphOutputNodePos.x <= newNodePosition.x)
        {
            pGraphOutputNode->setPos({ newNodePosition.x + offsetX, newNodePosition.y });
        }

        if (newNodePosition.x > mMaxNodePositionX)
        {
            mMaxNodePositionX = newNodePosition.x;
        }

        return newNodePosition;
    }

    void RenderGraphUI::updateDisplayData()
    {
        uint32_t nodeIndex = 1;

        mOutputToInputPins.clear();

        // set of field names that have a connection and are represented in the graph
        std::unordered_set<std::string> nodeConnectedInput;
        std::unordered_set<std::string> nodeConnectedOutput;
        std::unordered_map<std::string, uint32_t> previousGuiNodeIDs;
        std::unordered_set<uint32_t> existingIDs;

        for (const auto& currentRenderPassUI : mRenderPassUI)
        {
            existingIDs.insert(currentRenderPassUI.second.mGuiNodeID);
            previousGuiNodeIDs.insert(std::make_pair(currentRenderPassUI.first, currentRenderPassUI.second.mGuiNodeID));
        }

        mRenderGraphRef.resolveExecutionOrder();

        mRenderPassUI.clear();
        mInputPinStringToLinkID.clear();

        // build information for displaying graph
        for (const auto& nameToIndex : mRenderGraphRef.mNameToIndex)
        {
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

            uint32_t inputPinIndex = 0;
            uint32_t outputPinIndex = 0;

            // add all of the incoming connections
            for (uint32_t i = 0; i < pCurrentPass->getIncomingEdgeCount(); ++i)
            {
                uint32_t edgeID = pCurrentPass->getIncomingEdge(i);
                auto currentEdge = mRenderGraphRef.mEdgeData[edgeID];
                uint32_t pinIndex = 0;
                inputPinIndex = 0;

                while (pinIndex < renderPassUI.mReflection.getFieldCount())
                {
                    bool isInput = (static_cast<uint32_t>(renderPassUI.mReflection.getField(pinIndex).getType() & RenderPassReflection::Field::Type::Input) != 0);
                    if (isInput) 
                    { 
                        if (renderPassUI.mReflection.getField(pinIndex).getName() == currentEdge.dstField) { break;  }
                        inputPinIndex++;
                    }
                    pinIndex++;
                }

                auto pSourceNode = mRenderGraphRef.mNodeData.find( mRenderGraphRef.mpGraph->getEdge(edgeID)->getSourceNode());
                assert(pSourceNode != mRenderGraphRef.mNodeData.end());

                renderPassUI.addUIPin(currentEdge.dstField, inputPinIndex, true, currentEdge.srcField, pSourceNode->second.nodeName);
                
                std::string pinString = nameToIndex.first + "." + currentEdge.dstField;
                
                if (nodeConnectedInput.find(pinString) == nodeConnectedInput.end())
                {
                    nodeConnectedInput.insert(pinString);
                }

                mOutputToInputPins[pSourceNode->second.nodeName + "." + currentEdge.srcField].push_back(std::make_pair(inputPinIndex, renderPassUI.mGuiNodeID));
                mInputPinStringToLinkID.insert(std::make_pair(pinString, edgeID));
            }

            // add all of the outgoing connections
            for (uint32_t i = 0; i < pCurrentPass->getOutgoingEdgeCount(); ++i)
            {
                uint32_t edgeID = pCurrentPass->getOutgoingEdge(i);
                auto currentEdge = mRenderGraphRef.mEdgeData[edgeID];
                outputPinIndex = 0;

                std::string pinString = nameToIndex.first + "." + currentEdge.srcField;
                if (nodeConnectedOutput.find(pinString) != nodeConnectedOutput.end())
                {
                    break;
                }

                bool isGraphOutput = passGraphOutputs.find(currentEdge.srcField) != passGraphOutputs.end();
                uint32_t pinIndex = 0;

                while (pinIndex < renderPassUI.mReflection.getFieldCount())
                {
                    bool isOutput = (static_cast<uint32_t>(renderPassUI.mReflection.getField(pinIndex).getType() & RenderPassReflection::Field::Type::Output) != 0);
                    if (isOutput)
                    {
                        if (renderPassUI.mReflection.getField(pinIndex).getName() == currentEdge.srcField) { break; }
                        outputPinIndex++;
                    }
                    pinIndex++;
                }
                
                auto pDestNode = mRenderGraphRef.mNodeData.find(mRenderGraphRef.mpGraph->getEdge(edgeID)->getSourceNode());
                assert(pDestNode != mRenderGraphRef.mNodeData.end());

                renderPassUI.addUIPin(currentEdge.srcField, outputPinIndex, false, currentEdge.dstField, pDestNode->second.nodeName, isGraphOutput);
                nodeConnectedOutput.insert(pinString);
            }

            // Now we know which nodes are connected within the graph and not

            inputPinIndex = 0;
            outputPinIndex = 0;

            for (uint32_t i = 0; i < renderPassUI.mReflection.getFieldCount(); ++i)
            {
                const auto& currentField = renderPassUI.mReflection.getField(i);

                if (static_cast<uint32_t>(currentField.getType() & RenderPassReflection::Field::Type::Input) != 0)
                {
                    if (nodeConnectedInput.find(nameToIndex.first + "." + currentField.getName()) == nodeConnectedInput.end())
                    {
                        renderPassUI.addUIPin(currentField.getName(), inputPinIndex, true, "");
                    }

                    inputPinIndex++;
                }
                
                if (static_cast<uint32_t>(currentField.getType() & RenderPassReflection::Field::Type::Output) != 0)
                {
                    if (nodeConnectedOutput.find(nameToIndex.first + "." + currentField.getName()) == nodeConnectedOutput.end())
                    {
                        bool isGraphOutput = passGraphOutputs.find(currentField.getName()) != passGraphOutputs.end();
                        renderPassUI.addUIPin(currentField.getName(), outputPinIndex, false, "", "", isGraphOutput);
                    }

                    outputPinIndex++;
                }
            }

            mRenderPassUI.emplace(std::make_pair(nameToIndex.first, std::move(renderPassUI)));
        }
    }
}