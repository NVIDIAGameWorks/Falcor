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
#include "Utils/RenderGraphScripting.h"
#   define IMGUINODE_MAX_SLOT_NAME_LENGTH 255
#include "../Samples/Utils/RenderGraphEditor/dear_imgui_addons/imguinodegrapheditor/imguinodegrapheditor.h"
#include "Externals/dear_imgui/imgui.h"
// TODO Don't do this
#include "Externals/dear_imgui/imgui_internal.h"
#include <experimental/filesystem>
#include <fstream>
#include <functional>

namespace Falcor
{
    const float kUpdateTimeInterval = 2.0f;
    const float kPinRadius = 7.0f;

    class RenderGraphUI::NodeGraphEditorGui : public ImGui::NodeGraphEditor
    {
    public:
        RenderGraphUI::NodeGraphEditorGui(RenderGraphUI* pRenderGraphUI) : mpRenderGraphUI(pRenderGraphUI) {}

        // call on beginning a new frame
        void setCurrentGui(Gui* pGui) { mpGui = pGui; }

        Gui* getCurrentGui() const { assert(mpGui); return mpGui; }

        void reset() { inited = false; }

        glm::vec2 getOffsetPos() const { return { offset.x, offset.y }; }

        ImGui::NodeLink& getLink(int32_t index) { return links[index]; }
        
        void setOutputNode(ImGui::Node* pNode) { mpGraphOutputNode = pNode; }

        ImGui::Node* getOutputNode() { return mpGraphOutputNode; }

        RenderGraphUI* getRenderGraphUI() { return mpRenderGraphUI; }

        void setPopupNode(ImGui::Node* pFocusedNode) { mpFocusedNode = pFocusedNode;  }

        ImGui::Node* getPopupNode() { return mpFocusedNode; }

        void setPopupPin(uint32_t pinIndex, bool isInput) { mPinIndexToDisplay = pinIndex; }

        uint32_t getPopupPinIndex() const { return mPinIndexToDisplay; }

        bool isPopupPinInput() const { return mPopupPinIsInput; }
        
        const std::string& getRenderUINodeName() const { return mRenderUINodeName; }

        void setRenderUINodeName(const std::string& renderUINodeName) { mRenderUINodeName = renderUINodeName; }

        ImGui::Node*& getNodeFromID(uint32_t nodeID) { return mpIDtoNode[nodeID]; }

        // wraps around creating link to avoid setting static flag
        bool addLinkFromGraph(ImGui::Node* inputNode, int input_slot, ImGui::Node* outputNode, int output_slot, 
            bool checkIfAlreadyPresent = false, ImU32 col = GetStyle().color_link)
        {
            // tell addLink to call a different callback func
            auto oldCallback = linkCallback;
            linkCallback = setLinkFromGraph;

            bool insert = addLink(inputNode, input_slot, outputNode, output_slot, checkIfAlreadyPresent, col);
            linkCallback = oldCallback;
            return insert;
        }

        ImGui::Node* addAndInitNode(int nodeType, const std::string& name,
            const std::string& outputsString, const std::string& inputsString, uint32_t guiNodeID,
            RenderPass* pCurrentRenderPass, const ImVec2& pos = ImVec2(0, 0));

        void setNodeCallbackFunc(NodeCallback cb, void* pUserData)
        {
            mpCBUserData = pUserData;
            setNodeCallback(cb);
        }

        // callback function defined in derived class definition to access data without making class public to the rest of Falcor
        static void setNode(ImGui::Node*& node, ImGui::NodeGraphEditor::NodeState state, ImGui::NodeGraphEditor& editor);
        // callback for ImGui setting link between to nodes in the visual interface
        static void setLinkFromGui(const ImGui::NodeLink& link, ImGui::NodeGraphEditor::LinkState state, ImGui::NodeGraphEditor& editor);
        static void setLinkFromGraph(const ImGui::NodeLink& link, ImGui::NodeGraphEditor::LinkState state, ImGui::NodeGraphEditor& editor);
        static ImGui::Node* NodeGraphEditorGui::createNode(int, const ImVec2& pos, const ImGui::NodeGraphEditor&);

        RenderGraphUI* mpRenderGraphUI;

    private:
        friend class RenderGraphNode;

        Gui* mpGui = nullptr;
        void* mpCBUserData = nullptr;
        ImGui::Node* mpGraphOutputNode = nullptr;
        ImGui::Node* mpFocusedNode = nullptr;
        uint32_t mPinIndexToDisplay = uint32_t(-1);
        bool mPopupPinIsInput = false;
        std::string mRenderUINodeName;
        std::unordered_map<uint32_t, ImGui::Node*> mpIDtoNode;
    };

    class RenderGraphUI::RenderGraphNode : public ImGui::Node
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

        // Allow the renderGraphUI to set the position of each node
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
            // with this library there is no way of modifying the positioning of the labels on the node
            // manually making labels to align correctly from within the node

            // grab all of the fields again
            RenderGraphNode* pCurrentNode = static_cast<RenderGraphNode*>(field.userData);
            RenderGraphUI::NodeGraphEditorGui* pGraphEditorGui = static_cast<RenderGraphUI::NodeGraphEditorGui*>(&pCurrentNode->getNodeGraphEditor());
            if (pGraphEditorGui->isInited())  return false;

            Gui* pGui = pGraphEditorGui->getCurrentGui();
            RenderPass* pRenderPass = static_cast<RenderPass*>(pCurrentNode->mpRenderPass);
            int32_t paddingSpace = glm::max(pCurrentNode->OutputsCount, pCurrentNode->InputsCount) / 2 + 1;
            ImVec2 oldScreenPos = ImGui::GetCursorScreenPos();
            ImVec2 currentScreenPos{ pGraphEditorGui->offset.x  + pCurrentNode->Pos.x * ImGui::GetCurrentWindow()->FontWindowScale,
                pGraphEditorGui->offset.y + pCurrentNode->Pos.y * ImGui::GetCurrentWindow()->FontWindowScale };
            ImVec2 pinRectBoundsOffsetx{ -kPinRadius * 2.0f, kPinRadius * 4.0f };

            // TODO the pin colors need to be taken from the global style
            ImU32 pinColor = 0xFFFFFFFF;
            float slotNum = 1.0f;
            float pinOffsetx = kPinRadius * 2.0f;
            uint32_t pinCount = static_cast<uint32_t>(pCurrentNode->InputsCount);
            bool isInputs = true;

            std::string dummyText; dummyText.resize(32, ' ');
            pGui->addText(dummyText.c_str());
            pGui->addText(dummyText.c_str());
            pGui->addText(dummyText.c_str());
            pGui->addText(dummyText.c_str());

            if (pRenderPass && ImGui::IsMouseClicked(1))
            {
                std::string idString = std::string("Render UI##") + pCurrentNode->getName();
                
                // get hover rect of node 

                if (ImGui::IsMouseHoveringRect(currentScreenPos, ImVec2(currentScreenPos.x + pCurrentNode->Size.x, currentScreenPos.y + pCurrentNode->Size.y)))
                {
                    pGraphEditorGui->setRenderUINodeName(pCurrentNode->getName());
                }
                else
                {
                    uint32_t id = ImGui::GetCurrentWindow()->GetID(idString.c_str());
                    if (ImGui::IsPopupOpen(id))
                    {
                        ImGui::ClosePopup(id);
                        pGraphEditorGui->setRenderUINodeName("");
                    }
                }
            }

            for (int32_t i = 0; i < paddingSpace; ++i)
            {
                pGui->addText(dummyText.c_str());
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

                        if (pRenderPass && ImGui::IsMouseClicked(1))
                        {
                            pGraphEditorGui->setPopupNode(pCurrentNode);
                            pGraphEditorGui->setPopupPin(i, !static_cast<bool>(j));

                        }
                    }
                    else
                    {
                        ImGui::GetWindowDrawList()->AddCircle(ImVec2(inputPos.x, inputPos.y), kPinRadius, pinColor);
                    }
                    ImGui::SetCursorScreenPos({ inputPos.x + pinOffsetx - ((pinOffsetx < 0.0f) ? ImGui::CalcTextSize(isInputs ? pCurrentNode->InputNames[i] : pCurrentNode->OutputNames[i]).x : 0.0f), inputPos.y - kPinRadius });

                    slotNum++;
                    pGui->addText(isInputs ? pCurrentNode->InputNames[i] : pCurrentNode->OutputNames[i]);
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
                pGui->addText(dummyText.c_str());
            }

            pGui->addText(dummyText.c_str());

            return false;
        }

        void initialize(const std::string& name, const std::string& outputsString, 
            const std::string& inputsString, uint32_t guiNodeID, RenderPass* pRenderPass)
        {
            init(name.c_str(), Pos, inputsString.c_str(), outputsString.c_str(), guiNodeID);

            if (pRenderPass)
            {
                mpRenderPass = pRenderPass;
                const glm::vec4 nodeColor = Gui::pickUniqueColor(pRenderPass->getName());
                overrideTitleBgColor = ImGui::GetColorU32({ nodeColor.x, nodeColor.y, nodeColor.z, nodeColor.w });
            }

        }

        static RenderGraphNode* create(const ImVec2& pos)
        {
            RenderGraphNode* node = (RenderGraphNode*)ImGui::MemAlloc(sizeof(RenderGraphNode));
            IM_PLACEMENT_NEW(node) RenderGraphNode();

            node->fields.addFieldCustom(static_cast<ImGui::FieldInfo::RenderFieldDelegate>(renderUI), nullptr, node);
            node->Pos = pos;
            return node;
        }
    private:
    };

    ImGui::Node* RenderGraphUI::NodeGraphEditorGui::addAndInitNode(int nodeType, const std::string& name,
        const std::string& outputsString, const std::string& inputsString, uint32_t guiNodeID,
        RenderPass* pCurrentRenderPass, const ImVec2& pos)
    {
        // set init data for new node to obtain
        RenderGraphNode* newNode = static_cast<RenderGraphNode*>(addNode(nodeType, pos, nullptr));
        newNode->initialize(name, outputsString, inputsString, guiNodeID, pCurrentRenderPass);
        return newNode;
    }

    void RenderGraphUI::NodeGraphEditorGui::setLinkFromGui(const ImGui::NodeLink& link, ImGui::NodeGraphEditor::LinkState state, ImGui::NodeGraphEditor& editor)
    {
        if (state == ImGui::NodeGraphEditor::LinkState::LS_ADDED)
        {
            RenderGraphNode* inputNode = static_cast<RenderGraphNode*>(link.InputNode),
                *outputNode = static_cast<RenderGraphNode*>(link.OutputNode);
            RenderGraphUI::NodeGraphEditorGui* pGraphEditorGui = static_cast<RenderGraphUI::NodeGraphEditorGui*>(&editor);

            if (outputNode == pGraphEditorGui->mpGraphOutputNode)
            {
                editor.removeLink(link.InputNode, link.InputSlot, link.OutputNode, link.OutputSlot);
                pGraphEditorGui->getRenderGraphUI()->addOutput(inputNode->getName(), inputNode->getOutputName(link.InputSlot));
                return;
            }

            bool addStatus = false;
            addStatus = pGraphEditorGui->getRenderGraphUI()->addLink(
                inputNode->getName(), outputNode->getName(), inputNode->getOutputName(link.InputSlot), outputNode->getInputName(link.OutputSlot));

            // immediately remove link if it is not a legal edge in the render graph
            if (!addStatus && !editor.isInited()) //  only call after graph is setup
            {
                // does not call link callback surprisingly enough
                editor.removeLink(link.InputNode, link.InputSlot, link.OutputNode, link.OutputSlot);
                return;
            }
        }
    }

    // callback for ImGui setting link from render graph changes
    void RenderGraphUI::NodeGraphEditorGui::setLinkFromGraph(const ImGui::NodeLink& link, ImGui::NodeGraphEditor::LinkState state, ImGui::NodeGraphEditor& editor)
    {
        if (state == ImGui::NodeGraphEditor::LinkState::LS_ADDED)
        {
            RenderGraphNode* outputNode = static_cast<RenderGraphNode*>(link.OutputNode);

            if (outputNode == static_cast<RenderGraphUI::NodeGraphEditorGui*>(&editor)->mpGraphOutputNode) return;

            bool addStatus = false;

            // immediately remove link if it is not a legal edge in the render graph
            if (!addStatus && !editor.isInited()) //  only call after graph is setup
            {
                // does not call link callback surprisingly enough
                editor.removeLink(link.InputNode, link.InputSlot, link.OutputNode, link.OutputSlot);
                return;
            }
        }
    }

    ImGui::Node* RenderGraphUI::NodeGraphEditorGui::createNode(int, const ImVec2& pos, const ImGui::NodeGraphEditor&)
    {
        return RenderGraphUI::RenderGraphNode::create(pos);
    }

    bool RenderGraphUI::pushUpdateCommand(const std::string& commandString)
    {
        // make sure the graph is compiled
        mpRenderGraph->resolveExecutionOrder();

        // only send updates that we know are valid.
        if (mpRenderGraph->isValid(mLogString))
        {
            mLogString += commandString + " successful\n";
        }
        else
        {
            mLogString += "Graph is currently invalid\n";
        }

        // break apart multi-line command strings
        size_t offset = commandString.find_first_of('\n', 0);
        size_t lastOffset = 0;
        if (offset == std::string::npos) mCommandStrings.push_back(commandString);
        else
        {
            for (;;)
            {
                offset = commandString.find_first_of('\n', lastOffset);
                if (offset == std::string::npos)
                {
                    if (offset < commandString.size())
                    {
                        mCommandStrings.push_back(commandString.substr(lastOffset, commandString.size() - lastOffset));
                    }
                    break;
                }
                mCommandStrings.push_back(commandString.substr(lastOffset, offset - lastOffset));
                lastOffset = offset + 1;
            }
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

        mpRenderGraph->markGraphOutput(outputParam);
        pushUpdateCommand(std::string("AddGraphOutput ") + outputParam);
        mRebuildDisplayData = true;
    }

    void RenderGraphUI::addOutput(const std::string& outputPass, const std::string& outputField)
    {
        std::string outputParam = outputPass + "." + outputField;
        mpRenderGraph->markGraphOutput(outputParam);
        pushUpdateCommand(std::string("AddGraphOutput ") + outputParam);
        auto& passUI = mRenderPassUI[outputPass];
        passUI.mOutputPins[passUI.mNameToIndexOutput[outputField]].mIsGraphOutput = true;
        mRebuildDisplayData = true;
    }


    void RenderGraphUI::removeOutput(const std::string& outputPass, const std::string& outputField)
    {
        std::string outputParam = outputPass + "." + outputField;
        mpRenderGraph->unmarkGraphOutput(outputParam);
        pushUpdateCommand("RemoveGraphOutput " + outputParam);
        auto& passUI = mRenderPassUI[outputPass];
        passUI.mOutputPins[passUI.mNameToIndexOutput[outputField]].mIsGraphOutput = false;
    }

    bool RenderGraphUI::addLink(const std::string& srcPass, const std::string& dstPass, const std::string& srcField, const std::string& dstField)
    {
        // outputs warning if edge could not be created 
        std::string srcString = srcPass + "." + srcField, dstString = dstPass + "." + dstField;
        bool createdEdge = (mpRenderGraph->addEdge(srcString, dstString) != ((uint32_t)-1));

        pushUpdateCommand(std::string("AddEdge ") + srcString + " " + dstString);

        // update the ui to reflect the connections. This data is used for removal
        if (createdEdge)
        {
            RenderPassUI& srcRenderGraphUI = mRenderPassUI[srcPass];
            RenderPassUI& dstRenderGraphUI = mRenderPassUI[dstPass];

            uint32_t srcPinIndex = srcRenderGraphUI.mNameToIndexOutput[srcField];
            uint32_t dstPinIndex = dstRenderGraphUI.mNameToIndexInput[dstField];

            srcRenderGraphUI.mOutputPins[srcPinIndex].mConnectedPinName = dstField;
            srcRenderGraphUI.mOutputPins[srcPinIndex].mConnectedNodeName = dstPass;
            dstRenderGraphUI.mInputPins[dstPinIndex].mConnectedPinName = srcField;
            dstRenderGraphUI.mInputPins[dstPinIndex].mConnectedNodeName = srcPass;

            mRebuildDisplayData = true;
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
        mRebuildDisplayData = true;
        mpRenderGraph->removeRenderPass(name);
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

    RenderGraphUI::RenderGraphUI()
        : mNewNodeStartPosition(-40.0f, 100.0f)
    {
        mNextPassName.resize(255, 0);
    }

    RenderGraphUI::RenderGraphUI(const RenderGraph::SharedPtr& renderGraphRef, const std::string& renderGraphName)
        : mpRenderGraph(renderGraphRef), mNewNodeStartPosition(-40.0f, 100.0f), mRenderGraphName(renderGraphName)
    {
        mNextPassName.resize(255, 0);
        mpNodeGraphEditor = (new RenderGraphUI::NodeGraphEditorGui(this) );
    }

    RenderGraphUI::RenderGraphUI(RenderGraphUI&& rref)
        : mpRenderGraph(rref.mpRenderGraph), mEdgesColor(rref.mEdgesColor),
        mAutoGenEdgesColor(rref.mAutoGenEdgesColor), mAutoResolveEdgesColor(rref.mAutoResolveEdgesColor),
        mGraphOutputsColor(rref.mGraphOutputsColor), mNewNodeStartPosition(rref.mNewNodeStartPosition),
        mMaxNodePositionX(rref.mMaxNodePositionX), mAllNodeTypeStrings(rref.mAllNodeTypeStrings),
        mAllNodeTypes(rref.mAllNodeTypes), mRenderPassUI(rref.mRenderPassUI),
        mInputPinStringToLinkID(rref.mInputPinStringToLinkID), mOutputToInputPins(rref.mOutputToInputPins),
        mCommandStrings(rref.mCommandStrings), mTimeSinceLastUpdate(rref.mTimeSinceLastUpdate),
        mDisplayDragAndDropPopup(rref.mDisplayDragAndDropPopup),
        mNextPassName(rref.mNextPassName), mRenderGraphName(rref.mRenderGraphName),
        mShowWarningPopup(rref.mShowWarningPopup), mpNodeGraphEditor(rref.mpNodeGraphEditor)
    {
        rref.mpNodeGraphEditor = nullptr;
    }

    RenderGraphUI::~RenderGraphUI()
    {
        if (mpNodeGraphEditor) delete mpNodeGraphEditor;
        mpNodeGraphEditor = nullptr;
    }
    
    void RenderGraphUI::NodeGraphEditorGui::setNode(ImGui::Node*& node, ImGui::NodeGraphEditor::NodeState state, ImGui::NodeGraphEditor& editor)
    {
        RenderGraphNode* pRenderGraphNode = static_cast<RenderGraphNode*>(node);
        RenderGraphUI::NodeGraphEditorGui* pGraphEditor = static_cast<RenderGraphUI::NodeGraphEditorGui*>(&editor);
        if (!editor.isInited())
        {
            if (state == ImGui::NodeGraphEditor::NodeState::NS_DELETED)
            {
                pRenderGraphNode->getFields().clear();
                if (node == pGraphEditor->mpGraphOutputNode)
                {
                    pGraphEditor->mpGraphOutputNode = nullptr;
                    pGraphEditor->getRenderGraphUI()->mRebuildDisplayData = true;
                }
                else
                {
                    pGraphEditor->getRenderGraphUI()->removeRenderPass(node->getName());
                }
            }
        }
        if (state == ImGui::NodeGraphEditor::NodeState::NS_ADDED)
        {
            // always call the callback
            // PASS the initData into the callbacks
            pGraphEditor->getRenderGraphUI()->addRenderPass(node->getName(), pRenderGraphNode->mpRenderPass->getName());
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
            uint32_t popupWarningID = ImGui::GetCurrentWindow()->GetID("Auto-Resolve Warning");

            if (!mShowWarningPopup && checkInWindow && !ImGui::IsWindowHovered() && ImGui::IsMouseClicked(0))
            {
                mpNodeGraphEditor->setPopupNode(nullptr);
                mpNodeGraphEditor->setPopupPin(-1, false);
                mpNodeGraphEditor->selectedLink = -1;
                ImGui::EndPopup();
                ImGui::PopStyleVar();
                return;
            }

            if (mpNodeGraphEditor->getPopupNode())
            {
                RenderPassUI& renderPassUI = mRenderPassUI[mpNodeGraphEditor->getPopupNode()->getName()];
                renderPassUI.renderPinUI(pGui, mpNodeGraphEditor->getPopupPinIndex(), mpNodeGraphEditor->isPopupPinInput());
                ImGui::Separator();
            }

            if (mpNodeGraphEditor->selectedLink != -1)
            {
                ImGui::NodeLink& selectedLink = mpNodeGraphEditor->getLink(mpNodeGraphEditor->selectedLink);
                std::string srcPassName = std::string(selectedLink.InputNode->getName());
                std::string dstPassName = std::string(selectedLink.OutputNode->getName());
                ImGui::Text((std::string("Edge: ") + srcPassName + "-" + dstPassName).c_str());
                std::string inputString = srcPassName + "." + 
                    std::string(static_cast<RenderGraphNode*>(selectedLink.OutputNode)->getInputName(selectedLink.OutputSlot));
                uint32_t linkID = mInputPinStringToLinkID[inputString];
                auto edgeIt = mpRenderGraph->mEdgeData.find(linkID);

                // link exists, but is not an edge (such as graph output edge)
                if (edgeIt != mpRenderGraph->mEdgeData.end())
                {
                    RenderGraph::EdgeData& edgeData = edgeIt->second;
                    int32_t autoResolve = static_cast<int32_t>(edgeData.flags);

                    pGui->addText("Src Field : ");
                    pGui->addText(edgeData.srcField.c_str(), true);

                    pGui->addText("Dst Field : ");
                    pGui->addText(edgeData.dstField.c_str(), true);

                    pGui->addText("Auto-Generated : ");
                    pGui->addText(edgeData.autoGenerated ? "true" : "false", true);

                    // compare the sample count of the auto-resolve edge to display warning window
                    const RenderPassUI& srcPassUI = mRenderPassUI[srcPassName];
                    const RenderPassUI& dstPassUI = mRenderPassUI[dstPassName];
                    uint32_t srcSampleCount = srcPassUI.mReflection.getField(selectedLink.InputSlot).getSampleCount();
                    uint32_t dstSampleCount = dstPassUI.mReflection.getField(selectedLink.OutputSlot).getSampleCount();
                    bool setAutoResolve = false;

                    if (mShowWarningPopup)
                    {
                        if (ImGui::BeginPopup("Auto-Resolve Warning"))
                        {
                            pGui->addText("Auto-Resolve warning");

                            if (pGui->addButton("okay"))
                            {
                                setAutoResolve = true;
                                mShowWarningPopup = false;
                                edgeData.flags = static_cast<RenderGraph::EdgeData::Flags>(true);
                            }

                            if (pGui->addButton("cancel"))
                            {
                                setAutoResolve = false;
                                mShowWarningPopup = false;
                            }

                            ImGui::EndPopup();
                        }
                    }

                    if (pGui->addCheckBox("Auto-Resolve", autoResolve))
                    {
                        if (autoResolve && (srcSampleCount) && (srcSampleCount > dstSampleCount))
                        {
                            ImGui::OpenPopup("Auto-Resolve Warning");
                            mShowWarningPopup = true;
                        }
                        else
                        {
                            edgeData.flags = static_cast<RenderGraph::EdgeData::Flags>(autoResolve);
                            selectedLink.LinkColor = autoResolve ? mAutoResolveEdgesColor : mEdgesColor;
                        }
                    }
                }
            }
            
            ImGui::EndPopup();
        }
        ImGui::PopStyleVar();
    }

    void RenderGraphUI::renderUI(Gui* pGui)
    {
        static std::string dragAndDropText;
        ImGui::GetIO().FontAllowUserScaling = true;

        mpNodeGraphEditor->mpRenderGraphUI = this;
        mpNodeGraphEditor->setCurrentGui(pGui);
        mpNodeGraphEditor->show_top_pane = false;
        mpNodeGraphEditor->show_node_copy_paste_buttons = false;
        mpNodeGraphEditor->show_connection_names = false;
        mpNodeGraphEditor->show_left_pane = false;
        mpNodeGraphEditor->setLinkCallback(NodeGraphEditorGui::setLinkFromGui);
        mpNodeGraphEditor->setNodeCallback(NodeGraphEditorGui::setNode);
        
        const ImVec2& mousePos = ImGui::GetMousePos();
        bool bFromDragAndDrop = false;

        ImGui::NodeGraphEditor::Style& style = ImGui::NodeGraphEditor::GetStyle(); 
        style.color_node_frame_selected = ImGui::ColorConvertFloat4ToU32({ 226.0f / 255.0f, 190.0f / 255.0f, 42.0f / 255.0f, 0.8f });
        style.color_node_frame_active = style.color_node_frame_selected;
        style.node_slots_radius = kPinRadius;

        // update the deleted links from the GUI since the library doesn't call its own callback
        if (mRebuildDisplayData)
        {
            mRebuildDisplayData = false;
            mpNodeGraphEditor->setNodeCallback(nullptr);
            mpNodeGraphEditor->reset();
        }
        else
        {
            updatePins(false);
        }

        // push update commands for the open pop-up
        // TODO - get a callback working for only when data has changed
        if (mpNodeGraphEditor->getRenderUINodeName().size())
        {
            std::string idString = std::string("Render UI##") + mpNodeGraphEditor->getRenderUINodeName();
            if (!ImGui::IsPopupOpen(ImGui::GetCurrentWindow()->GetID(idString.c_str())))
            {
                ImGui::OpenPopup(idString.c_str());
            }
            if (ImGui::BeginPopup(idString.c_str()))
            {
                if (!ImGui::IsWindowHovered() && (ImGui::IsMouseClicked(0) || ImGui::IsMouseClicked(1)))
                {
                    mpNodeGraphEditor->setRenderUINodeName("");
                }
                else
                {
                    mpRenderGraph->getRenderPass(mpNodeGraphEditor->getRenderUINodeName())->renderUI(pGui, nullptr);
                }
                
                // push serialized data as update command for live preview
                
                // TODO -- execute command for click and drop
                //pushUpdateCommand(RenderGraphLoader::saveRenderGraphAsUpdateScript(*mpRenderGraph));

                ImGui::EndPopup();
            }
        }
        
        if (!mpNodeGraphEditor->isInited())
        {
            mpNodeGraphEditor->render();
            if (mpNodeGraphEditor->getPopupPinIndex() != uint32_t(-1) || (mpNodeGraphEditor->selectedLink != -1))
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
            if (pGui->dragDropDest("RenderPassType", statement))
            {
                dragAndDropText = statement;
                mNewNodeStartPosition = { -mpNodeGraphEditor->offset.x + mousePos.x, -mpNodeGraphEditor->offset.y + mousePos.y };
                mNextPassName = statement;
                mDisplayDragAndDropPopup = true;
            }

            if (mDisplayDragAndDropPopup)
            {
                pGui->pushWindow("CreateNewGraph", 256, 128, 
                    static_cast<uint32_t>(mNewNodeStartPosition.x), static_cast<uint32_t>(mNewNodeStartPosition.y));

                pGui->addTextBox("Pass Name", mNextPassName);
                if (pGui->addButton("create##renderpass")) // multiple buttons have create
                {
                    while (mpRenderGraph->renderPassExist(mNextPassName))
                    {
                        mNextPassName.push_back('_');
                    }

                    // TODO -- 
                    //RenderGraphLoader::ExecuteStatement("AddRenderPass " + mNextPassName + " " + dragAndDropText, *mpRenderGraph);
                    bFromDragAndDrop = true;
                    mRebuildDisplayData = true;
                    mDisplayDragAndDropPopup = false;
                    mNewNodeStartPosition /= ImGui::GetCurrentWindow()->FontWindowScale;
                    if (mMaxNodePositionX < mNewNodeStartPosition.x) mMaxNodePositionX = mNewNodeStartPosition.x;
                }
                if (pGui->addButton("cancel##renderPass"))
                {
                    mDisplayDragAndDropPopup = false;
                }

                pGui->popWindow();
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
            mpNodeGraphEditor->clear();
            mpNodeGraphEditor->render();
            mpNodeGraphEditor->setOutputNode(nullptr);

            return;
        }
        
        mAllNodeTypes.push_back("GraphOutputNode");
        for (auto& nodeTypeString : mAllNodeTypeStrings)
        {
            mAllNodeTypes.push_back(nodeTypeString.c_str());
        }

        mpNodeGraphEditor->registerNodeTypes(mAllNodeTypes.data(), static_cast<uint32_t>(mAllNodeTypes.size()), NodeGraphEditorGui::createNode, 0, -1, 0, 0);

        // create graph output node first
        if (!mpNodeGraphEditor->getOutputNode())
        {
            ImGui::Node* pNewNode = mpNodeGraphEditor->addAndInitNode(0, "GraphOutput", "", "inputs", 0, nullptr, { mMaxNodePositionX + 384.0f, mNewNodeStartPosition.y });
            mpNodeGraphEditor->setOutputNode(pNewNode);
        }
        else
        {
            RenderGraphNode* mpGraphOutputNode = static_cast<RenderGraphNode*>(mpNodeGraphEditor->getOutputNode());
            if (mpGraphOutputNode->getPos().x <= mMaxNodePositionX)
            {
                mpGraphOutputNode->setPos({ mMaxNodePositionX + 384.0f, mpGraphOutputNode->getPos().y });
            }
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
            RenderPass* pNodeRenderPass = mpRenderGraph->getRenderPass(currentPass.first).get();
            nameString = currentPass.first;
        
            if (!mpNodeGraphEditor->getAllNodesOfType(currentPassUI.mGuiNodeID, nullptr, false))
            {
                glm::vec2 nextPosition = getNextNodePosition(mpRenderGraph->getPassIndex(nameString));
        
                mpNodeGraphEditor->getNodeFromID(guiNodeID) = mpNodeGraphEditor->addAndInitNode(guiNodeID,
                    nameString, outputsString, inputsString, guiNodeID, pNodeRenderPass,
                    ImVec2(nextPosition.x, nextPosition.y));
                if (bFromDragAndDrop) addRenderPass(nameString, pNodeRenderPass->getName()); 
                bFromDragAndDrop = false;
            }
        }
        
        updatePins();
        
        mpNodeGraphEditor->render();
    }

    void RenderGraphUI::reset()
    {
        mpNodeGraphEditor->setOutputNode(nullptr);
        mpNodeGraphEditor->reset();
        mpNodeGraphEditor->clear();
        mRebuildDisplayData = true;
    }

    std::vector<uint32_t> RenderGraphUI::getExecutionOrder()
    {
        std::vector<uint32_t> executionOrder;
        std::map<float, uint32_t> posToIndex;

        for ( uint32_t i = 0; i < static_cast<uint32_t>(mpNodeGraphEditor->getNumNodes()); ++i)
        {
            RenderGraphNode* pCurrentGraphNode = static_cast<RenderGraphNode*>(mpNodeGraphEditor->getNode(i));
            std::string currentNodeName = pCurrentGraphNode->getName();
            if (currentNodeName == "GraphOutput") continue;
            float position = pCurrentGraphNode->getPos().x;
            posToIndex[position] = (mpRenderGraph->getPassIndex(currentNodeName));
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
                            if (!mpNodeGraphEditor->isLinkPresent(mpNodeGraphEditor->getNodeFromID(currentPassUI.mGuiNodeID), currentPinUI.mGuiPinID,
                                mpNodeGraphEditor->getNodeFromID(connectedPin.second), connectedPin.first))
                            {
                                // get edge data from the referenced graph
                                uint32_t edgeId = mInputPinStringToLinkID[currentPass.first + "." + mpNodeGraphEditor->getNodeFromID(connectedPin.second)->getName()];

                                // set color if autogenerated
                                uint32_t currentEdgeColor = mEdgesColor;
                                if (mpRenderGraph->mEdgeData[edgeId].autoGenerated) currentEdgeColor = mAutoGenEdgesColor;

                                mpNodeGraphEditor->addLinkFromGraph(mpNodeGraphEditor->getNodeFromID(currentPassUI.mGuiNodeID), currentPinUI.mGuiPinID,
                                    mpNodeGraphEditor->getNodeFromID(connectedPin.second), connectedPin.first, false, currentEdgeColor);

                                static_cast<RenderGraphNode*>(mpNodeGraphEditor->getNodeFromID(connectedPin.second))->mInputPinConnected[connectedPin.first] = true;
                                static_cast<RenderGraphNode*>(mpNodeGraphEditor->getNodeFromID(currentPassUI.mGuiNodeID))->mOutputPinConnected[currentPinUI.mGuiPinID] = true;
                            }
                        }
                    }

                    // mark graph outputs to graph output node
                    if (currentPinUI.mIsGraphOutput)
                    {
                        if (!mpNodeGraphEditor->isLinkPresent(mpNodeGraphEditor->getNodeFromID(currentPassUI.mGuiNodeID), currentPinUI.mGuiPinID,
                            mpNodeGraphEditor->getOutputNode(), 0))
                        {
                            ImU32 linkColor = ImGui::GetColorU32({ mGraphOutputsColor.x, mGraphOutputsColor.y, mGraphOutputsColor.z, mGraphOutputsColor.w });
                            mpNodeGraphEditor->addLinkFromGraph(mpNodeGraphEditor->getNodeFromID(currentPassUI.mGuiNodeID), currentPinUI.mGuiPinID,
                                mpNodeGraphEditor->getOutputNode(), 0, false, linkColor);
                        }
                    }
                }
                else
                {
                    // remove graph output links
                    if (currentPinUI.mIsGraphOutput)
                    {
                        if (!mpNodeGraphEditor->isLinkPresent(mpNodeGraphEditor->getNodeFromID(currentPassUI.mGuiNodeID), currentPinUI.mGuiPinID,
                            mpNodeGraphEditor->getOutputNode(), 0))
                        {
                            removeOutput(currentPass.first, currentPinUI.mPinName);
                            mpNodeGraphEditor->removeLink(mpNodeGraphEditor->getNodeFromID(currentPassUI.mGuiNodeID), currentPinUI.mGuiPinID,
                                mpNodeGraphEditor->getOutputNode(), 0);
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
                
                    if (!currentPinUI.mConnectedNodeName.size()) continue;

                    std::pair<uint32_t, uint32_t> inputIDs{ currentPinUI.mGuiPinID, currentPassUI.mGuiNodeID };
                    const auto& connectedNodeUI = mRenderPassUI[currentPinUI.mConnectedNodeName];
                    uint32_t inputPinID = connectedNodeUI.mNameToIndexOutput.find(currentPinUI.mConnectedPinName)->second;

                    if (!mpNodeGraphEditor->isLinkPresent(mpNodeGraphEditor->getNodeFromID(connectedNodeUI.mGuiNodeID), inputPinID,
                        mpNodeGraphEditor->getNodeFromID(inputIDs.second),inputIDs.first ))
                    {
                        auto edgeIt = mInputPinStringToLinkID.find(currentPass.first + "." + currentPinName);
                        assert(edgeIt != mInputPinStringToLinkID.end()); 
                        uint32_t edgeID = edgeIt->second;
                        
                        removeEdge(mpNodeGraphEditor->getNodeFromID(connectedNodeUI.mGuiNodeID)->getName(),
                            mpNodeGraphEditor->getNodeFromID(inputIDs.second)->getName(), mpRenderGraph->mEdgeData[edgeID].srcField, mpRenderGraph->mEdgeData[edgeID].dstField);
                        mpRenderGraph->removeEdge(edgeID);

                        currentPinUI.mConnectedNodeName = "";

                        static_cast<RenderGraphNode*>(mpNodeGraphEditor->getNodeFromID(inputIDs.second))->mInputPinConnected[inputIDs.first] = false;
                        static_cast<RenderGraphNode*>(mpNodeGraphEditor->getNodeFromID(connectedNodeUI.mGuiNodeID))->mOutputPinConnected[inputPinID] = false;

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

        if (std::find(mpRenderGraph->mExecutionList.begin(), mpRenderGraph->mExecutionList.end(), nodeID) == mpRenderGraph->mExecutionList.end())
        {
            return newNodePosition;
        }

        for (const auto& passID : mpRenderGraph->mExecutionList)
        {
            newNodePosition.x += offsetX;

            if (passID == nodeID)
            {
                const DirectedGraph::Node* pNode = mpRenderGraph->mpGraph->getNode(nodeID);
                for (uint32_t i = 0; i < pNode->getIncomingEdgeCount(); ++i)
                {
                    uint32_t outgoingEdgeCount = mpRenderGraph->mpGraph->getNode(mpRenderGraph->mpGraph->getEdge(pNode->getIncomingEdge(i))->getSourceNode())->getOutgoingEdgeCount();
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

        RenderGraphNode* mpGraphOutputNode = static_cast<RenderGraphNode*>(mpNodeGraphEditor->getOutputNode());
        glm::vec2 graphOutputNodePos = mpGraphOutputNode->getPos();
        if (graphOutputNodePos.x <= newNodePosition.x)
        {
            mpGraphOutputNode->setPos({ newNodePosition.x + offsetX, newNodePosition.y });
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

        mpRenderGraph->resolveExecutionOrder();

        mRenderPassUI.clear();
        mInputPinStringToLinkID.clear();

        // build information for displaying graph
        for (const auto& nameToIndex : mpRenderGraph->mNameToIndex)
        {
            auto pCurrentPass = mpRenderGraph->mpGraph->getNode(nameToIndex.second);
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
            mpRenderGraph->mNodeData[nameToIndex.second].pPass->reflect(renderPassUI.mReflection);

            // test to see if we have hit a graph output
            std::unordered_set<std::string> passGraphOutputs;

            for (const auto& output : mpRenderGraph->mOutputs)
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
                auto currentEdge = mpRenderGraph->mEdgeData[edgeID];
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

                auto pSourceNode = mpRenderGraph->mNodeData.find( mpRenderGraph->mpGraph->getEdge(edgeID)->getSourceNode());
                assert(pSourceNode != mpRenderGraph->mNodeData.end());

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
                auto currentEdge = mpRenderGraph->mEdgeData[edgeID];
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
                
                auto pDestNode = mpRenderGraph->mNodeData.find(mpRenderGraph->mpGraph->getEdge(edgeID)->getSourceNode());
                assert(pDestNode != mpRenderGraph->mNodeData.end());

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