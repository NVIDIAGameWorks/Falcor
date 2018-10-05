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
#include "Externals/dear_imgui/imgui.h"
#include "Externals/dear_imgui_addons/imguinodegrapheditor/imguinodegrapheditor.h"
#include "Externals/dear_imgui/imgui.h"
#include "Externals/dear_imgui/imgui_internal.h"
#include <experimental/filesystem>
#include <functional>

namespace Falcor
{
    const float kUpdateTimeInterval = 2.0f;
    const float kPinRadius = 7.0f;

    static const uint32_t kPinColor = 0xFFFFFFFF;
    static const uint32_t kEdgesColor = 0xFFFFFFFF;
    static const uint32_t kAutoGenEdgesColor = 0xFFFF0400;
    static const uint32_t kAutoResolveEdgesColor = 0xFF0104FF;
    static const uint32_t kGraphOutputsColor = 0xAF0101FF;

    class RenderGraphUI::NodeGraphEditorGui : public ImGui::NodeGraphEditor
    {
    public:
        NodeGraphEditorGui(RenderGraphUI* pRenderGraphUI) : mpRenderGraphUI(pRenderGraphUI) {}

        // call on beginning a new frame
        void setCurrentGui(Gui* pGui) { mpGui = pGui; }

        Gui* getCurrentGui() const { assert(mpGui); return mpGui; }

        void reset() { inited = false; }

        glm::vec2 getOffsetPos() const { return { offset.x, offset.y }; }

        ImGui::NodeLink& getLink(int32_t index) { return links[index]; }
        
        void setLinkColor(uint32_t index, uint32_t col) { links[index].LinkColor = col; }

        RenderGraphUI* getRenderGraphUI() { return mpRenderGraphUI; }

        void setPopupNode(ImGui::Node* pFocusedNode) { mpFocusedNode = pFocusedNode;  }

        ImGui::Node* getPopupNode() { return mpFocusedNode; }

        void setPopupPin(uint32_t pinIndex, bool isInput) { mPinIndexToDisplay = pinIndex; mPopupPinIsInput = isInput; }

        uint32_t getPopupPinIndex() const { return mPinIndexToDisplay; }

        bool isPopupPinInput() const { return mPopupPinIsInput; }
        
        const std::string& getRenderUINodeName() const { return mRenderUINodeName; }

        void setRenderUINodeName(const std::string& renderUINodeName) { mRenderUINodeName = renderUINodeName; }

        ImGui::Node*& getNodeFromID(uint32_t nodeID) { return mpIDtoNode[nodeID]; }

        // wraps around creating link to avoid setting static flag
        uint32_t addLinkFromGraph(ImGui::Node* inputNode, int input_slot, ImGui::Node* outputNode, int output_slot, bool checkIfAlreadyPresent = false, ImU32 col = GetStyle().color_link)
        {
            // tell addLink to call a different callback func
            auto oldCallback = linkCallback;
            linkCallback = setLinkFromGraph;

            bool insert = addLink(inputNode, input_slot, outputNode, output_slot, checkIfAlreadyPresent, col);
            linkCallback = oldCallback;
            return insert ? (links.size() - 1) : uint32_t(-1);
        }

        ImGui::Node* addAndInitNode(int nodeType, const std::string& name, const std::string& outputsString, const std::string& inputsString, uint32_t guiNodeID, RenderPass* pCurrentRenderPass, const ImVec2& pos = ImVec2(0, 0));

        void setNodeCallbackFunc(NodeCallback cb, void* pUserData)
        {
            mpCBUserData = pUserData;
            setNodeCallback(cb);
        }

        // callback function defined in derived class definition to access data without making class public to the rest of Falcor
        static void setNode(ImGui::Node*& node, ImGui::NodeGraphEditor::NodeState state, ImGui::NodeGraphEditor& editor);
        // callback for ImGui setting link between to nodes in the visual interface
        static void setLinkFromGui( ImGui::NodeLink& link, ImGui::NodeGraphEditor::LinkState state, ImGui::NodeGraphEditor& editor);
        static void setLinkFromGraph( ImGui::NodeLink& link, ImGui::NodeGraphEditor::LinkState state, ImGui::NodeGraphEditor& editor);
        static ImGui::Node* createNode(int, const ImVec2& pos, const ImGui::NodeGraphEditor&);

        RenderGraphUI* mpRenderGraphUI;

    private:
        friend class RenderGraphNode;

        Gui* mpGui = nullptr;
        void* mpCBUserData = nullptr;
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

        void setPinColor(uint32_t color, uint32_t index, bool isInput = false)
        {
            if (isInput) inputColors[index] = color;
            else outputColors[index] = color;
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
                    ImU32 pinColor = isInputs ? pCurrentNode->inputColors[i] : pCurrentNode->outputColors[i];

                    // fill in circle for the pin if connected to a link
                    if (pCurrentNode->pinIsConnected(i, isInputs))
                    {
                        ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(inputPos.x, inputPos.y), kPinRadius,pinColor);
                    }

                    if (ImGui::IsMouseHoveringRect(ImVec2(inputPos.x + pinRectBoundsOffsetx.x, inputPos.y - kPinRadius), ImVec2(inputPos.x + pinRectBoundsOffsetx.y, inputPos.y + kPinRadius)))
                    {
                        uint32_t hoveredPinColor = (pinColor == kGraphOutputsColor) ? (pinColor | 0xFF000000) : ImGui::GetColorU32(ImGui::NodeGraphEditor::GetStyle().color_node_title);
                        ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(inputPos.x, inputPos.y), kPinRadius, hoveredPinColor);

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

                    if (pinColor == kGraphOutputsColor)
                    {
                        ImVec2 arrowPoints[3] = { { inputPos.x + kPinRadius * 3.0f / 2.0f, inputPos.y + kPinRadius },
                        { inputPos.x + kPinRadius * 3.0f / 2.0f + kPinRadius, inputPos.y },
                        { inputPos.x + kPinRadius * 3.0f / 2.0f, inputPos.y - kPinRadius } };
                        ImGui::GetWindowDrawList()->AddPolyline(arrowPoints, 3, pinColor, false, 3.0f);
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
            
            bool isInputs = true;
            uint32_t pinCount = static_cast<uint32_t>(InputsCount);
            for (uint32_t j = 0; j < 2; ++j)
            {
                for (uint32_t i = 0; i < pinCount; ++i)
                {
                    if (isInputs) inputColors[i] = ImGui::NodeGraphEditor::GetStyle().color_node_input_slots;
                    else outputColors[i] = ImGui::NodeGraphEditor::GetStyle().color_node_output_slots;
                }
                pinCount = static_cast<uint32_t>(OutputsCount);
                isInputs = false;
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

    void RenderGraphUI::NodeGraphEditorGui::setLinkFromGui(ImGui::NodeLink& link, ImGui::NodeGraphEditor::LinkState state, ImGui::NodeGraphEditor& editor)
    {
        if (state == ImGui::NodeGraphEditor::LinkState::LS_ADDED)
        {
            RenderGraphNode* inputNode = static_cast<RenderGraphNode*>(link.InputNode),
                *outputNode = static_cast<RenderGraphNode*>(link.OutputNode);
            RenderGraphUI::NodeGraphEditorGui* pGraphEditorGui = static_cast<RenderGraphUI::NodeGraphEditorGui*>(&editor);

            bool addStatus = false;
            addStatus = pGraphEditorGui->getRenderGraphUI()->addLink(
                inputNode->getName(), outputNode->getName(), inputNode->getOutputName(link.InputSlot), outputNode->getInputName(link.OutputSlot), link.LinkColor);

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
    void RenderGraphUI::NodeGraphEditorGui::setLinkFromGraph(ImGui::NodeLink& link, ImGui::NodeGraphEditor::LinkState state, ImGui::NodeGraphEditor& editor)
    {
        if (state == ImGui::NodeGraphEditor::LinkState::LS_ADDED)
        {
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
    
    void RenderGraphUI::addRenderPass(const std::string& name, const std::string& nodeTypeName)
    {
        mpIr->addPass(name, nodeTypeName);
        mShouldUpdate = true;
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
        mpIr->markOutput(outputParam);
        mRebuildDisplayData = true;
        mShouldUpdate = true;
    }

    void RenderGraphUI::addOutput(const std::string& outputPass, const std::string& outputField)
    {
        std::string outputParam = outputPass + "." + outputField;
        addOutput(outputParam);
    }

    void RenderGraphUI::removeOutput(const std::string& outputPass, const std::string& outputField)
    {
        std::string outputParam = outputPass + "." + outputField;
        mpIr->unmarkOutput(outputParam);
        auto& passUI = mRenderPassUI[outputPass];
        passUI.mOutputPins[passUI.mNameToIndexOutput[outputField]].mIsGraphOutput = false;
        mShouldUpdate = true;
    }

    bool RenderGraphUI::autoResolveWarning(const std::string& srcString, const std::string& dstString)
    {
        std::string warningMsg = std::string("Warning: Edge ") + srcString + " - " + dstString + " can auto-resolve.\n";
        MsgBoxButton button = msgBox(warningMsg, MsgBoxType::OkCancel);

        if (button == MsgBoxButton::Ok)
        {
            mLogString += warningMsg;
            return true;
        }
        
        return false;
    }

    bool RenderGraphUI::addLink(const std::string& srcPass, const std::string& dstPass, const std::string& srcField, const std::string& dstField, uint32_t& color)
    {
        // outputs warning if edge could not be created 
        std::string srcString = srcPass + "." + srcField, dstString = dstPass + "." + dstField;
        bool canCreateEdge = (mpRenderGraph->getEdge(srcString, dstString) == ((uint32_t)-1));
        if (!canCreateEdge) return canCreateEdge;

        // update the ui to reflect the connections. This data is used for removal
        RenderPassUI& srcRenderPassUI = mRenderPassUI[srcPass];
        RenderPassUI& dstRenderPassUI = mRenderPassUI[dstPass];
        const auto outputIt = srcRenderPassUI.mNameToIndexOutput.find(srcField);
        const auto inputIt =  dstRenderPassUI.mNameToIndexInput.find(dstField);
        // check that link could exist
        canCreateEdge &= (outputIt != srcRenderPassUI.mNameToIndexOutput.end()) && 
            (inputIt != dstRenderPassUI.mNameToIndexInput.end());
        // check that the input is not already connected
        canCreateEdge &= (mInputPinStringToLinkID.find(dstString) == mInputPinStringToLinkID.end());

        if (canCreateEdge)
        {
            uint32_t srcPinIndex = outputIt->second;
            uint32_t dstPinIndex = inputIt->second;
            srcRenderPassUI.mOutputPins[srcPinIndex].mConnectedPinName = dstField;
            srcRenderPassUI.mOutputPins[srcPinIndex].mConnectedNodeName = dstPass;
            dstRenderPassUI.mInputPins[dstPinIndex].mConnectedPinName = srcField;
            dstRenderPassUI.mInputPins[dstPinIndex].mConnectedNodeName = srcPass;
            
            RenderPassReflection srcReflection = mpRenderGraph->mNodeData[mpRenderGraph->getPassIndex(srcPass)].pPass->reflect();
            RenderPassReflection dstReflection = mpRenderGraph->mNodeData[mpRenderGraph->getPassIndex(dstPass)].pPass->reflect();

            bool canAutoResolve = mpRenderGraph->canAutoResolve(srcReflection.getField(srcField), dstReflection.getField(dstField));
            if (canAutoResolve && mDisplayAutoResolvePopup) canCreateEdge = autoResolveWarning(srcString, dstString);
            color = canAutoResolve ? kAutoResolveEdgesColor : kEdgesColor;

            if (canCreateEdge)
            {
                mpIr->addEdge(srcString, dstString);
                mShouldUpdate = true;
            }
        }
        else
        {
            mLogString += "Unable to create edge for graph. \n";
        }

        return canCreateEdge;
    }

    void RenderGraphUI::removeEdge(const std::string& srcPass, const std::string& dstPass, const std::string& srcField, const std::string& dstField)
    {
        mpIr->removeEdge(srcPass + "." + srcField, dstPass + "." + dstField);
        mShouldUpdate = true;
    }

    void RenderGraphUI::removeRenderPass(const std::string& name)
    {
        mRebuildDisplayData = true;
        mpIr->removePass(name);
        mShouldUpdate = true;
    }

    void RenderGraphUI::updateGraph()
    {
        if (!mShouldUpdate) return;
        std::string newCommands = mpIr->getIR();
        mpIr = RenderGraphIR::create(mRenderGraphName, false); // reset
        if (mLastCommand == newCommands) return;

        // make sure the graph is compiled
        mpRenderGraph->resolveExecutionOrder();
        mLastCommand = newCommands;

        if (mRecordUpdates) mUpdateCommands += newCommands;

        // update reference graph to check if valid before sending to next 
        auto pScripting = RenderGraphScripting::create();
        pScripting->addGraph(mRenderGraphName, mpRenderGraph);
        pScripting->runScript(newCommands);

        if(newCommands.size()) mLogString += "Running: " + newCommands;

        // only send updates that we know are valid.
        if (!mpRenderGraph->isValid(mLogString))
        {
            mLogString += "Graph is currently invalid\n";
        }

        mShouldUpdate = false;
        mRebuildDisplayData = true;
    }

    void RenderGraphUI::writeUpdateScriptToFile(const std::string& filePath, float lastFrameTime)
    {
        if ((mTimeSinceLastUpdate += lastFrameTime) < kUpdateTimeInterval) return;
        mTimeSinceLastUpdate = 0.0f;
        if (!mUpdateCommands.size()) return;

        // only send delta of updates once the graph is valid
        std::string log;
        if (!mpRenderGraph->isValid(log)) return;

        std::ofstream outputFileStream(filePath, std::ios_base::out);
        outputFileStream << mUpdateCommands;
        mUpdateCommands.clear();
    }

    RenderGraphUI::RenderGraphUI()
        : mNewNodeStartPosition(-40.0f, 100.0f)
    {
        mNextPassName.resize(255, 0);
    }

    RenderGraphUI::RenderGraphUI(const RenderGraph::SharedPtr& pGraph, const std::string& graphName)
        : mpRenderGraph(pGraph), mNewNodeStartPosition(-40.0f, 100.0f), mRenderGraphName(graphName)
    {
        mNextPassName.resize(255, 0);
        mpNodeGraphEditor = std::make_shared<NodeGraphEditorGui>(this);
        mpIr = RenderGraphIR::create(graphName, false);
    }

    RenderGraphUI::~RenderGraphUI()
    {
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
                pGraphEditor->getRenderGraphUI()->removeRenderPass(node->getName());
            }
        }
        if (state == ImGui::NodeGraphEditor::NodeState::NS_ADDED)
        {
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

        PinUI& pinUIData = pinsRef[guiPinID];
        pinUIData.mPinName = fieldName;
        pinUIData.mGuiPinID = guiPinID;
        pinUIData.mConnectedPinName = connectedPinName;
        pinUIData.mConnectedNodeName = connectedNodeName;
        pinUIData.mIsGraphOutput = isGraphOutput;

        nameToIndexMapRef.insert(std::make_pair(fieldName, static_cast<uint32_t>(guiPinID) ));
    }

    void RenderPassUI::renderPinUI(Gui* pGui, const std::string& passName, RenderGraphUI* pGraphUI, uint32_t index, bool input)
    {
        RenderPassUI::PinUI& pinUI = input ? mInputPins[index] : mOutputPins[index];

        size_t fieldIndex = 0;
        for (size_t i = 0; i < mReflection.getFieldCount(); ++i)
        {
            if (mReflection.getField(i).getName() == pinUI.mPinName)
            {
                fieldIndex = i;
                break;
            }
        }
        
        // only render ui if is input or output
        const RenderPassReflection::Field& field = mReflection.getField(fieldIndex);
        if (is_set(field.getType(), RenderPassReflection::Field::Type::Input) || is_set(field.getType(), RenderPassReflection::Field::Type::Output))
        {
            pinUI.renderUI(pGui, field, pGraphUI, passName);
        }
    }

    void RenderPassUI::PinUI::renderUI(Gui* pGui, const RenderPassReflection::Field& field, RenderGraphUI* pGraphUI, const std::string& passName)
    {
        RenderPassReflection::Field::Type type = field.getType();
        uint32_t isInput = is_set(type, RenderPassReflection::Field::Type::Input);
        uint32_t isOutput = is_set(type, RenderPassReflection::Field::Type::Output);
        std::string displayName = mIsGraphOutput ? std::string("Graph Output : ") + mPinName : mPinName;

        pGui->addText(displayName.c_str());
        ImGui::Separator();

        pGui->addText("ResourceFlags : ");

        if (isInput && isOutput)    pGui->addText("InputOutput", true);
        else if (isInput)   pGui->addText("Input", true);
        else if (isOutput)  pGui->addText("Output", true);

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

        if (isOutput)
        {
            bool isGraphOutput = mIsGraphOutput;

            if (pGui->addCheckBox("Graph Output", mIsGraphOutput))
            {
                if (isGraphOutput && !mIsGraphOutput)
                {
                    pGraphUI->removeOutput(passName, mPinName);
                }
                else if (!isGraphOutput && mIsGraphOutput)
                {
                    pGraphUI->addOutput(passName, mPinName);
                }
            }
        }

        ImGui::Separator();
    }

    void RenderGraphUI::renderPopupMenu(Gui* pGui)
    {
        bool isPopupOpen = false;

        if (!(isPopupOpen = ImGui::IsPopupOpen(ImGui::GetCurrentWindow()->GetID("PinMenu"))))
        {
            ImGui::OpenPopup("PinMenu");
        }
        
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 8));
        if (ImGui::BeginPopup("PinMenu"))
        {
            ImGui::GetCurrentWindow()->GetID("Auto-Resolve Warning");

            if (isPopupOpen && !ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem | ImGuiHoveredFlags_ChildWindows) && (ImGui::IsMouseClicked(0) || ImGui::IsMouseClicked(1)))
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
                const std::string& passName = mpNodeGraphEditor->getPopupNode()->getName();
                RenderPassUI& renderPassUI = mRenderPassUI[passName];
                renderPassUI.renderPinUI(pGui, passName, this, mpNodeGraphEditor->getPopupPinIndex(), mpNodeGraphEditor->isPopupPinInput());

                ImGui::Separator();
            }

            if (mpNodeGraphEditor->selectedLink != -1)
            {
                ImGui::NodeLink& selectedLink = mpNodeGraphEditor->getLink(mpNodeGraphEditor->selectedLink);
                std::string srcPassName = std::string(selectedLink.InputNode->getName());
                std::string dstPassName = std::string(selectedLink.OutputNode->getName());
                pGui->addText((std::string("Edge: ") + srcPassName + '-' + dstPassName).c_str());
                std::string inputString = dstPassName + "." + 
                    std::string(static_cast<RenderGraphNode*>(selectedLink.OutputNode)->getInputName(selectedLink.OutputSlot));
                uint32_t linkID = mInputPinStringToLinkID[inputString];
                auto edgeIt = mpRenderGraph->mEdgeData.find(linkID);

                // link exists, but is not an edge (such as graph output edge)
                if (edgeIt != mpRenderGraph->mEdgeData.end())
                {
                    RenderGraph::EdgeData& edgeData = edgeIt->second;

                    pGui->addText("Src Field : ");
                    pGui->addText(edgeData.srcField.c_str(), true);

                    pGui->addText("Dst Field : ");
                    pGui->addText(edgeData.dstField.c_str(), true);

                    pGui->addText("Auto-Generated : ");
                    pGui->addText(edgeData.autoGenerated ? "true" : "false", true);
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
        
        ImVec2 mousePos = ImGui::GetMousePos();
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
        if (mpNodeGraphEditor->getRenderUINodeName().size())
        {
            std::string idString = std::string("Render UI##") + mpNodeGraphEditor->getRenderUINodeName();
            if (!ImGui::IsPopupOpen(ImGui::GetCurrentWindow()->GetID(idString.c_str())))
            {
                ImGui::OpenPopup(idString.c_str());
            }
            if (ImGui::BeginPopup(idString.c_str()))
            {
                if (!ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem | ImGuiHoveredFlags_ChildWindows) && (ImGui::IsMouseClicked(0) || ImGui::IsMouseClicked(1)))
                {
                    mpNodeGraphEditor->setRenderUINodeName("");
                }
                else
                {
                    std::string renderUIName = mpNodeGraphEditor->getRenderUINodeName();
                    auto pPass = mpRenderGraph->getPass(renderUIName);
                    pGui->addText((renderUIName + " UI ").c_str());
                    ImGui::Separator();
                    pPass->renderUI(pGui, nullptr); 
                    // TODO -- only call this with data change
                    mpIr->updatePass(renderUIName, pPass->getScriptingDictionary());
                    mShouldUpdate = true;
                }
                
                ImGui::EndPopup();
            }
        }
        
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

        if (!mpNodeGraphEditor->isInited())
        {
            mpNodeGraphEditor->render();
            
            std::string statement;
            bool addPass = false;
            if (pGui->dragDropDest("RenderPassType", statement))
            {
                dragAndDropText = statement;
                mNewNodeStartPosition = { -mpNodeGraphEditor->offset.x + mousePos.x, -mpNodeGraphEditor->offset.y + mousePos.y };
                mNewNodeStartPosition /= ImGui::GetCurrentWindow()->FontWindowScale;
                mNextPassName = statement;
                // only open pop-up if right clicked
                mDisplayDragAndDropPopup = ImGui::GetIO().KeyCtrl;
                addPass = !mDisplayDragAndDropPopup;
            }

            if (mDisplayDragAndDropPopup)
            {
                pGui->pushWindow("CreateNewGraph", 256, 128, 
                    static_cast<uint32_t>(mousePos.x), static_cast<uint32_t>(mousePos.y));

                pGui->addTextBox("Pass Name", mNextPassName);
                if (pGui->addButton("create##renderpass"))
                {
                    addPass = true;
                }
                if (pGui->addButton("cancel##renderPass"))
                {
                    mDisplayDragAndDropPopup = false;
                }

                pGui->popWindow();
            }

            if (addPass)
            {
                while (mpRenderGraph->doesPassExist(mNextPassName))
                {
                    mNextPassName.push_back('_');
                }

                mpIr->addPass(dragAndDropText, mNextPassName);
                bFromDragAndDrop = true;
                mDisplayDragAndDropPopup = false;
                mShouldUpdate = true;
                if (mMaxNodePositionX < mNewNodeStartPosition.x) mMaxNodePositionX = mNewNodeStartPosition.x;
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
            return;
        }
        
        for (auto& nodeTypeString : mAllNodeTypeStrings)
        {
            mAllNodeTypes.push_back(nodeTypeString.c_str());
        }

        mpNodeGraphEditor->registerNodeTypes(mAllNodeTypes.data(), static_cast<uint32_t>(mAllNodeTypes.size()), NodeGraphEditorGui::createNode, 0, -1, 0, 0);

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
            RenderPass* pNodeRenderPass = mpRenderGraph->getPass(currentPass.first).get();
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

    void RenderGraphUI::setRecordUpdates(bool recordUpdates)
    {
        mRecordUpdates = recordUpdates;
    }

    void RenderGraphUI::updatePins(bool addLinks)
    {
        //  Draw pin connections. All the nodes have to be added to the GUI before the connections can be drawn
        for (auto& currentPass : mRenderPassUI)
        {
            auto& currentPassUI = currentPass.second;

            for (auto& currentPinUI : currentPassUI.mOutputPins)
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
                                RenderGraphNode* pNode = static_cast<RenderGraphNode*>(mpNodeGraphEditor->getNodeFromID(connectedPin.second));
                                std::string dstName = pNode->getInputName(connectedPin.first);
                                std::string srcString = currentPass.first + "." + currentPinName;
                                std::string dstString = std::string(pNode->getName()) + "." + dstName;
                                const RenderPassReflection::Field& srcPin = currentPassUI.mReflection.getField(currentPinName);
                                const RenderPassReflection::Field& dstPin = mRenderPassUI[pNode->getName()].mReflection.getField(dstName);

                                uint32_t edgeColor = kEdgesColor;
                                if (mpRenderGraph->canAutoResolve(srcPin, dstPin))
                                {
                                    mLogString += std::string("Warning: Edge ") + srcString + " - " + dstName + " can auto-resolve.\n";
                                    edgeColor = kAutoResolveEdgesColor;
                                }
                                else
                                {
                                    uint32_t edgeId = mpRenderGraph->getEdge(srcString, dstString);
                                    if (mpRenderGraph->mEdgeData[edgeId].autoGenerated) edgeColor = kAutoGenEdgesColor;
                                }

                                mpNodeGraphEditor->addLinkFromGraph(mpNodeGraphEditor->getNodeFromID(currentPassUI.mGuiNodeID), currentPinUI.mGuiPinID,
                                    mpNodeGraphEditor->getNodeFromID(connectedPin.second), connectedPin.first, false, edgeColor);

                                static_cast<RenderGraphNode*>(mpNodeGraphEditor->getNodeFromID(connectedPin.second))->mInputPinConnected[connectedPin.first] = true;
                                static_cast<RenderGraphNode*>(mpNodeGraphEditor->getNodeFromID(currentPassUI.mGuiNodeID))->mOutputPinConnected[currentPinUI.mGuiPinID] = true;
                            }
                        }
                    }

                    // mark graph outputs to graph output node
                    if (currentPinUI.mIsGraphOutput)
                    {
                        static_cast<RenderGraphNode*>(mpNodeGraphEditor->getNodeFromID(currentPassUI.mGuiNodeID))->setPinColor(kGraphOutputsColor, currentPinUI.mGuiPinID);
                    }
                }
                else
                {
                    if (!currentPinUI.mIsGraphOutput)
                    {
                        static_cast<RenderGraphNode*>(mpNodeGraphEditor->getNodeFromID(currentPassUI.mGuiNodeID))->setPinColor(ImGui::NodeGraphEditor::GetStyle().color_node_output_slots, currentPinUI.mGuiPinID);
                    }
                }
            }

            if (!addLinks)
            {
                for (auto& currentPinUI : currentPassUI.mInputPins)
                {
                    const std::string& currentPinName = currentPinUI.mPinName;
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

        if (!mpRenderGraph->mExecutionList.size())
        {
            newNodePosition.x += offsetX * nodeID;
        }
        else
        {
            for (const auto& passID : mpRenderGraph->mExecutionList)
            {
                newNodePosition.x += offsetX;

                if (passID == nodeID)
                {
                    const DirectedGraph::Node* pNode = mpRenderGraph->mpGraph->getNode(nodeID);
                    if (!pNode->getIncomingEdgeCount())
                    {
                        newNodePosition.y += offsetY * pNode->getIncomingEdgeCount() * passID;
                        break;
                    }

                    for (uint32_t i = 0; i < pNode->getIncomingEdgeCount(); ++i)
                    {
                        uint32_t outgoingEdgeCount = mpRenderGraph->mpGraph->getNode(mpRenderGraph->mpGraph->getEdge(pNode->getIncomingEdge(i))->getSourceNode())->getOutgoingEdgeCount();
                        if (outgoingEdgeCount > pNode->getIncomingEdgeCount())
                        {
                            // move down by index in 
                            newNodePosition.y += offsetY * (outgoingEdgeCount - pNode->getIncomingEdgeCount());
                            break;
                        }
                    }

                    break;
                }
            }
        }
        if (newNodePosition.x > mMaxNodePositionX)
        {
            mMaxNodePositionX = newNodePosition.x;
        }

        return newNodePosition;
    }

    void RenderGraphUI::updateDisplayData()
    {
        uint32_t nodeIndex = 0;

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
            renderPassUI.mReflection = mpRenderGraph->mNodeData[nameToIndex.second].pPass->reflect();

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
                    bool isInput = is_set(renderPassUI.mReflection.getField(pinIndex).getType(),RenderPassReflection::Field::Type::Input);
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

                if (is_set(currentField.getType(), RenderPassReflection::Field::Type::Input))
                {
                    if (nodeConnectedInput.find(nameToIndex.first + "." + currentField.getName()) == nodeConnectedInput.end())
                    {
                        renderPassUI.addUIPin(currentField.getName(), inputPinIndex, true, "");
                    }

                    inputPinIndex++;
                }
                
                if (is_set(currentField.getType(), RenderPassReflection::Field::Type::Output))
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