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
#include "stdafx.h"
#include "RenderGraphUI.h"
#include "dear_imgui/imgui.h"
#include "dear_imgui_addons/imguinodegrapheditor/imguinodegrapheditor.h"
#include "dear_imgui/imgui_internal.h"
#include <fstream>
#include "RenderPassLibrary.h"

namespace Falcor
{
    const float kUpdateTimeInterval = 2.0f;
    const float kPinRadius = 7.0f;

    static const float kTimeTillPopup = 2.0f;
    static const uint32_t kPinColor = 0xFFFFFFFF;
    static const uint32_t kEdgesColor = 0xFFFFFFFF;
    static const uint32_t kAutoGenEdgesColor = 0xFFFF0400;
    static const uint32_t kAutoResolveEdgesColor = 0xFF0104FF;
    static const uint32_t kGraphOutputsColor = 0xAF0101FF;
    static const uint32_t kExecutionEdgeColor = 0xFF0AF1FF;
    static const std::string  kInPrefix = "##inEx_ ";
    static const std::string kOutPrefix = "##outEx_";

    class RenderGraphUI::NodeGraphEditorGui : public ImGui::NodeGraphEditor
    {
    public:
        NodeGraphEditorGui(RenderGraphUI* pRenderGraphUI) : mpRenderGraphUI(pRenderGraphUI) {}

        // call on beginning a new frame
        void setCurrentGui(Gui* pGui) { mpGui = pGui; }

        Gui* getCurrentGui() const { assert(mpGui); return mpGui; }

        void reset() { inited = false; }

        float2 getOffsetPos() const { return { offset.x, offset.y }; }

        ImGui::NodeLink& getLink(int32_t index) { return links[index]; }

        void setLinkColor(uint32_t index, uint32_t col) { links[index].LinkColor = col; }

        RenderGraphUI* getRenderGraphUI() { return mpRenderGraphUI; }

        void setPopupNode(ImGui::Node* pFocusedNode) { mpFocusedNode = pFocusedNode;  }

        ImGui::Node* getPopupNode() { return mpFocusedNode; }

        void setPopupPin(uint32_t pinIndex, bool isInput)
        {
            if (ImGui::IsAnyMouseDown())
            {
                mPopupPinHoverTime = 0.0f;
                mPinIndexToDisplay = -1;
            }

            if ((pinIndex != -1) )//&& pinIndex != mPinIndexToDisplay)
            {
                std::chrono::system_clock::time_point thisTime = std::chrono::system_clock::now();
                float timeDiff = (std::chrono::duration<float>(thisTime - mLastTime)).count();
                if(timeDiff < kTimeTillPopup) mPopupPinHoverTime += timeDiff;
                mLastTime = thisTime;
                if (mPopupPinHoverTime < kTimeTillPopup) return;
            }

            if (mPopupPinHoverTime >= kTimeTillPopup) mPopupPinHoverTime = 0.0f;
            mPinIndexToDisplay = pinIndex; mPopupPinIsInput = isInput;
        }

        void deselectPopupPin()
        {
            std::chrono::system_clock::time_point thisTime = std::chrono::system_clock::now();
            float timeDiff = (std::chrono::duration<float>(thisTime - mLastTime)).count();
            if (timeDiff < kTimeTillPopup) mNothingHoveredTime += timeDiff;
            mLastTimeDeselect = thisTime;

            if (mNothingHoveredTime >= kTimeTillPopup)
            {
                mNothingHoveredTime = 0.0f;
                mPinIndexToDisplay = static_cast<uint32_t>(-1);
            }
        }

        uint32_t getPopupPinIndex() const { return mPinIndexToDisplay; }

        bool isPopupPinInput() const { return mPopupPinIsInput; }

        ImGui::Node*& getNodeFromID(uint32_t nodeID) { return mpIDtoNode[nodeID]; }

        // wraps around creating link to avoid setting static flag
        uint32_t addLinkFromGraph(ImGui::Node* inputNode, int input_slot, ImGui::Node* outputNode, int output_slot, bool checkIfAlreadyPresent = false, ImU32 col = ImColor(200, 200, 100))
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

        float mPopupPinHoverTime = 0.0f;
        float mNothingHoveredTime = 0.0f;
        std::chrono::system_clock::time_point mLastTime = std::chrono::system_clock::now();
        std::chrono::system_clock::time_point mLastTimeDeselect = std::chrono::system_clock::now();
        Gui* mpGui = nullptr;
        void* mpCBUserData = nullptr;
        ImGui::Node* mpFocusedNode = nullptr;
        uint32_t mPinIndexToDisplay = uint32_t(-1);
        bool mPopupPinIsInput = false;
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
        void setPos(const float2& pos)
        {
            Pos.x = pos.x;
            Pos.y = pos.y;
        }

        float2 getPos()
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
            ImGui::TextUnformatted(dummyText.c_str());
            ImGui::TextUnformatted(dummyText.c_str());
            ImGui::TextUnformatted(dummyText.c_str());
            ImGui::TextUnformatted(dummyText.c_str());

            for (int32_t i = 0; i < paddingSpace; ++i)
            {
                ImGui::TextUnformatted(dummyText.c_str());
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
                        uint32_t hoveredPinColor = (pinColor == kGraphOutputsColor) ? (pinColor | 0xFF000000) : ImGui::GetColorU32(pGraphEditorGui->GetStyle().color_node_title);
                        ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(inputPos.x, inputPos.y), kPinRadius, hoveredPinColor);

                        if (pRenderPass)
                        {
                            pGraphEditorGui->setPopupNode(pCurrentNode);
                            pGraphEditorGui->setPopupPin(i, !static_cast<bool>(j));
                        }
                    }
                    else
                    {
                        ImGui::GetWindowDrawList()->AddCircle(ImVec2(inputPos.x, inputPos.y), kPinRadius, pinColor);
                    }

                    auto pText = isInputs ? pCurrentNode->InputNames[i] : pCurrentNode->OutputNames[i];
                    bool drawLabel = !(pText[0] == '#');

                    if (pinColor == kGraphOutputsColor)
                    {
                        ImVec2 arrowPoints[3] = { { inputPos.x + kPinRadius * 3.0f / 2.0f, inputPos.y + kPinRadius },
                        { inputPos.x + kPinRadius * 3.0f / 2.0f + kPinRadius, inputPos.y },
                        { inputPos.x + kPinRadius * 3.0f / 2.0f, inputPos.y - kPinRadius } };
                        ImGui::GetWindowDrawList()->AddPolyline(arrowPoints, 3, pinColor, false, 3.0f);
                    }
                    else if (!drawLabel && pinColor == kExecutionEdgeColor)
                    {
                        // we can draw anything special for execution edge inputs here
                    }

                    ImGui::SetCursorScreenPos({ inputPos.x + pinOffsetx - ((pinOffsetx < 0.0f) ? ImGui::CalcTextSize(isInputs ? pCurrentNode->InputNames[i] : pCurrentNode->OutputNames[i]).x : 0.0f), inputPos.y - kPinRadius });

                    slotNum++;
                    if (drawLabel) ImGui::TextUnformatted(pText);
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
                ImGui::TextUnformatted(dummyText.c_str());
            }

            ImGui::TextUnformatted(dummyText.c_str());

            return false;
        }

        void initialize(const std::string& name, const std::string& outputsString,
            const std::string& inputsString, uint32_t guiNodeID, RenderPass* pRenderPass)
        {
            init(name.c_str(), Pos, inputsString.c_str(), outputsString.c_str(), guiNodeID);

            if (pRenderPass)
            {
                mpRenderPass = pRenderPass;
                const float4 nodeColor = Gui::pickUniqueColor(pRenderPass->getName());
                overrideTitleBgColor = ImGui::GetColorU32({ nodeColor.x, nodeColor.y, nodeColor.z, nodeColor.w });
            }

            bool isInputs = true;
            uint32_t pinCount = static_cast<uint32_t>(InputsCount);
            for (uint32_t j = 0; j < 2; ++j)
            {
                for (uint32_t i = 0; i < pinCount; ++i)
                {
                    if (isInputs) inputColors[i] = ImColor(150, 150, 150, 150);
                    else outputColors[i] = ImColor(150, 150, 150, 150);
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
            std::string outputName = inputNode->getOutputName(link.InputSlot);
            addStatus = pGraphEditorGui->getRenderGraphUI()->addLink(
                inputNode->getName(), outputNode->getName(), outputName, outputNode->getInputName(link.OutputSlot), link.LinkColor);

            // immediately remove link if it is not a legal edge in the render graph
            if (!addStatus && !editor.isInited()) //  only call after graph is setup
            {
                // does not call link callback surprisingly enough
                editor.removeLink(link.InputNode, link.InputSlot, link.OutputNode, link.OutputSlot);
                return;
            }

            if (outputName[0] == '#')
            {
                static_cast<RenderGraphNode*>(link.InputNode)->setPinColor(link.LinkColor, link.InputSlot, false);
                static_cast<RenderGraphNode*>(link.OutputNode)->setPinColor(link.LinkColor, link.OutputSlot, true);
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
        mpIr->addPass(nodeTypeName, name);
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
        std::string srcString = srcPass + (srcField[0] == '#' ? "" : ".") + srcField;
        std::string dstString = dstPass + (dstField[0] == '#' ? "" : ".") + dstField;
        bool canCreateEdge = (mpRenderGraph->getEdge(srcString, dstString) == ((uint32_t)-1));
        if (!canCreateEdge) return canCreateEdge;

        // update the ui to reflect the connections. This data is used for removal
        RenderPassUI& srcRenderPassUI = mRenderPassUI[srcPass];
        RenderPassUI& dstRenderPassUI = mRenderPassUI[dstPass];
        const auto outputIt = srcRenderPassUI.mNameToIndexOutput.find(srcField);
        const auto inputIt =  dstRenderPassUI.mNameToIndexInput.find(dstField);
        // if one filed is empty, check that the other is as well
        if ((dstField[0] == '#') || (srcField[0] == '#'))
            canCreateEdge &= (srcField[0] == '#') && (dstField[0] == '#');
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

            if (!(dstField[0] == '#'))
            {
                RenderPassReflection srcReflection = mpRenderGraph->mNodeData[mpRenderGraph->getPassIndex(srcPass)].pPass->reflect({});
                RenderPassReflection dstReflection = mpRenderGraph->mNodeData[mpRenderGraph->getPassIndex(dstPass)].pPass->reflect({});

                bool canAutoResolve = false;// mpRenderGraph->canAutoResolve(srcReflection.getField(srcField), dstReflection.getField(dstField));
                if (canAutoResolve && mDisplayAutoResolvePopup) canCreateEdge = autoResolveWarning(srcString, dstString);
                color = canAutoResolve ? kAutoResolveEdgesColor : kEdgesColor;
            }

            if (canCreateEdge)
            {

                mShouldUpdate = true;

                if (dstField[0] == '#')
                {
                    // rebuilds data to avoid repeated code
                    mRebuildDisplayData = true;
                    color = kExecutionEdgeColor;
                    mpIr->addEdge(srcPass, dstPass);
                }
                else
                {
                    mpIr->addEdge(srcString, dstString);
                }
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
        if (dstField[0] == '#')
        {
            assert(srcField[0] == '#');
            mpIr->removeEdge(srcPass, dstPass);
        }
        else
        {
            mpIr->removeEdge(srcPass + "." + srcField, dstPass + "." + dstField);
        }
        mShouldUpdate = true;
    }

    void RenderGraphUI::removeRenderPass(const std::string& name)
    {
        mRebuildDisplayData = true;
        mpIr->removePass(name);
        mShouldUpdate = true;
    }

    void RenderGraphUI::updateGraph(RenderContext* pContext)
    {
        if (!mShouldUpdate) return;
        std::string newCommands = mpIr->getIR();
        mpIr = RenderGraphIR::create(mRenderGraphName, false); // reset
        if (mLastCommand == newCommands) return;

        mLastCommand = newCommands;
        if (mRecordUpdates) mUpdateCommands += newCommands;

        // update reference graph to check if valid before sending to next
        Scripting::getGlobalContext().setObject("g", mpRenderGraph);
        Scripting::runScript(newCommands);
        if(newCommands.size()) mLogString += newCommands;

        // only send updates that we know are valid.
        if (mpRenderGraph->compile(pContext) == false) mLogString += "Graph is currently invalid\n";
        mShouldUpdate = false;
        mRebuildDisplayData = true;
    }

    void RenderGraphUI::writeUpdateScriptToFile(RenderContext* pContext, const std::string& filePath, float lastFrameTime)
    {
        if ((mTimeSinceLastUpdate += lastFrameTime) < kUpdateTimeInterval) return;
        mTimeSinceLastUpdate = 0.0f;
        if (!mUpdateCommands.size()) return;

        // only send delta of updates once the graph is valid
        if (mpRenderGraph->compile(pContext) == false) return;
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
            pGraphEditor->getRenderGraphUI()->addRenderPass(node->getName(), getClassTypeName(pRenderGraphNode->mpRenderPass));
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

        nameToIndexMapRef.insert(std::make_pair(pinUIData.mPinName, static_cast<uint32_t>(guiPinID) ));
    }

    void RenderPassUI::renderPinUI(const std::string& passName, RenderGraphUI* pGraphUI, uint32_t index, bool input)
    {
        RenderPassUI::PinUI& pinUI = input ? mInputPins[index] : mOutputPins[index];

        size_t fieldIndex = -1;
        for (size_t i = 0; i < mReflection.getFieldCount(); ++i)
        {
            if (mReflection.getField(i)->getName() == pinUI.mPinName)
            {
                fieldIndex = i;
                break;
            }
        }

        if (fieldIndex != -1)
        {
            // only render ui if is input or output
            const RenderPassReflection::Field& field = *mReflection.getField(fieldIndex);
            if (is_set(field.getVisibility(), RenderPassReflection::Field::Visibility::Input) || is_set(field.getVisibility(), RenderPassReflection::Field::Visibility::Output))
            {
                pinUI.renderUI(field, pGraphUI, passName);
            }
        }
    }

    void RenderPassUI::PinUI::renderFieldInfo(const RenderPassReflection::Field& field, RenderGraphUI* pGraphUI, const std::string& passName, const std::string& fieldName)
    {
        RenderPassReflection::Field::Visibility type = field.getVisibility();
        uint32_t isInput = is_set(type, RenderPassReflection::Field::Visibility::Input);
        uint32_t isOutput = is_set(type, RenderPassReflection::Field::Visibility::Output);
        bool isExecutionPin = fieldName[0] == '#';

        if (isExecutionPin)
        {
            ImGui::TextUnformatted("Execution Pin");
            return;
        }

        ImGui::SameLine();
        ImGui::TextUnformatted((std::string("Field Name: ") + fieldName).c_str());
        ImGui::Separator();
        ImGui::TextUnformatted((field.getDesc() + "\n\n").c_str());

        ImGui::TextUnformatted("ResourceFlags : ");

        if (isInput && isOutput)
        {
            ImGui::SameLine();
            ImGui::TextUnformatted("InputOutput");
        }
        else if (isInput)
        {
            ImGui::SameLine();
            ImGui::TextUnformatted("Input");
        }
        else if (isOutput)
        {
            ImGui::SameLine();
            ImGui::TextUnformatted("Output");
        }

        ImGui::TextUnformatted("ResourceType : ");
        ImGui::SameLine();
        ImGui::TextUnformatted(to_string(field.getType()).c_str());

        ImGui::TextUnformatted("Width: ");
        ImGui::SameLine();
        ImGui::TextUnformatted(std::to_string(field.getWidth()).c_str());

        ImGui::TextUnformatted("Height: ");
        ImGui::SameLine();
        ImGui::TextUnformatted(std::to_string(field.getHeight()).c_str());

        ImGui::TextUnformatted("Depth: ");
        ImGui::SameLine();
        ImGui::TextUnformatted(std::to_string(field.getDepth()).c_str());

        ImGui::TextUnformatted("Sample Count: ");
        ImGui::SameLine();
        ImGui::TextUnformatted(std::to_string(field.getSampleCount()).c_str());

        ImGui::TextUnformatted("ResourceFormat: ");
        ImGui::SameLine();
        ImGui::TextUnformatted(to_string(field.getFormat()).c_str());

        ImGui::TextUnformatted("BindFlags: ");
        ImGui::SameLine();
        ImGui::TextUnformatted(to_string(field.getBindFlags()).c_str());

        ImGui::TextUnformatted("Flags: ");
        switch (field.getFlags())
        {
        case RenderPassReflection::Field::Flags::None:
            ImGui::SameLine();
            ImGui::TextUnformatted("None");
            break;
        case RenderPassReflection::Field::Flags::Optional:
            ImGui::SameLine();
            ImGui::TextUnformatted("Optional");
            break;
        case RenderPassReflection::Field::Flags::Persistent:
            ImGui::SameLine();
            ImGui::TextUnformatted("Persistent");
            break;
        default:
            should_not_get_here();
        }
    }

    void RenderPassUI::PinUI::renderUI(const RenderPassReflection::Field& field, RenderGraphUI* pGraphUI, const std::string& passName)
    {
        ImGui::TextUnformatted(mIsGraphOutput ? "Graph Output : " : "");
        renderFieldInfo(field, pGraphUI, passName, mPinName);
        bool isExecutionPin = mPinName[0] == '#';

        if (!isExecutionPin && is_set(field.getVisibility(), RenderPassReflection::Field::Visibility::Output))
        {
            bool isGraphOut = mIsGraphOutput;
            if (ImGui::Checkbox("Graph Output", &mIsGraphOutput))
            {
                if (isGraphOut && !mIsGraphOutput)
                {
                    pGraphUI->removeOutput(passName, mPinName);
                }
                else if (!isGraphOut && mIsGraphOutput)
                {
                    pGraphUI->addOutput(passName, mPinName);
                }
            }
        }

        ImGui::Separator();
    }

    void RenderGraphUI::renderPopupMenu()
    {
        bool isPopupOpen = false;
        bool first = false;

        if (!(isPopupOpen = ImGui::IsPopupOpen(ImGui::GetCurrentWindow()->GetID("PinMenu"))))
        {
            ImGui::OpenPopup("PinMenu");
            first = true;
        }

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8, 8));
        if (ImGui::BeginPopup("PinMenu"))
        {
            if (first) ImGui::SetWindowPos({ ImGui::GetWindowPos().x - 8, ImGui::GetWindowPos().y - 8 });
            else if (isPopupOpen && !ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem | ImGuiHoveredFlags_ChildWindows))
            {
                if (ImGui::IsMouseClicked(0) || ImGui::IsMouseClicked(1))
                {
                    mpNodeGraphEditor->selectedLink = -1;
                    mpNodeGraphEditor->setPopupNode(nullptr);
                    mpNodeGraphEditor->setPopupPin(-1, false);
                }
                else mpNodeGraphEditor->deselectPopupPin();
            }

            if (mpNodeGraphEditor->getPopupNode() && mpNodeGraphEditor->getPopupPinIndex() != -1)
            {
                const std::string& passName = mpNodeGraphEditor->getPopupNode()->getName();
                RenderPassUI& renderPassUI = mRenderPassUI[passName];
                renderPassUI.renderPinUI(passName, this, mpNodeGraphEditor->getPopupPinIndex(), mpNodeGraphEditor->isPopupPinInput());
                ImGui::Separator();
            }

            if (mpNodeGraphEditor->selectedLink != -1)
            {
                ImGui::NodeLink& selectedLink = mpNodeGraphEditor->getLink(mpNodeGraphEditor->selectedLink);
                std::string srcPassName = std::string(selectedLink.InputNode->getName());
                std::string dstPassName = std::string(selectedLink.OutputNode->getName());
                ImGui::TextUnformatted((std::string("Edge: ") + srcPassName + '-' + dstPassName).c_str());
                std::string inputName = std::string(static_cast<RenderGraphNode*>(selectedLink.OutputNode)->getInputName(selectedLink.OutputSlot));
                std::string inputString = dstPassName + (inputName.empty() ? "" : ".") + inputName;
                uint32_t linkID = mInputPinStringToLinkID[inputString];
                auto edgeIt = mpRenderGraph->mEdgeData.find(linkID);

                // link exists, but is not an edge (such as graph output edge)
                if (edgeIt != mpRenderGraph->mEdgeData.end())
                {
                    RenderGraph::EdgeData& edgeData = edgeIt->second;

                    if (edgeData.srcField.size())
                    {
                        ImGui::TextUnformatted("Src Field : ");
                        ImGui::SameLine();
                        ImGui::TextUnformatted(edgeData.srcField.c_str());
                    }

                    if (edgeData.dstField.size())
                    {
                        ImGui::TextUnformatted("Dst Field : ");
                        ImGui::SameLine();
                        ImGui::TextUnformatted(edgeData.dstField.c_str());
                    }

                    if (edgeData.dstField.size() || edgeData.srcField.size())
                    {
                        ImGui::TextUnformatted("Auto-Generated : ");
                        ImGui::SameLine();
                        ImGui::TextUnformatted(edgeData.autoGenerated ? "true" : "false");
                    }
                }
            }

            ImGui::EndPopup();
        }
        ImGui::PopStyleVar();
    }

    void RenderGraphUI::renderUI(RenderContext* pContext, Gui* pGui)
    {
        static std::string dragAndDropText;
        ImGui::GetIO().FontAllowUserScaling = true; // FIXME
        mpNodeGraphEditor->mpRenderGraphUI = this;
        mpNodeGraphEditor->setCurrentGui(pGui);
        mpNodeGraphEditor->show_top_pane = false;
        mpNodeGraphEditor->show_node_copy_paste_buttons = false;
        mpNodeGraphEditor->show_connection_names = false;
        mpNodeGraphEditor->show_left_pane = false;
        mpNodeGraphEditor->setLinkCallback(NodeGraphEditorGui::setLinkFromGui);
        mpNodeGraphEditor->setNodeCallback(NodeGraphEditorGui::setNode);

        ImVec2 mousePos = ImGui::GetMousePos();
        ImGui::NodeGraphEditor::Style& style = mpNodeGraphEditor->GetStyle();
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

        ImVector<const ImGui::Node*> selectedNodes;
        mpNodeGraphEditor->getSelectedNodes(selectedNodes);

        // push update commands for the open pop-up
        Gui::Window renderWindow(pGui, "Render UI", { 250, 200 });
        for (uint32_t i = 0 ; i  < static_cast<uint32_t>(selectedNodes.size()); ++i)
        {
            std::string renderUIName = selectedNodes.Data[i]->getName();

            if (auto renderGroup = renderWindow.group(renderUIName, true))
            {
                auto pPass = mpRenderGraph->getPass(renderUIName);
                bool internalResources = false;

                renderGroup.separator();
                std::string wrappedText = std::string("Description:  ") + RenderPassLibrary::getClassDescription(getClassTypeName(pPass.get()));
                ImGui::TextWrapped("%s", wrappedText.c_str());
                renderGroup.separator();

                pPass->renderUI(renderGroup);

                for (uint32_t i = 0; i < mRenderPassUI[renderUIName].mReflection.getFieldCount(); ++i)
                {
                    const auto& field = *mRenderPassUI[renderUIName].mReflection.getField(i);
                    if (is_set(field.getVisibility(), RenderPassReflection::Field::Visibility::Internal))
                    {
                        if (!internalResources)
                        {
                            renderGroup.separator(); renderGroup.text("\n\n"); renderGroup.separator();
                            renderGroup.text("Internal Resources:");
                            renderGroup.text("\n\n");
                            renderGroup.separator();
                        }
                        RenderPassUI::PinUI::renderFieldInfo(field, this, renderUIName, field.getName());
                        internalResources = true;
                    }
                }
                if (internalResources) renderGroup.separator();
                // TODO -- only call this with data change
                if (ImGui::IsWindowFocused())
                {
                    mpIr->updatePass(renderUIName, pPass->getScriptingDictionary());
                }
                mShouldUpdate = true;
            }
        }

        renderWindow.release();

        if (mpNodeGraphEditor->getPopupPinIndex() != uint32_t(-1) || (mpNodeGraphEditor->selectedLink != -1))
        {
            renderPopupMenu();
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
            bool b = false;
            if (ImGui::BeginDragDropTarget())
            {
                auto dragDropPayload = ImGui::AcceptDragDropPayload("RenderPassType");
                b = dragDropPayload && dragDropPayload->IsDataType("RenderPassType") && (dragDropPayload->Data != nullptr);
                if (b)
                {
                    statement.resize(dragDropPayload->DataSize);
                    std::memcpy(&statement.front(), dragDropPayload->Data, dragDropPayload->DataSize);
                }

                ImGui::EndDragDropTarget();
            }

            if (b)
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
                Gui::Window createGraphWindow(pGui, "CreateNewGraph", mDisplayDragAndDropPopup, { 256, 128 }, { (uint32_t)mousePos.x, (uint32_t)mousePos.y });

                createGraphWindow.textbox("Pass Name", mNextPassName);
                if (createGraphWindow.button("create##renderpass"))
                {
                    addPass = true;
                }
                if (createGraphWindow.button("cancel##renderPass"))
                {
                    mDisplayDragAndDropPopup = false;
                }

                createGraphWindow.release();
            }

            if (addPass)
            {
                while (mpRenderGraph->doesPassExist(mNextPassName))
                {
                    mNextPassName.push_back('_');
                }

                mpIr->addPass(dragAndDropText, mNextPassName);
                mAddedFromDragAndDrop = true;
                mDisplayDragAndDropPopup = false;
                mShouldUpdate = true;
                if (mMaxNodePositionX < mNewNodeStartPosition.x) mMaxNodePositionX = mNewNodeStartPosition.x;
            }

            //  set editor window behind other windows. set top menu bar to always be infront.this is repeated for other render call
            ImGui::BringWindowToDisplayBack(ImGui::FindWindowByName("Graph Editor"));
            ImGui::BringWindowToDisplayFront(ImGui::FindWindowByName("##MainMenuBar"));
            return;
        }

        updateDisplayData(pContext);

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
                float2 nextPosition = mAddedFromDragAndDrop ? mNewNodeStartPosition : getNextNodePosition(mpRenderGraph->getPassIndex(nameString));

                mpNodeGraphEditor->getNodeFromID(guiNodeID) = mpNodeGraphEditor->addAndInitNode(guiNodeID,
                    nameString, outputsString, inputsString, guiNodeID, pNodeRenderPass,
                    ImVec2(nextPosition.x, nextPosition.y));
                mAddedFromDragAndDrop = false;
            }
        }

        updatePins();
        mpNodeGraphEditor->render();
        ImGui::BringWindowToDisplayBack(ImGui::FindWindowByName("Graph Editor"));
        ImGui::BringWindowToDisplayFront(ImGui::FindWindowByName("##MainMenuBar"));
    }

    void RenderGraphUI::reset()
    {
        mpNodeGraphEditor->reset();
        mpNodeGraphEditor->clear();
        mRebuildDisplayData = true;
    }

    std::vector<uint32_t> RenderGraphUI::getPassOrder()
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
                                uint32_t edgeColor = kEdgesColor;
                                // execution edges
                                if (dstName[0] == '#')
                                {
                                    edgeColor = kExecutionEdgeColor;
                                }
                                else
                                {
                                    std::string srcString = currentPass.first + "." + currentPinName;
                                    std::string dstString = std::string(pNode->getName()) + "." + dstName;
                                    const RenderPassReflection::Field& srcPin = *currentPassUI.mReflection.getField(currentPinName);
                                    const RenderPassReflection::Field& dstPin = *mRenderPassUI[pNode->getName()].mReflection.getField(dstName);

                                    if (false/*mpRenderGraph->canAutoResolve(srcPin, dstPin)*/)
                                    {
                                        mLogString += std::string("Warning: Edge ") + srcString + " - " + dstName + " can auto-resolve.\n";
                                        edgeColor = kAutoResolveEdgesColor;
                                    }
                                    else
                                    {
                                        uint32_t edgeId = mpRenderGraph->getEdge(srcString, dstString);
                                        if (mpRenderGraph->mEdgeData[edgeId].autoGenerated) edgeColor = kAutoGenEdgesColor;
                                    }
                                }

                                mpNodeGraphEditor->addLinkFromGraph(mpNodeGraphEditor->getNodeFromID(currentPassUI.mGuiNodeID), currentPinUI.mGuiPinID,
                                    mpNodeGraphEditor->getNodeFromID(connectedPin.second), connectedPin.first, false, edgeColor);

                                RenderGraphNode* pDstGraphNode = static_cast<RenderGraphNode*>(mpNodeGraphEditor->getNodeFromID(connectedPin.second));
                                RenderGraphNode* pSrcGraphNode = static_cast<RenderGraphNode*>(mpNodeGraphEditor->getNodeFromID(currentPassUI.mGuiNodeID));
                                pDstGraphNode->mInputPinConnected[connectedPin.first] = true;
                                pSrcGraphNode->mOutputPinConnected[currentPinUI.mGuiPinID] = true;
                                if (dstName[0] == '#')
                                {
                                    pSrcGraphNode->setPinColor(kExecutionEdgeColor, currentPinUI.mGuiPinID, false);
                                    pDstGraphNode->setPinColor(kExecutionEdgeColor, connectedPin.first, true);
                                }
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
                        static_cast<RenderGraphNode*>(mpNodeGraphEditor->getNodeFromID(currentPassUI.mGuiNodeID))->setPinColor(ImColor(150, 150, 150, 150), currentPinUI.mGuiPinID);
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
                        RenderGraphNode* pDstGraphNode = static_cast<RenderGraphNode*>(mpNodeGraphEditor->getNodeFromID(inputIDs.second));
                        RenderGraphNode* pSrcGraphNode = static_cast<RenderGraphNode*>(mpNodeGraphEditor->getNodeFromID(connectedNodeUI.mGuiNodeID));

                        pDstGraphNode->mInputPinConnected[inputIDs.first] = false;
                        pSrcGraphNode->mOutputPinConnected[inputPinID] = false;
                        pSrcGraphNode->setPinColor(kPinColor, inputPinID, false);
                        pDstGraphNode->setPinColor(kPinColor, inputIDs.first, true);

                        continue;
                    }
                }
            }
        }
    }

    float2 RenderGraphUI::getNextNodePosition(uint32_t nodeID)
    {
        const float offsetX = 384.0f;
        const float offsetY = 128.0f;
        float2 newNodePosition = mNewNodeStartPosition;

        auto topologicalSort = DirectedGraphTopologicalSort::sort(mpRenderGraph->mpGraph.get());

        // For each object in the vector, if it's being used in the execution, put it in the list
        for (auto& node : topologicalSort)
        {
            newNodePosition.x += offsetX;

            if (node == nodeID)
            {
                const DirectedGraph::Node* pNode = mpRenderGraph->mpGraph->getNode(nodeID);
                if (!pNode->getIncomingEdgeCount())
                {
                    newNodePosition.y += offsetY * pNode->getIncomingEdgeCount() * nodeID;
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

        if (newNodePosition.x > mMaxNodePositionX) mMaxNodePositionX = newNodePosition.x;
        return newNodePosition;
    }

    void RenderGraphUI::updateDisplayData(RenderContext* pContext)
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

        mpRenderGraph->compile(pContext);
        mRenderPassUI.clear();
        mInputPinStringToLinkID.clear();

        // build information for displaying graph
        for (const auto& nameToIndex : mpRenderGraph->mNameToIndex)
        {
            auto pCurrentPass = mpRenderGraph->mpGraph->getNode(nameToIndex.second);
            RenderPassUI renderPassUI;
            bool  addedExecutionInput = false;
            bool addedExecutionOutput = false;

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
            renderPassUI.mReflection = mpRenderGraph->mNodeData[nameToIndex.second].pPass->reflect({});

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
                    bool isInput = is_set(renderPassUI.mReflection.getField(pinIndex)->getVisibility(),RenderPassReflection::Field::Visibility::Input);
                    if (isInput)
                    {
                        if (renderPassUI.mReflection.getField(pinIndex)->getName() == currentEdge.dstField) { break;  }
                        inputPinIndex++;
                    }
                    pinIndex++;
                }

                auto pSourceNode = mpRenderGraph->mNodeData.find( mpRenderGraph->mpGraph->getEdge(edgeID)->getSourceNode());
                assert(pSourceNode != mpRenderGraph->mNodeData.end());
                addedExecutionInput = currentEdge.dstField.empty();

                std::string dstFieldName = currentEdge.dstField.empty() ? kInPrefix + nameToIndex.first : currentEdge.dstField;
                std::string inputPinString  = nameToIndex.first + "." + dstFieldName;
                std::string srcFieldName = currentEdge.srcField.empty() ? kOutPrefix + pSourceNode->second.name : currentEdge.srcField;
                std::string outputPinString = pSourceNode->second.name + "." + srcFieldName;
                if (nodeConnectedInput.find(inputPinString) == nodeConnectedInput.end())
                {
                    nodeConnectedInput.insert(inputPinString);
                }

                renderPassUI.addUIPin(dstFieldName, inputPinIndex, true, srcFieldName, pSourceNode->second.name);
                mOutputToInputPins[outputPinString].push_back(std::make_pair(inputPinIndex, renderPassUI.mGuiNodeID));
                mInputPinStringToLinkID.insert(std::make_pair(inputPinString, edgeID));
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
                    bool isOutput = (static_cast<uint32_t>(renderPassUI.mReflection.getField(pinIndex)->getVisibility() & RenderPassReflection::Field::Visibility::Output) != 0);
                    if (isOutput)
                    {
                        if (renderPassUI.mReflection.getField(pinIndex)->getName() == currentEdge.srcField) { break; }
                        outputPinIndex++;
                    }
                    pinIndex++;
                }

                auto pDestNode = mpRenderGraph->mNodeData.find(mpRenderGraph->mpGraph->getEdge(edgeID)->getDestNode());
                assert(pDestNode != mpRenderGraph->mNodeData.end());
                addedExecutionOutput = currentEdge.dstField.empty();

                std::string dstFieldName = currentEdge.dstField.empty() ? kInPrefix + pDestNode->second.name : currentEdge.dstField;
                std::string inputPinString = nameToIndex.first + "." + dstFieldName;
                std::string srcFieldName = currentEdge.srcField.empty() ? kOutPrefix + nameToIndex.first : currentEdge.srcField;
                std::string outputPinString = pDestNode->second.name + "." + srcFieldName;

                renderPassUI.addUIPin(srcFieldName, outputPinIndex, false, dstFieldName, pDestNode->second.name, isGraphOutput);
                nodeConnectedOutput.insert(pinString);
            }

            // Now we know which nodes are connected within the graph and not
            inputPinIndex = 0;
            outputPinIndex = 0;

            for (uint32_t i = 0; i < renderPassUI.mReflection.getFieldCount(); ++i)
            {
                const auto& currentField = *renderPassUI.mReflection.getField(i);

                if (is_set(currentField.getVisibility(), RenderPassReflection::Field::Visibility::Input))
                {
                    if (nodeConnectedInput.find(nameToIndex.first + "." + currentField.getName()) == nodeConnectedInput.end())
                    {
                        renderPassUI.addUIPin(currentField.getName(), inputPinIndex, true, "");
                    }

                    inputPinIndex++;
                }

                if (is_set(currentField.getVisibility(), RenderPassReflection::Field::Visibility::Output))
                {
                    if (nodeConnectedOutput.find(nameToIndex.first + "." + currentField.getName()) == nodeConnectedOutput.end())
                    {
                        bool isGraphOutput = passGraphOutputs.find(currentField.getName()) != passGraphOutputs.end();
                        renderPassUI.addUIPin(currentField.getName(), outputPinIndex, false, "", "", isGraphOutput);
                    }

                    outputPinIndex++;
                }
            }

            // unconnected nodes will be renamed when they are connected
            if (!addedExecutionInput)  renderPassUI.addUIPin(kInPrefix  + nameToIndex.first,  static_cast<uint32_t>( renderPassUI.mInputPins.size()), true, "", "", false);
            if (!addedExecutionOutput) renderPassUI.addUIPin(kOutPrefix + nameToIndex.first, static_cast<uint32_t>(renderPassUI.mOutputPins.size()), false, "", "", false);

            mRenderPassUI.emplace(std::make_pair(nameToIndex.first, std::move(renderPassUI)));
        }
    }
}
