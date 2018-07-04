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

// TODO Don't do this
#include "Externals/dear_imgui/imgui.h"
#include "Externals/dear_imgui/imgui_internal.h"

namespace Falcor
{
    std::string gName;
    std::string gOutputsString;
    std::string gInputsString;
    uint32_t gGuiNodeID;
    
    class RenderGraphNode : public ImGui::Node
    {
    public:
        static RenderGraphNode* create(const ImVec2& pos)
        {
            RenderGraphNode* node = (RenderGraphNode*)ImGui::MemAlloc(sizeof(RenderGraphNode));
            IM_PLACEMENT_NEW(node) RenderGraphNode();

            node->init(gName.c_str(), pos, gOutputsString.c_str(), gInputsString.c_str(), gGuiNodeID);
            
            return node;
        }
    private:
    };

    static ImGui::Node* createNode(int, const ImVec2& pos, const ImGui::NodeGraphEditor&)
    {
        return RenderGraphNode::create(pos);
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

    void RenderGraphUI::renderUI(Gui* pGui)
    {
        static ImGui::NodeGraphEditor nodeGraphEditor;

        uint32_t specialPinIndex = static_cast<uint32_t>(static_cast<uint16_t>(-1)); // used for avoiding conflicts with creating additional unique pins.
        uint32_t linkIndex = 0;
        uint32_t mouseDragBezierPin = 0;
        uint32_t debugPinIndex = 0;

        constexpr uint32_t bezierColor = 0xFFFFFFFF;
        static bool draggingConnection = false;
        static unsigned draggingPinIndex = 0;

        const ImVec2& mousePos = ImGui::GetCurrentContext()->IO.MousePos;

        // handle deleting nodes and connections?
        std::unordered_set<uint32_t> nodeIDsToDelete;
        std::vector<std::string> nodeNamesToDelete;

        if (pGui->addButton("refresh"))
        {
            nodeGraphEditor.clear();
            nodeGraphEditor.inited = false;
        }

        if (!nodeGraphEditor.isInited())
        {
            nodeGraphEditor.render();
            return;
        }

        updateDisplayData();

        std::vector<const char*> nodeTypes;

        for (auto& currentPass : mRenderPassUI)
        {
            nodeTypes.push_back(currentPass.first.c_str());
            nodeGraphEditor.registerNodeTypes(nodeTypes.data(), static_cast<uint32_t>(nodeTypes.size()), createNode);
        }

        gOutputsString = "";
        gInputsString = "";

        for (auto& currentPass : mRenderPassUI)
        {
            auto& currentPassUI = currentPass.second;

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
                    ImGui::SameLine();
                    pGui->addText(currentPinName.c_str());
                    gOutputsString += currentPinName + ";";
                }
                else
                {
                    const auto& inputPins = mOutputToInputPins.find(currentPinName);
                    assert (inputPins != mOutputToInputPins.end());
                    for (uint32_t connectedPin : (inputPins->second)) 
                    {
                        // nodeGraphEditor.addLink(, connectedPin, , currentPinUI.mGuiPinID);
                    }

                    gInputsString += currentPinName + ";";
                }

                debugPinIndex++;
            }

            gName = currentPass.first;
            gGuiNodeID = currentPassUI.mGuiNodeID;

            nodeGraphEditor.addNode(gGuiNodeID, ImVec2(40, 150));

            // TODO -- get this within the node window
            currentPassUI.renderUI(pGui);
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
        uint32_t pinIndex = 0;

        mOutputToInputPins.clear();
        mRenderPassUI.clear();

        // set of field names that have a connection and are represented in the graph
        std::unordered_set<std::string> nodeConnected;

        // build information for displaying graph
        for (const auto& nameToIndex : mRenderGraphRef.mNameToIndex)
        {
            auto pCurrentPass = mRenderGraphRef.mpGraph->getNode(nameToIndex.second);
            RenderPassUI renderPassUI;
            renderPassUI.mGuiNodeID = nodeIndex++;

            // add all of the incoming connections
            for (uint32_t i = 0; i < pCurrentPass->getIncomingEdgeCount(); ++i)
            {
                auto currentEdge = mRenderGraphRef.mEdgeData[pCurrentPass->getIncomingEdge(i)];
                mOutputToInputPins[currentEdge.srcField].push_back(pinIndex);
                renderPassUI.addUIPin(currentEdge.dstField, pinIndex++, true);
                nodeConnected.insert(currentEdge.dstField);
                nodeConnected.insert(currentEdge.srcField);
            }

            // add all of the outgoing connections
            for (uint32_t i = 0; i < pCurrentPass->getOutgoingEdgeCount(); ++i)
            {
                auto currentEdge = mRenderGraphRef.mEdgeData[pCurrentPass->getOutgoingEdge(i)];
                renderPassUI.addUIPin(currentEdge.srcField, pinIndex++, false);
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
                    renderPassUI.addUIPin(inputNode.name, pinIndex++, true);
                }

                // add the details description for each pin

            }

            for (const auto& outputNode : passData.outputs)
            {
                if (nodeConnected.find(outputNode.name) == nodeConnected.end())
                {
                    renderPassUI.addUIPin(outputNode.name, pinIndex++, true);
                }

                // add the details description for each pin
            }

            mRenderPassUI.emplace(std::make_pair(nameToIndex.first, std::move(renderPassUI)));
        }
    }

    void RenderGraphUI::addRenderPassNode()
    {
        // redo this
        // insert properties for editing this node for the graph editor
        // mNodeProperties[mpPasses.back().get()][0] = StringProperty("Output Name",
        //     [this](Property* stringProperty) {
        //     addFieldDisplayData(reinterpret_cast<RenderPass*>(stringProperty->mpMetaData),
        //         static_cast<StringProperty*>(stringProperty)->mData[0], false);
        // },
        // { "" }, "Add Output"
        //     );
        // mNodeProperties[mpPasses.back().get()][0].mpMetaData = mpPasses.back().get();
        // 
        // mNodeProperties[mpPasses.back().get()][1] = StringProperty("Input Name",
        //     [this](Property* stringProperty) {
        //     addFieldDisplayData(reinterpret_cast<RenderPass*>(stringProperty->mpMetaData),
        //         static_cast<StringProperty*>(stringProperty)->mData[0], true);
        // },
        // { "" }, "Add Input"
        //     );
        // mNodeProperties[mpPasses.back().get()][1].mpMetaData = mpPasses.back().get();
    }

    void RenderGraphUI::deserializeJson(const rapidjson::Document& reader)
    {
        const char* memberArrayNames[2] = { "OutputFields", "InputFields" };
        bool isInput = false;
    
        // all of the fields types have the same type of schema
        for (uint32_t i = 0; i < 2; ++i)
        {
            // insert the display data for the fields with no connections
            if (reader.FindMember(memberArrayNames[i]) == reader.MemberEnd())
            {
                return;
            }
    
            auto fields = reader.FindMember(memberArrayNames[i])->value.GetArray();
            for (const auto& field : fields)
            {
                std::string passNameString(field.FindMember("RenderPassName")->value.GetString());
                std::string fieldNameString(field.FindMember("FieldName")->value.GetString());
                mRenderPassUI[passNameString].mPins[fieldNameString] = {};
            }
            isInput = true;
        }
    }
    
    void RenderGraphUI::serializeJson(rapidjson::Writer<rapidjson::OStreamWrapper>* writer) const
    {
        // write out the nodes and node data
        writer->String("RenderPassNodes");
        writer->StartArray();
    /*
        for (auto& nameIndexPair : mRenderGraphRef.mNameToIndex)
        {
            writer->StartObject();
    
            writer->String("RenderPassName");
            writer->String(nameIndexPair.first.c_str());
            writer->String("RenderPassType");
            writer->String(mRenderPassUI[nameIndexPair.first].->getTypeName().c_str());
    
            // serialize specialized data here ( ? )
    
            writer->EndObject();
        }
    
        writer->EndArray();
    
        // write out the fields that cannot be filled out by the connections
        const char* memberArrayNames[2] = { "OutputFields", "InputFields" };
        bool isInput = false;
    
        for (uint32_t i = 0; i < 2; ++i)
        {
            writer->String(memberArrayNames[i]);
            writer->StartArray();
    
            for (const auto& nameToIndexMap : mDisplayMap)
            {
                for (const auto& nameToIndexIt : nameToIndexMap.second)
                {
                    if (nameToIndexIt.second.second == isInput)
                    {
                        writer->StartObject();
    
                        writer->String("RenderPassName");
                        writer->String(mPassToName[nameToIndexMap.first].c_str());
                        writer->String("FieldName");
                        writer->String(nameToIndexIt.first.c_str());
    
                        writer->EndObject();
                    }
                }
    
            }
    
            writer->EndArray();
            isInput = true;
        }
    
        // write out the node connections
        writer->String("Edges");
        writer->StartArray();
    
        for (auto& edge : mRenderGraphRef.mpGraph->mEdges)
        {
    
            writer->String("SrcField");
            writer->String(edge.srcField.c_str());
            writer->String("DstField");
            writer->String(edge.dstField.c_str());
            writer->EndObject();
        }
        
        for (const auto& nameToIndex : mRenderGraphRef.mNameToIndex)
        {
            auto pCurrentPass = mRenderGraphRef.mpGraph->getNode(nameToIndex.second);

            writer->StartObject();

            // add all of the incoming connections
            for (uint32_t i = 0; i < pCurrentPass->getIncomingEdgeCount(); ++i)
            {

                writer->String("SrcRenderPassName");
                writer->String(mPassToName[edge.pSrc].c_str());
                writer->String("DstRenderPassName");
                writer->String(mPassToName[edge.pDst].c_str());

                auto currentEdge = mRenderGraphRef.mpGraph->getEdgeData(pCurrentPass->getIncomingEdge(i));

            }

            // add all of the outgoing connections
            for (uint32_t i = 0; i < pCurrentPass->getOutgoingEdgeCount(); ++i)
            {
                auto currentEdge = mRenderGraphRef.mpGraph->getEdgeData(pCurrentPass->getOutgoingEdge(i));
                renderPassUI.addUIPin(currentEdge->srcField, pinIndex++, false);
            }

            mRenderPassUI.emplace(std::make_pair(nameToIndex.first, std::move(renderPassUI)));
        }
        */
        writer->EndArray();
    }
}