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
#include "Externals/imgui-node-editor/NodeEditor/Include/NodeEditor.h"

// TODO Don't do this
#include "Externals/dear_imgui/imgui.h"
#include "Externals/dear_imgui/imgui_internal.h"

namespace Falcor
{
    void RenderGraphUI::renderUI(Gui* pGui)
    {
        uint32_t nodeIndex = 0;
        uint32_t pinIndex = 0;
        uint32_t specialPinIndex = static_cast<uint32_t>(static_cast<uint16_t>(-1)); // used for avoiding conflicts with creating additional unique pins.
        uint32_t linkIndex = 0;
        uint32_t mouseDragBezierPin = 0;

        constexpr uint32_t bezierColor = 0xFFFFFFFF;
        static bool draggingConnection = false;
        static unsigned draggingPinIndex = 0;

        const ImVec2& mousePos = ImGui::GetCurrentContext()->IO.MousePos;

        // handle deleting nodes and connections?
        std::unordered_set<uint32_t> nodeIDsToDelete;
        std::vector<std::string> nodeNamesToDelete;

        ax::NodeEditor::BeginDelete();

        ax::NodeEditor::NodeId deletedNodeId = 0;
        while (ax::NodeEditor::QueryDeletedNode(&deletedNodeId))
        {
            nodeIDsToDelete.insert(static_cast<uint32_t>(deletedNodeId.Get()));
        }

        for (const auto& currentPassData : mRenderGraphRef.mpGraphmpPasses)
        {
            // only worry about the GUI for the node if no deletion
            if (nodeIDsToDelete.find(nodeIndex) != nodeIDsToDelete.end())
            {
                nodeNamesToDelete.push_back(mRenderPassUI[currentPass.get()]);
            }

            ax::NodeEditor::BeginNode(nodeIndex);

            // display name for the render pass
            pGui->addText(mRenderPassUI[currentPass.get()].c_str());

            const auto& nameToIndexMap = mRenderPassUI[currentPass.get()];

            // for attempting to create a new edge in the editor. Name of the first node
            static std::string firstConnectionName;

            for (const auto& nameToIndexIt : nameToIndexMap)
            {
                // Connect the graph nodes for each of the edges
                // need to iterate in here in order to use the right indices
                bool isInput = nameToIndexIt.second.second;

                ax::NodeEditor::PushStyleVar(ax::NodeEditor::StyleVar::StyleVar_SourceDirection, ImVec2(-1, 0)); // input
                ax::NodeEditor::PushStyleVar(ax::NodeEditor::StyleVar::StyleVar_TargetDirection, ImVec2(1, 0));  // output

                ax::NodeEditor::BeginPin(nameToIndexIt.second.first, static_cast<ax::NodeEditor::PinKind>(isInput));

                // draw the pin output for input / output
                if (!isInput)
                {
                    pGui->addText(nameToIndexIt.first.c_str());
                    ImGui::SameLine();
                }

                ImGui::InvisibleButton(nameToIndexIt.first.c_str(), { 6.0f, 6.0f });

                const ImVec2& nodeSize = ax::NodeEditor::GetNodeSize(nodeIndex);
                ImVec2 buttonAlignment = (ImVec2(nodeSize.x - nodeSize.x / 8.0f, ImGui::GetCursorScreenPos().y));
                auto lastItemRect = ImGui::GetCurrentWindow()->DC.LastItemRect; // might make this a helper function
                auto pWindowDrawList = ImGui::GetCurrentWindow()->DrawList;

                // if (ax::NodeEditor::GetSe)

                //ImGui::PushStyleColor(0, { 255, 255, 255, 255});
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenBlockedByActiveItem)) {
                    pWindowDrawList->AddCircleFilled(lastItemRect.GetCenter(), 6.0f, bezierColor);

                    if (draggingConnection && draggingPinIndex != pinIndex)
                    {
                        //attempt to add connection
                        addEdge(firstConnectionName, mPassToName[currentPass.get()] + "." + nameToIndexIt.first);
                        draggingConnection = false;
                    }

                    if (ImGui::IsItemClicked())
                    {
                        draggingConnection = true;
                        draggingPinIndex = pinIndex;
                    }
                }
                else
                {
                    pWindowDrawList->AddCircle(lastItemRect.GetCenter(), 6.0f, bezierColor);
                }

                if (!ImGui::IsMouseDown(0))
                {
                    draggingConnection = false;
                }

                ax::NodeEditor::PinRect(ImVec2(lastItemRect.GetCenter().x - 1.0f, lastItemRect.GetCenter().y + 1.0f),
                    ImVec2(lastItemRect.GetCenter().x + 1.0f, lastItemRect.GetCenter().y - 1.0f));

                if (isInput)
                {
                    ImGui::SameLine();
                    pGui->addText(nameToIndexIt.first.c_str());
                }

                ax::NodeEditor::EndPin();

                if ((draggingPinIndex == pinIndex) && draggingConnection)
                {
                    firstConnectionName = mPassToName[currentPass.get()] + "." + nameToIndexIt.first;
                    mouseDragBezierPin = nameToIndexIt.second.first;
                    // use the same bezier with picking by creating another link
                }

                pinIndex++;
            }

            mNodeProperties[currentPass.get()][0].renderUI(pGui);
            mNodeProperties[currentPass.get()][1].renderUI(pGui);

            currentPass->renderUI(pGui, mPassToName[currentPass.get()]);

            ax::NodeEditor::EndNode();
            nodeIndex++;
        }

        ax::NodeEditor::EndDelete();

        // draw connections
        for (const auto& nameIndexPair : mNameToIndex)
        {
            for (const auto& renderGraphEdge : mEdges)
            {
                ax::NodeEditor::Link(linkIndex++, mDisplayMap[renderGraphEdge.pSrc][renderGraphEdge.srcField].first,
                    mDisplayMap[renderGraphEdge.pDst][renderGraphEdge.dstField].first);
            }
        }

        // add output of the graph as the last output

        if (mouseDragBezierPin)
        {
            ax::NodeEditor::LinkToPos(linkIndex++, mouseDragBezierPin, ImGui::GetCurrentContext()->IO.MousePos);
        }


        // get rid of nodes for next frame
        for (const std::string& passName : nodeNamesToDelete)
        {
            mRenderGraphRef.removeRenderPass(passName);
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

        
        // Fill out data for displaying GUI of render graph
        auto& nameToIndexMap = mDisplayMap[newEdge.pSrc];
        auto& nameToIndexIt = nameToIndexMap.find(newEdge.srcField);
        if (nameToIndexIt == nameToIndexMap.end())
        {
            nameToIndexMap.insert(std::make_pair(newEdge.srcField, std::make_pair(mDisplayPinIndex++, false)));
        }

        auto& nameToIndexMapDst = mDisplayMap[newEdge.pDst];
        nameToIndexIt = nameToIndexMapDst.find(newEdge.dstField);
        if (nameToIndexIt == nameToIndexMapDst.end())
        {
            nameToIndexMapDst.insert(std::make_pair(newEdge.dstField, std::make_pair(mDisplayPinIndex++, true)));
        }
    }

    // void RenderGraphUI::setEdgeViewport(const std::string& input, const std::string& output, const glm::vec3& viewportBounds)
    // {
    //     RenderPass::PassData::Field overrideData{};
    //     overrideData.width  = static_cast<uint32_t>(viewportBounds.x);
    //     overrideData.height = static_cast<uint32_t>(viewportBounds.y);
    //     overrideData.depth  = static_cast<uint32_t>(viewportBounds.z);
    // 
    //     str_pair fieldNamePair;
    //     parseFieldName(input, fieldNamePair);
    //     overrideData.name = fieldNamePair.second;
    // 
    //     overrideData.format = ResourceFormat::Unknown;
    //     overrideData.sampleCount = 1;
    //     overrideData.bindFlags = Texture::BindFlags::RenderTarget;
    // }

    void RenderGraphUI::addFieldDisplayData(RenderPass* pRenderPass, const std::string& displayName, bool isInput)
    {
        assert(pRenderPass);
        mDisplayMap[pRenderPass].insert(std::make_pair(displayName, std::make_pair(mDisplayPinIndex++, isInput)));
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
                RenderPass::SharedPtr pFieldPass = mpPasses[mNameToIndex[std::string(field.FindMember("RenderPassName")->value.GetString())]];
                std::string fieldNameString(field.FindMember("FieldName")->value.GetString());
                addFieldDisplayData(pFieldPass.get(), fieldNameString, isInput);
            }
            isInput = true;
        }
    
        // TODO - need to mark output from the file as well but need way of mapping to resources or resource handles
    }
    
    void RenderGraphUI::serializeJson(rapidjson::Writer<rapidjson::OStreamWrapper>* writer) const
    {
    #define string_2_json(_string) \
            rapidjson::StringRef(_string.c_str())
    
        // write out the nodes and node data
        writer->String("RenderPassNodes");
        writer->StartArray();
    
        for (auto& nameIndexPair : mRenderGraphRef.mNameToIndex)
        {
            writer->StartObject();
    
            writer->String("RenderPassName");
            writer->String(nameIndexPair.first.c_str());
            writer->String("RenderPassType");
            writer->String(mpPasses[nameIndexPair.second]->getTypeName().c_str());
    
            // serialize custom data here
    
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
    
            // maybe make another structure for free pins??? (Avoids duplicates like now, but will have additional memory (maybe slot map??))
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
    
        for (auto& edge : mEdges)
        {
            writer->StartObject();
    
            writer->String("SrcRenderPassName");
            writer->String(mPassToName[edge.pSrc].c_str());
            writer->String("DstRenderPassName");
            writer->String(mPassToName[edge.pDst].c_str());
    
            writer->String("SrcField");
            writer->String(edge.srcField.c_str());
            writer->String("DstField");
            writer->String(edge.dstField.c_str());
            writer->EndObject();
        }
    
        writer->EndArray();
    
    #undef string_2_json
    }
}