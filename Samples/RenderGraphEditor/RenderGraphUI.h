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

#include "Utils/GuiProperty.h"
#include "Graphics/RenderGraph/RenderGraph.h"
#include "Graphics/RenderGraph/RenderPass.h"

#include <array>

// need for document passed in. may move entire file serialization to serialize json
#include "Externals/RapidJson/include/rapidjson/rapidjson.h"
#include "Externals/RapidJson/include/rapidjson/writer.h"
#include "Externals/RapidJson/include/rapidjson/ostreamwrapper.h"
#include "Externals/RapidJson/include/rapidjson/document.h"

namespace Falcor
{
    class RenderPassUI
    {
    public:

        struct PinUIData
        {
            std::string mFieldName; // might be temp
            uint32_t mGuiPinID;
            bool mIsInput;
        };

        void renderUI(Gui *pGui);

        friend class RenderGraphUI;

    private:

        std::vector<PinUIData> mInputPins;
        std::vector<PinUIData> mOutputPins;
        uint32_t mGuiNodeID;

    };

    class RenderGraphUI
    {
    public:

        RenderGraphUI(RenderGraph& renderGraphRef) : mRenderGraphRef(renderGraphRef) {}

        /** Display enter graph in GUI.
        */
        void renderUI(Gui *pGui);

        /** Serialization function. Serialize full graph into json file.
        */
        void serializeJson(rapidjson::Writer<rapidjson::OStreamWrapper>* document) const;

        /** De-serialize function for deserializing graph and building data for GUI viewing
        */
        void deserializeJson(const rapidjson::Document& reader);

        void addFieldDisplayData(RenderPass* pRenderPass, const std::string& displayName, bool isInput);

        /** Set bounds for the inputs and receiving outputs of a given edge within the graph
        */
        // void setEdgeViewport(const std::string& input, const std::string& output, const glm::vec3& viewportBounds);


        void addFieldDisplayData(RenderPass* pRenderPass, const std::string& displayName, bool isInput);

        /**  Add a new display node for the graph representing a render pass 
          */
        void addRenderPassNode();

    private:
        
        // TODO -- remove this for now
        // std::unordered_map<std::string, RenderPass::PassData::Field> mOverridePassDatas;

        // map the pointers to their names to get destination name for editor.
        // std::unordered_map<RenderPass*, std::string> mPassToName;


        // start with reference of render graph
        RenderGraph& mRenderGraphRef;

        // Display data for node editor
        uint32_t mDisplayPinIndex = 0;

        std::unordered_map<std::string, RenderPassUI> mRenderPassUI;

        std::unordered_map<RenderPass*, StringProperty[2] > mNodeProperties; // ideally more generic as nodes gain more data
    };
}
