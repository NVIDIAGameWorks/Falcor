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
#pragma once

#define _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
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

        // pin type enum?

        struct PinUIData
        {
            uint32_t mGuiPinID;
            bool mIsInput;
        };

        void addUIPin(const std::string& fieldName, uint32_t guiPinID, bool isInput);

        void renderUI(Gui *pGui);

        friend class RenderGraphUI;

    private:

        std::unordered_map<std::string, PinUIData> mPins; // should this be a map? this probably can be a vec
        uint32_t mGuiNodeID;

    };

    class RenderGraphUI
    {
    public:

        RenderGraphUI(RenderGraph& renderGraphRef) : mRenderGraphRef(renderGraphRef) {}

        /** Display enter graph in GUI.
        */
        void renderUI(Gui *pGui);
        
        /**  Add a new display node for the graph representing a render pass 
          */
        void addRenderPassNode();

        // TODO -- move these out of the UI code

        /** Serialization function. Serialize full graph into json file.
        */
        void serializeJson(rapidjson::Writer<rapidjson::OStreamWrapper>* document) const;

        /** De-serialize function for deserializing graph and building data for GUI viewing
        */
        void deserializeJson(const rapidjson::Document& reader);

    private:

        /** Updates structure for drawing the gui graph
        */
        void updateDisplayData();

        // start with reference of render graph
        RenderGraph& mRenderGraphRef;

        std::unordered_map <std::string, RenderPassUI> mRenderPassUI;

        // maps output pin name to input pin ids
        std::unordered_map <std::string, std::vector<uint32_t> > mOutputToInputPins;

        uint32_t mDisplayPinIndex = 0;
    };
}
