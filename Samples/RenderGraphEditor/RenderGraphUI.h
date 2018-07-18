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
#include "Graphics/RenderGraph/RenderGraph.h"
#include "Graphics/RenderGraph/RenderPass.h"
#include "Graphics/RenderGraph/RenderPassReflection.h"

#include <array>

namespace Falcor
{
    class RenderPassUI
    {
    public:
        // pin type enum?

        struct PinUIData
        {
            std::string mPinName;
            uint32_t mGuiPinID;
            bool mIsInput;
            std::string mConnectedPinName;
            std::string mConnectedNodeName;
            bool mIsGraphOutput;
        };

        void addUIPin(const std::string& fieldName, uint32_t guiPinID, bool isInput, const std::string& connectedPinName = "", const std::string& connectedNodeName = "", bool isGraphOutput = false);

        void renderUI(Gui *pGui);

        friend class RenderGraphUI;

    private:

        std::vector<PinUIData> mInputPins;
        std::unordered_map<std::string, uint32_t> mNameToIndexInput;

        std::vector<PinUIData> mOutputPins;
        std::unordered_map<std::string, uint32_t> mNameToIndexOutput;

        uint32_t mGuiNodeID;
        RenderPassReflection mReflection;
    };

    class RenderGraphUI
    {
    public:

        RenderGraphUI(RenderGraph& renderGraphRef);

        /** Display enter graph in GUI.
        */
        void renderUI(Gui *pGui);

        void reset();

        /** static functions used for GUI callbacks required to be static
         */
        static bool addLink(const std::string& srcPass, const std::string& dstPass, const std::string& srcField, const std::string& dstField);

        static void removeLink();

        static void removeRenderPass(const std::string& name);

        static bool sRebuildDisplayData;

    private:

        /** Updates structure for drawing the GUI graph
        */
        void updateDisplayData();
        
        void drawPins(bool addLinks = true);

        /** Helper function to calculate position of the next node in execution order
         */
        glm::vec2 getNextNodePosition(uint32_t nodeID);

        glm::vec2 mNewNodeStartPosition{ -40.0f, 100.0f };

        // start with reference of render graph
        RenderGraph& mRenderGraphRef;

        std::unordered_set<std::string> mAllNodeTypeStrings;
        std::vector<const char*> mAllNodeTypes;

        std::unordered_map <std::string, RenderPassUI> mRenderPassUI;

        std::unordered_map <std::string, uint32_t> mInputPinStringToLinkID;

        // maps output pin name to input pin ids. Pair first is pin id, second is node id
        std::unordered_map <std::string, std::vector< std::pair<uint32_t, uint32_t > > > mOutputToInputPins;
    };
}
