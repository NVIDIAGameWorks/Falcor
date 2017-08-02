/***************************************************************************
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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
#include "Falcor.h"
#include <string>
#include <map>
#include <vector>
#include <set>
#include <random>
#include <algorithm> // std::random_shuffle
#include <ctime>     // std::time
#include <cstdlib>   // std::rand, std::srand

namespace Falcor
{
    namespace CommonShaderDescs
    {
        //  Data.
        struct CommonResourceDesc;
        struct ArrayDesc;
        struct ResourceAttachmentDesc;
        struct ShaderStructDesc;
        struct HLSLSemanticDesc;
        struct GLSLSemanticDesc;

        //  Access Type.
        enum class AccessType : uint32_t
        {
            Read = 0u,
            ReadWrite = 1u
        };

        //  Shader Array Desc.
        struct ArrayDesc
        {
            //  Whether or not it is an Array.
            bool isArray = false;

            //  The Dimensions of the Array - this is empty when the array is unbounded. Also, setting a dimension to 0 will cause that dimension to be unbounded.
            std::vector<uint32_t> dimensions = {};
        };

        //  Register Desc.
        struct ResourceAttachmentDesc
        {
            //
            std::string registerType = "t";

            //  Attachment Subpoint.
            uint32_t attachmentSubpoint = 0;

            //  Attachment point.
            uint32_t attachmentPoint = 0;

            //  Attachment Sub-point (Binding / Index)
            bool isAttachmentSubpointExplicit = false;

            //  Attachment Point (Set / Space)
            bool isAttachmentPointExplicit = false;
        };

        //  Shader Resource Desc.
        struct CommonResourceDesc
        {
            CommonResourceDesc(const std::string &newResourceVariable)
            {
                resourceVariable = newResourceVariable;
            }

            //
            std::string resourceVariable;

            //  Access Type.
            AccessType accessType = AccessType::Read;

            //  Resource Register Desc.
            ArrayDesc arrayDesc;

            //  Resource Register Desc.
            ResourceAttachmentDesc attachmentDesc;
        };


        //  Return the lines joined together.
        std::string getCodeBlock(const std::vector<std::string> &lines);

    };
};
