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
#include "Graphics/Model/Model.h"

namespace Falcor
{
    class RtModel : public Model, public inherit_shared_from_this<Model, RtModel>
    {
    public:
        using SharedPtr = std::shared_ptr<RtModel>;
        using SharedConstPtr = std::shared_ptr<const RtModel>;

        static RtModel::SharedPtr createFromFile(const char* filename, RtBuildFlags buildFlags = RtBuildFlags::None, Model::LoadFlags flags = Model::LoadFlags::None);
        static RtModel::SharedPtr createFromModel(const Model& model, RtBuildFlags buildFlags = RtBuildFlags::None);
        RtBuildFlags getBuildFlags() const { return mBuildFlags; }

        struct BottomLevelData
        {
            uint32_t meshBaseIndex = 0;
            uint32_t meshCount = 0;
            bool isStatic = true;
#ifdef FALCOR_VK
            AccelerationStructureHandle pBlas;
#else
            Buffer::SharedPtr pBlas; 
#endif
        };

        uint32_t getBottomLevelDataCount() const { return (uint32_t)mBottomLevelData.size(); }
        const BottomLevelData& getBottomLevelData(uint32_t index) const { return mBottomLevelData[index]; }

    protected:
        RtModel(const Model& model, RtBuildFlags buildFlags);
        bool update() override;            // Override update() from Model, which updates vertices for skinned models
        void buildAccelerationStructure();

        std::vector<BottomLevelData> mBottomLevelData;
        RtBuildFlags mBuildFlags;
        void createBottomLevelData();
    };
}
