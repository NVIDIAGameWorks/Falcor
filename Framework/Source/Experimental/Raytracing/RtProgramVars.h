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
#include "RtProgram/RtProgram.h"
#include "RtScene.h"
#include "API/Buffer.h"
#include "Graphics/Program/ProgramVars.h"

namespace Falcor
{
    class RenderContext;
    class RtStateObject;
    class RtVarsCmdList;
    class RtVarsContext;

    class RtProgramVars : public std::enable_shared_from_this<RtProgramVars>
    {
    public:
        using SharedPtr = std::shared_ptr<RtProgramVars>;
        using SharedConstPtr = std::shared_ptr<const RtProgramVars>;

        using VarsVector = std::vector<GraphicsVars::SharedPtr>;

        static SharedPtr create(RtProgram::SharedPtr pProgram, RtScene::SharedPtr pScene);

        VarsVector& getHitVars(uint32_t rayID) { return mHitVars[rayID]; }
        const GraphicsVars::SharedPtr& getRayGenVars() { return mRayGenVars; }
        const GraphicsVars::SharedPtr& getMissVars(uint32_t rayID) { return mMissVars[rayID]; }
        const RaytracingVars::SharedPtr& getGlobalVars() { return mpGlobalVars; }

        bool apply(RenderContext* pCtx, RtStateObject* pRtso);

        Buffer::SharedPtr getShaderTable() const { return mpShaderTable; }
        uint32_t getRecordSize() const { return mRecordSize; }
        uint32_t getRayGenRecordIndex() const { return kRayGenRecordIndex; }
        uint32_t getFirstMissRecordIndex() const { return kFirstMissRecordIndex; }
        uint32_t getFirstHitRecordIndex() const { return mFirstHitVarEntry; }
        uint32_t getHitProgramsCount() const { return mHitProgCount; }
        uint32_t getMissProgramsCount() const { return mMissProgCount; }
        uint32_t getHitRecordsCount() const { return mHitRecordCount; }

        static uint32_t getProgramIdentifierSize();
        
    private:
        static const uint32_t kRayGenRecordIndex = 0;
        static const uint32_t kFirstMissRecordIndex = 1;
        uint32_t mMissProgCount = 0;
        uint32_t mHitProgCount = 0;
        uint32_t mHitRecordCount = 0;       ///< Total number of hit records in shader table
        uint32_t mFirstHitVarEntry = 0;

        RtProgramVars(RtProgram::SharedPtr pProgram, RtScene::SharedPtr pScene);
        RtProgram::SharedPtr mpProgram;
        RtScene::SharedPtr mpScene;
        uint32_t mProgramIdentifierSize;
        uint32_t mRecordSize;
        Buffer::SharedPtr mpShaderTable;

        uint8_t* getRayGenRecordPtr();
        uint8_t* getMissRecordPtr(uint32_t missId);
        uint8_t* getHitRecordPtr(uint32_t hitId, uint32_t meshId);

        bool init();
        bool applyRtProgramVars(uint8_t* pRecord, const RtProgramVersion* pProgVersion, const RtStateObject* pRtso, ProgramVars* pVars, RtVarsContext* pContext);

        RaytracingVars::SharedPtr mpGlobalVars;
        GraphicsVars::SharedPtr mRayGenVars;
        std::vector<VarsVector> mHitVars;
        std::vector<uint8_t> mShaderTableData;
        VarsVector mMissVars;
        std::shared_ptr<RtVarsContext> mpRtVarsHelper;
    };

    class RtVarsContext : public CopyContext, public inherit_shared_from_this<CopyContext, RtVarsContext>
    {
    public:
        using SharedPtr = std::shared_ptr<RtVarsContext>;
        ~RtVarsContext();

        static SharedPtr create();

        const LowLevelContextData::SharedPtr& getLowLevelData() const override { return mpLowLevelData; }
        void resourceBarrier(const Resource* pResource, Resource::State newState, const ResourceViewInfo* pViewInfo = nullptr) override;
        std::shared_ptr<RtVarsCmdList> getRtVarsCmdList() const { return mpList; }
    private:
        RtVarsContext();
        void apiInit();
        std::shared_ptr<RtVarsCmdList> mpList;
    };
}
