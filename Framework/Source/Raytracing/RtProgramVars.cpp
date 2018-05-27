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
#include "RtProgramVars.h"
#include "API/Device.h"
#include "RtStateObject.h"

namespace Falcor
{
    static bool checkParams(RtProgram::SharedPtr pProgram, RtScene::SharedPtr pScene)
    {
        if (pScene == nullptr)
        {
            logError("RtProgramVars must have a scene attached to it");
            return false;
        }

        if (pProgram->getRayGenProgram() == nullptr)
        {
            logError("RtProgramVars must have a ray-gen program attached to it");
            return false;
        }
        return true;
    }
    
    RtProgramVars::RtProgramVars(RtProgram::SharedPtr pProgram, RtScene::SharedPtr pScene) : mpProgram(pProgram), mpScene(pScene)
    {
        mpRtVarsHelper = RtVarsContext::create(gpDevice->getRenderContext());
    }

    RtProgramVars::SharedPtr RtProgramVars::create(RtProgram::SharedPtr pProgram, RtScene::SharedPtr pScene)
    {
        SharedPtr pVars = SharedPtr(new RtProgramVars(pProgram, pScene));
        if ((checkParams(pProgram, pScene) == false) || (pVars->init() == false))
        {
            return nullptr;
        }
        return pVars;
    }

    template<typename ProgType>
    void createVars(ProgType pProg, GraphicsVars::SharedPtr pVars[], uint32_t varCount)
    {
        for(uint32_t i = 0 ; i < varCount ; i++)
        {
            pVars[i] = GraphicsVars::create(pProg->getLocalReflector(), true);
        }
    }

    bool RtProgramVars::init()
    {
        // Find the max root-signature size and create the programVars
        uint32_t maxRootSigSize = 0;
        createVars(mpProgram->getRayGenProgram(), &mRayGenVars, 1);

        mHitProgCount = mpProgram->getHitProgramCount();
        mMissProgCount = mpProgram->getMissProgramCount();
        mFirstHitVarEntry = kFirstMissRecordIndex + mMissProgCount;
        mMissVars.resize(mMissProgCount);
        mHitVars.resize(mHitProgCount);
        uint32_t recordCountPerHit = mpScene->getGeometryCount(mHitProgCount);

        for (uint32_t i = 0 ; i < mHitProgCount; i++)
        {
            if(mpProgram->getHitProgram(i))
            {
                mHitVars[i].resize(recordCountPerHit);
                createVars(mpProgram->getHitProgram(i), mHitVars[i].data(), recordCountPerHit);
            }
        }

        for (uint32_t i = 0; i < mMissProgCount; i++)
        {
            if(mpProgram->getMissProgram(i))
            {
                createVars(mpProgram->getMissProgram(i), &mMissVars[i], 1);
            }
        }
        
        // Get the program identifier size
        ID3D12DeviceRaytracingPrototypePtr pRtDevice = gpDevice->getApiHandle();
        mProgramIdentifierSize = pRtDevice->GetShaderIdentifierSize();

        // Create the global variables
        mpGlobalVars = GraphicsVars::create(mpProgram->getGlobalReflector(), true);

        return true;
    }

    void RtProgramVars::updateShaderTableRecordSize(int newSize)
    {
        newSize = align_to(D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT, newSize);
        if (newSize != mRecordSize)
        {
            assert(newSize != 0);

            mRecordSize = newSize;
            uint32_t recordCountPerHit = mpScene->getGeometryCount(mHitProgCount);
            
            // Create the shader-table buffer
            uint32_t hitEntries = recordCountPerHit * mHitProgCount;
            uint32_t numEntries = mMissProgCount + hitEntries + 1; // 1 is for the ray-gen
            
            // Create the buffer and allocate the temporary storage
            mpShaderTable = Buffer::create(numEntries * mRecordSize, Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None);
            assert(mpShaderTable);
            mShaderTableData.resize(mpShaderTable->getSize());
        }
    }

    // We are using the following layout for the shader-table:
    //
    // +------------+---------+---------+-----+--------+---------+--------+-----+--------+--------+-----+--------+-----+--------+--------+-----+--------+
    // |            |         |         | ... |        |         |        | ... |        |        | ... |        | ... |        |        | ... |        |
    // |   RayGen   |   Ray0  |   Ray1  | ... |  RayN  |   Ray0  |  Ray1  | ... |  RayN  |  Ray0  | ... |  RayN  | ... |  Ray0  |  Ray0  | ... |  RayN  |   
    // |   Entry    |   Miss  |   Miss  | ... |  Miss  |   Hit   |   Hit  | ... |  Hit   |  Hit   | ... |  Hit   | ... |  Hit   |  Hit   | ... |  Hit   |
    // |            |         |         | ... |        |  Mesh0  |  Mesh0 | ... |  Mesh0 |  Mesh1 | ... |  Mesh1 | ... | MeshN  |  MeshN | ... |  MeshN |
    // +------------+---------+---------+-----+--------+---------+--------+-----+--------+--------+-----+--------+-----+--------+--------+-----+--------+
    //
    // The first record is the ray gen, followed by the miss records, followed by the meshes records.
    // For each mesh we have N hit records, N == number of mesh instances in the model
    // The size of each record is mRecordSize
    // 
    // If this layout changes, we also need to change the constants kRayGenRecordIndex and kFirstMissRecordIndex

    uint8_t* RtProgramVars::getRayGenRecordPtr()
    {
        return mShaderTableData.data() + (kRayGenRecordIndex * mRecordSize);
    }

    uint8_t* RtProgramVars::getMissRecordPtr(uint32_t missId)
    {
        assert(missId < mMissProgCount);
        uint32_t offset = mRecordSize * (kFirstMissRecordIndex + missId);
        return mShaderTableData.data() + offset;
    }

    uint8_t* RtProgramVars::getHitRecordPtr(uint32_t hitId, uint32_t meshId)
    {   
        assert(hitId < mHitProgCount);
        uint32_t meshIndex = mFirstHitVarEntry + mHitProgCount * meshId;    // base record of the requested mesh
        uint32_t recordIndex = meshIndex + hitId;
        return mShaderTableData.data() + (recordIndex * mRecordSize);
    }

    bool applyRtProgramVars(uint8_t* pRecord, const RtProgramKernels* pProgVersion, const RtStateObject* pRtso, uint32_t progIdSize, ProgramVars* pVars, RtVarsContext* pContext)
    {
        MAKE_SMART_COM_PTR(ID3D12StateObjectPropertiesPrototype);
        ID3D12StateObjectPropertiesPrototypePtr pRtsoPtr = pRtso->getApiHandle();
        memcpy(pRecord, pRtsoPtr->GetShaderIdentifier(pProgVersion->getExportName().c_str()), progIdSize);
        pRecord += progIdSize;
        pContext->getRtVarsCmdList()->setRootParams(pProgVersion->getLocalRootSignature(), pRecord);
        return pVars->applyProgramVarsCommon<true>(pContext, true, pProgVersion->getLocalRootSignature().get());
    }

    bool RtProgramVars::apply(RenderContext* pCtx, RtStateObject* pRtso)
    {
        // Find max root signature size
        uint32_t sigSize = 0;
        for (auto prog : pRtso->getProgramList())
            sigSize = std::max(sigSize, prog->getLocalRootSignature()->getSizeInBytes());
        updateShaderTableRecordSize(mProgramIdentifierSize + sigSize);

        // We always have a ray-gen program, apply it first
        uint8_t* pRayGenRecord = getRayGenRecordPtr();
        applyRtProgramVars(pRayGenRecord, pRtso->getProgramList()[0].get(), pRtso, mProgramIdentifierSize, getRayGenVars().get(), mpRtVarsHelper.get());

        // Loop over the rays
        uint32_t hitCount = mpProgram->getHitProgramCount();
        uint32_t pi = 1;
        for (uint32_t h = 0; h < hitCount; h++)
        {
            if(mpProgram->getHitProgram(h))
            {
                for (uint32_t i = 0; i < mpScene->getGeometryCount(hitCount); i++)
                {
                    uint8_t* pHitRecord = getHitRecordPtr(h, i);
                    if (!applyRtProgramVars(pHitRecord, pRtso->getProgramList()[pi].get(), pRtso,
                        mProgramIdentifierSize, getHitVars(h)[i].get(), mpRtVarsHelper.get()))
                    {
                        return false;
                    }
                    pi++;
                }
            }
        }

        for (uint32_t m = 0; m < mpProgram->getMissProgramCount(); m++)
        {
            if(mpProgram->getMissProgram(m))
            {
                uint8_t* pMissRecord = getMissRecordPtr(m);
                if (!applyRtProgramVars(pMissRecord, pRtso->getProgramList()[pi].get(), pRtso, mProgramIdentifierSize, getMissVars(m).get(), mpRtVarsHelper.get()))
                {
                    return false;
                }
                pi++;
            }
        }

        mpGlobalVars->applyProgramVarsCommon<false>(pCtx, true, pRtso->getGlobalRootSignature().get());
        pCtx->updateBuffer(mpShaderTable.get(), mShaderTableData.data());
        return true;
    }
}
