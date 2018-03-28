/************************************************************************************************************************************\
|*                                                                                                                                    *|
|*     Copyright © 2017 NVIDIA Corporation.  All rights reserved.                                                                     *|
|*                                                                                                                                    *|
|*  NOTICE TO USER:                                                                                                                   *|
|*                                                                                                                                    *|
|*  This software is subject to NVIDIA ownership rights under U.S. and international Copyright laws.                                  *|
|*                                                                                                                                    *|
|*  This software and the information contained herein are PROPRIETARY and CONFIDENTIAL to NVIDIA                                     *|
|*  and are being provided solely under the terms and conditions of an NVIDIA software license agreement                              *|
|*  and / or non-disclosure agreement.  Otherwise, you have no rights to use or access this software in any manner.                   *|
|*                                                                                                                                    *|
|*  If not covered by the applicable NVIDIA software license agreement:                                                               *|
|*  NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOFTWARE FOR ANY PURPOSE.                                            *|
|*  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.                                                           *|
|*  NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,                                                                     *|
|*  INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.                       *|
|*  IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES,                               *|
|*  OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT,                         *|
|*  NEGLIGENCE OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOURCE CODE.            *|
|*                                                                                                                                    *|
|*  U.S. Government End Users.                                                                                                        *|
|*  This software is a "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT 1995),                                       *|
|*  consisting  of "commercial computer  software"  and "commercial computer software documentation"                                  *|
|*  as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government only as a commercial end item.     *|
|*  Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995),                                          *|
|*  all U.S. Government End Users acquire the software with only those rights set forth herein.                                       *|
|*                                                                                                                                    *|
|*  Any use of this software in individual and commercial software must include,                                                      *|
|*  in the user documentation and internal comments to the code,                                                                      *|
|*  the above Disclaimer (as applicable) and U.S. Government End Users Notice.                                                        *|
|*                                                                                                                                    *|
 \************************************************************************************************************************************/
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
    void getSigSizeAndCreateVars(ProgType pProg, uint32_t& maxRootSigSize, GraphicsVars::SharedPtr pVars[], uint32_t varCount)
    {
        RtProgramVersion::SharedConstPtr pVersion = pProg->getActiveVersion();
        maxRootSigSize = max(pVersion->getRootSignature()->getSizeInBytes(), maxRootSigSize);
        for(uint32_t i = 0 ; i < varCount ; i++)
        {
            pVars[i] = GraphicsVars::create(pVersion->getReflector(), true, pVersion->getRootSignature());
        }
    }

    bool RtProgramVars::init()
    {
        // Find the max root-signature size and create the programVars
        uint32_t maxRootSigSize = 0;
        getSigSizeAndCreateVars(mpProgram->getRayGenProgram(), maxRootSigSize, &mRayGenVars, 1);

        mHitProgCount = mpProgram->getHitProgramCount();
        mMissProgCount = mpProgram->getMissProgramCount();
        mFirstHitVarEntry = kFirstMissSbtRecordIndex + mMissProgCount;
        mMissVars.resize(mMissProgCount);
        mHitVars.resize(mHitProgCount);
        uint32_t recordCountPerHit = mpScene->getGeometryCount(mHitProgCount);

        for (uint32_t i = 0 ; i < mHitProgCount; i++)
        {
            mHitVars[i].resize(recordCountPerHit);
            getSigSizeAndCreateVars(mpProgram->getHitProgram(i), maxRootSigSize, mHitVars[i].data(), recordCountPerHit);
        }

        for (uint32_t i = 0; i < mMissProgCount; i++)
        {
            getSigSizeAndCreateVars(mpProgram->getMissProgram(i), maxRootSigSize, &mMissVars[i], 1);
        }


        // Get the program identifier size
        ID3D12DeviceRaytracingPrototypePtr pRtDevice = gpDevice->getApiHandle();
        mProgramIdentifierSize = pRtDevice->GetShaderIdentifierSize();

        // Create the SBT buffer
        uint32_t hitEntries = recordCountPerHit * mHitProgCount;
        uint32_t numEntries = mMissProgCount + hitEntries + 1; // 1 is for the ray-gen

        // Calculate the record size
        mRecordSize = mProgramIdentifierSize + maxRootSigSize;
        mRecordSize = align_to(D3D12_RAYTRACING_SHADER_RECORD_BYTE_ALIGNMENT, mRecordSize);
        assert(mRecordSize != 0);

        // Create the buffer and allocate the temporary storage
        mpSBT = Buffer::create(numEntries * mRecordSize, Resource::BindFlags::ShaderResource, Buffer::CpuAccess::None);
        assert(mpSBT);
        mSbtData.resize(mpSBT->getSize());

        return true;
    }

    // We are using the following layout for the SBT:
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
    // If this layout changes, we also need to change the constants kRayGenSbtRecordIndex and kFirstMissSbtRecordIndex

    uint8_t* RtProgramVars::getRayGenRecordPtr()
    {
        return mSbtData.data() + (kRayGenSbtRecordIndex * mRecordSize);
    }

    uint8_t* RtProgramVars::getMissRecordPtr(uint32_t missId)
    {
        assert(missId < mMissProgCount);
        uint32_t offset = mRecordSize * (kFirstMissSbtRecordIndex + missId);
        return mSbtData.data() + offset;
    }

    uint8_t* RtProgramVars::getHitRecordPtr(uint32_t hitId, uint32_t meshId)
    {   
        assert(hitId < mHitProgCount);
        uint32_t meshIndex = mFirstHitVarEntry + mHitProgCount * meshId;    // base record of the requested mesh
        uint32_t recordIndex = meshIndex + hitId;
        return mSbtData.data() + (recordIndex * mRecordSize);
    }

    bool applyRtProgramVars(uint8_t* pRecord, const RtProgramVersion* pProgVersion, const RtStateObject* pRtso, uint32_t progIdSize, ProgramVars* pVars, RtVarsContext* pContext)
    {
        MAKE_SMART_COM_PTR(ID3D12StateObjectPropertiesPrototype);
        ID3D12StateObjectPropertiesPrototypePtr pRtsoPtr = pRtso->getApiHandle();
        memcpy(pRecord, pRtsoPtr->GetShaderIdentifier(pProgVersion->getExportName().c_str()), progIdSize);
        pRecord += progIdSize;
        pContext->getRtVarsCmdList()->setRootParams(pProgVersion->getRootSignature(), pRecord);
        return pVars->applyProgramVarsCommon<true>(pContext, true);
    }

    bool RtProgramVars::apply(RenderContext* pCtx, RtStateObject* pRtso)
    {
        // We always have a ray-gen program, apply it first
        uint8_t* pRayGenRecord = getRayGenRecordPtr();
        applyRtProgramVars(pRayGenRecord, mpProgram->getRayGenProgram()->getActiveVersion().get(), pRtso, mProgramIdentifierSize, getRayGenVars().get(), mpRtVarsHelper.get());

        // Loop over the rays
        uint32_t hitCount = mpProgram->getHitProgramCount();
        for (uint32_t h = 0; h < hitCount; h++)
        {
            for (uint32_t i = 0; i < mpScene->getGeometryCount(hitCount); i++)
            {
                uint8_t* pHitRecord = getHitRecordPtr(h, i);
                if(!applyRtProgramVars(pHitRecord, mpProgram->getHitProgram(h)->getActiveVersion().get(), pRtso, mProgramIdentifierSize, getHitVars(h)[i].get(), mpRtVarsHelper.get()))
                {
                    return false;
                }
            }
        }

        for (uint32_t m = 0; m < mpProgram->getMissProgramCount(); m++)
        {
            uint8_t* pMissRecord = getMissRecordPtr(m);
            if(!applyRtProgramVars(pMissRecord, mpProgram->getMissProgram(m)->getActiveVersion().get(), pRtso, mProgramIdentifierSize, getMissVars(m).get(), mpRtVarsHelper.get()))
            {
                return false;
            }
        }

        pCtx->updateBuffer(mpSBT.get(), mSbtData.data());
        return true;
    }
}
