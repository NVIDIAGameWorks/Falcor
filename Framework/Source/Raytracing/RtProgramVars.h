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
#pragma once
#include "RtProgram/RtProgram.h"
#include "RtScene.h"
#include "API/Buffer.h"
#include "Graphics/Program/ProgramVars.h"
#include "RtProgramVarsHelper.h"

namespace Falcor
{
    class RenderContext;
    class RtStateObject;

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

        bool apply(RenderContext* pCtx, RtStateObject* pRtso);

        Buffer::SharedPtr getSBT() const { return mpSBT; }
        uint32_t getRecordSize() const { return mRecordSize; }
        uint32_t getRayGenSbtRecordIndex() const { return kRayGenSbtRecordIndex; }
        uint32_t getFirstMissSbtRecordIndex() const { return kFirstMissSbtRecordIndex; }
        uint32_t getFirstHitSbtRecordIndex() const { return mFirstHitVarEntry; }
        uint32_t getHitProgramsCount() const { return mHitProgCount; }
        uint32_t getMissProgramsCount() const { return mMissProgCount; }

    private:
        static const uint32_t kRayGenSbtRecordIndex = 0;
        static const uint32_t kFirstMissSbtRecordIndex = 1;
        uint32_t mMissProgCount = 0;
        uint32_t mHitProgCount = 0;
        uint32_t mFirstHitVarEntry = 0;

        RtProgramVars(RtProgram::SharedPtr pProgram, RtScene::SharedPtr pScene);
        RtProgram::SharedPtr mpProgram;
        RtScene::SharedPtr mpScene;
        uint32_t mRecordSize;
        uint32_t mProgramIdentifierSize;
        Buffer::SharedPtr mpSBT;

        uint8_t* getRayGenRecordPtr();
        uint8_t* getMissRecordPtr(uint32_t missId);
        uint8_t* getHitRecordPtr(uint32_t hitId, uint32_t meshId);

        bool init();

        GraphicsVars::SharedPtr mRayGenVars;
        std::vector<VarsVector> mHitVars;
        std::vector<uint8_t> mSbtData;
        VarsVector mMissVars;
        RtVarsContext::SharedPtr mpRtVarsHelper;
    };
}