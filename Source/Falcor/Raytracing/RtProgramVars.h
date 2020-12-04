/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
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
 **************************************************************************/
#pragma once
#include "Core/Program/ProgramVars.h"
#include "RtProgram/RtProgram.h"
#include "RtProgramVarsHelper.h"

namespace Falcor
{
    class RtStateObject;

    class dlldecl RtProgramVars : public ProgramVars
    {
    public:
        using SharedPtr = ParameterBlockSharedPtr<RtProgramVars>;
        using SharedConstPtr = ParameterBlockSharedPtr<const RtProgramVars>;

        /** Create a new ray tracing vars object.
            \param[in] pProgram The ray tracing program.
            \param[in] pScene The scene.
            \return A new object, or an exception is thrown if creation failed.
        */
        static SharedPtr create(const RtProgram::SharedPtr& pProgram, const Scene::SharedPtr& pScene);

        const EntryPointGroupVars::SharedPtr& getRayGenVars(uint32_t index = 0) { return mRayGenVars[index].pVars; }
        const EntryPointGroupVars::SharedPtr& getMissVars(uint32_t rayID) { return mMissVars[rayID].pVars; }
        const EntryPointGroupVars::SharedPtr& getHitVars(uint32_t rayID, uint32_t meshID) { return mHitVars[meshID * mDescHitGroupCount + rayID].pVars; }
        const EntryPointGroupVars::SharedPtr& getAABBHitVars(uint32_t rayID, uint32_t primitiveIndex) { return mAABBHitVars[primitiveIndex * mDescHitGroupCount + rayID].pVars; }

        bool apply(RenderContext* pCtx, RtStateObject* pRtso);

        ShaderTable::SharedPtr getShaderTable() const { return mpShaderTable; }

        uint32_t getRayGenVarsCount() const { return uint32_t(mRayGenVars.size()); }
        uint32_t getMissVarsCount() const { return uint32_t(mMissVars.size()); }
        uint32_t getTotalHitVarsCount() const { return uint32_t(mHitVars.size()); }
        uint32_t getAABBHitVarsCount() const { return uint32_t(mAABBHitVars.size()); }
        uint32_t getDescHitGroupCount() const { return mDescHitGroupCount; }

        Scene::SharedPtr getSceneForGeometryIndices() const { return mpSceneForGeometryIndices.lock(); }
        void setSceneForGeometryIndices(const Scene::SharedPtr& scene) { mpSceneForGeometryIndices = scene; }

    private:
        struct EntryPointGroupInfo
        {
            EntryPointGroupVars::SharedPtr  pVars;
            ChangeEpoch                     lastObservedChangeEpoch = 0;
        };

        using VarsVector = std::vector<EntryPointGroupInfo>;

        RtProgramVars(
            const RtProgram::SharedPtr& pProgram,
            const Scene::SharedPtr& pScene);

        void init();
        bool applyVarsToTable(ShaderTable::SubTableType type, uint32_t tableOffset, VarsVector& varsVec, const RtStateObject* pRtso);

        Scene::SharedPtr mpScene;
        uint32_t mDescHitGroupCount = 0;
        mutable ShaderTable::SharedPtr mpShaderTable;

        VarsVector mRayGenVars;
        VarsVector mMissVars;
        VarsVector mHitVars;
        VarsVector mAABBHitVars;

        RtVarsContext::SharedPtr mpRtVarsHelper;

        std::weak_ptr<Scene> mpSceneForGeometryIndices;
    };
}
