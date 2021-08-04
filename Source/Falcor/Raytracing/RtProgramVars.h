/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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
#include "RtBindingTable.h"

namespace Falcor
{
    class RtStateObject;

    /** This class manages a raytracing program's reflection and variable assignment.
    */
    class dlldecl RtProgramVars : public ProgramVars
    {
    public:
        using SharedPtr = ParameterBlockSharedPtr<RtProgramVars>;
        using SharedConstPtr = ParameterBlockSharedPtr<const RtProgramVars>;

        /** Create a new ray tracing vars object.
            \param[in] pProgram The ray tracing program.
            \param[in] pBindingTable The raytracing binding table.
            \return A new object, or an exception is thrown if creation failed.
        */
        static SharedPtr create(const RtProgram::SharedPtr& pProgram, const RtBindingTable::SharedPtr& pBindingTable);

        const EntryPointGroupVars::SharedPtr& getRayGenVars()
        {
            assert(mRayGenVars.size() == 1);
            return mRayGenVars[0].pVars;
        }
        const EntryPointGroupVars::SharedPtr& getMissVars(uint32_t missIndex)
        {
            assert(missIndex < mMissVars.size());
            return mMissVars[missIndex].pVars;
        }
        const EntryPointGroupVars::SharedPtr& getHitVars(uint32_t rayType, uint32_t geometryID)
        {
            assert(rayType < mRayTypeCount&& geometryID < mGeometryCount);
            return mHitVars[mRayTypeCount * geometryID + rayType].pVars;
        }

        bool apply(RenderContext* pCtx, RtStateObject* pRtso);

        ShaderTable::SharedPtr getShaderTable() const { return mpShaderTable; }

        uint32_t getMissVarsCount() const { return uint32_t(mMissVars.size()); }
        uint32_t getTotalHitVarsCount() const { return uint32_t(mHitVars.size()); }
        uint32_t getRayTypeCount() const { return mRayTypeCount; }
        uint32_t getGeometryCount() const { return mGeometryCount; }

        const std::vector<int32_t>& getUniqueEntryPointGroupIndices() const { return mUniqueEntryPointGroupIndices; }

    private:
        struct EntryPointGroupInfo
        {
            EntryPointGroupVars::SharedPtr pVars;
            ChangeEpoch lastObservedChangeEpoch = 0;
        };

        using VarsVector = std::vector<EntryPointGroupInfo>;

        RtProgramVars(const RtProgram::SharedPtr& pProgram, const RtBindingTable::SharedPtr& pBindingTable);

        void init(const RtBindingTable::SharedPtr& pBindingTable);
        bool applyVarsToTable(ShaderTable::SubTableType type, uint32_t tableOffset, VarsVector& varsVec, const RtStateObject* pRtso);

        uint32_t mRayTypeCount = 0;                         ///< Number of ray types (= number of hit groups per geometry).
        uint32_t mGeometryCount = 0;                        ///< Number of geometries.
        std::vector<int32_t> mUniqueEntryPointGroupIndices; ///< Indices of all unique entry point groups that we use in the associated program.

        mutable ShaderTable::SharedPtr mpShaderTable;       ///< GPU shader table.

        VarsVector mRayGenVars;
        VarsVector mMissVars;
        VarsVector mHitVars;

        RtVarsContext::SharedPtr mpRtVarsHelper;
    };
}
