/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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
#include "RtBindingTable.h"
#include "Core/Macros.h"
#include "Core/API/ParameterBlock.h"
#include "Core/API/ShaderTable.h"
#include <memory>
#include <vector>

namespace Falcor
{
    class GraphicsProgram;
    class ComputeProgram;
    class ComputeContext;

    class FALCOR_API EntryPointGroupVars  : public ParameterBlock
    {
    public:
        using SharedPtr = ParameterBlockSharedPtr<EntryPointGroupVars>;
        using SharedConstPtr = std::shared_ptr<const EntryPointGroupVars>;

        /** Create a new entry point group vars object.
            \param[in] pReflector The reflection object.
            \param[in] groupIndexInProgram Group index.
            \return New object, or throws an exception if creation failed.
        */
        static SharedPtr create(const EntryPointGroupReflection::SharedConstPtr& pReflector, uint32_t groupIndexInProgram)
        {
            FALCOR_ASSERT(pReflector);
            return SharedPtr(new EntryPointGroupVars(pReflector, groupIndexInProgram));
        }

        uint32_t getGroupIndexInProgram() const { return mGroupIndexInProgram; }

    protected:
        EntryPointGroupVars(const EntryPointGroupReflection::SharedConstPtr& pReflector, uint32_t groupIndexInProgram)
            : ParameterBlock(pReflector->getProgramVersion(), pReflector)
            , mGroupIndexInProgram(groupIndexInProgram)
        {
            FALCOR_ASSERT(pReflector);
        }

    private:
        uint32_t mGroupIndexInProgram;
    };

    /** This class manages a program's reflection and variable assignment.
        It's a high-level abstraction of variables-related concepts such as CBs, texture and sampler assignments, root-signature, descriptor tables, etc.
    */
    class FALCOR_API ProgramVars : public ParameterBlock
    {
    public:
        using SharedPtr = ParameterBlockSharedPtr<ProgramVars>;
        using SharedConstPtr = std::shared_ptr<const ProgramVars>;

        /** Get the program reflection interface
        */
        const ProgramReflection::SharedConstPtr& getReflection() const { return mpReflector; }

#ifdef FALCOR_D3D12
        virtual bool updateSpecializationImpl() const override;
#endif

        uint32_t getEntryPointGroupCount() const { return uint32_t(mpEntryPointGroupVars.size()); }
        EntryPointGroupVars* getEntryPointGroupVars(uint32_t index) const
        {
            return mpEntryPointGroupVars[index].get();
        }

    protected:
        ProgramVars(const ProgramReflection::SharedConstPtr& pReflector);

        ProgramReflection::SharedConstPtr mpReflector;

        void addSimpleEntryPointGroups();

        std::vector<EntryPointGroupVars::SharedPtr> mpEntryPointGroupVars;
    };

    class FALCOR_API GraphicsVars : public ProgramVars
    {
    public:
        using SharedPtr = ParameterBlockSharedPtr<GraphicsVars>;
        using SharedConstPtr = std::shared_ptr<const GraphicsVars>;

        /** Create a new graphics vars object.
            \param[in] pReflector A program reflection object containing the requested declarations.
            \return A new object, or an exception is thrown if creation failed.
        */
        static SharedPtr create(const ProgramReflection::SharedConstPtr& pReflector);

        /** Create a new graphics vars object.
            \param[in] pProg A program containing the requested declarations. The active version of the program is used.
            \return A new object, or an exception is thrown if creation failed.
        */
        static SharedPtr create(const GraphicsProgram* pProg);

#ifdef FALCOR_D3D12
        virtual bool apply(RenderContext* pContext, bool bindRootSig, const ProgramKernels* pProgramKernels);
#endif

    protected:
        GraphicsVars(const ProgramReflection::SharedConstPtr& pReflector);
    };

    template<bool forGraphics>
    bool applyProgramVarsCommon(ParameterBlock* pVars, CopyContext* pContext, bool bindRootSig, ProgramKernels* pProgramKernels);

    class FALCOR_API ComputeVars : public ProgramVars
    {
    public:
        using SharedPtr = ParameterBlockSharedPtr<ComputeVars>;
        using SharedConstPtr = std::shared_ptr<const ComputeVars>;

        /** Create a new compute vars object.
            \param[in] pReflector A program reflection object containing the requested declarations.
            \return A new object, or an exception is thrown if creation failed.
        */
        static SharedPtr create(const ProgramReflection::SharedConstPtr& pReflector);

        /** Create a new compute vars object.
            \param[in] pProg A program containing the requested declarations. The active version of the program is used.
            \return A new object, or an exception is thrown if creation failed.
        */
        static SharedPtr create(const ComputeProgram* pProg);

#ifdef FALCOR_D3D12
        virtual bool apply(ComputeContext* pContext, bool bindRootSig, const ProgramKernels* pProgramKernels);
#endif

        /** Dispatch the program using the argument values set in this object.
        */
        void dispatchCompute(ComputeContext* pContext, uint3 const& threadGroupCount);

    protected:
        ComputeVars(const ProgramReflection::SharedConstPtr& pReflector);
    };

    class RtStateObject;

    /** This class manages a raytracing program's reflection and variable assignment.
    */
    class FALCOR_API RtProgramVars : public ProgramVars
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

#ifdef FALCOR_D3D12
        bool apply(RenderContext* pCtx, RtStateObject* pRtso);
#endif
#ifdef FALCOR_GFX
        bool prepareShaderTable(RenderContext* pCtx, RtStateObject* pRtso);
#endif

        ShaderTablePtr getShaderTable() const { return mpShaderTable; }
        uint32_t getMissVarsCount() const { return uint32_t(mMissVars.size()); }
        uint32_t getTotalHitVarsCount() const { return uint32_t(mHitVars.size()); }
        uint32_t getRayTypeCount() const { return mRayTypeCount; }
        uint32_t getGeometryCount() const { return mGeometryCount; }

        const std::vector<int32_t>& getUniqueEntryPointGroupIndices() const { return mUniqueEntryPointGroupIndices; }

    private:
        struct EntryPointGroupInfo
        {
#ifdef FALCOR_D3D12
            EntryPointGroupVars::SharedPtr pVars;
            ChangeEpoch lastObservedChangeEpoch = 0;
#elif defined(FALCOR_GFX)
            int32_t entryPointGroupIndex = -1;
#endif
        };

        using VarsVector = std::vector<EntryPointGroupInfo>;

        RtProgramVars(const RtProgram::SharedPtr& pProgram, const RtBindingTable::SharedPtr& pBindingTable);

        void init(const RtBindingTable::SharedPtr& pBindingTable);

#ifdef FALCOR_D3D12
        bool applyVarsToTable(ShaderTable::SubTableType type, uint32_t tableOffset, VarsVector& varsVec, const RtStateObject* pRtso);
#endif
        static RtEntryPointGroupKernels* getUniqueRtEntryPointGroupKernels(const ProgramKernels::SharedConstPtr& pKernels, int32_t uniqueEntryPointGroupIndex);

        uint32_t mRayTypeCount = 0;                         ///< Number of ray types (= number of hit groups per geometry).
        uint32_t mGeometryCount = 0;                        ///< Number of geometries.
        std::vector<int32_t> mUniqueEntryPointGroupIndices; ///< Indices of all unique entry point groups that we use in the associated program.

        mutable ShaderTablePtr mpShaderTable;       ///< GPU shader table.

#ifdef FALCOR_GFX
        mutable RtStateObject* mpCurrentRtStateObject = nullptr; ///< The RtStateObject used to create the current shader table.
#endif
        VarsVector mRayGenVars;
        VarsVector mMissVars;
        VarsVector mHitVars;
    };

}
