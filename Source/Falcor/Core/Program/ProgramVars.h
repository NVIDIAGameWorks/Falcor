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
#include "Core/API/RootSignature.h"
#include "ShaderVar.h"

namespace Falcor
{
    class GraphicsProgram;
    class ComputeProgram;
    class ComputeContext;

    class dlldecl EntryPointGroupVars  : public ParameterBlock
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
            assert(pReflector);
            return SharedPtr(new EntryPointGroupVars(pReflector, groupIndexInProgram));
        }

        uint32_t getGroupIndexInProgram() const { return mGroupIndexInProgram; }

    protected:
        EntryPointGroupVars(const EntryPointGroupReflection::SharedConstPtr& pReflector, uint32_t groupIndexInProgram)
            : ParameterBlock(pReflector->getProgramVersion(), pReflector)
            , mGroupIndexInProgram(groupIndexInProgram)
        {
            assert(pReflector);
        }

    private:
        uint32_t mGroupIndexInProgram;
    };

    /** This class manages a program's reflection and variable assignment.
        It's a high-level abstraction of variables-related concepts such as CBs, texture and sampler assignments, root-signature, descriptor tables, etc.
    */
    class dlldecl ProgramVars : public ParameterBlock
    {
    public:
        using SharedPtr = ParameterBlockSharedPtr<ProgramVars>;
        using SharedConstPtr = std::shared_ptr<const ProgramVars>;

        /** Get the program reflection interface
        */
        const ProgramReflection::SharedConstPtr& getReflection() const { return mpReflector; }

        virtual bool updateSpecializationImpl() const override;

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

    class dlldecl GraphicsVars : public ProgramVars
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

        virtual bool apply(RenderContext* pContext, bool bindRootSig, RootSignature* pRootSignature);

    protected:
        GraphicsVars(const ProgramReflection::SharedConstPtr& pReflector);
    };

    template<bool forGraphics>
    bool applyProgramVarsCommon(ParameterBlock* pVars, CopyContext* pContext, bool bindRootSig, RootSignature* pRootSignature);

    class dlldecl ComputeVars : public ProgramVars
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

        virtual bool apply(ComputeContext* pContext, bool bindRootSig, RootSignature* pRootSignature);

    protected:
        ComputeVars(const ProgramReflection::SharedConstPtr& pReflector);
    };
}
