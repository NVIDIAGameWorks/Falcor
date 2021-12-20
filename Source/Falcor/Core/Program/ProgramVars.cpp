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
#include "stdafx.h"
#include "Core/Program/ProgramVars.h"
#include "Core/Program/GraphicsProgram.h"
#include "Core/Program/ComputeProgram.h"
#include "Core/API/ComputeContext.h"
#include "Core/API/RenderContext.h"

#include <slang/slang.h>

namespace Falcor
{
    void ProgramVars::addSimpleEntryPointGroups()
    {
        auto& entryPointGroups = mpReflector->getEntryPointGroups();
        auto groupCount = entryPointGroups.size();
        for( size_t gg = 0; gg < groupCount; ++gg )
        {
            auto pGroup = entryPointGroups[gg];
            auto pGroupVars = EntryPointGroupVars::create(pGroup, uint32_t(gg));
            mpEntryPointGroupVars.push_back(pGroupVars);
        }
    }

    GraphicsVars::GraphicsVars(const ProgramReflection::SharedConstPtr& pReflector)
        : ProgramVars(pReflector)
    {
        addSimpleEntryPointGroups();
    }

    GraphicsVars::SharedPtr GraphicsVars::create(const ProgramReflection::SharedConstPtr& pReflector)
    {
        if (pReflector == nullptr) ArgumentError("Can't create a GraphicsVars object without a program reflector");
        return SharedPtr(new GraphicsVars(pReflector));
    }

    GraphicsVars::SharedPtr GraphicsVars::create(const GraphicsProgram* pProg)
    {
        if (pProg == nullptr) ArgumentError("Can't create a GraphicsVars object without a program");
        return create(pProg->getReflector());
    }

    ComputeVars::SharedPtr ComputeVars::create(const ProgramReflection::SharedConstPtr& pReflector)
    {
        if (pReflector == nullptr) ArgumentError("Can't create a ComputeVars object without a program reflector");
        return SharedPtr(new ComputeVars(pReflector));
    }

    ComputeVars::SharedPtr ComputeVars::create(const ComputeProgram* pProg)
    {
        if (pProg == nullptr) ArgumentError("Can't create a ComputeVars object without a program");
        return create(pProg->getReflector());
    }

    ComputeVars::ComputeVars(const ProgramReflection::SharedConstPtr& pReflector)
        : ProgramVars(pReflector)
    {
        addSimpleEntryPointGroups();
    }


    RtProgramVars::RtProgramVars(const RtProgram::SharedPtr& pProgram, const RtBindingTable::SharedPtr& pBindingTable)
        : ProgramVars(pProgram->getReflector())
    {
        if (pProgram == nullptr)
        {
            throw ArgumentError("RtProgramVars must have a raytracing program attached to it");
        }
        if (pBindingTable == nullptr || !pBindingTable->getRayGen().isValid())
        {
            throw ArgumentError("RtProgramVars must have a raygen program attached to it");
        }

        init(pBindingTable);
    }

    RtProgramVars::SharedPtr RtProgramVars::create(const RtProgram::SharedPtr& pProgram, const RtBindingTable::SharedPtr& pBindingTable)
    {
        return SharedPtr(new RtProgramVars(pProgram, pBindingTable));
    }
}
