/***************************************************************************
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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
#include "Framework.h"
#include "ProgramReflection.h"
#include "Utils/StringUtils.h"
#include <unordered_set>
using namespace slang;

namespace Falcor
{
    static std::unordered_set<std::string> gParameterBlocksRegistry;

    bool isParameterBlock(VariableLayoutReflection* pSlangVar)
    {
        // A candidate for a parameter block must be a top-level constant buffer containing a single struct
        TypeLayoutReflection* pSlangType = pSlangVar->getTypeLayout();
        if (pSlangType->getTotalArrayElementCount() == 0 && pSlangType->unwrapArray()->getKind() == TypeReflection::Kind::ConstantBuffer)
        {
            if (pSlangType->unwrapArray()->getElementCount() == 1)
            {
                TypeLayoutReflection* pFieldLayout = pSlangType->unwrapArray()->getFieldByIndex(0)->getTypeLayout();
                if (pFieldLayout->getTotalArrayElementCount() == 0 && pFieldLayout->unwrapArray()->getKind() == TypeReflection::Kind::Struct)
                {
                    return true;
                }
            }                
        }
        return false;
    }

    ProgramReflection::SharedPtr ProgramReflection::create(slang::ShaderReflection* pSlangReflector, std::string& log)
    {
        return SharedPtr(new ProgramReflection(pSlangReflector, log));
    }

    ProgramReflection::ProgramReflection(slang::ShaderReflection* pSlangReflector, std::string& log)
    {

    }
}