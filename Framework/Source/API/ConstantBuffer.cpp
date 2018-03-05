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
#include "Framework.h"
#include "ConstantBuffer.h"
#include "Graphics/Program/ProgramVersion.h"
#include "Buffer.h"
#include "glm/glm.hpp"
#include "Texture.h"
#include "Graphics/Program/ProgramReflection.h"
#include "API/Device.h"

namespace Falcor
{
    ConstantBuffer::~ConstantBuffer() = default;

    ConstantBuffer::ConstantBuffer(const std::string& name, const ReflectionResourceType::SharedConstPtr& pReflectionType, size_t size) :
        VariablesBuffer(name, pReflectionType, size, 1, Buffer::BindFlags::Constant, Buffer::CpuAccess::Write)
    {
    }

    ConstantBuffer::SharedPtr ConstantBuffer::create(const std::string& name, const ReflectionResourceType::SharedConstPtr& pReflectionType, size_t overrideSize)
    {
        size_t size = (overrideSize == 0) ? pReflectionType->getSize() : overrideSize;
        SharedPtr pBuffer = SharedPtr(new ConstantBuffer(name, pReflectionType, size));
        return pBuffer;
    }

    ConstantBuffer::SharedPtr ConstantBuffer::create(Program::SharedPtr& pProgram, const std::string& name, size_t overrideSize)
    {
        const auto& pProgReflector = pProgram->getActiveVersion()->getReflector();
        const auto& pParamBlockReflection = pProgReflector->getDefaultParameterBlock();
        ReflectionVar::SharedConstPtr pBufferReflector = pParamBlockReflection ? pParamBlockReflection->getResource(name) : nullptr;

        if (pBufferReflector)
        {
            ReflectionResourceType::SharedConstPtr pResType = pBufferReflector->getType()->asResourceType()->inherit_shared_from_this::shared_from_this();
            if(pResType && pResType->getType() == ReflectionResourceType::Type::ConstantBuffer)
            {
                return create(name, pResType, overrideSize);
            }
        }
        logError("Can't find a constant buffer named \"" + name + "\" in the program");
        return nullptr;
    }

    bool ConstantBuffer::uploadToGPU(size_t offset, size_t size)
    {
        if (mDirty) mpCbv = nullptr;
        return VariablesBuffer::uploadToGPU(offset, size);
    }

    ConstantBufferView::SharedPtr ConstantBuffer::getCbv() const
    {
        if (mpCbv == nullptr)
        {
            mpCbv = ConstantBufferView::create(Resource::shared_from_this());
        }
        return mpCbv;
    }
}