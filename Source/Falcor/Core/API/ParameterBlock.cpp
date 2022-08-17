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
#include "ParameterBlock.h"
#include "CopyContext.h"
#include "Core/Assert.h"
#include "Core/Errors.h"
#include "Core/Program/ProgramVersion.h"
#include "Core/BufferTypes/VariablesBufferUI.h"

namespace Falcor
{
    ParameterBlock::SharedPtr ParameterBlock::create(const std::shared_ptr<const ProgramVersion>& pProgramVersion, const ReflectionType::SharedConstPtr& pElementType)
    {
        if (!pElementType) throw ArgumentError("Can't create a parameter block without type information");
        auto pReflection = ParameterBlockReflection::create(pProgramVersion.get(), pElementType);
        return create(pReflection);
    }

    ParameterBlock::SharedPtr ParameterBlock::create(const ParameterBlockReflection::SharedConstPtr& pReflection)
    {
        FALCOR_ASSERT(pReflection);
        return SharedPtr(new ParameterBlock(pReflection->getProgramVersion(), pReflection));
    }

    ParameterBlock::SharedPtr ParameterBlock::create(const std::shared_ptr<const ProgramVersion>& pProgramVersion, const std::string& typeName)
    {
        FALCOR_ASSERT(pProgramVersion);
        return ParameterBlock::create(pProgramVersion, pProgramVersion->getReflector()->findType(typeName));
    }

    ShaderVar ParameterBlock::getRootVar() const
    {
        return ShaderVar(const_cast<ParameterBlock*>(this));
    }

    ShaderVar ParameterBlock::findMember(const std::string& varName) const
    {
        return getRootVar().findMember(varName);
    }

    ShaderVar ParameterBlock::findMember(uint32_t index) const
    {
        return getRootVar().findMember(index);
    }

    size_t ParameterBlock::getElementSize() const
    {
        return mpReflector->getElementType()->getByteSize();
    }

    UniformShaderVarOffset ParameterBlock::getVariableOffset(const std::string& varName) const
    {
        return getElementType()->getZeroOffset()[varName];
    }

    void ParameterBlock::createConstantBuffers(const ShaderVar& var)
    {
        auto pType = var.getType();
        if (pType->getResourceRangeCount() == 0) return;

        switch (pType->getKind())
        {
        case ReflectionType::Kind::Struct:
        {
            auto pStructType = pType->asStructType();
            uint32_t memberCount = pStructType->getMemberCount();
            for (uint32_t i = 0; i < memberCount; ++i) createConstantBuffers(var[i]);
        }
        break;
        case ReflectionType::Kind::Resource:
        {
            auto pResourceType = pType->asResourceType();
            switch (pResourceType->getType())
            {
            case ReflectionResourceType::Type::ConstantBuffer:
            {
                auto pCB = ParameterBlock::create(pResourceType->getParameterBlockReflector());
                var.setParameterBlock(pCB);
            }
            break;

            default:
                break;
            }
        }
        break;

        default:
            break;
        }
    }

    void ParameterBlock::renderUI(Gui::Widgets& widget)
    {
        VariablesBufferUI variablesBufferUI(*this);
        variablesBufferUI.renderUI(widget);
    }

    void ParameterBlock::prepareResource(
        CopyContext* pContext,
        Resource* pResource,
        bool isUav)
    {
        if (!pResource) return;

        // If it's a buffer with a UAV counter, insert a UAV barrier
        const Buffer* pBuffer = pResource->asBuffer().get();
        if (isUav && pBuffer && pBuffer->getUAVCounter())
        {
            pContext->resourceBarrier(pBuffer->getUAVCounter().get(), Resource::State::UnorderedAccess);
            pContext->uavBarrier(pBuffer->getUAVCounter().get());
        }

        bool insertBarrier = true;
        insertBarrier = (is_set(pResource->getBindFlags(), Resource::BindFlags::AccelerationStructure) == false);
        if (insertBarrier)
        {
            insertBarrier = !pContext->resourceBarrier(pResource, isUav ? Resource::State::UnorderedAccess : Resource::State::ShaderResource);
        }

        // Insert UAV barrier automatically if the resource is an UAV that is already in UnorderedAccess state.
        // Otherwise the user would have to insert barriers explicitly between passes accessing UAVs, which is easily forgotten.
        if (insertBarrier && isUav) pContext->uavBarrier(pResource);
    }

    // Template specialization to allow setting booleans on a parameter block.
    // On the host side a bool is 1B and the device 4B. We cast bools to 32-bit integers here.
    // Note that this applies to our boolN vectors as well, which are currently 1B per element.

    template<> FALCOR_API bool ParameterBlock::setVariable(UniformShaderVarOffset offset, const bool& value)
    {
        int32_t v = value ? 1 : 0;
        return setVariable(offset, v);
    }

    template<> FALCOR_API bool ParameterBlock::setVariable(UniformShaderVarOffset offset, const bool2& value)
    {
        int2 v = { value.x ? 1 : 0, value.y ? 1 : 0 };
        return setVariable(offset, v);
    }

    template<> FALCOR_API bool ParameterBlock::setVariable(UniformShaderVarOffset offset, const bool3& value)
    {
        int3 v = { value.x ? 1 : 0, value.y ? 1 : 0, value.z ? 1 : 0 };
        return setVariable(offset, v);
    }

    template<> FALCOR_API bool ParameterBlock::setVariable(UniformShaderVarOffset offset, const bool4& value)
    {
        int4 v = { value.x ? 1 : 0, value.y ? 1 : 0, value.z ? 1 : 0, value.w ? 1 : 0 };
        return setVariable(offset, v);
    }
}
