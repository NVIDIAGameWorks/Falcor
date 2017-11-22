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
#include "VariablesBuffer.h"
#include "ProgramVersion.h"
#include "Buffer.h"
#include "glm/glm.hpp"
#include "Texture.h"
#include "API/ProgramReflection.h"
#include "API/Device.h"
#include <cstring>

namespace Falcor
{
    VariablesBuffer::~VariablesBuffer() = default;

    VariablesBuffer::VariablesBuffer(const ProgramReflection::BufferReflection::SharedConstPtr& pReflector, size_t elementSize, size_t elementCount, BindFlags bindFlags, CpuAccess cpuAccess) :
        mpReflector(pReflector), Buffer(elementSize * elementCount, bindFlags, cpuAccess), mElementCount(elementCount), mElementSize(elementSize)
    {
        Buffer::apiInit(false);
        mData.assign(mSize, 0);
    }

    size_t VariablesBuffer::getVariableOffset(const std::string& varName) const
    {
        size_t offset;
        mpReflector->getVariableData(varName, offset);
        return offset;
    }

    bool VariablesBuffer::uploadToGPU(size_t offset, size_t size)
    {
        if(mDirty == false)
        {
            return false;
        }

        if(size == -1)
        {
            size = mSize - offset;
        }

        if(size + offset > mSize)
        {
            logWarning("VariablesBuffer::uploadToGPU() - trying to upload more data than what the buffer contains. Call is ignored.");
            return false;
        }

        updateData(mData.data(), offset, size);
        mDirty = false;
        return true;
    }

    
#define verify_element_index() if(elementIndex >= mElementCount) {logWarning(std::string(__FUNCTION__) + ": elementIndex is out-of-bound. Ignoring call."); return;}

    template<typename VarType> 
    void VariablesBuffer::setVariable(size_t offset, size_t elementIndex, const VarType& value)
    {
        verify_element_index();
        if(checkVariableByOffset<VarType>(offset, 1, mpReflector.get()))
        {
            const uint8_t* pVar = mData.data() + offset + elementIndex * mElementSize;
            *(VarType*)pVar = value;
            mDirty = true;
        }
    }

#define set_constant_by_offset(_t) template void VariablesBuffer::setVariable(size_t offset, size_t elementIndex, const _t& value)
    set_constant_by_offset(bool);
    set_constant_by_offset(glm::bvec2);
    set_constant_by_offset(glm::bvec3);
    set_constant_by_offset(glm::bvec4);

    set_constant_by_offset(uint32_t);
    set_constant_by_offset(glm::uvec2);
    set_constant_by_offset(glm::uvec3);
    set_constant_by_offset(glm::uvec4);

    set_constant_by_offset(int32_t);
    set_constant_by_offset(glm::ivec2);
    set_constant_by_offset(glm::ivec3);
    set_constant_by_offset(glm::ivec4);

    set_constant_by_offset(float);
    set_constant_by_offset(glm::vec2);
    set_constant_by_offset(glm::vec3);
    set_constant_by_offset(glm::vec4);

    set_constant_by_offset(glm::mat2);
    set_constant_by_offset(glm::mat2x3);
    set_constant_by_offset(glm::mat2x4);

    set_constant_by_offset(glm::mat3);
    set_constant_by_offset(glm::mat3x2);
    set_constant_by_offset(glm::mat3x4);

    set_constant_by_offset(glm::mat4);
    set_constant_by_offset(glm::mat4x2);
    set_constant_by_offset(glm::mat4x3);

    set_constant_by_offset(uint64_t);

#undef set_constant_by_offset

    template<typename VarType>
    void VariablesBuffer::setVariable(const std::string& name, size_t element, const VarType& value)
    {
        size_t offset;
        const auto* pVar = mpReflector->getVariableData(name, offset);
        if((_LOG_ENABLED == 0) || (offset != ProgramReflection::kInvalidLocation && checkVariableType<VarType>(pVar->type, name, mpReflector->getName())))
        {
            setVariable<VarType>(offset, element, value);
        }
    }

#define set_constant_by_name(_t) template void VariablesBuffer::setVariable(const std::string& name, size_t element, const _t& value)

    set_constant_by_name(bool);
    set_constant_by_name(glm::bvec2);
    set_constant_by_name(glm::bvec3);
    set_constant_by_name(glm::bvec4);

    set_constant_by_name(uint32_t);
    set_constant_by_name(glm::uvec2);
    set_constant_by_name(glm::uvec3);
    set_constant_by_name(glm::uvec4);

    set_constant_by_name(int32_t);
    set_constant_by_name(glm::ivec2);
    set_constant_by_name(glm::ivec3);
    set_constant_by_name(glm::ivec4);

    set_constant_by_name(float);
    set_constant_by_name(glm::vec2);
    set_constant_by_name(glm::vec3);
    set_constant_by_name(glm::vec4);

    set_constant_by_name(glm::mat2);
    set_constant_by_name(glm::mat2x3);
    set_constant_by_name(glm::mat2x4);

    set_constant_by_name(glm::mat3);
    set_constant_by_name(glm::mat3x2);
    set_constant_by_name(glm::mat3x4);

    set_constant_by_name(glm::mat4);
    set_constant_by_name(glm::mat4x2);
    set_constant_by_name(glm::mat4x3);

    set_constant_by_name(uint64_t);
#undef set_constant_by_name

    template<typename VarType> 
    void VariablesBuffer::setVariableArray(size_t offset, size_t elementIndex, const VarType* pValue, size_t count)
    {
        verify_element_index();
        if(checkVariableByOffset<VarType>(offset, count, mpReflector.get()))
        {
            const uint8_t* pVar = mData.data() + offset;
            VarType* pData = (VarType*)pVar + elementIndex * mElementSize;
            for(size_t i = 0; i < count; i++)
            {
                pData[i] = pValue[i];
            }
            mDirty = true;
        }
    }

#define set_constant_array_by_offset(_t) template void VariablesBuffer::setVariableArray(size_t offset, size_t elementIndex, const _t* pValue, size_t count)

    set_constant_array_by_offset(bool);
    set_constant_array_by_offset(glm::bvec2);
    set_constant_array_by_offset(glm::bvec3);
    set_constant_array_by_offset(glm::bvec4);

    set_constant_array_by_offset(uint32_t);
    set_constant_array_by_offset(glm::uvec2);
    set_constant_array_by_offset(glm::uvec3);
    set_constant_array_by_offset(glm::uvec4);

    set_constant_array_by_offset(int32_t);
    set_constant_array_by_offset(glm::ivec2);
    set_constant_array_by_offset(glm::ivec3);
    set_constant_array_by_offset(glm::ivec4);

    set_constant_array_by_offset(float);
    set_constant_array_by_offset(glm::vec2);
    set_constant_array_by_offset(glm::vec3);
    set_constant_array_by_offset(glm::vec4);

    set_constant_array_by_offset(glm::mat2);
    set_constant_array_by_offset(glm::mat2x3);
    set_constant_array_by_offset(glm::mat2x4);

    set_constant_array_by_offset(glm::mat3);
    set_constant_array_by_offset(glm::mat3x2);
    set_constant_array_by_offset(glm::mat3x4);

    set_constant_array_by_offset(glm::mat4);
    set_constant_array_by_offset(glm::mat4x2);
    set_constant_array_by_offset(glm::mat4x3);

    set_constant_array_by_offset(uint64_t);

#undef set_constant_array_by_offset

    template<typename VarType>
    void VariablesBuffer::setVariableArray(const std::string& name, size_t elementIndex, const VarType* pValue, size_t count)
    {
        size_t offset;
        const auto& pVarDesc = mpReflector->getVariableData(name, offset);
        if( _LOG_ENABLED == 0 || (offset != ProgramReflection::kInvalidLocation && checkVariableType<VarType>(pVarDesc->type, name, mpReflector->getName())))
        {
            setVariableArray(offset, elementIndex, pValue, count);
        }
    }
    
#define set_constant_array_by_string(_t) template void VariablesBuffer::setVariableArray(const std::string& name, size_t elementIndex, const _t* pValue, size_t count)

    set_constant_array_by_string(bool);
    set_constant_array_by_string(glm::bvec2);
    set_constant_array_by_string(glm::bvec3);
    set_constant_array_by_string(glm::bvec4);

    set_constant_array_by_string(uint32_t);
    set_constant_array_by_string(glm::uvec2);
    set_constant_array_by_string(glm::uvec3);
    set_constant_array_by_string(glm::uvec4);

    set_constant_array_by_string(int32_t);
    set_constant_array_by_string(glm::ivec2);
    set_constant_array_by_string(glm::ivec3);
    set_constant_array_by_string(glm::ivec4);

    set_constant_array_by_string(float);
    set_constant_array_by_string(glm::vec2);
    set_constant_array_by_string(glm::vec3);
    set_constant_array_by_string(glm::vec4);

    set_constant_array_by_string(glm::mat2);
    set_constant_array_by_string(glm::mat2x3);
    set_constant_array_by_string(glm::mat2x4);

    set_constant_array_by_string(glm::mat3);
    set_constant_array_by_string(glm::mat3x2);
    set_constant_array_by_string(glm::mat3x4);

    set_constant_array_by_string(glm::mat4);
    set_constant_array_by_string(glm::mat4x2);
    set_constant_array_by_string(glm::mat4x3);

    set_constant_array_by_string(uint64_t);

#undef set_constant_array_by_string

    void VariablesBuffer::setBlob(const void* pSrc, size_t offset, size_t size)
    {
        if((_LOG_ENABLED != 0) && (offset + size > mSize))
        {
            std::string Msg("Error when setting blob to buffer\"");
            Msg += mpReflector->getName() + "\". Blob to large and will result in overflow. Ignoring call.";
            logError(Msg);
            return;
        }
        std::memcpy(mData.data() + offset, pSrc, size);
        mDirty = true;
    }

    bool checkResourceDimension(const Texture* pTexture, const ProgramReflection::Resource* pResourceDesc, const std::string& name, const std::string& bufferName)
    {
#if _LOG_ENABLED

        bool dimsMatch = false;
        bool formatMatch = false;

        Texture::Type texDim = pTexture->getType();
        bool isArray = pTexture->getArraySize() > 1;

        // Check if the dimensions match
        switch(pResourceDesc->dims)
        {
        case ProgramReflection::Resource::Dimensions::Texture1D:
            dimsMatch = (texDim == Texture::Type::Texture1D) && (isArray == false);
            break;
        case ProgramReflection::Resource::Dimensions::Texture2D:
            dimsMatch = (texDim == Texture::Type::Texture2D) && (isArray == false);
            break;
        case ProgramReflection::Resource::Dimensions::Texture3D:
            dimsMatch = (texDim == Texture::Type::Texture3D) && (isArray == false);
            break;
        case ProgramReflection::Resource::Dimensions::TextureCube:
            dimsMatch = (texDim == Texture::Type::TextureCube) && (isArray == false);
            break;
        case ProgramReflection::Resource::Dimensions::Texture1DArray:
            dimsMatch = (texDim == Texture::Type::Texture1D) && isArray;
            break;
        case ProgramReflection::Resource::Dimensions::Texture2DArray:
            dimsMatch = (texDim == Texture::Type::Texture2D);
            break;
        case ProgramReflection::Resource::Dimensions::Texture2DMS:
            dimsMatch = (texDim == Texture::Type::Texture2DMultisample) && (isArray == false);
            break;
        case ProgramReflection::Resource::Dimensions::Texture2DMSArray:
            dimsMatch = (texDim == Texture::Type::Texture2DMultisample);
            break;
        case ProgramReflection::Resource::Dimensions::TextureCubeArray:
            dimsMatch = (texDim == Texture::Type::TextureCube) && isArray;
            break;
        case ProgramReflection::Resource::Dimensions::Buffer:
            break;
        default:
            should_not_get_here();
        }

        // Check if the resource Type match
        FormatType texFormatType = getFormatType(pTexture->getFormat());

        switch(pResourceDesc->retType)
        {
        case ProgramReflection::Resource::ReturnType::Float:
            formatMatch = (texFormatType == FormatType::Float) || (texFormatType == FormatType::Snorm) || (texFormatType == FormatType::Unorm) || (texFormatType == FormatType::UnormSrgb);
            break;
        case ProgramReflection::Resource::ReturnType::Int:
            formatMatch = (texFormatType == FormatType::Sint);
            break;
        case ProgramReflection::Resource::ReturnType::Uint:
            formatMatch = (texFormatType == FormatType::Uint);
            break;
        default:
            should_not_get_here();
        }
        if((dimsMatch && formatMatch) == false)
        {
            std::string msg("Error when setting texture \"");
            msg += name + "\".\n";
            if(dimsMatch == false)
            {
                msg += "Dimensions mismatch.\nTexture has Type " + to_string(texDim) + (isArray ? "Array" : "") + ".\nVariable has Type " + to_string(pResourceDesc->dims) + ".\n";
            }

            if(formatMatch == false)
            {
                msg += "Format mismatch.\nTexture has format Type " + to_string(texFormatType) + ".\nVariable has Type " + to_string(pResourceDesc->retType) + ".\n";
            }

            msg += "\nError when setting resource to buffer " + bufferName;
            logError(msg);
            return false;
        }
#endif
        return true;
    }

    const ProgramReflection::Resource* getResourceDesc(const std::string& name, const ProgramReflection::BufferReflection* pReflector)
    {
        auto pResource = pReflector->getResourceData(name);
#ifdef FALCOR_D3D11
        // If it's not found and this is DX, search for out internal struct
        if(pResource = nullptr)
        {
            pResource = pReflector->getResourceData(name + ".t");
        }
#endif
        return pResource;
    }

    void VariablesBuffer::setTexture(size_t offset, const Texture* pTexture, const Sampler* pSampler)
    {
        bool bOK = true;
#if _LOG_ENABLED
        // Debug checks
        if(pTexture)
        {
            for(auto a = mpReflector->varBegin() ; a != mpReflector->varEnd() ; a++)
            {
                const auto& varDesc = a->second;
                const auto& varName = a->first;
                if(varDesc.type == ProgramReflection::Variable::Type::Resource)
                {
                    size_t ArrayIndex = 0;
                    bool bCheck = (varDesc.location == offset);

                    // Check arrays
                    if(varDesc.arrayStride > 0 && offset > varDesc.location)
                    {
                        size_t Stride = offset - varDesc.location;
                        if((Stride % varDesc.arrayStride) == 0)
                        {
                            ArrayIndex = Stride / varDesc.arrayStride;
                            if(ArrayIndex < varDesc.arraySize)
                            {
                                bCheck = true;
                            }
                        }
                    }

                    if(bCheck)
                    {
                        const auto& pResource = getResourceDesc(varName, mpReflector.get());
                        assert(pResource != nullptr);                
                        if(pResource->type != ProgramReflection::Resource::ResourceType::Sampler)
                        {
                            bOK = checkResourceDimension(pTexture, pResource, varName, mpReflector->getName());
                            break;
                        }
                    }
                }
            }

            if(bOK == false)
            {
                std::string msg("Error when setting texture by offset. No variable found at offset ");
                msg += std::to_string(offset) + ". Ignoring call";
                logError(msg);
            }
        }
#endif

        if(bOK)
        {
            mDirty = true;
            setTextureInternal(offset, pTexture, pSampler);
        }
    }

    void VariablesBuffer::setTexture(const std::string& name, const Texture* pTexture, const Sampler* pSampler)
    {
        size_t offset;
        mpReflector->getVariableData(name, offset);
        if(offset != ProgramReflection::kInvalidLocation)
        {
            bool bOK = true;
#if _LOG_ENABLED == 1
            if(pTexture != nullptr)
            {
                const auto pDesc = getResourceDesc(name, mpReflector.get());
                bOK = (pDesc != nullptr) && checkResourceDimension(pTexture, pDesc, name, mpReflector->getName());
            }
#endif
            if(bOK)
            {
                setTexture(offset, pTexture, pSampler);
            }
        }
    }

    void VariablesBuffer::setTextureArray(const std::string& name, const Texture* pTexture[], const Sampler* pSampler, size_t count)
    {
        size_t offset;
        const auto& pVarDesc = mpReflector->getVariableData(name, offset);

        if(pVarDesc)
        {
            if(pVarDesc->arraySize < count)
            {
                logWarning("Error when setting textures array. 'count' is larger than the array size. Ignoring out-of-bound elements");
                count = pVarDesc->arraySize;
            }

            for(uint32_t i = 0; i < count; i++)
            {
                bool bOK = true;
#if _LOG_ENABLED == 1
                if(pTexture[i] != nullptr)
                {
                    const auto& pDesc = getResourceDesc(name, mpReflector.get());
                    bOK = (pDesc) && checkResourceDimension(pTexture[i], pDesc, name, mpReflector->getName());
                }
#endif
                if(bOK)
                {
                    setTexture(offset + i * sizeof(uint64_t), pTexture[i], pSampler);
                }
            }
        }
    }

    void VariablesBuffer::setTextureInternal(size_t offset, const Texture* pTexture, const Sampler* pSampler)
    {

    }
}
