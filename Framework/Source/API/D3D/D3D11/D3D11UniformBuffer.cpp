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
#include "API/ConstantBuffer.h"
#include "glm/glm.hpp"
#include "API/Buffer.h"
#include "API/ProgramVersion.h"
#include "ShaderReflectionD3D11.h"
#include "API/Texture.h"
#include "utils/StringUtils.h"
#include "API/Sampler.h"

namespace Falcor
{
    struct AssignedResources
    {
        std::map<uint32_t, ID3D11ShaderResourceViewPtr> srv;
        std::map<uint32_t, ID3D11SamplerStatePtr> sampler;
    };

    std::map<const ProgramVersion*, AssignedResources> gAssignedResourcesMap;

    using namespace ShaderReflection;
    ConstantBuffer::~ConstantBuffer() = default;

    bool ConstantBuffer::apiInit(bool isConstantBuffer)
    {
        bool bufferFound = false;
        // We don't necessarily have this buffer. This can happen if the buffer only contains textures.
        assert(isConstantBuffer);
        for(uint32_t i = 0; i < (uint32_t)ShaderType::Count; i++)
        {
            const Shader* pShader = pProgram->getShader((ShaderType)i);
            if(pShader)
            {
                ID3D11ShaderReflectionPtr pReflection = pShader->getReflectionInterface();

                // Check if the buffer is available in this shader
                ID3D11ShaderReflectionConstantBuffer* pCB = pReflection->GetConstantBufferByName(bufferName.c_str());
                D3D11_SHADER_BUFFER_DESC bufferDesc;
                if(pCB->GetDesc(&bufferDesc) == S_OK)
                {
                    mSize = bufferDesc.Size;
                    // Found the buffer. Reflect it's variables
                    reflectBufferVariables(pCB, mVariables);
                    bufferFound = true;
                    break;
                }
            }
        }

        // Initialize resources
        if(gAssignedResourcesMap.find(pProgram) == gAssignedResourcesMap.end())
        {
            gAssignedResourcesMap[pProgram] = AssignedResources(); // First time we encountered the program
        }

        mAssignedResourcesMap = &gAssignedResourcesMap[pProgram].srv;
        mAssignedSamplersMap = &gAssignedResourcesMap[pProgram].sampler;

        // Reflect the resources in the entire program
        for(uint32_t i = 0; i < (uint32_t)ShaderType::Count; i++)
        {
            const Shader* pShader = pProgram->getShader((ShaderType)i);
            if(pShader)
            {
                reflectResources(pShader->getReflectionInterface(), mResources);
            }
        }

        // Merge the resources with the variables
        for(const auto& resIt : mResources)
        {
            const auto& res = resIt.second;
            VariableDesc var;
            var.arraySize = res.arraySize;
            var.arrayStride = 1;
            var.isRowMajor = false;
            var.offset = res.offset;
            var.type = VariableDesc::Type::Resource;
            mVariables[resIt.first] = var;
        }

        if(mVariables.size() == 0)
        {
            assert(bufferFound == false);
            logError("ConstantBuffer::ApiInit() - Buffer '" + bufferName + "' not found in program " + pProgram->getName() + " and no resources were declared in the program.");
            return false;
        }
        else if(bufferFound == false)
        {
            Logger::log(Logger::Level::Warning, "ConstantBuffer::ApiInit() - Buffer '" + bufferName + "' not found in program " + pProgram->getName() + ".\nThis can happen if the buffer only contains textures.");
        }

        return true;
    }

    void ConstantBuffer::setTextureInternal(size_t offset, const Texture* pTexture, const Sampler* pSampler)
    {
#ifdef _LOG_ENABLED
        if(pSampler)
        {
            // Make sure that the names for the texture and resource match
            const ShaderResourceDesc* pTexDesc = nullptr;

            bool bOK = false;
            std::string error;
            for(const auto& res : mResources)
            {
                const auto& desc = res.second;
                if(desc.type == ShaderResourceDesc::ResourceType::Texture)
                {
                    if((offset >= desc.offset) && (offset < (desc.offset + desc.arraySize)))
                    {
                        static const std::string suffix(".Texture");
                        std::string varName = res.first;
                        if(hasSuffix(varName, suffix))
                        {
                            varName.replace(varName.find(suffix), suffix.size(), ".Sampler");
                            const auto& samplerDesc = mResources.find(varName);
                            if((samplerDesc == mResources.end()) || (samplerDesc->second.type != ShaderResourceDesc::ResourceType::Sampler))
                            {
                                error = "Can't find matching sampler to texture " + res.first;
                            }
                            else if(samplerDesc->second.offset != offset)
                            {
                                varName.erase(varName.find(".Sampler"));
                                error = "The sampler state index (" + std::to_string(samplerDesc->second.offset) + ") is different than resource index (" + std::to_string(offset) + ") for texture " + varName + ".\n";
                                error += "This can happen if you defined some textures or samplers without using the Tex1D/2D* shadre macros, or if you manually set incorrect registers. Inconvenient, I know, but you'll need to fix it.";
                            }
                            else
                            {
                                bOK = true;
                            }
                            break;
                        }
                        else
                        {
                            error = "Can't find matching sampler to texture " + varName + ". It appears you are not using STexture from 'HlslCommon.h' to define your HLSL resources.";
                        }

                        break;
                    }
                }
            }
            if(bOK == false)
            {
                logError("Error when trying to bind sampler to constant buffer " + mName + ".\n" + error);
                return;
            }            
        }
#endif
        (*mAssignedResourcesMap)[(uint32_t)offset] = pTexture ? pTexture->getShaderResourceView() : nullptr;
        (*mAssignedSamplersMap)[(uint32_t)offset] = pSampler ? pSampler->getApiHandle() : nullptr;
    }
}
