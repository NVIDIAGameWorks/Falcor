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
#include "API/ProgramVersion.h"
#include "ShaderReflectionD3D11.h"
#include <algorithm>
#include <iterator>

namespace Falcor
{
    using namespace ShaderReflection;

    void ProgramVersion::deleteApiHandle()
    {

    }

    bool validateBufferDeclaration(const BufferDesc& prevDesc, 
        const BufferDesc& currentDesc, 
        ID3D11ShaderReflectionConstantBuffer* pPrevBuffer, 
        ID3D11ShaderReflectionConstantBuffer* pCurrentBuffer, 
        std::string& log)
    {
        bool bMatch = true;
#define error_msg(msg_) std::string(msg_) + " mismatch.\n";

#define test_field(field_, msg_)                        \
            if(prevDesc.field_ != currentDesc.field_)   \
            {                                           \
                log += error_msg(msg_);                 \
                bMatch = false;                         \
            }

        test_field(bufferSlot, "buffer slot");
        test_field(variableCount, "variable count");
        test_field(sizeInBytes, "size");
        test_field(type, "Type");
#undef test_field

        VariableDescMap prevMap, currentMap;
        reflectBufferVariables(pPrevBuffer, prevMap);
        reflectBufferVariables(pCurrentBuffer, currentMap);
        if(prevMap.size() != currentMap.size())
        {
            log += error_msg("variable count");
            bMatch = false;
        }

        auto prevVar = prevMap.begin();
        auto currentVar = currentMap.begin();
        while(bMatch && (prevVar != prevMap.end()))
        {
            if(prevVar->first != currentVar->first)
            {
                log += error_msg("variable name") + ". First seen as " + prevVar->first + ", now called " + currentVar->first;
                bMatch = false;
            }
            else
            {
#define test_field(field_, msg_)                                      \
            if(prevVar->second.field_ != currentVar->second.field_)   \
            {                                                         \
                log += error_msg(prevVar->first + " " + msg_)         \
                bMatch = false;                                       \
            }

                test_field(offset, "offset");
                test_field(arraySize, "variable count");
                test_field(isRowMajor, "row major");
                test_field(type, "Type");
#undef test_field
            }

            prevVar++;
            currentVar++;
        }

        return bMatch;
    }

    static bool validateResourceDeclarations(const ShaderResourceDescMap& programMap, const ShaderResourceDescMap& shaderMap, const Shader* pShader, const std::map<std::string, const Shader*> resourceFirstShaderMap, std::string& log)
    {
        for(const auto& shaderRes : shaderMap)
        {
            // Loop over the program map
            for(const auto programRes : programMap)
            {
                if(shaderRes.first == programRes.first)
                {
                    bool bMatch = true;
                    // Name matches. Everything else should match to.
#define test_field(field_, msg_)                                      \
            if(shaderRes.second.field_ != programRes.second.field_)   \
            {                                                         \
                log += error_msg(shaderRes.first + " " + msg_)        \
                bMatch = false;                                       \
            }

                    test_field(dims, "resource dimensions");
                    test_field(retType, "return Type");
                    test_field(type, "resource Type");
                    test_field(offset, "bind point");
                    test_field(arraySize, "array size");
#undef test_field
                    if(bMatch == false)
                    {
                        const Shader* pFirstShader = resourceFirstShaderMap.at(shaderRes.first);
                        log = "Resource '" + shaderRes.first + "' declaration mismatch between " + to_string((pFirstShader->getType())) + " shader and " + to_string(pShader->getType()) + " shader.\n" + log;
                        return false;
                    }
                    break;
                }
                else if(shaderRes.second.offset == programRes.second.offset)
                {
                    // Different names, same bind point. Error
                    std::string firstName = programRes.first;
                    std::string secondName = shaderRes.first;
                    const Shader* pFirstShader = resourceFirstShaderMap.at(firstName);
                    log = "Resource bind-point mismatch. Bind point " + std::to_string(shaderRes.second.offset) + " first use in " + to_string(pFirstShader->getType()) + " shader as ' " + firstName + "'.";
                    log += "Second use in " + to_string(pShader->getType()) + " shader as '" + secondName + "'.\nNames must match.";
                    return false;
                }
            }
        }
        return true;
    }

    static ProgramVersion::SharedPtr returnError(std::string& log, const std::string& msg, const Shader* pPrevShader, const Shader* pCurShader, ProgramVersion::SharedPtr& pProgram)
    {
        // Declarations mismatch. Return error.
        const std::string prevShaderType = to_string(pPrevShader->getType());
        const std::string curShaderType = to_string(pCurShader->getType());
        log = msg + "\nMismatch between " + prevShaderType + " shader and " + curShaderType + " shader.";
        pProgram = nullptr;
        return pProgram;
    }

    static void mergeResourceMaps(ShaderResourceDescMap& programMap, ShaderResourceDescMap& shaderMap, const Shader* pShader, std::map<std::string, const Shader*>& resourceFirstShaderMap)
    {
        ShaderResourceDescMap diff;
        std::set_difference(shaderMap.begin(), shaderMap.end(), programMap.begin(), programMap.end(), std::inserter(diff, diff.begin()), shaderMap.value_comp());
        for(const auto& d : diff)
        {
            resourceFirstShaderMap[d.first] = pShader;
        }
        programMap.insert(shaderMap.begin(), shaderMap.end());
    }

    bool ProgramVersion::init(std::string& log, const std::string& name)
    {
        return true;
    }
}
