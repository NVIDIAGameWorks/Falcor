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
#include "API/VAO.h"
#include <map>

namespace Falcor
{
    using VaoElementDesc = std::vector < D3D11_INPUT_ELEMENT_DESC > ;
    struct VaoData
    {
        VaoElementDesc ElementDesc;
        std::map<ID3DBlobPtr, ID3D11InputLayoutPtr> pLayouts;
    };


    bool Vao::initialize()
    {
        VaoData* pData = new VaoData;
        mpPrivateData = pData;

        for(size_t vb = 0; vb < mpVBs.size(); vb++)
        {
            auto& pLayout = mpVBs[vb].pLayout;
            for(uint32_t elemID = 0; elemID < pLayout->getElementCount(); elemID++)
            {
                D3D11_INPUT_ELEMENT_DESC element;
                element.AlignedByteOffset = pLayout->getElementOffset(elemID);
                element.Format = getDxgiFormat(pLayout->getElementFormat(elemID));
                element.InputSlot = (uint32_t)vb;
                element.InputSlotClass = getDxInputClass(pLayout->getInputClass());
                element.InstanceDataStepRate = pLayout->getInstanceStepRate();
                const auto& SemanticName = pLayout->getElementName(elemID);

                for(uint32_t arrayIndex = 0; arrayIndex < pLayout->getElementArraySize(elemID); arrayIndex++)
                {
                    // Reallocating name for each array index simplifies the destructor
                    char* name = new char[SemanticName.size() + 1];
                    memcpy(name, SemanticName.c_str(), SemanticName.size());
                    name[SemanticName.size()] = 0;
                    element.SemanticName = name;
                    element.SemanticIndex = arrayIndex;
                    pData->ElementDesc.push_back(element);

                    element.AlignedByteOffset += getFormatBytesPerBlock(pLayout->getElementFormat(elemID));
                }
            }
        }

        return true;
    }

    Vao::~Vao()
    {
        VaoData* pData = (VaoData*)mpPrivateData;
        for(auto& a : pData->ElementDesc)
        {
            safe_delete_array(a.SemanticName);
        }

        safe_delete(pData);
    }

    VaoHandle Vao::getApiHandle() const
    {
        UNSUPPORTED_IN_D3D11("CVao doesn't have an API handle");
        return mApiHandle;
    }

    ID3D11InputLayoutPtr Vao::getInputLayout(ID3DBlob* pVsBlob) const
    {
        VaoData* pData = (VaoData*)mpPrivateData;
        const auto& it = pData->pLayouts.find(pVsBlob);
        if(it != pData->pLayouts.end())
        {
            return it->second;
        }
        else
        { 
            ID3D11InputLayoutPtr pLayout;
            d3d_call(getD3D11Device()->CreateInputLayout(pData->ElementDesc.data(), (uint32_t)pData->ElementDesc.size(), pVsBlob->GetBufferPointer(), pVsBlob->GetBufferSize(), &pLayout));
            pData->pLayouts[pVsBlob] = pLayout;
            return pLayout;
        }
    }
}
