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
#include "stdafx.h"
#include "VAO.h"

namespace Falcor
{
    bool checkVaoParams(const Vao::BufferVec& vbDesc, const VertexLayout* pLayout, Buffer* pIB, ResourceFormat ibFormat)
    {
        // TODO: Check number of vertex buffers match with pLayout.
        if (pIB)
        {
            if (ibFormat != ResourceFormat::R16Uint && ibFormat != ResourceFormat::R32Uint)
            {
                logError("Invalid index buffer format (" + to_string(ibFormat) + ")");
                return false;
            }
        }

        return true;
    }

    Vao::Vao(const BufferVec& pVBs, const VertexLayout::SharedPtr& pLayout, const Buffer::SharedPtr& pIB, ResourceFormat ibFormat, Topology topology)
        : mIbFormat(ibFormat)
        , mpVBs(pVBs)
        , mpIB(pIB)
        , mpVertexLayout(pLayout)
        , mTopology(topology)
    {
    }

    Vao::SharedPtr Vao::create(Topology topology, const VertexLayout::SharedPtr& pLayout, const BufferVec& pVBs, const Buffer::SharedPtr& pIB, ResourceFormat ibFormat)
    {
        if (pLayout != nullptr)
        {
            if (checkVaoParams(pVBs, pLayout.get(), pIB.get(), ibFormat) == false)
            {
                throw std::exception("Failed to create VAO");
            }
        }

        SharedPtr pVao = SharedPtr(new Vao(pVBs, pLayout, pIB, ibFormat, topology));
        return pVao;
    }

    Vao::ElementDesc Vao::getElementIndexByLocation(uint32_t elementLocaion) const
    {
        ElementDesc desc;

        for(uint32_t bufId = 0; bufId < getVertexBuffersCount(); ++bufId)
        {
            const VertexBufferLayout* pVbLayout = mpVertexLayout->getBufferLayout(bufId).get();
            assert(pVbLayout);

            for(uint32_t i = 0; i < pVbLayout->getElementCount(); ++i)
            {
                if(pVbLayout->getElementShaderLocation(i) == elementLocaion)
                {
                    desc.vbIndex = bufId;
                    desc.elementIndex = i;
                    return desc;
                }
            }
        }
        return desc;
    }

    SCRIPT_BINDING(Vao)
    {
        pybind11::class_<Vao, Vao::SharedPtr>(m, "Vao");
    }
}
