/***************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
#include "Experimental/Raytracing/RtModel.h"
#include "API/Device.h"
#include "API/RenderContext.h"
#include "API/LowLevel/LowLevelContextData.h"
#include "API/VAO.h"

namespace Falcor
{
    void RtModel::buildAccelerationStructure()
    {
        RenderContext* pContext = gpDevice->getRenderContext();

        auto dxrFlags = getDxrBuildFlags(mBuildFlags);

        // Create an AS for each mesh-group
        for (auto& blasData : mBottomLevelData)
        {
            std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> geomDesc(blasData.meshCount);
            for (size_t meshIndex = blasData.meshBaseIndex; meshIndex < blasData.meshBaseIndex + blasData.meshCount; meshIndex++)
            {
                assert(meshIndex < mMeshes.size());
                const Mesh* pMesh = getMesh((uint32_t)meshIndex).get();

                D3D12_RAYTRACING_GEOMETRY_DESC& desc = geomDesc[meshIndex - blasData.meshBaseIndex];
                desc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
                desc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_NONE;
                desc.Triangles.Transform3x4 = 0;

                // Get the position VB
                const Vao* pVao = getMeshVao(pMesh).get();
                const auto& elemDesc = pVao->getElementIndexByLocation(VERTEX_POSITION_LOC);
                const auto& pVbLayout = pVao->getVertexLayout()->getBufferLayout(elemDesc.vbIndex);

                const Buffer* pVB = pVao->getVertexBuffer(elemDesc.vbIndex).get();
                pContext->resourceBarrier(pVB, Resource::State::NonPixelShader);
                desc.Triangles.VertexBuffer.StartAddress = pVB->getGpuAddress() + pVbLayout->getElementOffset(elemDesc.elementIndex);
                desc.Triangles.VertexBuffer.StrideInBytes = pVbLayout->getStride();
                desc.Triangles.VertexCount = pMesh->getVertexCount();
                desc.Triangles.VertexFormat = getDxgiFormat(pVbLayout->getElementFormat(elemDesc.elementIndex));

                // Get the IB
                const Buffer* pIB = pVao->getIndexBuffer().get();
                pContext->resourceBarrier(pIB, Resource::State::NonPixelShader);
                desc.Triangles.IndexBuffer = pIB->getGpuAddress();
                desc.Triangles.IndexCount = pMesh->getIndexCount();
                desc.Triangles.IndexFormat = getDxgiFormat(pVao->getIndexBufferFormat());

                // If this is an opaque mesh, set the opaque flag
                if (pMesh->getMaterial()->getAlphaMode() == AlphaModeOpaque)
                {
                    desc.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
                }
            }

            // Create the acceleration and aux buffers
            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs = {};
            inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
            inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_NONE;
            inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
            inputs.NumDescs = (uint32_t)geomDesc.size();
            inputs.pGeometryDescs = geomDesc.data();

            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO info;
            GET_COM_INTERFACE(gpDevice->getApiHandle(), ID3D12Device5, pDevice5);
            pDevice5->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &info);

            Buffer::SharedPtr pScratchBuffer = Buffer::create(info.ScratchDataSizeInBytes, Buffer::BindFlags::UnorderedAccess, Buffer::CpuAccess::None);
            blasData.pBlas = Buffer::create(info.ResultDataMaxSizeInBytes, Buffer::BindFlags::AccelerationStructure, Buffer::CpuAccess::None);

            // Build the AS
            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC asDesc = {};
            asDesc.Inputs = inputs;
            asDesc.DestAccelerationStructureData = blasData.pBlas->getGpuAddress();
            asDesc.ScratchAccelerationStructureData = pScratchBuffer->getGpuAddress();

            GET_COM_INTERFACE(pContext->getLowLevelData()->getCommandList(), ID3D12GraphicsCommandList4, pList4);
            pList4->BuildRaytracingAccelerationStructure(&asDesc, 0, nullptr);

            // Insert a UAV barrier
            pContext->uavBarrier(blasData.pBlas.get());
        }
    }
};
