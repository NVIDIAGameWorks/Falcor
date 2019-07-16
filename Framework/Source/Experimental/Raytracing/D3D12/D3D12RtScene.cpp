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
#include "Experimental/Raytracing/RtScene.h"
#include "API/Device.h"

namespace Falcor
{
    // TODO: Cache TLAS per hitProgCount, as some render pipelines need multiple TLAS:es with different #hit progs in same frame, currently that trigger rebuild every frame. See issue #365.
    void RtScene::createTlas(uint32_t hitProgCount)
    {
        if (mTlasHitProgCount == hitProgCount) return;
        mTlasHitProgCount = hitProgCount;

        // Early out if hit program count is zero or if scene is empty.
        if (hitProgCount == 0 || getModelCount() == 0)
        {
            mModelInstanceData.clear();
            mpTopLevelAS = nullptr;
            mTlasSrv = nullptr;
            mGeometryCount = 0;
            mInstanceCount = 0;
            mRefit = false;
            return;
        }

        // todo: move this somewhere fair.
        mRtFlags |= RtBuildFlags::AllowUpdate;

        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS dxrFlags = getDxrBuildFlags(mRtFlags);
        RenderContext* pContext = gpDevice->getRenderContext();
        std::vector<D3D12_RAYTRACING_INSTANCE_DESC> instanceDesc = createInstanceDesc(this, hitProgCount);

        // todo: improve this check - make sure things have not changed much and update was enabled last time
        bool isRefitPossible = mRefit && mpTopLevelAS && (mInstanceCount == (uint32_t)instanceDesc.size());

        mInstanceCount = (uint32_t)instanceDesc.size();

        // Create the top-level acceleration buffers
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS inputs = {};
        inputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
        inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        inputs.NumDescs = mInstanceCount;
        inputs.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;

        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO info;
        GET_COM_INTERFACE(gpDevice->getApiHandle(), ID3D12Device5, pDevice5);
        pDevice5->GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &info);

        Buffer::SharedPtr pScratchBuffer = Buffer::create(align_to(D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT, info.ScratchDataSizeInBytes), Buffer::BindFlags::UnorderedAccess, Buffer::CpuAccess::None);

        if (!isRefitPossible)
        {
            mpTopLevelAS = Buffer::create(align_to(D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT, info.ResultDataMaxSizeInBytes), Buffer::BindFlags::AccelerationStructure, Buffer::CpuAccess::None);
        }
        else
        {
            pContext->uavBarrier(mpTopLevelAS.get());
        }

        Buffer::SharedPtr pInstanceData = Buffer::create(mInstanceCount * sizeof(D3D12_RAYTRACING_INSTANCE_DESC), Buffer::BindFlags::None, Buffer::CpuAccess::None, instanceDesc.data());
        assert((mInstanceCount != 0) && pInstanceData->getApiHandle() && mpTopLevelAS->getApiHandle() && pScratchBuffer->getApiHandle());

        // Create the TLAS
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC asDesc = {};
        asDesc.Inputs = inputs;
        asDesc.Inputs.InstanceDescs = pInstanceData->getGpuAddress();
        asDesc.DestAccelerationStructureData = mpTopLevelAS->getGpuAddress();
        asDesc.ScratchAccelerationStructureData = pScratchBuffer->getGpuAddress();

        if (isRefitPossible)
        {
            asDesc.Inputs.Flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
            asDesc.SourceAccelerationStructureData = asDesc.DestAccelerationStructureData;
        }

        GET_COM_INTERFACE(pContext->getLowLevelData()->getCommandList(), ID3D12GraphicsCommandList4, pList4);
        pContext->resourceBarrier(pInstanceData.get(), Resource::State::NonPixelShader);
        pList4->BuildRaytracingAccelerationStructure(&asDesc, 0, nullptr);
        pContext->uavBarrier(mpTopLevelAS.get());

        // Create the SRV
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc = {};
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.RaytracingAccelerationStructure.Location = mpTopLevelAS->getGpuAddress();

        DescriptorSet::Layout layout;
        layout.addRange(DescriptorSet::Type::TextureSrv, 0, 1);
        DescriptorSet::SharedPtr pSet = DescriptorSet::create(gpDevice->getCpuDescriptorPool(), layout);
        assert(pSet);
        gpDevice->getApiHandle()->CreateShaderResourceView(nullptr, &srvDesc, pSet->getCpuHandle(0));

        ResourceWeakPtr pWeak = mpTopLevelAS;
        mTlasSrv = std::make_shared<ShaderResourceView>(pWeak, pSet, 0, 1, 0, 1);

        mRefit = false;
    }
}
