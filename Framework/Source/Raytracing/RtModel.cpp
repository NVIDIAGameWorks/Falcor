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
#include "RtModel.h"
#include "API/Device.h"
#include "API/RenderContext.h"
#include "API/LowLevel/LowLevelContextData.h"
#include "API/VAO.h"

namespace Falcor
{
    RtModel::RtModel(const Model& model, RtBuildFlags buildFlags) : mBuildFlags(buildFlags), Model(model)
    {
    }

    void RtModel::createBottomLevelData()
    {
        // The logic works as follows:
        //  - Static meshes come before dynamic meshes
        //  - Meshes that have a single instance and have the same matrix are contiguous

        // Store the meshes into lists, grouped based on their transformation matrix
        std::list<MeshInstanceList> staticMeshes;
        std::list<MeshInstanceList> dynamicMeshes;
        for (const auto& instanceList : mMeshes)
        {
            assert(instanceList.size() > 0);
            if (instanceList[0]->getObject()->hasBones())
            {
                dynamicMeshes.push_back(instanceList);
                continue;
            }

            // If we have multiple instances, push it to the end of the list
            if (instanceList.size() > 1)
            {
                staticMeshes.push_back(instanceList);
            }
            else
            {
                bool handled = false;
                // Find the insert location. Should have a single instance and the matrix should match
                for (auto& it = staticMeshes.begin(); it != staticMeshes.end(); it++)
                {
                    if (it->size() > 1) break;
                    if ((*it)[0]->getTransformMatrix() == instanceList[0]->getTransformMatrix())
                    {
                        handled = true;
                        staticMeshes.insert(it, instanceList);
                        break;
                    }
                }
                if (!handled) staticMeshes.push_front(instanceList);
            }
        }

        // Copy the lists into the vectors. Static meshes first
        size_t count = mMeshes.size();
        assert(staticMeshes.size() + dynamicMeshes.size() == count);
        mMeshes.clear();
        mMeshes.reserve(count);

        auto insertFunc = [this](const auto& meshList, bool isStatic)
        {
            if (meshList.size())
            {
                BottomLevelData data;
                data.isStatic = isStatic;
                mat4 transformation = (*meshList.begin())[0]->getTransformMatrix();
                data.meshCount = 0;
                data.meshBaseIndex = (uint32_t)mMeshes.size();
                bool instanced = false;
                for (auto& it : meshList)
                {
                    // The logic works as follows:
                    //  - Dynamic meshes all go in the same group, as they don't have individual mesh-instance transforms. The skinning code computes their vertices.
                    //  - Static non-instanced meshes are grouped per instance transform
                    //  - Static instanced meshes all go in invididual groups, and come after non-instanced meshes in the list.

                    if (isStatic)
                    {
                        // Validate that instanced meshes are last
                        if (it.size() > 1) instanced = true;
                        else assert(!instanced);

                        // If mesh is instanced or the transform has changed, start a new mesh group
                        if (it.size() > 1 || it[0]->getTransformMatrix() != transformation)
                        {
                            if (data.meshCount > 0) // The first mesh could be instanced...
                            {
                                mBottomLevelData.push_back(data);
                            }

                            transformation = it[0]->getTransformMatrix();
                            data.meshBaseIndex = (uint32_t)mMeshes.size();
                            data.meshCount = 0;
                        }
                    }
                    else
                    {
                        // We don't handle dynamic instanced meshes
                        assert(it.size() == 1);
                    }

                    mMeshes.push_back(it);
                    data.meshCount++;
                }
                mBottomLevelData.push_back(data);
            }
        };

        insertFunc(staticMeshes, true);
        insertFunc(dynamicMeshes, false);

        // Validate that mBottomLevelData represents a contiguous range that includes all meshes, and that grouped meshes are non-instanced
        uint32_t baseIdx = 0;
        for (auto& it : mBottomLevelData)
        {
            assert(it.meshCount > 0);
            assert(it.meshBaseIndex + it.meshCount <= mMeshes.size());
            for (uint32_t idx = it.meshBaseIndex; idx < it.meshBaseIndex + it.meshCount; idx++)
            {
                assert(it.meshCount == 1 || mMeshes[idx].size() == 1);
            }
            assert(it.meshBaseIndex == baseIdx);
            baseIdx += it.meshCount;
        }
        assert(baseIdx == mMeshes.size());
    }

    RtModel::SharedPtr RtModel::createFromModel(const Model& model, RtBuildFlags buildFlags)
    {
        SharedPtr pRtModel = SharedPtr(new RtModel(model, buildFlags));
        pRtModel->createBottomLevelData();

        // If model is skinned, postpone build until after animate() so we have valid skinned vertices
        if (!pRtModel->hasBones())
        {
            pRtModel->buildAccelerationStructure();
        }
        return pRtModel;
    }

    bool RtModel::update()
    {
        // Call base class to compute skinned vertices
        if (Model::update())
        {
            buildAccelerationStructure();
            return true;
        }
        return false;
    }

    void RtModel::buildAccelerationStructure()
    {
        RenderContext* pContext = gpDevice->getRenderContext().get();
        ID3D12CommandListRaytracingPrototypePtr pRtCmdList = pContext->getLowLevelData()->getCommandList();
        ID3D12DeviceRaytracingPrototypePtr pRtDevice = gpDevice->getApiHandle();

        auto dxrFlags = getDxrBuildFlags(mBuildFlags);

        // Create an AS for each mesh-group
        for (auto& blasData : mBottomLevelData)
        {
            std::vector<D3D12_RAYTRACING_GEOMETRY_DESC> geomDesc(blasData.meshCount);
            for (size_t meshIndex = blasData.meshBaseIndex; meshIndex < blasData.meshBaseIndex + blasData.meshCount; meshIndex++)   // PETRIK: Fixed loop to loop over range [meshIndex...meshIndex + meshCount)
            {
                assert(meshIndex < mMeshes.size());
                const Mesh* pMesh = getMesh((uint32_t)meshIndex).get();

                D3D12_RAYTRACING_GEOMETRY_DESC& desc = geomDesc[meshIndex - blasData.meshBaseIndex];
                desc.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;

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
            D3D12_GET_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO_DESC prebuildDesc;
            prebuildDesc.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
            prebuildDesc.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_NONE;
            prebuildDesc.NumDescs = (uint32_t)geomDesc.size();
            prebuildDesc.pGeometryDescs = geomDesc.data();
            prebuildDesc.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;

            D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO info;
            pRtDevice->GetRaytracingAccelerationStructurePrebuildInfo(&prebuildDesc, &info);

            Buffer::SharedPtr pScratchBuffer = Buffer::create(info.ScratchDataSizeInBytes, Buffer::BindFlags::UnorderedAccess, Buffer::CpuAccess::None);
            blasData.pBlas = Buffer::create(info.ResultDataMaxSizeInBytes, Buffer::BindFlags::AccelerationStructure, Buffer::CpuAccess::None);

            // Build the AS
            D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC asDesc = {};
            asDesc.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
            asDesc.NumDescs = prebuildDesc.NumDescs;
            asDesc.Flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_NONE;
            asDesc.pGeometryDescs = prebuildDesc.pGeometryDescs;
            asDesc.DestAccelerationStructureData.StartAddress = blasData.pBlas->getGpuAddress();
            asDesc.DestAccelerationStructureData.SizeInBytes = info.ResultDataMaxSizeInBytes;
            asDesc.ScratchAccelerationStructureData.StartAddress = pScratchBuffer->getGpuAddress();
            asDesc.ScratchAccelerationStructureData.SizeInBytes = info.ScratchDataSizeInBytes;
            asDesc.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;

            pRtCmdList->BuildRaytracingAccelerationStructure(&asDesc);

            // Insert a UAV barrier
            pContext->uavBarrier(blasData.pBlas.get());
        }
    }

    RtModel::SharedPtr RtModel::createFromFile(const char* filename, RtBuildFlags buildFlags, Model::LoadFlags flags)
    {
        Model::SharedPtr pModel = Model::createFromFile(filename, flags);
        if (!pModel) return nullptr;

        return createFromModel(*pModel, buildFlags);
    }
};