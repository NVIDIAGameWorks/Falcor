/************************************************************************************************************************************\
|*                                                                                                                                    *|
|*     Copyright © 2017 NVIDIA Corporation.  All rights reserved.                                                                     *|
|*                                                                                                                                    *|
|*  NOTICE TO USER:                                                                                                                   *|
|*                                                                                                                                    *|
|*  This software is subject to NVIDIA ownership rights under U.S. and international Copyright laws.                                  *|
|*                                                                                                                                    *|
|*  This software and the information contained herein are PROPRIETARY and CONFIDENTIAL to NVIDIA                                     *|
|*  and are being provided solely under the terms and conditions of an NVIDIA software license agreement                              *|
|*  and / or non-disclosure agreement.  Otherwise, you have no rights to use or access this software in any manner.                   *|
|*                                                                                                                                    *|
|*  If not covered by the applicable NVIDIA software license agreement:                                                               *|
|*  NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOFTWARE FOR ANY PURPOSE.                                            *|
|*  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.                                                           *|
|*  NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,                                                                     *|
|*  INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.                       *|
|*  IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES,                               *|
|*  OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT,                         *|
|*  NEGLIGENCE OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOURCE CODE.            *|
|*                                                                                                                                    *|
|*  U.S. Government End Users.                                                                                                        *|
|*  This software is a "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT 1995),                                       *|
|*  consisting  of "commercial computer  software"  and "commercial computer software documentation"                                  *|
|*  as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government only as a commercial end item.     *|
|*  Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995),                                          *|
|*  all U.S. Government End Users acquire the software with only those rights set forth herein.                                       *|
|*                                                                                                                                    *|
|*  Any use of this software in individual and commercial software must include,                                                      *|
|*  in the user documentation and internal comments to the code,                                                                      *|
|*  the above Disclaimer (as applicable) and U.S. Government End Users Notice.                                                        *|
|*                                                                                                                                    *|
 \************************************************************************************************************************************/
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
                for (auto& it = staticMeshes.begin() ; it != staticMeshes.end() ; it++)
                {
                    if(it->size() > 1) break;
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
                for (auto& it : meshList)
                {
                    mMeshes.push_back(it);
                    data.meshCount++;

                    // If mesh is instanced or the transform has changed, start a new mesh group
                    if (it.size() > 1 || it[0]->getTransformMatrix() != transformation)
                    {
                        mBottomLevelData.push_back(data);
                        data.meshBaseIndex = (uint32_t)mMeshes.size();
                        data.meshCount = 0;
                    }
                }
                if (data.meshCount > 0)
                {
                    mBottomLevelData.push_back(data);
                }
            }
        };

        insertFunc(staticMeshes, true);
        insertFunc(dynamicMeshes, false);

        // Validate that mBottomLevelData represents a contiguous range that includes all meshes
        uint32_t baseIdx = 0;
        for (auto& it : mBottomLevelData)
        {
            assert(it.meshCount > 0);
            assert(it.meshBaseIndex + it.meshCount <= mMeshes.size());
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
        ID3D12CommandListRaytracingPrototypePtr pRtCmdList = gpDevice->getRenderContext()->getLowLevelData()->getCommandList();
        ID3D12DeviceRaytracingPrototypePtr pRtDevice = gpDevice->getApiHandle();

        auto dxrFlags = getDxrBuildFlags(mBuildFlags);

        // Create an AS for each mesh-group
        for(auto& blasData : mBottomLevelData)
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

                desc.Triangles.VertexBuffer.StartAddress = pVao->getVertexBuffer(elemDesc.vbIndex)->getGpuAddress() + pVbLayout->getElementOffset(elemDesc.elementIndex);
                desc.Triangles.VertexBuffer.StrideInBytes = pVbLayout->getStride();
                desc.Triangles.VertexCount = pMesh->getVertexCount();
                desc.Triangles.VertexFormat = getDxgiFormat(pVbLayout->getElementFormat(elemDesc.elementIndex));

                // Get the IB
                desc.Triangles.IndexBuffer = pVao->getIndexBuffer()->getGpuAddress();
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
        }
    }

    RtModel::SharedPtr RtModel::createFromFile(const char* filename, RtBuildFlags buildFlags, Model::LoadFlags flags)
    {
        Model::SharedPtr pModel = Model::createFromFile(filename, flags);
        if (!pModel) return nullptr;

        return createFromModel(*pModel, buildFlags);
    }
};