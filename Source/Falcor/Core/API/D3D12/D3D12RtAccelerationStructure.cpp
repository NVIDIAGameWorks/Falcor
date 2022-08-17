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
#include "D3D12RtAccelerationStructure.h"
#include "Core/API/RtAccelerationStructure.h"
#include "Core/API/Device.h"
#include "Core/API/D3D12/D3D12API.h"

namespace Falcor
{
    static_assert(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BYTE_ALIGNMENT == kAccelerationStructureByteAlignment);

    static_assert(sizeof(RtInstanceDesc) == sizeof(D3D12_RAYTRACING_INSTANCE_DESC));

    // Static asserts to ensure RtGeometryInstanceFlags values are consistent with D3D12 definitions.
    static_assert((uint32_t)RtGeometryInstanceFlags::TriangleFacingCullDisable == (uint32_t)D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_CULL_DISABLE);
    static_assert((uint32_t)RtGeometryInstanceFlags::TriangleFrontCounterClockwise == (uint32_t)D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_FRONT_COUNTERCLOCKWISE);
    static_assert((uint32_t)RtGeometryInstanceFlags::ForceOpaque == (uint32_t)D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_OPAQUE);
    static_assert((uint32_t)RtGeometryInstanceFlags::NoOpaque == (uint32_t)D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_NON_OPAQUE);

    // Static asserts to ensure RtAccelerationStructureBuildFlags values are consistent with D3D12 definitions.
    static_assert((uint32_t)RtAccelerationStructureBuildFlags::None == (uint32_t)D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_NONE);
    static_assert((uint32_t)RtAccelerationStructureBuildFlags::AllowUpdate == (uint32_t)D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE);
    static_assert((uint32_t)RtAccelerationStructureBuildFlags::AllowCompaction == (uint32_t)D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION);
    static_assert((uint32_t)RtAccelerationStructureBuildFlags::PreferFastTrace == (uint32_t)D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE);
    static_assert((uint32_t)RtAccelerationStructureBuildFlags::PreferFastBuild == (uint32_t)D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD);
    static_assert((uint32_t)RtAccelerationStructureBuildFlags::MinimizeMemory == (uint32_t)D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_MINIMIZE_MEMORY);
    static_assert((uint32_t)RtAccelerationStructureBuildFlags::PerformUpdate == (uint32_t)D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE);

    // Static asserts to ensure RtGeometryFlags values are consistent with D3D12 definitions.
    static_assert((uint32_t)RtGeometryFlags::None == (uint32_t)D3D12_RAYTRACING_GEOMETRY_FLAG_NONE);
    static_assert((uint32_t)RtGeometryFlags::Opaque == (uint32_t)D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE);
    static_assert((uint32_t)RtGeometryFlags::NoDuplicateAnyHitInvocation == (uint32_t)D3D12_RAYTRACING_GEOMETRY_FLAG_NO_DUPLICATE_ANYHIT_INVOCATION);

    RtAccelerationStructure::RtAccelerationStructure(const RtAccelerationStructure::Desc& desc)
        : mDesc(desc)
    {
    }

    RtAccelerationStructure::~RtAccelerationStructure() = default;

    ShaderResourceView::SharedPtr RtAccelerationStructure::getShaderResourceView()
    {
        if (!mApiHandle)
        {
            // `ShaderResourceView::createViewForAccelerationStructure` current does not support
            // creating a view into the middle of a buffer.
            if (mDesc.getOffset() != 0)
            {
                throw RuntimeError("RtAccelerationStructure::getShaderResourceView does not support acceleration structures with a non-zero offset.");
            }

            mApiHandle = ShaderResourceView::createViewForAccelerationStructure(mDesc.getBuffer());
        }
        return mApiHandle;
    }

    bool RtAccelerationStructure::apiInit()
    {
        return true;
    }

    AccelerationStructureHandle RtAccelerationStructure::getApiHandle() const
    {
        return const_cast<RtAccelerationStructure*>(this)->getShaderResourceView();
    }

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS& D3D12AccelerationStructureBuildInputsTranslator::translate(const RtAccelerationStructureBuildInputs& buildInputs)
    {
        if (buildInputs.geometryDescs)
        {
            mGeomDescs.resize(buildInputs.descCount);
            for (size_t i = 0; i < mGeomDescs.size(); i++)
            {
                auto& inputGeomDesc = buildInputs.geometryDescs[i];
                mGeomDescs[i].Flags = translateGeometryFlags(inputGeomDesc.flags);

                switch (inputGeomDesc.type)
                {
                case RtGeometryType::Triangles:
                    mGeomDescs[i].Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
                    mGeomDescs[i].Triangles.IndexBuffer = inputGeomDesc.content.triangles.indexData;
                    mGeomDescs[i].Triangles.IndexCount = inputGeomDesc.content.triangles.indexCount;
                    mGeomDescs[i].Triangles.IndexFormat = getDxgiFormat(inputGeomDesc.content.triangles.indexFormat);
                    mGeomDescs[i].Triangles.Transform3x4 = inputGeomDesc.content.triangles.transform3x4;
                    mGeomDescs[i].Triangles.VertexBuffer.StartAddress = inputGeomDesc.content.triangles.vertexData;
                    mGeomDescs[i].Triangles.VertexBuffer.StrideInBytes = inputGeomDesc.content.triangles.vertexStride;
                    mGeomDescs[i].Triangles.VertexCount = inputGeomDesc.content.triangles.vertexCount;
                    mGeomDescs[i].Triangles.VertexFormat = getDxgiFormat(inputGeomDesc.content.triangles.vertexFormat);
                    break;

                case RtGeometryType::ProcedurePrimitives:
                    mGeomDescs[i].Type = D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS;
                    mGeomDescs[i].AABBs.AABBCount = inputGeomDesc.content.proceduralAABBs.count;
                    mGeomDescs[i].AABBs.AABBs.StartAddress = inputGeomDesc.content.proceduralAABBs.data;
                    mGeomDescs[i].AABBs.AABBs.StrideInBytes = inputGeomDesc.content.proceduralAABBs.stride;
                    break;

                default:
                    FALCOR_UNREACHABLE();
                }
            }
        }

        mDesc.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        mDesc.NumDescs = buildInputs.descCount;

        switch (buildInputs.kind)
        {
        case RtAccelerationStructureKind::TopLevel:
            mDesc.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
            mDesc.InstanceDescs = buildInputs.instanceDescs;
            break;

        case RtAccelerationStructureKind::BottomLevel:
            mDesc.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
            mDesc.pGeometryDescs = &mGeomDescs[0];
            break;
        }

        mDesc.Flags = (D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS)buildInputs.flags;

        return mDesc;
    }

    RtAccelerationStructurePrebuildInfo RtAccelerationStructure::getPrebuildInfo(const RtAccelerationStructureBuildInputs& inputs)
    {
        static_assert(sizeof(RtAccelerationStructurePrebuildInfo) == sizeof(D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO));

        D3D12AccelerationStructureBuildInputsTranslator translator;
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS& d3dInputs = translator.translate(inputs);
        FALCOR_GET_COM_INTERFACE(gpDevice->getApiHandle(), ID3D12Device5, pDevice5);
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO d3dPrebuildInfo;
        pDevice5->GetRaytracingAccelerationStructurePrebuildInfo(&d3dInputs, &d3dPrebuildInfo);

        RtAccelerationStructurePrebuildInfo result = {};
        result.resultDataMaxSize = d3dPrebuildInfo.ResultDataMaxSizeInBytes;
        result.scratchDataSize = d3dPrebuildInfo.ScratchDataSizeInBytes;
        result.updateScratchDataSize = d3dPrebuildInfo.UpdateScratchDataSizeInBytes;
        return result;
    }

}
