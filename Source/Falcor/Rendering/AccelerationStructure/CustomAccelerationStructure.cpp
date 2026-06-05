/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
#include "CustomAccelerationStructure.h"
#include "Core/Error.h"
#include "Core/API/RenderContext.h"
#include "Utils/Logger.h"
#include "Utils/Math/Common.h"
#include "Utils/Scripting/ScriptBindings.h"
#include <fstd/bit.h> // TODO C++20: Replace with <bit>

namespace Falcor
{
namespace
{
std::string kAABBClearShaderFile = "Rendering/AccelerationStructure/ClearAABBs.cs.slang";
}

CustomAccelerationStructure::CustomAccelerationStructure(
    ref<Device> pDevice,
    const uint64_t aabbCount,
    const uint64_t aabbGpuAddress,
    const BuildMode buildMode,
    const UpdateMode updateMode
)
{
    mpDevice = pDevice;
    if (!mpDevice->isFeatureSupported(Device::SupportedFeatures::Raytracing))
    {
        throw std::exception("Raytracing is not supported by the current device");
    }

    mBuildMode = buildMode;
    mUpdateMode = updateMode;

    const std::vector count = {aabbCount};
    const std::vector gpuAddress = {aabbGpuAddress};

    createAccelerationStructure(count, gpuAddress);
}

CustomAccelerationStructure::CustomAccelerationStructure(
    ref<Device> pDevice,
    const std::vector<uint64_t>& aabbCount,
    const std::vector<uint64_t>& aabbGpuAddress,
    const BuildMode buildMode,
    const UpdateMode updateMode
)
{
    mpDevice = pDevice;
    if (!mpDevice->isFeatureSupported(Device::SupportedFeatures::Raytracing))
    {
        throw std::exception("Raytracing is not supported by the current device");
    }

    mBuildMode = buildMode;
    mUpdateMode = updateMode;

    createAccelerationStructure(aabbCount, aabbGpuAddress);
}

CustomAccelerationStructure::~CustomAccelerationStructure()
{
    clearData();
}

void CustomAccelerationStructure::update(RenderContext* pRenderContext)
{
    const std::vector<uint64_t> count = {0u};
    buildAccelerationStructure(pRenderContext, count, false);
}

void CustomAccelerationStructure::update(RenderContext* pRenderContext, const uint64_t aabbCount)
{
    const std::vector<uint64_t> count = {aabbCount};
    buildAccelerationStructure(pRenderContext, count, true);
}

void CustomAccelerationStructure::update(RenderContext* pRenderContext, const std::vector<uint64_t>& aabbCount)
{
    buildAccelerationStructure(pRenderContext, aabbCount, true);
}

void CustomAccelerationStructure::bindTlas(const ShaderVar& rootVar, std::string shaderName)
{
    rootVar[shaderName].setAccelerationStructure(mTlas.pTlasObject);
}

void CustomAccelerationStructure::createAccelerationStructure(
    const std::vector<uint64_t>& aabbCount,
    const std::vector<uint64_t>& aabbGpuAddress
)
{
    clearData();

    FALCOR_ASSERT(aabbCount.size() == aabbGpuAddress.size());

    mNumberBlas = aabbCount.size();

    createBottomLevelAS(aabbCount, aabbGpuAddress);
    createTopLevelAS();
    mAccelerationStructureWasBuild = false;
}

void CustomAccelerationStructure::clearData()
{
    // clear all previous data
    for (uint i = 0; i < mBlas.size(); i++)
    {
        mBlas[i].reset();
        mBlasObjects[i].reset();
    }

    mBlas.clear();
    mBlasData.clear();
    mBlasObjects.clear();
    mInstanceDesc.clear();
    mTlasScratch.reset();
    mTlas.pInstanceDescs.reset();
    mTlas.pTlas.reset();
    mTlas.pTlasObject.reset();
}

void CustomAccelerationStructure::createBottomLevelAS(const std::vector<uint64_t> aabbCount, const std::vector<uint64_t> aabbGpuAddress)
{
    // Create Number of desired blas and reset max size
    mBlasData.resize(mNumberBlas);
    mBlas.resize(mNumberBlas);
    mBlasObjects.resize(mNumberBlas);
    mBlasScratchMaxSize = 0;

    FALCOR_ASSERT(aabbCount.size() >= mNumberBlas);
    FALCOR_ASSERT(aabbGpuAddress.size() >= mNumberBlas);

    // Prebuild
    for (size_t i = 0; i < mNumberBlas; i++)
    {
        auto& blas = mBlasData[i];

        // Create geometry description
        auto& desc = blas.geomDescs;
        desc.type = RtGeometryType::ProcedurePrimitives;
        desc.flags = RtGeometryFlags::NoDuplicateAnyHitInvocation; //< Important! So that photons are not collected multiple times

        desc.content.proceduralAABBs.count = aabbCount[i];
        desc.content.proceduralAABBs.data = aabbGpuAddress[i];
        desc.content.proceduralAABBs.stride = sizeof(AABB);

        // Create input for blas
        auto& inputs = blas.buildInputs;
        inputs.kind = RtAccelerationStructureKind::BottomLevel;
        inputs.descCount = 1;
        inputs.geometryDescs = &desc;
        inputs.flags = RtAccelerationStructureBuildFlags::None;

        // Build option flags
        if (mBuildMode == BuildMode::FastBuild)
            inputs.flags |= RtAccelerationStructureBuildFlags::PreferFastBuild;
        else if (mBuildMode == BuildMode::FastTrace)
            inputs.flags |= RtAccelerationStructureBuildFlags::PreferFastTrace;

        if (mUpdateMode == UpdateMode::BLASOnly || (mUpdateMode == UpdateMode::All))
            inputs.flags |= RtAccelerationStructureBuildFlags::AllowUpdate;

        // Get prebuild Info
        blas.prebuildInfo = RtAccelerationStructure::getPrebuildInfo(mpDevice.get(), inputs);

        // Figure out the padded allocation sizes to have proper alignment.
        FALCOR_ASSERT(blas.prebuildInfo.resultDataMaxSize > 0);
        blas.blasByteSize = align_to((uint64_t)kAccelerationStructureByteAlignment, blas.prebuildInfo.resultDataMaxSize);

        uint64_t scratchByteSize = std::max(blas.prebuildInfo.scratchDataSize, blas.prebuildInfo.updateScratchDataSize);
        blas.scratchByteSize = align_to((uint64_t)kAccelerationStructureByteAlignment, scratchByteSize);

        mBlasScratchMaxSize = std::max(blas.scratchByteSize, mBlasScratchMaxSize);
    }

    // Create the scratch and blas buffers
    mBlasScratch = mpDevice->createBuffer(mBlasScratchMaxSize, ResourceBindFlags::UnorderedAccess);
    mBlasScratch->setName("CustomAS::BlasScratch");

    for (uint i = 0; i < mNumberBlas; i++)
    {
        mBlasData[i].blasByteSize = align_to((uint64_t)kAccelerationStructureByteAlignment, mBlasData[i].blasByteSize);
        mBlas[i] = mpDevice->createBuffer(mBlasData[i].blasByteSize, ResourceBindFlags::AccelerationStructure); //MemoryType::DeviceLocal
        mBlas[i]->setName("CustomAS::BlasBuffer" + std::to_string(i));

        // Create API object
        RtAccelerationStructure::Desc blasDesc = {};
        blasDesc.setBuffer(mBlas[i], 0u, mBlasData[i].blasByteSize);
        blasDesc.setKind(RtAccelerationStructureKind::BottomLevel);
        mBlasObjects[i] = RtAccelerationStructure::create(mpDevice, blasDesc);
    }
}

void CustomAccelerationStructure::createTopLevelAS()
{
    mInstanceDesc.clear(); // clear to be sure that it is empty

    // fill the instance description if empty
    for (int i = 0; i < mNumberBlas; i++)
    {
        RtInstanceDesc desc = {};
        desc.accelerationStructure = mBlas[i]->getGpuAddress();
        desc.flags = RtGeometryInstanceFlags::None;
        desc.instanceID = i;
        desc.instanceMask = mNumberBlas < 8 ? 1 << i : 0xFF; // For up to 8 they are instanced seperatly
        desc.instanceContributionToHitGroupIndex = 0;

        // Create a identity matrix for the transform and copy it to the instance desc
        float4x4 identityMat = float4x4::identity();
        std::memcpy(desc.transform, &identityMat, sizeof(desc.transform)); // Copy transform
        mInstanceDesc.push_back(desc);
    }

    auto& inputs = mTlas.buildInputs;
    inputs = {};
    inputs.kind = RtAccelerationStructureKind::TopLevel;
    inputs.descCount = (uint32_t)mNumberBlas;
    inputs.flags = RtAccelerationStructureBuildFlags::None;

    if (mBuildMode == BuildMode::FastBuild)
        inputs.flags |= RtAccelerationStructureBuildFlags::PreferFastBuild;
    else if (mBuildMode == BuildMode::FastTrace)
        inputs.flags |= RtAccelerationStructureBuildFlags::PreferFastTrace;

    if (mUpdateMode == UpdateMode::TLASOnly || (mUpdateMode == UpdateMode::All))
        inputs.flags |= RtAccelerationStructureBuildFlags::AllowUpdate;

    // Prebuild
    mTlasPrebuildInfo = RtAccelerationStructure::getPrebuildInfo(mpDevice.get(), inputs);
    mTlasScratch = mpDevice->createBuffer(mTlasPrebuildInfo.scratchDataSize, ResourceBindFlags::UnorderedAccess, MemoryType::DeviceLocal);
    mTlasScratch->setName("CustomAS::TLAS_Scratch");

    // Create buffers for the TLAS
    mTlas.pTlas = mpDevice->createBuffer(mTlasPrebuildInfo.resultDataMaxSize, ResourceBindFlags::AccelerationStructure, MemoryType::DeviceLocal);
    mTlas.pTlas->setName("CustomAS::TLAS");
    mTlas.pInstanceDescs = mpDevice->createBuffer((uint32_t)mInstanceDesc.size() * sizeof(RtInstanceDesc), ResourceBindFlags::None, MemoryType::Upload, mInstanceDesc.data());
    mTlas.pInstanceDescs->setName("CustomAS::TLAS_Instance_Description");

    // Create API object
    RtAccelerationStructure::Desc tlasDesc = {};
    tlasDesc.setBuffer(mTlas.pTlas, 0u, mTlasPrebuildInfo.resultDataMaxSize);
    tlasDesc.setKind(RtAccelerationStructureKind::TopLevel);
    mTlas.pTlasObject = RtAccelerationStructure::create(mpDevice, tlasDesc);
}

void CustomAccelerationStructure::buildAccelerationStructure(
    RenderContext* pRenderContext,
    const std::vector<uint64_t>& aabbCount,
    bool updateAABBCount
)
{
    // Check if buffers exists
    FALCOR_ASSERT(mBlas[0]);
    FALCOR_ASSERT(mTlas.pTlas);

    FALCOR_ASSERT(aabbCount.size() >= mNumberBlas)

    buildBottomLevelAS(pRenderContext, aabbCount, updateAABBCount);
    buildTopLevelAS(pRenderContext);
    mAccelerationStructureWasBuild = true;
}

void CustomAccelerationStructure::buildBottomLevelAS(
    RenderContext* pRenderContext,
    const std::vector<uint64_t>& aabbCount,
    bool updateAABBCount
)
{
    FALCOR_PROFILE(pRenderContext, "buildCustomBlas");

    for (size_t i = 0; i < mNumberBlas; i++)
    {
        // Skip
        if (updateAABBCount && aabbCount[i] <= mMinUpdateAABBCount)
            continue;
        auto& blas = mBlasData[i];

        // barriers for the scratch and blas buffer
        pRenderContext->uavBarrier(mBlasScratch.get());
        pRenderContext->uavBarrier(mBlas[i].get());

        if (updateAABBCount)
            blas.geomDescs.content.proceduralAABBs.count = aabbCount[i];

        // Fill the build desc struct
        RtAccelerationStructure::BuildDesc asDesc = {};
        asDesc.inputs = blas.buildInputs;
        asDesc.scratchData = mBlasScratch->getGpuAddress();
        asDesc.dest = mBlasObjects[i].get();

        if (mAccelerationStructureWasBuild && (mUpdateMode == UpdateMode::BLASOnly || (mUpdateMode == UpdateMode::All)))
        {
            asDesc.source = asDesc.dest;
            asDesc.inputs.flags |= RtAccelerationStructureBuildFlags::PerformUpdate;
        }

        // Build the acceleration structure
        pRenderContext->buildAccelerationStructure(asDesc, 0, nullptr);

        // Barrier for the blas
        pRenderContext->uavBarrier(mBlas[i].get());
    }
}

void CustomAccelerationStructure::buildTopLevelAS(RenderContext* pRenderContext)
{
    FALCOR_PROFILE(pRenderContext, "buildCustomTlas");

    RtAccelerationStructure::BuildDesc asDesc = {};
    asDesc.inputs = mTlas.buildInputs;
    asDesc.inputs.instanceDescs = mTlas.pInstanceDescs->getGpuAddress();
    asDesc.scratchData = mTlasScratch->getGpuAddress();
    asDesc.dest = mTlas.pTlasObject.get();

    if (mAccelerationStructureWasBuild && (mUpdateMode == UpdateMode::TLASOnly || (mUpdateMode == UpdateMode::All)))
    {
        asDesc.source = asDesc.dest;
        asDesc.inputs.flags |= RtAccelerationStructureBuildFlags::PerformUpdate;
    }

    // Create TLAS
    if (mTlas.pInstanceDescs)
    {
        pRenderContext->resourceBarrier(mTlas.pInstanceDescs.get(), Resource::State::NonPixelShader);
    }

    // Build the acceleration structure
    pRenderContext->buildAccelerationStructure(asDesc, 0, nullptr);

    // Barrier for the blas
    pRenderContext->uavBarrier(mTlas.pTlas.get());
}

void CustomAccelerationStructure::clearAABBBuffers(
    RenderContext* pRenderContext,
    const ref<Buffer> pAABBBuffer,
    bool clearToNaN,
    ref<Buffer> pCounterBuffer
)
{
    std::vector<ref<Buffer>> pAABBs = {pAABBBuffer};
    clearAABBBuffers(pRenderContext, pAABBs, clearToNaN, pCounterBuffer);
}

void CustomAccelerationStructure::clearAABBBuffers(RenderContext* pRenderContext, const std::vector<ref<Buffer>>& pAABBBuffers, bool clearToNaN, ref<Buffer> pCounterBuffer)
{
    FALCOR_PROFILE(pRenderContext, "ClearAccelAABBBuffers");

    if (pAABBBuffers.empty())
        return;

    // Create compute pass if invalid
    if (!mpClearAABBsPass)
    {
        ProgramDesc desc;
        desc.addShaderLibrary(kAABBClearShaderFile).csEntry("main").setShaderModel(ShaderModel::SM6_6);

        DefineList defines;
        defines.add("USE_COUNTER_TO_CLEAR", pCounterBuffer ? "1" : "0");

        mpClearAABBsPass = ComputePass::create(mpDevice, desc, defines, true);
    }

    mpClearAABBsPass->getProgram()->addDefine("USE_COUNTER_TO_CLEAR", pCounterBuffer ? "1" : "0");

    auto var = mpClearAABBsPass->getRootVar();

    for (uint i = 0; i < pAABBBuffers.size(); i++)
    {
        auto& pAABB = pAABBBuffers[i];
        uint3 dispatchSize = uint3(1);
        if (pAABB->isStructured() || pAABB->isTyped())
            dispatchSize.x = (pAABB->getElementCount());
        else
            dispatchSize.x = pAABB->getElementCount() / sizeof(AABB);

        var["CB"]["gMax"] = dispatchSize.x;
        var["CB"]["gOffset"] = 0;
        var["CB"]["gCounterIdx"] = i;
        var["CB"]["gClearToNaN"] = clearToNaN;

        var["gCounter"] = pCounterBuffer; // Can be nullptr
        var["gAABB"] = pAABB;

        mpClearAABBsPass->execute(pRenderContext, dispatchSize);
        pRenderContext->resourceBarrier(pAABB.get(), Resource::State::ShaderResource);
    }
}

} // namespace Falcor
