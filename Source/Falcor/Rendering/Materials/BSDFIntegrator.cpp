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
#include "BSDFIntegrator.h"
#include "Core/Assert.h"
#include "Core/Errors.h"
#include "Core/API/RenderContext.h"
#include "Utils/Logger.h"

namespace Falcor
{
    namespace
    {
        const char kShaderFile[] = "Rendering/Materials/BSDFIntegrator.cs.slang";
        const char kParameterBlock[] = "gIntegrator";

        /// Integration grid size. Do not change, the programs are specialized for this size.
        /// The shader takes currently 8x8 stratified samples per grid cell, for a total of 4096x4096 samples over the hemisphere.
        const uint2 kGridSize = { 512, 512 };
    }

    BSDFIntegrator::SharedPtr BSDFIntegrator::create(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
    {
        return SharedPtr(new BSDFIntegrator(pRenderContext, pScene));
    }

    BSDFIntegrator::BSDFIntegrator(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
        : mpScene(pScene)
    {
        checkArgument(pScene != nullptr, "'pScene' must be a valid scene");

        // Create programs.
        Program::Desc desc;
        desc.setShaderModel("6_6");
        desc.addShaderModules(pScene->getShaderModules());
        desc.addShaderLibrary(kShaderFile);
        desc.addTypeConformances(pScene->getTypeConformances());
        auto defines = pScene->getSceneDefines();
        Program::Desc descFinal = desc;

        desc.csEntry("mainIntegration");
        mpIntegrationPass = ComputePass::create(desc, defines);
        descFinal.csEntry("mainFinal");
        mpFinalPass = ComputePass::create(descFinal, defines);

        // Compute number of intermediate results.
        uint3 groupSize = mpIntegrationPass->getThreadGroupSize();
        FALCOR_ASSERT(groupSize.x == 32 && groupSize.y == 32 && groupSize.z == 1);
        uint groupThreadCount = groupSize.x * groupSize.y * groupSize.z;
        mResultCount = (kGridSize.x * kGridSize.y) / groupThreadCount;
        FALCOR_ASSERT(mResultCount * groupThreadCount == kGridSize.x * kGridSize.y);

        uint3 finalGroupSize = mpFinalPass->getThreadGroupSize();
        FALCOR_ASSERT(finalGroupSize.x == 256 && finalGroupSize.y == 1 && finalGroupSize.z == 1);
        FALCOR_ASSERT(finalGroupSize.x == mResultCount);

        mpFence = GpuFence::create();
    }

    float3 BSDFIntegrator::integrateIsotropic(RenderContext* pRenderContext, const MaterialID materialID, float cosTheta)
    {
        std::vector<float> cosThetas(1, cosTheta);
        auto results = integrateIsotropic(pRenderContext, materialID, cosThetas);
        return results[0];
    }

    std::vector<float3> BSDFIntegrator::integrateIsotropic(RenderContext* pRenderContext, const MaterialID materialID, const std::vector<float>& cosThetas)
    {
        FALCOR_ASSERT(mpScene);
        checkArgument(materialID.get() < mpScene->getMaterialCount(), "'materialID' is out of range");
        checkArgument(!cosThetas.empty(), "'cosThetas' array is empty");

        CpuTimer timer;
        timer.update();

        // Upload cos theta angles.
        FALCOR_ASSERT(cosThetas.size() <= std::numeric_limits<uint32_t>::max());
        const uint32_t gridCount = (uint32_t)cosThetas.size();

        if (!mpCosThetaBuffer || mpCosThetaBuffer->getElementCount() < gridCount)
        {
            mpCosThetaBuffer = Buffer::createStructured(sizeof(float), gridCount, ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, cosThetas.data(), false);
        }
        else
        {
            mpCosThetaBuffer->setBlob(cosThetas.data(), 0, cosThetas.size() * sizeof(cosThetas[0]));
        }

        // Allocate buffer for intermediate and final results.
        uint32_t elemCount = gridCount * mResultCount;
        if (!mpResultBuffer || mpResultBuffer->getElementCount() < elemCount)
        {
            mpResultBuffer = Buffer::createStructured(sizeof(float3), elemCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, Buffer::CpuAccess::None, nullptr, false);
        }
        if (!mpFinalResultBuffer || mpFinalResultBuffer->getElementCount() < gridCount)
        {
            mpFinalResultBuffer = Buffer::createStructured(sizeof(float3), gridCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess, Buffer::CpuAccess::None, nullptr, false);
            mpStagingBuffer = Buffer::createStructured(sizeof(float3), gridCount, ResourceBindFlags::None, Buffer::CpuAccess::Read, nullptr, false);
        }

        // Execute GPU passes.
        integrationPass(pRenderContext, materialID, gridCount);
        finalPass(pRenderContext, gridCount);

        // Copy result to staging buffer.
        pRenderContext->copyBufferRegion(mpStagingBuffer.get(), 0, mpFinalResultBuffer.get(), 0, sizeof(float3) * gridCount);

        // Flush GPU and wait for results to be available.
        pRenderContext->flush(false);
        mpFence->gpuSignal(pRenderContext->getLowLevelData()->getCommandQueue());
        mpFence->syncCpu();

        // Read back final results.
        const float3* finalResults = reinterpret_cast<const float3*>(mpStagingBuffer->map(Buffer::MapType::Read));
        std::vector<float3> output(finalResults, finalResults + gridCount);
        mpStagingBuffer->unmap();

        timer.update();
        logInfo("Finished BSDF integration for {} incident directions in {} seconds.", cosThetas.size(), timer.delta());

        return output;
    }

    void BSDFIntegrator::integrationPass(RenderContext* pRenderContext, const MaterialID materialID, const uint32_t gridCount) const
    {
        FALCOR_ASSERT(mpIntegrationPass);
        auto var = mpIntegrationPass->getRootVar()[kParameterBlock];
        var["gridSize"] = kGridSize;
        var["gridCount"] = gridCount;
        var["resultCount"] = mResultCount;
        var["materialID"] = materialID.getSlang();
        var["cosThetas"] = mpCosThetaBuffer;
        var["results"] = mpResultBuffer;

        mpIntegrationPass["gScene"] = mpScene->getParameterBlock();
        mpIntegrationPass->execute(pRenderContext, uint3(kGridSize, gridCount));
    }

    void BSDFIntegrator::finalPass(RenderContext* pRenderContext, const uint32_t gridCount) const
    {
        FALCOR_ASSERT(mpFinalPass);
        auto var = mpFinalPass->getRootVar()[kParameterBlock];
        var["gridCount"] = gridCount;
        var["resultCount"] = mResultCount;
        var["results"] = mpResultBuffer;
        var["finalResults"] = mpFinalResultBuffer;

        mpFinalPass->execute(pRenderContext, uint3(mResultCount, gridCount, 1));

#if 0
        // DEBUG: Final accumulation on the CPU.
        const float3* results = reinterpret_cast<const float3*>(mpResultBuffer->map(Buffer::MapType::Read));
        for (uint32_t gridIdx = 0; gridIdx < gridCount; gridIdx++)
        {
            float3 sum = {};
            for (size_t i = 0; i < mResultCount; i++)
            {
                sum += results[mResultCount * gridIdx + i];
            }
            float3 result = sum / (float)mResultCount;
        }
        mpResultBuffer->unmap();
#endif
    }
}
