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
#include "RGLAcquisition.h"
#include "Core/API/RenderContext.h"
#include "Utils/Logger.h"
#include "Utils/Math/Vector.h"
#include "Scene/Material/RGLCommon.h"

namespace Falcor
{
    namespace
    {
        const char kShaderFile[] = "Rendering/Materials/RGLAcquisition.cs.slang";
        const char kParameterBlock[] = "gAcquisition";

        // Number of data points in the incident-phi domain. We do isotropic only, so 1 sample is enough.
        const uint kPhiSize = 1;
        // Number of data points in the incident-theta domain.
        const uint kThetaSize = 8;
        // Number of data points for half vector (theta, phi) domain.
        const uint2 kNDFSize = uint2(128, 2);
        // Number of grid points to sample for numerically integrating the NDF cross section.
        const uint2 kSigmaIntegrationGrid = uint2(128, 128);

        // Size of the VNDF and measured data in the wi and wo domain. wi is dictated by phiSize/thetaSize.
        // We choose the same defaults as the RGL paper for the wo domain.
        const uint4 kVNDFSize = uint4(kPhiSize, kThetaSize, 128, 128);
        const uint4 kLumiSize = uint4(kPhiSize, kThetaSize, 32, 32);
        // Number of power iterations to use for extracting the largest eigen vector.
        // Same number as in the RGL paper (seems to be enough).
        const uint kEigenVectorPowerIterations = 4;
        // There does not seem to be a good way of finding out the wave size from the host side.
        // Assume a minimum of 16 lanes. Underestimating the true number just costs more memory,
        // overestimating it causes out-of-bounds access.
        const uint kMinWaveSize = 16;

        // Precomputed number of elements in each of the arrays, for convenience.
        const uint kNDFN = kNDFSize.x * kNDFSize.y;
        const uint kSigmaIntegrationN = kSigmaIntegrationGrid.x * kSigmaIntegrationGrid.y;
        const uint kLumiN = kLumiSize.x * kLumiSize.y * kLumiSize.z * kLumiSize.w;
        const uint kVNDFN = kVNDFSize.x * kVNDFSize.y * kVNDFSize.z * kVNDFSize.w;
    }

    RGLAcquisition::SharedPtr RGLAcquisition::create(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
    {
        return SharedPtr(new RGLAcquisition(pRenderContext, pScene));
    }

    RGLAcquisition::RGLAcquisition(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
        : mpScene(pScene)
    {
        checkArgument(pScene != nullptr, "'pScene' must be a valid scene");

        Program::Desc descBase;
        descBase.addShaderModules(pScene->getShaderModules());
        descBase.addShaderLibrary(kShaderFile);
        descBase.addTypeConformances(pScene->getTypeConformances());
        auto defines = pScene->getSceneDefines();

        auto addEntryPoint = [&](const char* name)
        {
            auto desc = descBase;
            desc.csEntry(name);
            return ComputePass::create(desc, defines);
        };

        mpRetroReflectionPass = addEntryPoint("measureRetroreflection");
        mpBuildKernelPass     = addEntryPoint("buildPowerIterationKernel");
        mpPowerIterationPass  = addEntryPoint("powerIteration");
        mpIntegrateSigmaPass  = addEntryPoint("integrateSigma");
        mpSumSigmaPass        = addEntryPoint("sumSigma");
        mpComputeThetaPass    = addEntryPoint("computeTheta");
        mpComputeVNDFPass     = addEntryPoint("computeVNDF");
        mpAcquireBRDFPass     = addEntryPoint("acquireBRDF");

        auto createStructured = [&](size_t elemSize, size_t count, const void* srcData = nullptr)
        {
            return Buffer::createStructured(
                uint32_t(elemSize),
                uint32_t(count),
                ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                Buffer::CpuAccess::None,
                srcData,
                false
            );
        };

        mpNDFDirectionsBuffer = createStructured(sizeof(float3), kNDFN);
        mpRetroBuffer         = createStructured(sizeof(float), kNDFN);
        mpNDFKernelBuffer     = createStructured(sizeof(float), kNDFN * kNDFN);
        mpNDFBufferTmp        = createStructured(sizeof(float), kNDFN);
        mpNDFBuffer           = createStructured(sizeof(float), kNDFN);
        mpSigmaBuffer         = createStructured(sizeof(float), kSigmaIntegrationN * kNDFN / kMinWaveSize);
        mpThetaBuffer         = createStructured(sizeof(float), kThetaSize);
        mpPhiBuffer           = createStructured(sizeof(float), kPhiSize);
        mpVNDFBuffer          = createStructured(sizeof(float), kVNDFN);
        mpLumiBuffer          = createStructured(sizeof(float), kLumiN);
        mpRGBBuffer           = createStructured(sizeof(float), kLumiN * 3);
        mpVNDFCondBuffer      = createStructured(sizeof(float), kVNDFN);
        mpVNDFMargBuffer      = createStructured(sizeof(float), kVNDFN / kVNDFSize.z);
    }

    void RGLAcquisition::acquireIsotropic(RenderContext* pRenderContext, const MaterialID materialID)
    {
        FALCOR_ASSERT(mpScene);
        checkArgument(materialID.get() < mpScene->getMaterialCount(), "'materialID' is out of range");

        CpuTimer timer;
        timer.update();

        auto setupParams = [&](const ComputePass::SharedPtr& pPass)
        {
            auto var = pPass->getRootVar()[kParameterBlock];
            var["materialID"] = materialID.getSlang();

            var["ndfSize"] = kNDFSize;
            var["phiSize"] = kPhiSize;
            var["vndfSize"] = kVNDFSize;
            var["lumiSize"] = kLumiSize;
            var["thetaSize"] = kThetaSize;
            var["sigmaIntegrationGrid"] = kSigmaIntegrationGrid;

            var["retroReflection"] = mpRetroBuffer;
            var["ndfDirections"] = mpNDFDirectionsBuffer;
            var["ndfKernel"] = mpNDFKernelBuffer;
            var["ndf"] = mpNDFBuffer;
            var["sigma"] = mpSigmaBuffer;
            var["phis"] = mpPhiBuffer;
            var["thetas"] = mpThetaBuffer;
            var["vndfBuf"] = mpVNDFBuffer;
            var["vndfMarginalBuf"] = mpVNDFMargBuffer;
            var["vndfConditionalBuf"] = mpVNDFCondBuffer;
            var["lumi"] = mpLumiBuffer;
            var["rgb"] = mpRGBBuffer;
            pPass["gScene"] = mpScene->getParameterBlock();
        };

        setupParams(mpRetroReflectionPass);
        setupParams(mpBuildKernelPass);
        setupParams(mpPowerIterationPass);
        setupParams(mpIntegrateSigmaPass);
        setupParams(mpSumSigmaPass);
        setupParams(mpComputeThetaPass);
        setupParams(mpComputeVNDFPass);
        setupParams(mpAcquireBRDFPass);

        // Phase 1: Measure retroreflections and build kernel matrix of Fredholm problem.
        mpRetroReflectionPass->execute(pRenderContext, uint3(kNDFSize, 1));
        mpBuildKernelPass->execute(pRenderContext, uint3(kNDFN, kNDFN, 1));

        // Numerically extract the biggest eigen vector of kernel matrix; this is the NDF.
        std::vector<float> ones(kNDFN, 1.0f);
        mpNDFBuffer->setBlob(ones.data(), 0, sizeof(float) * kNDFN);
        auto var = mpPowerIterationPass->getRootVar()[kParameterBlock];
        for (uint i = 0; i < kEigenVectorPowerIterations; ++i)
        {
            var["ndf"] = mpNDFBuffer;
            var["ndfTmp"] = mpNDFBufferTmp;
            mpPowerIterationPass->execute(pRenderContext, uint3(kNDFSize, 1));
            std::swap(mpNDFBuffer, mpNDFBufferTmp);
        }

        // Phase 2: Numerically integrate the projected microfacet area corresponding to NDF.
        mpIntegrateSigmaPass->execute(pRenderContext, uint3(kSigmaIntegrationGrid.x, kSigmaIntegrationGrid.y, kNDFN));
        mpSumSigmaPass->execute(pRenderContext, uint3(kNDFSize.x, kNDFSize.y, 1));

        // Normalize NDF and projected areas.
        std::vector<float> ndfCPU(kNDFN), sigmaCPU(kNDFN);
        std::memcpy(  ndfCPU.data(), mpNDFBuffer  ->map(Buffer::MapType::Read), kNDFN * sizeof(float));
        std::memcpy(sigmaCPU.data(), mpSigmaBuffer->map(Buffer::MapType::Read), kNDFN * sizeof(float));
        mpNDFBuffer  ->unmap();
        mpSigmaBuffer->unmap();
        for (uint y = 0; y < kNDFSize.y; ++y)
        {
            // First entry of sigma in each row corresponds to integrating <wi, wm> NDF(wm) with theta_i=0
            // and is equivalent to the normalization constant for the NDF.
            // Use this factor to properly normalize NDF now and adjust sigmas accordingly.
            float norm = 1.0f / sigmaCPU[y * kNDFSize.x];
            for (uint x = 0; x < kNDFSize.x; ++x)
            {
                sigmaCPU[x + y * kNDFSize.x] *= norm;
                  ndfCPU[x + y * kNDFSize.x] *= norm;
            }
        }
        mpSigmaBuffer->setBlob(sigmaCPU.data(), 0, sigmaCPU.size() * sizeof(float));
        mpNDFBuffer  ->setBlob(  ndfCPU.data(), 0,   ndfCPU.size() * sizeof(float));

        // Phase 3: Determine which angles to measure (currently just theta - we don't support anisotropy).
        mpComputeThetaPass->execute(pRenderContext, uint3(kThetaSize, 1, 1));

        // Phase 4: Compute the VNDF and make it samplable.
        mpComputeVNDFPass->execute(pRenderContext, uint3(kVNDFSize.x, kVNDFSize.y, kVNDFSize.z * kVNDFSize.w));

        const float* vndfBuf = reinterpret_cast<const float*>(mpVNDFBuffer->map(Buffer::MapType::Read));
        SamplableDistribution4D vndfDist(vndfBuf, kVNDFSize);
        mpVNDFBuffer->unmap();
        mpVNDFBuffer    ->setBlob(vndfDist.getPDF(),         0, sizeof(float) * kVNDFN);
        mpVNDFCondBuffer->setBlob(vndfDist.getConditional(), 0, sizeof(float) * kVNDFN);
        mpVNDFMargBuffer->setBlob(vndfDist.getMarginal(),    0, sizeof(float) * kVNDFN / kVNDFSize.z);

        // Phase 5: Perform the actual BRDF measurement.
        mpAcquireBRDFPass->execute(pRenderContext, uint3(kLumiSize.x, kLumiSize.y, kLumiSize.z * kLumiSize.w));

        timer.update();
        logInfo("Finished BSDF acquisition in {} seconds.", timer.delta());
    }

    RGLFile RGLAcquisition::toRGLFile()
    {
        RGLFile result;

        auto theta = mpThetaBuffer->map(Buffer::MapType::Read);
        auto phi   = mpPhiBuffer  ->map(Buffer::MapType::Read);
        auto sigma = mpSigmaBuffer->map(Buffer::MapType::Read);
        auto ndf   = mpNDFBuffer  ->map(Buffer::MapType::Read);
        auto vndf  = mpVNDFBuffer ->map(Buffer::MapType::Read);
        auto rgb   = mpRGBBuffer  ->map(Buffer::MapType::Read);
        auto lumi  = mpLumiBuffer ->map(Buffer::MapType::Read);

        std::string description = "Virtually measured BRDF";
        result.addField("description", RGLFile::UInt8,   std::vector<uint>{{uint(description.size())}}, description.c_str());
        result.addField("phi_i",       RGLFile::Float32, std::vector<uint>{{kPhiSize}}, phi);
        result.addField("theta_i",     RGLFile::Float32, std::vector<uint>{{kThetaSize}}, theta);
        result.addField("sigma",       RGLFile::Float32, std::vector<uint>{{kNDFSize.y, kNDFSize.x}}, sigma);
        result.addField("ndf",         RGLFile::Float32, std::vector<uint>{{kNDFSize.y, kNDFSize.x}}, ndf);
        result.addField("vndf",        RGLFile::Float32, std::vector<uint>{{kVNDFSize.x, kVNDFSize.y, kVNDFSize.z, kVNDFSize.w}}, vndf);
        result.addField("luminance",   RGLFile::Float32, std::vector<uint>{{kLumiSize.x, kLumiSize.y, kLumiSize.z, kLumiSize.w}}, lumi);
        result.addField("rgb",         RGLFile::Float32, std::vector<uint>{{kLumiSize.x, kLumiSize.y, 3, kLumiSize.z, kLumiSize.w}}, rgb);

        mpThetaBuffer->unmap();
        mpPhiBuffer  ->unmap();
        mpSigmaBuffer->unmap();
        mpNDFBuffer  ->unmap();
        mpVNDFBuffer ->unmap();
        mpRGBBuffer  ->unmap();
        mpLumiBuffer ->unmap();

        return std::move(result);
    }
}
