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
#include "MERLFile.h"
#include "Utils/Logger.h"
#include "Utils/Image/ImageIO.h"
#include "Scene/Material/MERLMaterial.h"
#include "Scene/Material/DiffuseSpecularUtils.h"
#include "Rendering/Materials/BSDFIntegrator.h"
#include <fstream>

namespace Falcor
{
    namespace
    {
        // Angular sampling resolution of the measured data.
        const size_t kBRDFSamplingResThetaH = 90;
        const size_t kBRDFSamplingResThetaD = 90;
        const size_t kBRDFSamplingResPhiD = 360;

        // Scale factors for the RGB channels of the measured data.
        const double kRedScale = 1.0 / 1500.0;
        const double kGreenScale = 1.15 / 1500.0;
        const double kBlueScale = 1.66 / 1500.0;

        const uint32_t kAlbedoLUTSize = MERLMaterialData::kAlbedoLUTSize;
    }

    MERLFile::MERLFile(const std::filesystem::path& path)
    {
        if (!loadBRDF(path))
            FALCOR_THROW("Failed to load MERL BRDF from '{}'", path);
    }

    bool MERLFile::loadBRDF(const std::filesystem::path& path)
    {
        mDesc = {};
        mData.clear();
        mAlbedoLUT.clear();

        std::ifstream ifs(path, std::ios_base::in | std::ios_base::binary);
        if (!ifs.good())
        {
            logWarning("MERLFile: Failed to open file '{}'.", path);
            return false;
        }

        // Load header.
        int dims[3] = {};
        ifs.read(reinterpret_cast<char*>(dims), sizeof(int) * 3);

        size_t n = (size_t)dims[0] * dims[1] * dims[2];
        if (n != kBRDFSamplingResThetaH * kBRDFSamplingResThetaD * kBRDFSamplingResPhiD / 2)
        {
            logWarning("MERLFile: Dimensions don't match in file '{}'.", path);
            return false;
        }

        // Load BRDF data.
        std::vector<double> data(3 * n);
        ifs.read(reinterpret_cast<char*>(data.data()), sizeof(double) * 3 * n);
        if (!ifs.good())
        {
            logWarning("MERLFile: Failed to load BRDF data from file '{}'.", path);
            return false;
        }

        mDesc.path = path;
        mDesc.name = path.stem().string();

        prepareData(dims, data);

        // Load JSON sidecar file if it exists.
        const auto jsonPath = std::filesystem::path(path).replace_extension("json");
        if (!DiffuseSpecularUtils::loadJSONData(jsonPath, mDesc.extraData))
            logWarning("MERLFile: Failed to load associated JSON data for BRDF '{}'.", mDesc.name);

        logInfo("Loaded MERL BRDF '{}'.", mDesc.name);
        return true;
    }

    void MERLFile::prepareData(const int dims[3], const std::vector<double>& data)
    {
        // Convert BRDF samples to fp32 precision and interleave RGB channels.
        const size_t n = (size_t)dims[0] * dims[1] * dims[2];

        FALCOR_ASSERT(data.size() == 3 * n);
        mData.resize(n);

        size_t negCount = 0;
        size_t infCount = 0;
        size_t nanCount = 0;

        for (size_t i = 0; i < n; i++)
        {
            float3& v = mData[i];

            // Extract RGB and apply scaling.
            v.x = static_cast<float>(data[i] * kRedScale);
            v.y = static_cast<float>(data[i + n] * kGreenScale);
            v.z = static_cast<float>(data[i + 2 * n] * kBlueScale);

            // Validate data point and set to zero if invalid.
            bool isNeg = v.x < 0.f || v.y < 0.f || v.z < 0.f;
            bool isInf = std::isinf(v.x) || std::isinf(v.y) || std::isinf(v.z);
            bool isNaN = std::isnan(v.x) || std::isnan(v.y) || std::isnan(v.z);

            if (isNeg) negCount++;
            if (isInf) infCount++;
            if (isNaN) nanCount++;

            if (isInf || isNaN) v = float3(0.f);
            else if (isNeg) v = max(v, float3(0.f));
        }

        if (negCount > 0) logWarning("MERL BRDF {} has {} samples with negative values. Clamped to zero.", mDesc.name, negCount);
        if (infCount > 0) logWarning("MERL BRDF {} has {} samples with inf values. Sample set to zero.", mDesc.name, infCount);
        if (nanCount > 0) logWarning("MERL BRDF {} has {} samples with NaN values. Sample set to zero.", mDesc.name, nanCount);
    }

    const std::vector<float4>& MERLFile::prepareAlbedoLUT(ref<Device> pDevice)
    {
        if (!mAlbedoLUT.empty())
            return mAlbedoLUT;

        FALCOR_CHECK(!mDesc.path.empty(), "No BRDF loaded");
        const auto texPath = mDesc.path.replace_extension("dds");

        // Try loading cached albedo lookup table.
        if (std::filesystem::is_regular_file(texPath))
        {
            const auto albedoLut = ImageIO::loadBitmapFromDDS(texPath);

            if (albedoLut->getFormat() == kAlbedoLUTFormat &&
                albedoLut->getWidth() == kAlbedoLUTSize && albedoLut->getHeight() == 1)
            {
                const float4* data = reinterpret_cast<const float4*>(albedoLut->getData());
                mAlbedoLUT.resize(kAlbedoLUTSize);
                std::copy(data, data + kAlbedoLUTSize, mAlbedoLUT.begin());

                logInfo("Loaded albedo LUT from '{}'.", texPath.string());
                return mAlbedoLUT;
            }
        }

        // Failed to load a valid lookup table. We'll recompute it.
        computeAlbedoLUT(pDevice, kAlbedoLUTSize);
        FALCOR_ASSERT(mAlbedoLUT.size() == kAlbedoLUTSize);

        // Cache lookup table as texture on disk.
        {
            const uint8_t* data = reinterpret_cast<const uint8_t*>(mAlbedoLUT.data());
            const auto albedoLut = Bitmap::create(mAlbedoLUT.size(), 1, kAlbedoLUTFormat, data);

            ImageIO::saveToDDS(texPath, *albedoLut, ImageIO::CompressionMode::None, false);
            logInfo("Saved albedo LUT to '{}'.", texPath);
        }

        return mAlbedoLUT;
    }

    void MERLFile::computeAlbedoLUT(ref<Device> pDevice, const size_t binCount)
    {
        logInfo("MERLFile: Computing albedo LUT for MERL BRDF '{}'...", mDesc.name);

        std::vector<float> cosThetas(binCount);
        for (uint32_t i = 0; i < binCount; i++) cosThetas[i] = (float)(i + 1) / binCount;

        // Create MERL material based on loaded data.
        ref<MERLMaterial> pMaterial = make_ref<MERLMaterial>(pDevice, *this);

        // Create and update dummy scene containing the material.
        Scene::SceneData sceneData;
        sceneData.pMaterials = std::make_unique<MaterialSystem>(pDevice);
        MaterialID materialID = sceneData.pMaterials->addMaterial(pMaterial);

        ref<Scene> pScene = Scene::create(pDevice, std::move(sceneData));
        pScene->update(pDevice->getRenderContext(), 0.0);

        // Create BSDF integrator utility.
        BSDFIntegrator integrator(pDevice, pScene);

        // Integrate BSDF.
        auto albedos = integrator.integrateIsotropic(pDevice->getRenderContext(), materialID, cosThetas);

        // Copy result into RGBA format needed for texture creation.
        mAlbedoLUT.resize(binCount);
        for (uint32_t i = 0; i < binCount; i++)
            mAlbedoLUT[i] = float4(albedos[i], 1.f);
    }
}
