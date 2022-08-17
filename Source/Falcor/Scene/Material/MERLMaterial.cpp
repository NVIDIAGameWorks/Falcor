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
#include "MERLMaterial.h"
#include "Core/Renderer.h"
#include "Utils/Logger.h"
#include "Utils/Image/ImageIO.h"
#include "Utils/Scripting/ScriptBindings.h"
#include "Rendering/Materials/BSDFIntegrator.h"
#include <fstream>

namespace Falcor
{
    namespace
    {
        static_assert((sizeof(MaterialHeader) + sizeof(MERLMaterialData)) <= sizeof(MaterialDataBlob), "MERLMaterialData is too large");

        const char kShaderFile[] = "Rendering/Materials/MERLMaterial.slang";

        // Angular sampling resolution of the measured data.
        const size_t kBRDFSamplingResThetaH = 90;
        const size_t kBRDFSamplingResThetaD = 90;
        const size_t kBRDFSamplingResPhiD = 360;

        // Scale factors for the RGB channels of the measured data.
        const double kRedScale = 1.0 / 1500.0;
        const double kGreenScale = 1.15 / 1500.0;
        const double kBlueScale = 1.66 / 1500.0;

        const uint32_t kAlbedoLUTSize = MERLMaterialData::kAlbedoLUTSize;
        const ResourceFormat kAlbedoLUTFormat = ResourceFormat::RGBA32Float;
    }

    MERLMaterial::SharedPtr MERLMaterial::create(const std::string& name, const std::filesystem::path& path)
    {
        return SharedPtr(new MERLMaterial(name, path));
    }

    MERLMaterial::MERLMaterial(const std::string& name, const std::filesystem::path& path)
        : Material(name, MaterialType::MERL)
    {
        if (!loadBRDF(path))
        {
            throw RuntimeError("MERLMaterial() - Failed to load BRDF from '{}'.", path);
        }

        // Create resources for albedo lookup table.
        Sampler::Desc desc;
        desc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Point, Sampler::Filter::Point);
        desc.setAddressingMode(Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp);
        desc.setMaxAnisotropy(1);
        mpLUTSampler = Sampler::create(desc);

        prepareAlbedoLUT(gpFramework->getRenderContext());
    }

    bool MERLMaterial::renderUI(Gui::Widgets& widget)
    {
        widget.text("MERL BRDF " + mBRDFName);
        widget.tooltip("Full path the BRDF was loaded from:\n" + mPath.string(), true);

        return false;
    }

    Material::UpdateFlags MERLMaterial::update(MaterialSystem* pOwner)
    {
        FALCOR_ASSERT(pOwner);

        auto flags = Material::UpdateFlags::None;
        if (mUpdates != Material::UpdateFlags::None)
        {
            uint32_t bufferID = pOwner->addBuffer(mpBRDFData);
            uint32_t samplerID = pOwner->addTextureSampler(mpLUTSampler);

            if (mData.bufferID != bufferID || mData.samplerID != samplerID)
            {
                mUpdates |= Material::UpdateFlags::DataChanged;
            }
            mData.bufferID = bufferID;
            mData.samplerID = samplerID;

            updateTextureHandle(pOwner, mpAlbedoLUT, mData.texAlbedoLUT);

            flags |= mUpdates;
            mUpdates = Material::UpdateFlags::None;
        }

        return flags;
    }

    bool MERLMaterial::isEqual(const Material::SharedPtr& pOther) const
    {
        auto other = std::dynamic_pointer_cast<MERLMaterial>(pOther);
        if (!other) return false;

        if (!isBaseEqual(*other)) return false;
        if (mPath != other->mPath) return false;

        return true;
    }

    Program::ShaderModuleList MERLMaterial::getShaderModules() const
    {
        return { Program::ShaderModule(kShaderFile) };
    }

    Program::TypeConformanceList MERLMaterial::getTypeConformances() const
    {
        return { {{"MERLMaterial", "IMaterial"}, (uint32_t)MaterialType::MERL} };
    }

    bool MERLMaterial::loadBRDF(const std::filesystem::path& path)
    {
        std::filesystem::path fullPath;
        if (!findFileInDataDirectories(path, fullPath))
        {
            logWarning("MERLMaterial::loadBRDF() - Can't find file '{}'.", path);
            return false;
        }

        std::ifstream ifs(fullPath, std::ios_base::in | std::ios_base::binary);
        if (!ifs.good())
        {
            logWarning("MERLMaterial::loadBRDF() - Failed to open file '{}'.", path);
            return false;
        }

        // Load header.
        int dims[3] = {};
        ifs.read(reinterpret_cast<char*>(dims), sizeof(int) * 3);

        size_t n = (size_t)dims[0] * dims[1] * dims[2];
        if (n != kBRDFSamplingResThetaH * kBRDFSamplingResThetaD * kBRDFSamplingResPhiD / 2)
        {
            logWarning("MERLMaterial::loadBRDF() - Dimensions don't match in file '{}'.", path);
            return false;
        }

        // Load BRDF data.
        std::vector<double> data(3 * n);
        ifs.read(reinterpret_cast<char*>(data.data()), sizeof(double) * 3 * n);
        if (!ifs.good())
        {
            logWarning("MERLMaterial::loadBRDF() - Failed to load BRDF data from file '{}'.", path);
            return false;
        }

        mPath = fullPath;
        mBRDFName = fullPath.stem().string();
        prepareData(dims, data);
        markUpdates(Material::UpdateFlags::ResourcesChanged);

        logInfo("Loaded MERL BRDF '{}'.", mBRDFName);

        return true;
    }

    void MERLMaterial::prepareData(const int dims[3], const std::vector<double>& data)
    {
        // Convert BRDF samples to fp32 precision and interleave RGB channels.
        const size_t n = (size_t)dims[0] * dims[1] * dims[2];

        FALCOR_ASSERT(data.size() == 3 * n);
        std::vector<float3> brdf(n);

        size_t negCount = 0;
        size_t infCount = 0;
        size_t nanCount = 0;

        for (size_t i = 0; i < n; i++)
        {
            float3& v = brdf[i];

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

        if (negCount > 0) logWarning("MERL BRDF {} has {} samples with negative values. Clamped to zero.", mBRDFName, negCount);
        if (infCount > 0) logWarning("MERL BRDF {} has {} samples with inf values. Sample set to zero.", mBRDFName, infCount);
        if (nanCount > 0) logWarning("MERL BRDF {} has {} samples with NaN values. Sample set to zero.", mBRDFName, nanCount);

        // Create GPU buffer.
        FALCOR_ASSERT(sizeof(brdf[0]) == sizeof(float3));
        mpBRDFData = Buffer::create(brdf.size() * sizeof(brdf[0]), ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, brdf.data());
    }

    void MERLMaterial::prepareAlbedoLUT(RenderContext* pRenderContext)
    {
        const auto texPath = mPath.replace_extension("dds");

        // Try loading albedo lookup table.
        if (std::filesystem::is_regular_file(texPath))
        {
            // Load 1D texture in non-SRGB format, no mips.
            // If successful, verify dimensions/format/etc. match the expectations.
            mpAlbedoLUT = Texture::createFromFile(texPath, false, false, ResourceBindFlags::ShaderResource);

            if (mpAlbedoLUT)
            {
                if (mpAlbedoLUT->getFormat() == kAlbedoLUTFormat &&
                    mpAlbedoLUT->getWidth() == kAlbedoLUTSize &&
                    mpAlbedoLUT->getHeight() == 1 && mpAlbedoLUT->getDepth() == 1 &&
                    mpAlbedoLUT->getMipCount() == 1 && mpAlbedoLUT->getArraySize() == 1)
                {
                    logInfo("Loaded albedo LUT from '{}'.", texPath.string());
                    return;
                }
            }
        }

        // Failed to load a valid lookup table. We'll recompute it.
        computeAlbedoLUT(pRenderContext);

        // Cache lookup table in texture on disk.
        // TODO: Capture texture to DDS is not yet supported. Calling ImageIO directly for now.
        //mpAlbedoLUT->captureToFile(0, 0, texPath, Bitmap::FileFormat::DdsFile, Bitmap::ExportFlags::Uncompressed);
        FALCOR_ASSERT(mpAlbedoLUT);
        ImageIO::saveToDDS(gpFramework->getRenderContext(), texPath, mpAlbedoLUT, ImageIO::CompressionMode::None, false);

        logInfo("Saved albedo LUT to '{}'.", texPath);
    }

    void MERLMaterial::computeAlbedoLUT(RenderContext* pRenderContext)
    {
        logInfo("Computing albedo LUT for MERL BRDF '{}'...", mBRDFName);

        std::vector<float> cosThetas(kAlbedoLUTSize);
        for (uint32_t i = 0; i < kAlbedoLUTSize; i++) cosThetas[i] = (float)(i + 1) / kAlbedoLUTSize;

        // Create copy of material to avoid changing our local state.
        auto pMaterial = SharedPtr(new MERLMaterial(*this));

        // Create and update scene containing the material.
        Scene::SceneData sceneData;
        sceneData.pMaterials = MaterialSystem::create();
        MaterialID materialID = sceneData.pMaterials->addMaterial(pMaterial);

        Scene::SharedPtr pScene = Scene::create(std::move(sceneData));
        pScene->update(pRenderContext, 0.0);

        // Create BSDF integrator utility.
        auto pIntegrator = BSDFIntegrator::create(pRenderContext, pScene);

        // Integreate BSDF.
        auto albedos = pIntegrator->integrateIsotropic(pRenderContext, materialID, cosThetas);

        // Copy result into format needed for texture creation.
        static_assert(kAlbedoLUTFormat == ResourceFormat::RGBA32Float);
        std::vector<float4> initData(kAlbedoLUTSize, float4(0.f));
        for (uint32_t i = 0; i < kAlbedoLUTSize; i++) initData[i] = float4(albedos[i], 1.f);

        // Create albedo LUT texture.
        mpAlbedoLUT = Texture::create2D(kAlbedoLUTSize, 1, kAlbedoLUTFormat, 1, 1, initData.data(), ResourceBindFlags::ShaderResource);
    }

    FALCOR_SCRIPT_BINDING(MERLMaterial)
    {
        using namespace pybind11::literals;

        FALCOR_SCRIPT_BINDING_DEPENDENCY(Material)

        pybind11::class_<MERLMaterial, Material, MERLMaterial::SharedPtr> material(m, "MERLMaterial");
        material.def(pybind11::init(&MERLMaterial::create), "name"_a, "path"_a);
    }
}
