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
#include "RGLMaterial.h"
#include "RGLFile.h"
#include "RGLCommon.h"
#include "Core/API/Device.h"
#include "Utils/Logger.h"
#include "Utils/Image/ImageIO.h"
#include "Utils/Scripting/ScriptBindings.h"
#include "GlobalState.h"
#include "Rendering/Materials/BSDFIntegrator.h"
#include <fstream>

namespace Falcor
{
    namespace
    {
        static_assert((sizeof(MaterialHeader) + sizeof(RGLMaterialData)) <= sizeof(MaterialDataBlob), "RGLMaterialData is too large");

        const char kShaderFile[] = "Rendering/Materials/RGLMaterial.slang";

        const uint32_t kAlbedoLUTSize = RGLMaterialData::kAlbedoLUTSize;
        const ResourceFormat kAlbedoLUTFormat = ResourceFormat::RGBA32Float;

        const std::string kLoadFile = "load";
    }

    RGLMaterial::RGLMaterial(ref<Device> pDevice, const std::string& name, const std::filesystem::path& path)
        : Material(pDevice, name, MaterialType::RGL)
    {
        if (!loadBRDF(path))
        {
            FALCOR_THROW("RGLMaterial() - Failed to load BRDF from '{}'.", path);
        }

        // Create resources for albedo lookup table.
        Sampler::Desc desc;
        desc.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Point, TextureFilteringMode::Point);
        desc.setAddressingMode(TextureAddressingMode::Clamp, TextureAddressingMode::Clamp, TextureAddressingMode::Clamp);
        desc.setMaxAnisotropy(1);
        mpSampler = mpDevice->createSampler(desc);

        prepareAlbedoLUT(mpDevice->getRenderContext());
    }

    bool RGLMaterial::renderUI(Gui::Widgets& widget)
    {
        widget.text("RGL BRDF " + mBRDFName);
        widget.text(mBRDFDescription);
        widget.tooltip("Full path the BRDF was loaded from:\n" + mPath.string(), true);

        return false;
    }

    Material::UpdateFlags RGLMaterial::update(MaterialSystem* pOwner)
    {
        FALCOR_ASSERT(pOwner);

        auto flags = Material::UpdateFlags::None;
        if (mUpdates != Material::UpdateFlags::None)
        {
            auto checkBuffer = [&](ref<Buffer>& buf, uint& handle)
            {
                // If a BRDF was already loaded and we're just updating data,
                // then buffer handles are already assigned.
                // Replace the buffer contents instead of adding a new buffer.
                if (mBRDFUploaded)
                {
                    pOwner->replaceBuffer(handle, buf);
                }
                else
                {
                    uint32_t bufferID = pOwner->addBuffer(buf);
                    if (bufferID != handle) mUpdates |= Material::UpdateFlags::DataChanged;
                    handle = bufferID;
                }
            };

            checkBuffer(mpThetaBuf, mData.thetaBufID);
            checkBuffer(mpPhiBuf,   mData.phiBufID);
            checkBuffer(mpSigmaBuf, mData.sigmaBufID);
            checkBuffer(mpNDFBuf,   mData.ndfBufID);
            checkBuffer(mpVNDFBuf,  mData.vndfBufID);
            checkBuffer(mpLumiBuf,  mData.lumiBufID);
            checkBuffer(mpRGBBuf,   mData.rgbBufID);
            checkBuffer(mpVNDFMarginalBuf, mData.vndfMarginalBufID);
            checkBuffer(mpLumiMarginalBuf, mData.lumiMarginalBufID);
            checkBuffer(mpVNDFConditionalBuf, mData.vndfConditionalBufID);
            checkBuffer(mpLumiConditionalBuf, mData.lumiConditionalBufID);
            updateTextureHandle(pOwner, mpAlbedoLUT, mData.texAlbedoLUT);
            mBRDFUploaded = true;

            flags |= mUpdates;
            mUpdates = Material::UpdateFlags::None;
        }

        return flags;
    }

    bool RGLMaterial::isEqual(const ref<Material>& pOther) const
    {
        auto other = dynamic_ref_cast<RGLMaterial>(pOther);
        if (!other) return false;

        if (!isBaseEqual(*other)) return false;
        if (mPath != other->mPath) return false;

        return true;
    }

    ProgramDesc::ShaderModuleList RGLMaterial::getShaderModules() const
    {
        return { ProgramDesc::ShaderModule::fromFile(kShaderFile) };
    }

    TypeConformanceList RGLMaterial::getTypeConformances() const
    {
        return { {{"RGLMaterial", "IMaterial"}, (uint32_t)MaterialType::RGL} };
    }

    bool RGLMaterial::loadBRDF(const std::filesystem::path& path)
    {
        std::ifstream ifs(path, std::ios_base::in | std::ios_base::binary);
        if (!ifs.good())
        {
            logWarning("RGLMaterial::loadBRDF() - Failed to open file '{}'.", path);
            return false;
        }

        std::unique_ptr<RGLFile> file;
        try
        {
            file.reset(new RGLFile(ifs));
        }
        catch(const RuntimeError& e)
        {
            logWarning("RGLMaterial::loadBRDF() - Failed to parse RGL file '{}': {}.", path, e.what());
            return false;
        }

        if (!ifs.good())
        {
            logWarning("RGLMaterial::loadBRDF() - Failed to load BRDF data from file '{}': Read error.", path);
            return false;
        }

        auto theta = file->data().thetaI;
        auto phi   = file->data().phiI;
        auto sigma = file->data().sigma;
        auto ndf   = file->data().ndf;
        auto vndf  = file->data().vndf;
        auto lumi  = file->data().luminance;
        auto rgb   = file->data().rgb;

        const uint64_t kMaxResolution = RGLMaterialData::kMaxResolution;
        if (phi->shape[0] > kMaxResolution || theta->shape[0] > kMaxResolution || std::max(sigma->shape[0], sigma->shape[1]) > kMaxResolution
            || std::max(ndf->shape[0], ndf->shape[1]) > kMaxResolution || std::max(vndf->shape[2], vndf->shape[3]) > kMaxResolution
            || std::max(lumi->shape[2], lumi->shape[3]) > kMaxResolution)
        {
            logWarning("RGLMaterial::loadBRDF() - Failed to process BRDF data: Measurement resolution too large.", path);
            return false;
        }

        mPath = path;
        mBRDFName = std::filesystem::path(path).stem().string();
        mBRDFDescription = file->data().description;

        mData.phiSize = uint(phi->shape[0]);
        mData.thetaSize = uint(theta->shape[0]);
        mData.sigmaSize = uint2(sigma->shape[1], sigma->shape[0]);
        mData.  ndfSize = uint2(ndf  ->shape[1], ndf  ->shape[0]);
        mData. vndfSize = uint2(vndf ->shape[3], vndf ->shape[2]);
        mData. lumiSize = uint2(lumi ->shape[3], lumi ->shape[2]);

        uint4 vndfSize = uint4(mData.phiSize, mData.thetaSize, mData.vndfSize.x, mData.vndfSize.y);
        uint4 lumiSize = uint4(mData.phiSize, mData.thetaSize, mData.lumiSize.x, mData.lumiSize.y);
        auto prod3 = [&](uint4 v) { return v.x * v.y * v.z; };
        auto prod4 = [&](uint4 v) { return v.x * v.y * v.z * v.w; };

        SamplableDistribution4D vndfDist(reinterpret_cast<float*>(vndf->data.get()), vndfSize);
        SamplableDistribution4D lumiDist(reinterpret_cast<float*>(lumi->data.get()), lumiSize);

        mpVNDFMarginalBuf    = mpDevice->createBuffer(prod3(vndfSize) * 4, ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, vndfDist.getMarginal());
        mpLumiMarginalBuf    = mpDevice->createBuffer(prod3(lumiSize) * 4, ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, lumiDist.getMarginal());
        mpVNDFConditionalBuf = mpDevice->createBuffer(prod4(vndfSize) * 4, ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, vndfDist.getConditional());
        mpLumiConditionalBuf = mpDevice->createBuffer(prod4(lumiSize) * 4, ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, lumiDist.getConditional());

        mpThetaBuf = mpDevice->createBuffer(theta->numElems * sizeof(float), ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, theta->data.get());
        mpPhiBuf   = mpDevice->createBuffer(phi  ->numElems * sizeof(float), ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, phi  ->data.get());
        mpSigmaBuf = mpDevice->createBuffer(sigma->numElems * sizeof(float), ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, sigma->data.get());
        mpNDFBuf   = mpDevice->createBuffer(ndf  ->numElems * sizeof(float), ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, ndf  ->data.get());
        mpVNDFBuf  = mpDevice->createBuffer(vndf ->numElems * sizeof(float), ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, vndfDist.getPDF());
        mpLumiBuf  = mpDevice->createBuffer(lumi ->numElems * sizeof(float), ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, lumiDist.getPDF());
        mpRGBBuf   = mpDevice->createBuffer(rgb  ->numElems * sizeof(float), ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, rgb  ->data.get());

        markUpdates(Material::UpdateFlags::ResourcesChanged);

        logInfo("Loaded RGL BRDF '{}': {}.", mBRDFName, mBRDFDescription);

        return true;
    }

    void RGLMaterial::prepareAlbedoLUT(RenderContext* pRenderContext) // TODO
    {
        const auto texPath = mPath.replace_extension("dds");

        // Try loading albedo lookup table.
        if (std::filesystem::is_regular_file(texPath))
        {
            // Load 1D texture in non-SRGB format, no mips.
            // If successful, verify dimensions/format/etc. match the expectations.
            mpAlbedoLUT = Texture::createFromFile(mpDevice, texPath.string(), false, false, ResourceBindFlags::ShaderResource);

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
        //mpAlbedoLUT->captureToFile(0, 0, texPath.string(), Bitmap::FileFormat::DdsFile, Bitmap::ExportFlags::Uncompressed);
        FALCOR_ASSERT(mpAlbedoLUT);
        ImageIO::saveToDDS(pRenderContext, texPath.string(), mpAlbedoLUT, ImageIO::CompressionMode::None, false);

        logInfo("Saved albedo LUT to '{}'.", texPath.string());
    }

    void RGLMaterial::computeAlbedoLUT(RenderContext* pRenderContext) // TODO
    {
        logInfo("Computing albedo LUT for RGL BRDF '{}'...", mBRDFName);

        std::vector<float> cosThetas(kAlbedoLUTSize);
        for (uint32_t i = 0; i < kAlbedoLUTSize; i++) cosThetas[i] = (float)(i + 1) / kAlbedoLUTSize;

        // Create copy of material to avoid changing our local state.
        auto pMaterial = make_ref<RGLMaterial>(*this);

        // Create and update scene containing the material.
        Scene::SceneData sceneData;
        sceneData.pMaterials = std::make_unique<MaterialSystem>(mpDevice);
        MaterialID materialID = sceneData.pMaterials->addMaterial(pMaterial);

        ref<Scene> pScene = Scene::create(mpDevice, std::move(sceneData));
        pScene->update(pRenderContext, 0.0);

        // Create BSDF integrator utility.
        BSDFIntegrator integrator(mpDevice, pScene);

        // Integrate BSDF.
        // TODO: Measured BRDFs could potentially be anisotropic.
        // It's unlikely this would affect the albedo significantly, and doing the integration
        // properly would be more trouble than its worth.
        auto albedos = integrator.integrateIsotropic(pRenderContext, materialID, cosThetas);

        // Copy result into format needed for texture creation.
        static_assert(kAlbedoLUTFormat == ResourceFormat::RGBA32Float);
        std::vector<float4> initData(kAlbedoLUTSize, float4(0.f));
        for (uint32_t i = 0; i < kAlbedoLUTSize; i++) initData[i] = float4(albedos[i], 1.f);

        // Create albedo LUT texture.
        mpAlbedoLUT = mpDevice->createTexture2D(kAlbedoLUTSize, 1, kAlbedoLUTFormat, 1, 1, initData.data(), ResourceBindFlags::ShaderResource);
    }

    FALCOR_SCRIPT_BINDING(RGLMaterial)
    {
        using namespace pybind11::literals;

        FALCOR_SCRIPT_BINDING_DEPENDENCY(Material)

        pybind11::class_<RGLMaterial, Material, ref<RGLMaterial>> material(m, "RGLMaterial");
        auto create = [] (const std::string& name, const std::filesystem::path& path)
        {
            return RGLMaterial::create(accessActivePythonSceneBuilder().getDevice(), name, getActiveAssetResolver().resolvePath(path));
        };
        material.def(pybind11::init(create), "name"_a, "path"_a); // PYTHONDEPRECATED
        material.def(kLoadFile.c_str(), &RGLMaterial::loadBRDF, "path"_a);
    }
}
