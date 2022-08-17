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
#include "RGLMaterial.h"
#include "RGLFile.h"
#include "RGLCommon.h"
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
        static_assert((sizeof(MaterialHeader) + sizeof(RGLMaterialData)) <= sizeof(MaterialDataBlob), "RGLMaterialData is too large");

        const char kShaderFile[] = "Rendering/Materials/RGLMaterial.slang";

        const uint32_t kAlbedoLUTSize = RGLMaterialData::kAlbedoLUTSize;
        const ResourceFormat kAlbedoLUTFormat = ResourceFormat::RGBA32Float;

        const std::string kLoadFile = "load";
    }

    RGLMaterial::SharedPtr RGLMaterial::create(const std::string& name, const std::filesystem::path& path)
    {
        return SharedPtr(new RGLMaterial(name, path));
    }

    RGLMaterial::RGLMaterial(const std::string& name, const std::filesystem::path& path)
        : Material(name, MaterialType::RGL)
    {
        if (!loadBRDF(path))
        {
            throw RuntimeError("RGLMaterial() - Failed to load BRDF from '{}'.", path);
        }

        // Create resources for albedo lookup table.
        Sampler::Desc desc;
        desc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Point, Sampler::Filter::Point);
        desc.setAddressingMode(Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp);
        desc.setMaxAnisotropy(1);
        mpSampler = Sampler::create(desc);

        prepareAlbedoLUT(gpFramework->getRenderContext());
    }

    bool RGLMaterial::renderUI(Gui::Widgets& widget)
    {
        widget.text("RGL BRDF " + mBRDFName);
        widget.text(mBRDFDescription);
        widget.tooltip("Full path the BRDF was loaded from:\n" + mFilePath.string(), true);

        return false;
    }

    Material::UpdateFlags RGLMaterial::update(MaterialSystem* pOwner)
    {
        FALCOR_ASSERT(pOwner);

        auto flags = Material::UpdateFlags::None;
        if (mUpdates != Material::UpdateFlags::None)
        {
            auto checkBuffer = [&](Buffer::SharedPtr& buf, uint& handle)
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

    bool RGLMaterial::isEqual(const Material::SharedPtr& pOther) const
    {
        auto other = std::dynamic_pointer_cast<RGLMaterial>(pOther);
        if (!other) return false;

        if (!isBaseEqual(*other)) return false;
        if (mFilePath != other->mFilePath) return false;

        return true;
    }

    Program::ShaderModuleList RGLMaterial::getShaderModules() const
    {
        return { Program::ShaderModule(kShaderFile) };
    }

    Program::TypeConformanceList RGLMaterial::getTypeConformances() const
    {
        return { {{"RGLMaterial", "IMaterial"}, (uint32_t)MaterialType::RGL} };
    }

    bool RGLMaterial::loadBRDF(const std::filesystem::path& path)
    {
        std::filesystem::path fullPath;
        if (!findFileInDataDirectories(path, fullPath))
        {
            logWarning("RGLMaterial::loadBRDF() - Can't find file '{}'.", path);
            return false;
        }

        std::ifstream ifs(fullPath, std::ios_base::in | std::ios_base::binary);
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

        mFilePath = fullPath;
        mBRDFName = std::filesystem::path(fullPath).stem().string();
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

        mpVNDFMarginalBuf    = Buffer::create(prod3(vndfSize) * 4, ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, vndfDist.getMarginal());
        mpLumiMarginalBuf    = Buffer::create(prod3(lumiSize) * 4, ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, lumiDist.getMarginal());
        mpVNDFConditionalBuf = Buffer::create(prod4(vndfSize) * 4, ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, vndfDist.getConditional());
        mpLumiConditionalBuf = Buffer::create(prod4(lumiSize) * 4, ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, lumiDist.getConditional());

        mpThetaBuf = Buffer::create(theta->numElems * sizeof(float), ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, theta->data.get());
        mpPhiBuf   = Buffer::create(phi  ->numElems * sizeof(float), ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, phi  ->data.get());
        mpSigmaBuf = Buffer::create(sigma->numElems * sizeof(float), ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, sigma->data.get());
        mpNDFBuf   = Buffer::create(ndf  ->numElems * sizeof(float), ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, ndf  ->data.get());
        mpVNDFBuf  = Buffer::create(vndf ->numElems * sizeof(float), ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, vndfDist.getPDF());
        mpLumiBuf  = Buffer::create(lumi ->numElems * sizeof(float), ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, lumiDist.getPDF());
        mpRGBBuf   = Buffer::create(rgb  ->numElems * sizeof(float), ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, rgb  ->data.get());

        markUpdates(Material::UpdateFlags::ResourcesChanged);

        logInfo("Loaded RGL BRDF '{}': {}.", mBRDFName, mBRDFDescription);

        return true;
    }

    void RGLMaterial::prepareAlbedoLUT(RenderContext* pRenderContext) // TODO
    {
        const auto texPath = mFilePath.replace_extension("dds");

        // Try loading albedo lookup table.
        if (std::filesystem::is_regular_file(texPath))
        {
            // Load 1D texture in non-SRGB format, no mips.
            // If successful, verify dimensions/format/etc. match the expectations.
            mpAlbedoLUT = Texture::createFromFile(texPath.string(), false, false, ResourceBindFlags::ShaderResource);

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
        ImageIO::saveToDDS(gpFramework->getRenderContext(), texPath.string(), mpAlbedoLUT, ImageIO::CompressionMode::None, false);

        logInfo("Saved albedo LUT to '{}'.", texPath.string());
    }

    void RGLMaterial::computeAlbedoLUT(RenderContext* pRenderContext) // TODO
    {
        logInfo("Computing albedo LUT for RGL BRDF '{}'...", mBRDFName);

        std::vector<float> cosThetas(kAlbedoLUTSize);
        for (uint32_t i = 0; i < kAlbedoLUTSize; i++) cosThetas[i] = (float)(i + 1) / kAlbedoLUTSize;

        // Create copy of material to avoid changing our local state.
        auto pMaterial = SharedPtr(new RGLMaterial(*this));

        // Create and update scene containing the material.
        Scene::SceneData sceneData;
        sceneData.pMaterials = MaterialSystem::create();
        MaterialID materialID = sceneData.pMaterials->addMaterial(pMaterial);

        Scene::SharedPtr pScene = Scene::create(std::move(sceneData));
        pScene->update(pRenderContext, 0.0);

        // Create BSDF integrator utility.
        auto pIntegrator = BSDFIntegrator::create(pRenderContext, pScene);

        // Integrate BSDF.
        // TODO: Measured BRDFs could potentially be anisotropic.
        // It's unlikely this would affect the albedo significantly, and doing the integration
        // properly would be more trouble than its worth.
        auto albedos = pIntegrator->integrateIsotropic(pRenderContext, materialID, cosThetas);

        // Copy result into format needed for texture creation.
        static_assert(kAlbedoLUTFormat == ResourceFormat::RGBA32Float);
        std::vector<float4> initData(kAlbedoLUTSize, float4(0.f));
        for (uint32_t i = 0; i < kAlbedoLUTSize; i++) initData[i] = float4(albedos[i], 1.f);

        // Create albedo LUT texture.
        mpAlbedoLUT = Texture::create2D(kAlbedoLUTSize, 1, kAlbedoLUTFormat, 1, 1, initData.data(), ResourceBindFlags::ShaderResource);
    }

    FALCOR_SCRIPT_BINDING(RGLMaterial)
    {
        using namespace pybind11::literals;

        FALCOR_SCRIPT_BINDING_DEPENDENCY(Material)

        pybind11::class_<RGLMaterial, Material, RGLMaterial::SharedPtr> material(m, "RGLMaterial");
        material.def(pybind11::init(&RGLMaterial::create), "name"_a, "path"_a);
        material.def(kLoadFile.c_str(), &RGLMaterial::loadBRDF, "path"_a);
    }
}
