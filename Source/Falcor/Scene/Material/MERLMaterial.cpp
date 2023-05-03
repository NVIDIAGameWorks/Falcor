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
#include "MERLMaterial.h"
#include "Core/API/Device.h"
#include "Utils/Logger.h"
#include "Utils/Scripting/ScriptBindings.h"
#include "Scene/SceneBuilderAccess.h"
#include "Scene/Material/MERLFile.h"
#include "Scene/Material/MaterialSystem.h"
#include "Scene/Material/DiffuseSpecularUtils.h"

namespace Falcor
{
    namespace
    {
        static_assert((sizeof(MaterialHeader) + sizeof(MERLMaterialData)) <= sizeof(MaterialDataBlob), "MERLMaterialData is too large");

        const char kShaderFile[] = "Rendering/Materials/MERLMaterial.slang";
    }

    MERLMaterial::SharedPtr MERLMaterial::create(std::shared_ptr<Device> pDevice, const std::string& name, const std::filesystem::path& path)
    {
        return SharedPtr(new MERLMaterial(std::move(pDevice), name, path));
    }

    MERLMaterial::MERLMaterial(std::shared_ptr<Device> pDevice, const std::string& name, const std::filesystem::path& path)
        : Material(std::move(pDevice), name, MaterialType::MERL)
    {
        std::filesystem::path fullPath;
        if (!findFileInDataDirectories(path, fullPath))
        {
            throw RuntimeError("MERLMaterial: Can't find file '{}'.", path);
        }

        MERLFile merlFile(fullPath);
        init(merlFile);

        // Create albedo LUT texture.
        auto lut = merlFile.prepareAlbedoLUT(mpDevice);
        checkInvariant(!lut.empty() && sizeof(lut[0]) == sizeof(float4), "Expected albedo LUT in float4 format.");
        static_assert(MERLFile::kAlbedoLUTFormat == ResourceFormat::RGBA32Float);
        mpAlbedoLUT = Texture::create2D(mpDevice.get(), (uint32_t)lut.size(), 1, MERLFile::kAlbedoLUTFormat, 1, 1, lut.data(), ResourceBindFlags::ShaderResource);
    }

    MERLMaterial::MERLMaterial(std::shared_ptr<Device> pDevice, const MERLFile& merlFile)
        : Material(std::move(pDevice), "", MaterialType::MERL)
    {
        init(merlFile);
    }

    void MERLMaterial::init(const MERLFile& merlFile)
    {
        mPath = merlFile.getDesc().path;
        mBRDFName = merlFile.getDesc().name;
        mData.extraData = merlFile.getDesc().extraData;

        // Create GPU buffer.
        const auto& brdf = merlFile.getData();
        checkInvariant(!brdf.empty() && sizeof(brdf[0]) == sizeof(float3), "Expected BRDF data in float3 format.");
        mpBRDFData = Buffer::create(mpDevice.get(), brdf.size() * sizeof(brdf[0]), ResourceBindFlags::ShaderResource, Buffer::CpuAccess::None, brdf.data());

        // Create sampler for albedo LUT.
        Sampler::Desc desc;
        desc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Point, Sampler::Filter::Point);
        desc.setAddressingMode(Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp);
        desc.setMaxAnisotropy(1);
        mpLUTSampler = Sampler::create(mpDevice.get(), desc);

        markUpdates(Material::UpdateFlags::ResourcesChanged);
    }

    bool MERLMaterial::renderUI(Gui::Widgets& widget)
    {
        bool changed = false;

        widget.text("MERL BRDF " + mBRDFName);
        widget.tooltip("Full path the BRDF was loaded from:\n" + mPath.string(), true);

        if (auto g = widget.group("Approx diffuse/specular sampling"))
        {
            if (DiffuseSpecularUtils::renderUI(g, mData.extraData))
            {
                markUpdates(Material::UpdateFlags::DataChanged);
                changed = true;
            }
        }

        return changed;
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

    FALCOR_SCRIPT_BINDING(MERLMaterial)
    {
        using namespace pybind11::literals;

        FALCOR_SCRIPT_BINDING_DEPENDENCY(Material)

        pybind11::class_<MERLMaterial, Material, MERLMaterial::SharedPtr> material(m, "MERLMaterial");
        auto create = [] (const std::string& name, const std::filesystem::path& path)
        {
            return MERLMaterial::create(getActivePythonSceneBuilder().getDevice(), name, path);
        };
        material.def(pybind11::init(create), "name"_a, "path"_a); // PYTHONDEPRECATED
    }
}
