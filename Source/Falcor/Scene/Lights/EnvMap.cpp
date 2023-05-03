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
#include "EnvMap.h"
#include "Core/API/Device.h"
#include "Core/Program/ShaderVar.h"
#include "Utils/Scripting/ScriptBindings.h"
#include "Scene/SceneBuilderAccess.h"
#include <glm/gtc/integer.hpp>
#include <glm/gtx/euler_angles.hpp>

namespace Falcor
{
    EnvMap::SharedPtr EnvMap::create(std::shared_ptr<Device> pDevice, const Texture::SharedPtr& pTexture)
    {
        return SharedPtr(new EnvMap(std::move(pDevice), pTexture));
    }

    EnvMap::SharedPtr EnvMap::createFromFile(std::shared_ptr<Device> pDevice, const std::filesystem::path& path)
    {
        // Load environment map from file. Set it to generate mips and use linear color.
        auto pTexture = Texture::createFromFile(pDevice.get(), path, true, false);
        if (!pTexture) return nullptr;
        return create(std::move(pDevice), pTexture);
    }

    void EnvMap::renderUI(Gui::Widgets& widgets)
    {
        auto rotation = getRotation();
        if (widgets.var("Rotation XYZ", rotation, -360.f, 360.f, 0.5f)) setRotation(rotation);
        widgets.var("Intensity", mData.intensity, 0.f, 1000000.f);
        widgets.var("Color tint", mData.tint, 0.f, 1.f);
        widgets.text("EnvMap: " + mpEnvMap->getSourcePath().string());
    }

    void EnvMap::setRotation(float3 degreesXYZ)
    {
        if (degreesXYZ != mRotation)
        {
            mRotation = degreesXYZ;

            rmcv::mat4 transform = rmcv::eulerAngleXYZ(glm::radians(mRotation.x), glm::radians(mRotation.y), glm::radians(mRotation.z));

            mData.transform = transform;
            mData.invTransform = rmcv::inverse(transform);
        }
    }

    void EnvMap::setIntensity(float intensity)
    {
        mData.intensity = intensity;
    }

    void EnvMap::setTint(const float3& tint)
    {
        mData.tint = tint;
    }

    void EnvMap::setShaderData(const ShaderVar& var) const
    {
        FALCOR_ASSERT(var.isValid());

        // Set variables.
        var["data"].setBlob(mData);

        // Bind resources.
        var["envMap"].setTexture(mpEnvMap);
        var["envSampler"].setSampler(mpEnvSampler);
    }

    EnvMap::Changes EnvMap::beginFrame()
    {
        mChanges = Changes::None;

        if (mData.transform != mPrevData.transform) mChanges |= Changes::Transform;
        if (mData.intensity != mPrevData.intensity) mChanges |= Changes::Intensity;
        if (mData.tint != mPrevData.tint) mChanges |= Changes::Intensity;

        mPrevData = mData;

        return getChanges();
    }

    uint64_t EnvMap::getMemoryUsageInBytes() const
    {
        return mpEnvMap ? mpEnvMap->getTextureSizeInBytes() : 0;
    }

    EnvMap::EnvMap(std::shared_ptr<Device> pDevice, const Texture::SharedPtr& pTexture)
        : mpDevice(std::move(pDevice))
    {
        checkArgument(mpDevice != nullptr, "'pDevice' must be a valid device");
        checkArgument(pTexture != nullptr, "'pTexture' must be a valid texture");

        mpEnvMap = pTexture;

        // Create sampler.
        // The lat-long map wraps around horizontally, but not vertically. Set the sampler to only wrap in U.
        Sampler::Desc samplerDesc;
        samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Linear);
        samplerDesc.setAddressingMode(Sampler::AddressMode::Wrap, Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp);
        mpEnvSampler = Sampler::create(mpDevice.get(), samplerDesc);
    }

    FALCOR_SCRIPT_BINDING(EnvMap)
    {
        using namespace pybind11::literals;

        pybind11::class_<EnvMap, EnvMap::SharedPtr> envMap(m, "EnvMap");
        auto createFromFile = [](const std::filesystem::path &path) {
            return EnvMap::createFromFile(getActivePythonSceneBuilder().getDevice(), path);
        };
        envMap.def(pybind11::init(createFromFile), "path"_a); // PYTHONDEPRECATED
        envMap.def_static("createFromFile", createFromFile, "path"_a);
        envMap.def_property_readonly("path", &EnvMap::getPath);
        envMap.def_property("rotation", &EnvMap::getRotation, &EnvMap::setRotation);
        envMap.def_property("intensity", &EnvMap::getIntensity, &EnvMap::setIntensity);
        envMap.def_property("tint", &EnvMap::getTint, &EnvMap::setTint);
    }
}
