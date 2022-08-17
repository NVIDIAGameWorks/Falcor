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
#pragma once
#include "EnvMapData.slang"
#include "Core/Macros.h"
#include "Core/API/Texture.h"
#include "Core/API/Sampler.h"
#include "Utils/Math/Vector.h"
#include "Utils/UI/Gui.h"
#include <memory>
#include <filesystem>

namespace Falcor
{
    struct ShaderVar;

    /** Environment map based radiance probe.
        Utily class for evaluating radiance stored in an lat-long environment map.
    */
    class FALCOR_API EnvMap
    {
    public:
        using SharedPtr = std::shared_ptr<EnvMap>;

        virtual ~EnvMap() = default;

        /** Create a new environment map.
            \param[in] texture The environment map texture.
        */
        static SharedPtr create(const Texture::SharedPtr& texture);

        /** Create a new environment map from file.
            \param[in] path The environment map texture file path.
            \return A new object, or nullptr if the environment map failed to load.
        */
        static SharedPtr createFromFile(const std::filesystem::path& path);

        /** Render the GUI.
        */
        void renderUI(Gui::Widgets& widgets);

        /** Set rotation angles.
            Rotation is applied as rotation around Z, Y and X axes, in that order.
            Note that glm::extractEulerAngleXYZ() may be used to extract these angles from
            a transformation matrix.
            \param[in] degreesXYZ Rotation angles in degrees for XYZ.
        */
        void setRotation(float3 degreesXYZ);

        /** Get rotation angles.
        */
        float3 getRotation() const { return mRotation; }

        /** Set intensity (scalar multiplier).
        */
        void setIntensity(float intensity);

        /** Set color tint (rgb multiplier).
        */
        void setTint(const float3& tint);

        /** Get intensity.
        */
        float getIntensity() const { return mData.intensity; }

        /** Get color tint.
        */
        float3 getTint() const { return mData.tint; }

        /** Get the file path of the environment map texture.
        */
        const std::filesystem::path& getPath() const { return mpEnvMap->getSourcePath(); }

        const Texture::SharedPtr& getEnvMap() const { return mpEnvMap; }
        const Sampler::SharedPtr& getEnvSampler() const { return mpEnvSampler; }

        /** Bind the environment map to a given shader variable.
            \param[in] var Shader variable.
        */
        void setShaderData(const ShaderVar& var) const;

        enum class Changes
        {
            None            = 0x0,
            Transform       = 0x1,
            Intensity       = 0x2,
        };

        /** Begin frame. Should be called once at the start of each frame.
        */
        Changes beginFrame();

        /** Get the environment map changes that happened in since the previous frame.
        */
        Changes getChanges() const { return mChanges; }

        /** Get the total GPU memory usage in bytes.
        */
        uint64_t getMemoryUsageInBytes() const;

    protected:
        EnvMap(const Texture::SharedPtr& texture);

        Texture::SharedPtr      mpEnvMap;           ///< Loaded environment map (RGB).
        Sampler::SharedPtr      mpEnvSampler;

        EnvMapData              mData;
        EnvMapData              mPrevData;

        float3                  mRotation = { 0.f, 0.f, 0.f };

        Changes                 mChanges = Changes::None;

        friend class SceneCache;
    };

    FALCOR_ENUM_CLASS_OPERATORS(EnvMap::Changes);
}
