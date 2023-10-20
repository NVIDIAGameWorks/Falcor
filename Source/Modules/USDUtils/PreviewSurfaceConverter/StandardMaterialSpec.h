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
#pragma once
#include "Core/API/Texture.h"
#include "Core/API/Sampler.h"
#include "Scene/Material/Material.h"
#include "Scene/Material/StandardMaterial.h"
#include "Utils/Logger.h"

#include <string>

namespace Falcor
{
/**
 * We avoid creating duplicate material instances by maintaing a cache that maps from StandardMaterialSpec to
 * Falcor::StandardMaterial. Doing so requires that we compute a hash for a StandardMaterialSpec. The code here, and
 * below in the std namespace, allows us to do so.
 */

// Recursion bottom-out
inline void hash_combine(std::size_t& seed) {}

// Varadic helper function to combine hash values. Adapted from boost::hash_combine
template<typename T, typename... Args>
inline void hash_combine(std::size_t& seed, const T& v, Args... args)
{
    seed ^= std::hash<T>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    hash_combine(seed, args...);
}

/**
 * Falcor::StandardMaterial specification.
 * Used to hold parameters needed to construct a StandardMaterial instance prior to actually doing so.
 * Can be hashed for use in, for example, a std::unordered_map, to avoid creating duplicate materials.
 */
struct StandardMaterialSpec
{
    struct TextureTransform
    {
        float2 scale = float2(1.f, 1.f);
        float2 translate = float2(0.f, 0.f);
        float rotate = 0.f;

        bool isIdentity() const { return rotate == 0.f && scale.x == 1.f && scale.y && 1.f && translate.x == 0.f && translate.y == 0.f; }

        bool operator==(const TextureTransform& other) const
        {
            return all(scale == other.scale) && all(translate == other.translate) && rotate == other.rotate;
        }

        bool operator!=(const TextureTransform& other) const
        {
            return any(scale != other.scale) || any(translate != other.translate) || rotate != other.rotate;
        }
    };

    struct ConvertedInput
    {
        ConvertedInput() : uniformValue(float4(0.f, 0.f, 0.f, 1.f)) {}
        ConvertedInput(float v) : uniformValue(float4(v, 0.f, 0.f, 1.f)) {}
        ConvertedInput(float4 v) : uniformValue(v) {}

        bool isTextured() const { return !texturePath.empty(); }

        bool operator==(const ConvertedInput& o) const
        {
            return texturePath == o.texturePath && texTransform == o.texTransform && all(textureScale == o.textureScale) &&
                   channels == o.channels && all(uniformValue == o.uniformValue) && loadSRGB == o.loadSRGB;
        }

        float4 uniformValue = float4(0.f, 0.f, 0.f, 0.f);         ///< Uniform value, may only hold a single valid component for
                                                                  ///< a float, and so on.
        std::string texturePath;                                  ///< Path to texture file.
        Falcor::Transform texTransform;                           ///< 2D transformation applied to texture coordinates
        float4 textureScale = float4(1.f, 1.f, 1.f, 1.f);         ///< Texture value scale; valid only for emissive component.
                                                                  ///< Alpha may be specified, but is ignored.
        TextureChannelFlags channels = TextureChannelFlags::None; ///< Texture channels that hold the data.
        bool loadSRGB = false;                                    ///< If true, texture should be assumed to hold sRGB data.
    };

    StandardMaterialSpec() {}
    StandardMaterialSpec(const std::string& name) : name(name) {}

    /**
     * Update texTransform based on TextureTransform associated with the given ConvertedInput.
     *
     * USD generally supports specifying a unique transform per texture, whereas Falcor assumes a single transform is
     * applied to all textures in a material. This function causes a warning to be emitted if the USD is not compatible
     * with Falcor.
     */
    void updateTexTransform(StandardMaterialSpec::ConvertedInput& input)
    {
        if (!input.isTextured())
            return;
        const Falcor::Transform& newTransform = input.texTransform;
        if (newTransform.getMatrix() != Falcor::float4x4::identity() && texTransform.getMatrix() == Falcor::float4x4::identity())
        {
            texTransform = newTransform;
        }
        else if (newTransform.getMatrix() != texTransform.getMatrix())
        {
            Falcor::logWarning(
                "Material '{}' uses more than one unique texture transform, which is not supported. Applying the first "
                "encountered non-idenity transform to all textures.",
                name
            );
        }
    };

    bool operator==(const StandardMaterialSpec& o) const
    {
        return texTransform == o.texTransform && baseColor == o.baseColor && normal == o.normal && metallic == o.metallic &&
               roughness == o.roughness && opacity == o.opacity && emission == o.emission && disp == o.disp &&
               opacityThreshold == o.opacityThreshold && ior == o.ior;
    }

    std::string name;
    Transform texTransform;
    ConvertedInput baseColor;
    ConvertedInput normal;
    ConvertedInput metallic;
    ConvertedInput roughness;
    ConvertedInput opacity = {1.f};
    ConvertedInput emission;
    ConvertedInput disp;
    float opacityThreshold = 0.f;
    float ior = 1.5f;
};

/**
 * Hash object for use by hashed containers.
 */
class SpecHash
{
public:
    size_t operator()(const StandardMaterialSpec& o) const
    {
        size_t hash = 0;
        hash_combine(
            hash, o.texTransform, o.baseColor, o.normal, o.metallic, o.roughness, o.opacity, o.emission, o.disp, o.opacityThreshold, o.ior
        );
        return hash;
    }
};
} // namespace Falcor

namespace std
{
// Hash objects for types comprising StandardMaterialSpec

template<>
struct hash<Falcor::StandardMaterialSpec::TextureTransform>
{
    size_t operator()(const Falcor::StandardMaterialSpec::TextureTransform& t) const
    {
        size_t hash = 0;
        Falcor::hash_combine(hash, t.scale, t.translate, t.rotate);
        return hash;
    }
};

template<>
struct hash<Falcor::Transform>
{
    size_t operator()(const Falcor::Transform& t) const
    {
        size_t hash = 0;
        Falcor::hash_combine(hash, t.getTranslation(), t.getScaling(), t.getRotation(), static_cast<uint32_t>(t.getCompositionOrder()));
        return hash;
    }
};

template<>
struct hash<Falcor::StandardMaterialSpec::ConvertedInput>
{
    size_t operator()(const Falcor::StandardMaterialSpec::ConvertedInput& i) const
    {
        size_t hash = 0;
        Falcor::hash_combine(
            hash, i.uniformValue, i.texturePath, i.texTransform, i.textureScale, static_cast<uint32_t>(i.channels), i.loadSRGB
        );
        return hash;
    }
};
} // namespace std
