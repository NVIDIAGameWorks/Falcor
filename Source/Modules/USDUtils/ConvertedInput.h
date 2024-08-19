/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
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

#include "Utils/Math/Vector.h"
#include "Scene/Transform.h"
#include "USDUtils/USDHelpers.h"
#include "USDUtils/USDUtils.h"
#include "Core/API/Formats.h"
#include <functional>
#include <string>
BEGIN_DISABLE_USD_WARNINGS
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usdShade/shader.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/base/tf/token.h>
END_DISABLE_USD_WARNINGS

namespace Falcor
{

struct ConvertedInput;

struct ConvertedTexTransform
{
    Transform transform;

    /**
     * Update transform based on TextureTransform associated with the given ConvertedInput.
     *
     * USD generally supports specifying a unique transform per texture, whereas Falcor assumes a single transform is
     * applied to all textures in a material. This function causes a warning to be emitted if the USD is not compatible
     * with Falcor.
     */
    void update(const ConvertedInput& input);
};

struct ConvertedInput
{
    enum class TextureEncoding : uint8_t
    {
        Unknown = 0,
        Linear,
        Srgb,
        Normal
    };

    ConvertedInput() : uniformValue(float4(0.f, 0.f, 0.f, 1.f)) {}
    ConvertedInput(float v) : uniformValue(float4(v, 0.f, 0.f, 1.f)) {}
    ConvertedInput(float4 v) : uniformValue(v) {}

    static ConvertedInput convertTexture(
        const pxr::UsdShadeInput& input,
        const pxr::TfToken& outputName,
        const TextureEncoding& texEncoding,
        ConvertedTexTransform& texTransform
    );

    static ConvertedInput convertFloat(const pxr::UsdShadeInput& input, const pxr::TfToken& outputName)
    {
        ConvertedTexTransform texTransform; // throwaway
        return convertFloat(input, outputName, texTransform);
    }

    static ConvertedInput convertFloat(
        const pxr::UsdShadeInput& input,
        const pxr::TfToken& outputName,
        ConvertedTexTransform& texTransform
    );

    static ConvertedInput convertColor(const pxr::UsdShadeInput& input, const pxr::TfToken& outputName)
    {
        ConvertedTexTransform texTransform; // throwaway
        return convertColor(input, outputName, TextureEncoding::Linear, texTransform);
    }

    static ConvertedInput convertColor(
        const pxr::UsdShadeInput& input,
        const pxr::TfToken& outputName,
        const TextureEncoding& texEncoding,
        ConvertedTexTransform& texTransform
    );
    static bool convertTextureCoords(const pxr::UsdShadeShader& shader, Falcor::Transform& xform);

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
    UsdShadeInput shadeInput; ///< Input associated with value (for warning/error messages, not in operator==)
};

} // namespace Falcor

namespace std
{

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
struct hash<Falcor::ConvertedInput>
{
    size_t operator()(const Falcor::ConvertedInput& i) const
    {
        size_t hash = 0;
        Falcor::hash_combine(
            hash, i.uniformValue, i.texturePath, i.texTransform, i.textureScale, static_cast<uint32_t>(i.channels), i.loadSRGB
        );
        return hash;
    }
};
} // namespace std
