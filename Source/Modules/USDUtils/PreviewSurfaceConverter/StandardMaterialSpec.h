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
#include "Core/API/Texture.h"
#include "Core/API/Sampler.h"
#include "Scene/Material/Material.h"
#include "Scene/Material/StandardMaterial.h"
#include "Utils/Logger.h"
#include "USDUtils/USDUtils.h"
#include "USDUtils/ConvertedInput.h"

#include <string>

namespace Falcor
{
/**
 * Falcor::StandardMaterial specification.
 * Used to hold parameters needed to construct a StandardMaterial instance prior to actually doing so.
 * Can be hashed for use in, for example, a std::unordered_map, to avoid creating duplicate materials.
 */
struct StandardMaterialSpec
{
    StandardMaterialSpec() {}
    StandardMaterialSpec(const std::string& name) : name(name) {}

    bool operator==(const StandardMaterialSpec& o) const
    {
        return texTransform.transform == o.texTransform.transform && baseColor == o.baseColor && normal == o.normal &&
               metallic == o.metallic && roughness == o.roughness && opacity == o.opacity && emission == o.emission && disp == o.disp &&
               volumeAbsorption == o.volumeAbsorption && volumeScattering == o.volumeScattering && opacityThreshold == o.opacityThreshold &&
               ior == o.ior;
    }

    std::string name;
    ConvertedTexTransform texTransform;
    ConvertedInput baseColor;
    ConvertedInput normal;
    ConvertedInput metallic;
    ConvertedInput roughness;
    ConvertedInput opacity = {1.f};
    ConvertedInput emission;
    ConvertedInput disp;
    ConvertedInput volumeAbsorption;
    ConvertedInput volumeScattering;
    float opacityThreshold = 0.f;
    float ior = 1.5f;
};

/**
 * Hash object for use by hashed containers.
 */
class StandardMaterialSpecHash
{
public:
    size_t operator()(const StandardMaterialSpec& o) const
    {
        size_t hash = 0;
        hash_combine(
            hash,
            o.texTransform.transform,
            o.baseColor,
            o.normal,
            o.metallic,
            o.roughness,
            o.opacity,
            o.emission,
            o.disp,
            o.volumeAbsorption,
            o.volumeScattering,
            o.opacityThreshold,
            o.ior
        );
        return hash;
    }
};
} // namespace Falcor
