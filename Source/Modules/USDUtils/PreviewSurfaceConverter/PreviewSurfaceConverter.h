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
#include "Core/Pass/ComputePass.h"
#include "Scene/Material/Material.h"
#include "Scene/Material/StandardMaterial.h"
#include "StandardMaterialSpec.h"

#include "USDUtils/USDUtils.h"
#include "USDUtils/USDHelpers.h"
#include "USDUtils/ConvertedMaterialCache.h"

BEGIN_DISABLE_USD_WARNINGS
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usdShade/material.h>
END_DISABLE_USD_WARNINGS

#include <mutex>
#include <string>

namespace Falcor
{

/**
 * Class to create Falcor::Material instances from UsdPreviewSurface.
 * We extend UsdPreviewSurface by supporting additional inputs:
 *   - uniform float3 volumeAbsorption: Volume absorption coefficients
 *   - uniform float3 volumeScattering; Volume scattering coefficients
 *   - float2 normal2: 2-channel normal map
 */
class PreviewSurfaceConverter
{
public:
    /**
     * Create a new converter, compiling required compute shaders
     */
    PreviewSurfaceConverter(ref<Device> pDevice);

    /**
     * Create a Falcor material from a USD material containing a UsdPreviewSurface shader.
     * \param material UsdShadeMaterial that contains the UsdPreviewSurface to be converted
     * \return A Falcor material instance representing the UsdPreviewSurface specified in the given material, if any, nullptr otherwise.
     */
    ref<Material> convert(const pxr::UsdShadeMaterial& material, RenderContext* pRenderContext);

private:
    StandardMaterialSpec createSpec(const std::string& name, const UsdShadeShader& shader) const;
    ref<Texture> loadTexture(const ConvertedInput& ci);

    ref<Texture> createSpecularTransmissionTexture(ConvertedInput& opacity, ref<Texture> opacityTexture, RenderContext* pRenderContext);
    ref<Texture> packBaseColorAlpha(
        ConvertedInput& baseColor,
        ref<Texture> baseColorTexture,
        ConvertedInput& opacity,
        ref<Texture> opacityTexture,
        RenderContext* pRenderContext
    );
    ref<Texture> createSpecularTexture(
        ConvertedInput& roughness,
        ref<Texture> roughnessTexture,
        ConvertedInput& metallic,
        ref<Texture> metallicTexture,
        RenderContext* pRenderContext
    );

    ref<Device> mpDevice;

    ref<ComputePass> mpSpecTransPass; ///< Pass to convert opacity to transparency
    ref<ComputePass> mpPackAlphaPass; ///< Pass to convert separate RGB and A to RGBA
    ref<ComputePass> mpSpecularPass;  ///< Pass to create ORM texture

    ref<Sampler> mpSampler; ///< Bilinear clamp sampler

    ConvertedMaterialCache<StandardMaterial, StandardMaterialSpec, StandardMaterialSpecHash> mMaterialCache;

    std::mutex mMutex; ///< Mutex to ensure serial invocation of calls that are not thread safe (e.g., texture creation
                       ///< and compute program execution).
};
} // namespace Falcor
