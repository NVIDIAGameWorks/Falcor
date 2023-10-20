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
#include "Core/Pass/ComputePass.h"
#include "Scene/Material/Material.h"
#include "Scene/Material/StandardMaterial.h"
#include "StandardMaterialSpec.h"

#include "USDUtils/USDUtils.h"
#include "USDUtils/USDHelpers.h"

BEGIN_DISABLE_USD_WARNINGS
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usdShade/material.h>
END_DISABLE_USD_WARNINGS

#include <condition_variable>
#include <mutex>
#include <string>
#include <unordered_map>

namespace Falcor
{

/** Class to create Falcor::Material instances from UsdPreviewSurfaces.
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
     * \param primName Name of the primitive to which the material is bound (for warning message reporting only).
     * \return A Falcor material instance representing the UsdPreviewSurface specified in the given material, if any, nullptr otherwise.
     */
    ref<Material> convert(const pxr::UsdShadeMaterial& material, const std::string& primName, RenderContext* pRenderContext);

private:
    StandardMaterialSpec createSpec(const std::string& name, const UsdShadeShader& shader) const;
    ref<Texture> loadTexture(const StandardMaterialSpec::ConvertedInput& ci);

    ref<Texture> createSpecularTransmissionTexture(
        StandardMaterialSpec::ConvertedInput& opacity,
        ref<Texture> opacityTexture,
        RenderContext* pRenderContext
    );
    ref<Texture> packBaseColorAlpha(
        StandardMaterialSpec::ConvertedInput& baseColor,
        ref<Texture> baseColorTexture,
        StandardMaterialSpec::ConvertedInput& opacity,
        ref<Texture> opacityTexture,
        RenderContext* pRenderContext
    );
    ref<Texture> createSpecularTexture(
        StandardMaterialSpec::ConvertedInput& roughness,
        ref<Texture> roughnessTexture,
        StandardMaterialSpec::ConvertedInput& metallic,
        ref<Texture> metallicTexture,
        RenderContext* pRenderContext
    );

    StandardMaterialSpec::ConvertedInput convertTexture(
        const pxr::UsdShadeInput& input,
        const TfToken& outputName,
        bool assumeSrgb = false,
        bool scaleSupported = false
    ) const;
    StandardMaterialSpec::ConvertedInput convertFloat(const pxr::UsdShadeInput& input, const TfToken& outputName) const;
    StandardMaterialSpec::ConvertedInput convertColor(
        const pxr::UsdShadeInput& input,
        const TfToken& outputName,
        bool assumeSrgb = false,
        bool scaleSupported = false
    ) const;

    ref<StandardMaterial> getCachedMaterial(const pxr::UsdShadeShader& shader);
    ref<StandardMaterial> getCachedMaterial(const StandardMaterialSpec& spec);

    void cacheMaterial(const UsdShadeShader& shader, ref<StandardMaterial> pMaterial);
    void cacheMaterial(const StandardMaterialSpec& spec, ref<StandardMaterial> pMaterial);

    ref<Device> mpDevice;

    ref<ComputePass> mpSpecTransPass; ///< Pass to convert opacity to transparency
    ref<ComputePass> mpPackAlphaPass; ///< Pass to convert separate RGB and A to RGBA
    ref<ComputePass> mpSpecularPass;  ///< Pass to create ORM texture

    ref<Sampler> mpSampler; ///< Bilinear clamp sampler

    ///< Map from UsdPreviewSurface-defining UsdShadeShader to Falcor material instance. An entry with a null instance
    ///< indicates in-progress conversion.
    std::unordered_map<pxr::UsdPrim, ref<StandardMaterial>, UsdObjHash> mPrimMaterialCache;

    ///< Map from StandardMaterialSpec to Falcor material instance. An entry with a null instance indicates in-progress
    ///< conversion.
    std::unordered_map<StandardMaterialSpec, ref<StandardMaterial>, SpecHash> mSpecMaterialCache;

    std::mutex mMutex;      ///< Mutex to ensure serial invocation of calls that are not thread safe (e.g., texture creation
                            ///< and compute program execution).
    std::mutex mCacheMutex; ///< Mutex controlling access to the material caches.
    std::condition_variable mPrimCacheUpdated; ///< Condition variable for threads waiting on by-prim cache update.
    std::condition_variable mSpecCacheUpdated; ///< Condition variable for threads waiting on by-spec cache update.
};
} // namespace Falcor
