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
#include "Utils.h"
#include "USDHelpers.h"
#include "Core/API/Texture.h"
#include "Core/API/Sampler.h"
#include "RenderGraph/BasePasses/ComputePass.h"
#include "Scene/Material/Material.h"
#include "Scene/Material/StandardMaterial.h"

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
        /** Create a new converter, compiling required compute shaders.
        */
        PreviewSurfaceConverter();

        /** Create a Falcor material from a material containing a UsdPreviewSurface.
            \param material UsdShadeMaterial that contains the UsdPreviewSurface to be converted
            \param primName Name of the primitive to which the material is bound (for warning message reporting only).
            \return A Falcor material instance representing the UsdPreviewSurface specified in the given material, if any, nullptr otherwise.
        */
        Material::SharedPtr convert(const pxr::UsdShadeMaterial& material, const std::string& primName, RenderContext* pRenderContext);

    private:
        struct ConvertedInput
        {
            ConvertedInput() :
                uniformValue(float4(0.f, 0.f, 0.f, 1.f))
            {
            }
            ConvertedInput(float v) :
                uniformValue(float4(v, 0.f, 0.f, 1.f))
            {
            }
            ConvertedInput(float4 v) :
                uniformValue(v)
            {
            }
            Texture::SharedPtr pTexture;
            TextureChannelFlags channels = TextureChannelFlags::None;
            float4 uniformValue;
        };

        Texture::SharedPtr createSpecularTransmissionTexture(ConvertedInput& opacity, RenderContext* pRenderContext);
        Texture::SharedPtr packBaseColorAlpha(ConvertedInput& baseColor, ConvertedInput& opacity, RenderContext* pRenderContext);
        Texture::SharedPtr createSpecularTexture(ConvertedInput& roughness, ConvertedInput& metallic, RenderContext* pRenderContext);

        ConvertedInput convertTexture(const pxr::UsdShadeInput& input, bool assumeSrgb = false);
        ConvertedInput convertFloat(const pxr::UsdShadeInput& input);
        ConvertedInput convertColor(const pxr::UsdShadeInput& input, bool assumeSrgb = false);

        StandardMaterial::SharedPtr getCachedMaterial(const pxr::UsdShadeShader& shader);
        void cacheMaterial(const UsdShadeShader& shader, StandardMaterial::SharedPtr pMaterial);

        ComputePass::SharedPtr mpSpecTransPass;                 //< Pass to convert opacity to transparency
        ComputePass::SharedPtr mpPackAlphaPass;                 //< Pass to convert separate RGB and A to RGBA
        ComputePass::SharedPtr mpSpecularPass;                  //< Pass to create ORM texture

        Sampler::SharedPtr mpSampler;                           //< Bilinear clamp sampler

        std::unordered_map<pxr::UsdPrim, StandardMaterial::SharedPtr, UsdObjHash> mMaterialCache;    //< Map from UsdPreviewSurface-defining UsdShadeShader to Falcor material instance. An entry with a null instance indicates in-progress conversion.

        std::mutex mMutex;                                      //< Mutex to ensure serial invocation of calls that are not thread safe (e.g., texture creation and compute program execution).
        std::mutex mCacheMutex;                                 //< Mutex controlling access to the material cache.
        std::condition_variable mCacheUpdated;                  //< Condition variable for threads waiting on cache update.
    };
}
