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
#include "PreviewSurfaceConverter.h"
#include "Core/API/RenderContext.h"
#include "Utils/Image/ImageIO.h"

#include "USDUtils/USDUtils.h"
#include "USDUtils/USDHelpers.h"

BEGIN_DISABLE_USD_WARNINGS
#include <pxr/usd/usdGeom/xformCommonAPI.h>
#include <pxr/usd/usdRender/tokens.h>
#include <pxr/usd/ar/resolver.h>
#include <pxr/usd/sdf/layerUtils.h>
END_DISABLE_USD_WARNINGS

using namespace pxr;

namespace Falcor
{
namespace
{
const char kSpecTransShaderFile[]("Modules/USDUtils/PreviewSurfaceConverter/CreateSpecularTransmissionTexture.cs.slang");
const char kSpecTransShaderEntry[]("createSpecularTransmissionTexture");

const char kPackAlphaShaderFile[]("Modules/USDUtils/PreviewSurfaceConverter/PackBaseColorAlpha.cs.slang");
const char kPackAlphaShaderEntry[]("packBaseColorAlpha");

const char kSpecularShaderFile[]("Modules/USDUtils/PreviewSurfaceConverter/CreateSpecularTexture.cs.slang");
const char kSpecularShaderEntry[]("createSpecularTexture");

inline int32_t getChannelIndex(TextureChannelFlags flags)
{
    switch (flags)
    {
    case TextureChannelFlags::None:
        return -1;
    case TextureChannelFlags::Red:
        return 0;
    case TextureChannelFlags::Green:
        return 1;
    case TextureChannelFlags::Blue:
        return 2;
    case TextureChannelFlags::Alpha:
        return 3;
    default:
        FALCOR_UNREACHABLE();
    }
    return -1;
}
} // namespace

PreviewSurfaceConverter::PreviewSurfaceConverter(ref<Device> pDevice) : mpDevice(pDevice)
{
    mpSpecTransPass = ComputePass::create(mpDevice, kSpecTransShaderFile, kSpecTransShaderEntry);

    mpPackAlphaPass = ComputePass::create(mpDevice, kPackAlphaShaderFile, kPackAlphaShaderEntry);

    mpSpecularPass = ComputePass::create(mpDevice, kSpecularShaderFile, kSpecularShaderEntry);

    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Point);
    samplerDesc.setAddressingMode(TextureAddressingMode::Clamp, TextureAddressingMode::Clamp, TextureAddressingMode::Clamp);

    mpSampler = mpDevice->createSampler(samplerDesc);
}

// Convert textured opacity to textured specular transparency.
ref<Texture> PreviewSurfaceConverter::createSpecularTransmissionTexture(
    ConvertedInput& opacity,
    ref<Texture> opacityTexture,
    RenderContext* pRenderContext
)
{
    std::scoped_lock lock(mMutex);

    if (popcount((uint32_t)opacity.channels) > 1)
    {
        logWarning("Cannot create transmission texture; opacity texture provides more than one channel of data.");
        return nullptr;
    }

    if (opacityTexture == nullptr)
    {
        return nullptr;
    }

    uint2 resolution(opacityTexture->getWidth(), opacityTexture->getHeight());
    ref<Texture> pTexture = mpDevice->createTexture2D(
        resolution.x,
        resolution.y,
        ResourceFormat::RGBA8Unorm,
        1,
        Texture::kMaxPossible,
        nullptr,
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::RenderTarget
    );

    auto var = mpSpecTransPass->getRootVar();
    var["CB"]["outDim"] = resolution;
    var["CB"]["opacityChannel"] = getChannelIndex(opacity.channels);
    var["opacityTexture"] = opacityTexture;
    var["outputTexture"] = pTexture;

    mpSpecTransPass->execute(pRenderContext, resolution.x, resolution.y);

    pTexture->generateMips(pRenderContext);

    return pTexture;
}

/**
 * Combine base color and alpha, one or both of which may be textured, into a single texture.
 * If both are textured, they may be of different resolutions.
 */
ref<Texture> PreviewSurfaceConverter::packBaseColorAlpha(
    ConvertedInput& baseColor,
    ref<Texture> baseColorTexture,
    ConvertedInput& opacity,
    ref<Texture> opacityTexture,
    RenderContext* pRenderContext
)
{
    std::scoped_lock lock(mMutex);

    if (opacityTexture && popcount((uint32_t)opacity.channels) > 1)
    {
        logWarning("Cannot set alpha channel; opacity texture provides more than one channel.");
        return nullptr;
    }

    // Set output resolution to the maxium of the input dimensions
    uint32_t width =
        std::max<uint32_t>((opacityTexture ? opacityTexture->getWidth() : 0), (baseColorTexture ? baseColorTexture->getWidth() : 0));
    uint32_t height =
        std::max<uint32_t>((opacityTexture ? opacityTexture->getHeight() : 0), (baseColorTexture ? baseColorTexture->getHeight() : 0));
    uint2 resolution(width, height);

    // sRGB format textures can't be bound as UAVs. Render to an RGBA32Float intermediate texture, and then blit to
    // RGBA8UnormSrgb to preserve precision near zero.
    ref<Texture> pTexture = mpDevice->createTexture2D(
        resolution.x,
        resolution.y,
        ResourceFormat::RGBA32Float,
        1,
        1,
        nullptr,
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
    );

    auto var = mpPackAlphaPass->getRootVar();
    var["CB"]["opacityChannel"] = opacityTexture ? getChannelIndex(opacity.channels) : -1;
    var["CB"]["opacityValue"] = opacity.uniformValue.r;
    var["CB"]["baseColorTextureValid"] = baseColorTexture ? 1 : 0;
    var["CB"]["baseColorValue"] = baseColor.uniformValue;
    var["CB"]["outDim"] = resolution;
    var["opacityTexture"] = opacityTexture;
    var["baseColorTexture"] = baseColorTexture;
    var["sampler"] = mpSampler;
    var["outputTexture"] = pTexture;

    mpPackAlphaPass->execute(pRenderContext, resolution.x, resolution.y);

    // Create the output mipmapped sRGB texture
    ref<Texture> pFinal = mpDevice->createTexture2D(
        resolution.x,
        resolution.y,
        ResourceFormat::RGBA8UnormSrgb,
        1,
        Texture::kMaxPossible,
        nullptr,
        ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget
    );

    // Blit the intermediate texture to the output texture to perform format conversion
    pRenderContext->blit(pTexture->getSRV(), pFinal->getRTV());

    pFinal->generateMips(pRenderContext);

    return pFinal;
}

// Combine roughness and metallic parameters, one or both of which are textured, into a specular/ORM texture.
// If both are textured, they may be of different resolutions.
ref<Texture> PreviewSurfaceConverter::createSpecularTexture(
    ConvertedInput& roughness,
    ref<Texture> roughnessTexture,
    ConvertedInput& metallic,
    ref<Texture> metallicTexture,
    RenderContext* pRenderContext
)
{
    std::scoped_lock lock(mMutex);

    if (roughnessTexture && popcount((uint32_t)roughness.channels) > 1)
    {
        logWarning("Cannot create specular texture; roughness texture provides more than one channel.");
        return nullptr;
    }
    if (metallicTexture && popcount((uint32_t)metallic.channels) > 1)
    {
        logWarning("Cannot create specular texture; metallic texture provides more than one channel.");
        return nullptr;
    }

    // Set output resolution to the maxium of the input dimensions
    uint32_t width =
        std::max<uint32_t>((roughnessTexture ? roughnessTexture->getWidth() : 0), (metallicTexture ? metallicTexture->getWidth() : 0));
    uint32_t height =
        std::max<uint32_t>((roughnessTexture ? roughnessTexture->getHeight() : 0), (metallicTexture ? metallicTexture->getHeight() : 0));
    uint2 resolution(width, height);

    ref<Texture> pTexture = mpDevice->createTexture2D(
        width,
        height,
        ResourceFormat::RGBA8Unorm,
        1,
        Texture::kMaxPossible,
        nullptr,
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::RenderTarget
    );

    auto var = mpSpecularPass->getRootVar();
    var["CB"]["roughnessChannel"] = roughnessTexture ? getChannelIndex(roughness.channels) : -1;
    var["CB"]["roughnessValue"] = roughness.uniformValue.r;
    var["CB"]["metallicChannel"] = metallicTexture ? getChannelIndex(metallic.channels) : -1;
    var["CB"]["metallicValue"] = metallic.uniformValue.r;
    var["CB"]["outDim"] = resolution;
    var["metallicTexture"] = metallicTexture;
    var["roughnessTexture"] = roughnessTexture;
    var["sampler"] = mpSampler;
    var["outputTexture"] = pTexture;

    mpSpecularPass->execute(pRenderContext, resolution.x, resolution.y);

    pTexture->generateMips(pRenderContext);

    return pTexture;
}

ref<Texture> PreviewSurfaceConverter::loadTexture(const ConvertedInput& ci)
{
    if (ci.texturePath.empty())
    {
        return nullptr;
    }
    if (hasExtension(ci.texturePath, ".dds"))
    {
        // It would be better if we could separate texture file reading, which is relatlvely
        // slow but can run in parallel, from texture creation, which must be single-threaded,
        // as done below.
        std::scoped_lock lock(mMutex);
        return ImageIO::loadTextureFromDDS(mpDevice, ci.texturePath, ci.loadSRGB);
    }
    else
    {
        // Create the texture by first reading the image (which is relatively slow) outside of the mutex,
        // and then creating the texture itself inside it.
        Bitmap::UniqueConstPtr pBitmap(Bitmap::createFromFile(ci.texturePath, true /* topDown */));
        if (pBitmap)
        {
            ResourceFormat format = pBitmap->getFormat();
            if (ci.loadSRGB)
            {
                format = linearToSrgbFormat(format);
            }
            {
                std::scoped_lock lock(mMutex);
                return mpDevice->createTexture2D(
                    pBitmap->getWidth(), pBitmap->getHeight(), format, 1, Texture::kMaxPossible, pBitmap->getData()
                );
            }
        }
    }
    return nullptr;
}

// Traverse the graph rooted at the given UsdPreviewSurface shader node, construcing and returning a corresponding
// StandardMaterialSpec.
StandardMaterialSpec PreviewSurfaceConverter::createSpec(const std::string& name, const UsdShadeShader& shader) const
{
    StandardMaterialSpec spec(name);

    std::vector<UsdShadeInput> shadeInputs(shader.GetInputs());

    bool useNormal2 =
        std::find_if(shadeInputs.begin(), shadeInputs.end(), [](const UsdShadeInput& input) { return input.GetBaseName() == "normal2"; }) !=
        shadeInputs.end();

    for (auto& curInput : shadeInputs)
    {
        logDebug("UsdPreviewSurface '{}' has input: {}", shader.GetPath().GetString(), curInput.GetBaseName().GetString());

        // Convert the input name to lowercase to allow for mixed capitalization
        std::string inputName = curInput.GetBaseName();
        std::transform(inputName.begin(), inputName.end(), inputName.begin(), ::tolower);

        // Get the most-upstream source of the input.
        UsdShadeConnectableAPI source;
        TfToken sourceName;
        UsdShadeAttributeType sourceType;
        UsdShadeInput input = getSourceInput(curInput, source, sourceName, sourceType);

        SdfValueTypeName typeName(input.GetTypeName());
        logDebug(
            "UsdPreviewSurface '{}' input '{}' has type '{}'.", shader.GetPath().GetString(), inputName, typeName.GetAsToken().GetString()
        );

        if (inputName == "diffusecolor")
        {
            // Get diffuse color value(s), assume that textures are sRGB by default.
            spec.baseColor = ConvertedInput::convertColor(input, sourceName, ConvertedInput::TextureEncoding::Srgb, spec.texTransform);
        }
        else if (inputName == "emissivecolor")
        {
            // Get color value(s), assume that textures are sRGB by default.
            spec.emission = ConvertedInput::convertColor(input, sourceName);
        }
        else if (inputName == "usespecularworkflow")
        {
            if (typeName == SdfValueTypeNames->Int)
            {
                int specularWorkflow;
                if (input.Get<int>(&specularWorkflow, UsdTimeCode::EarliestTime()))
                {
                    if (specularWorkflow)
                    {
                        logWarning("Specular workflow is not supported.");
                    }
                }
            }
            else
            {
                logWarning("Unsupported specular workflow value type: '{}'.", typeName.GetAsToken().GetString());
            }
        }
        else if (inputName == "metallic")
        {
            spec.metallic = ConvertedInput::convertFloat(input, sourceName, spec.texTransform);
        }
        else if (inputName == "specularcolor")
        {
            if (typeName == SdfValueTypeNames->Color3f)
            {
                logInfo("Ignoring specular color.");
            }
            else if (typeName == SdfValueTypeNames->Asset)
            {
                logInfo("Ignoring specular texture.");
            }
            else
            {
                logInfo("Unsupported specular color value type: '{}'.", typeName.GetAsToken().GetString());
            }
        }
        else if (inputName == "roughness")
        {
            spec.roughness = ConvertedInput::convertFloat(input, sourceName, spec.texTransform);
        }
        else if (inputName == "clearcoat")
        {
            logWarning("Falcor's standard material does not support clearcoat.");
        }
        else if (inputName == "clearcoatroughness")
        {
            logWarning("Falcor's standard material does not support clearcoat roughness.");
        }
        else if (inputName == "opacity")
        {
            spec.opacity = ConvertedInput::convertFloat(input, sourceName, spec.texTransform);
        }
        else if (inputName == "opacitythreshold")
        {
            if (typeName == SdfValueTypeNames->Float)
            {
                input.Get<float>(&spec.opacityThreshold, UsdTimeCode::EarliestTime());
            }
            else if (typeName == SdfValueTypeNames->Asset)
            {
                logWarning("Falcor's standard material does not support textured opacity threshold.");
            }
            else
            {
                logWarning("Unsupported opacity threshold value type: '{}'.", typeName.GetAsToken().GetString());
            }
        }
        else if (inputName == "ior")
        {
            if (typeName == SdfValueTypeNames->Float)
            {
                input.Get<float>(&spec.ior, UsdTimeCode::EarliestTime());
            }
            else if (typeName == SdfValueTypeNames->Asset)
            {
                logWarning("Falcor's standard material does not support textured IoR.");
            }
            else
            {
                logWarning("Unsupported ior value type: '{}'.", typeName.GetAsToken().GetString());
            }
        }
        else if (inputName == "normal" || inputName == "normal2")
        {
            // The extra logic here allows us to suppress the 'unsupported' warning message below.
            if ((inputName == "normal" && !useNormal2) || (inputName == "normal2" && useNormal2))
            {
                if (typeName == SdfValueTypeNames->Asset)
                {
                    spec.normal =
                        ConvertedInput::convertTexture(input, sourceName, ConvertedInput::TextureEncoding::Normal, spec.texTransform);
                }
                else
                {
                    logWarning("Falcor's standard material does not support uniform normal values.");
                }
            }
        }
        else if (inputName == "displacement")
        {
            spec.disp = ConvertedInput::convertFloat(input, sourceName, spec.texTransform);
            if (!spec.disp.isTextured())
            {
                logWarning("Falcor's standard material does not support uniform displacement.");
            }
        }
        else if (inputName == "occlusion")
        {
            logWarning("Falcor's standard material does not support an occlusion parameter.");
        }
        else if (inputName == "volumeabsorption")
        {
            spec.volumeAbsorption = ConvertedInput::convertColor(input, sourceName);
        }
        else if (inputName == "volumescattering")
        {
            spec.volumeScattering = ConvertedInput::convertColor(input, sourceName);
        }
        else
        {
            logWarning("Unsupported UsdPreviewSurface input '{}'.", inputName);
        }
    }
    return spec;
}

ref<Material> PreviewSurfaceConverter::convert(const UsdShadeMaterial& material, RenderContext* pRenderContext)
{
    UsdShadeShader shader = material.ComputeSurfaceSource();
    std::string materialName = material.GetPath().GetString();

    if (!shader)
    {
        logDebug("Material '{}': could not find surface output UsdShadeShader.", materialName);
        return nullptr;
    }

    TfToken surfaceId = getAttribute(shader.GetPrim().GetAttribute(UsdShadeTokens->infoId), TfToken());
    if (surfaceId != TfToken("UsdPreviewSurface"))
    {
        logDebug("Material '{}' has a surface output node of type '{}', not UsdPreviewSurface.", materialName, surfaceId.GetString());
        return nullptr;
    }

    logDebug("Material '{}' has output '{}'.", materialName, shader.GetPath().GetString());

    // Is there a valid cached instance for this material prim?  If so, simply return it.
    // Note that this call will block if another thread is concurrently converting the same material.
    ref<StandardMaterial> pMaterial = mMaterialCache.get(shader);
    if (pMaterial)
    {
        return pMaterial;
    }

    // Due to the fact that threads may be waiting for material conversion to complete, the following code must always
    // create non-null by-prim and by-spec material cache entries.

    // Construct a StandardMaterialSpec before creating the StandardMaterial itself from it in order to
    // avoid creating duplicate material instances.
    StandardMaterialSpec spec = createSpec(materialName, shader);

    // Does there already exist a material matching this spec? If so, return it.
    pMaterial = mMaterialCache.get(spec, spec.name);
    if (pMaterial)
    {
        // There is an entry for this material in the by-spec cache. Ensure that there is also an entry in the by-prim
        // cache.
        mMaterialCache.add(shader, spec, pMaterial);
        return pMaterial;
    }

    // Create the output material
    pMaterial = StandardMaterial::create(mpDevice, materialName);

    // In USD, double-sidedness is a geometric property, while in Falcor it's a material property.
    // For now, force all materials to be double-sided.
    pMaterial->setDoubleSided(true);

    ref<Texture> baseColorTexture = loadTexture(spec.baseColor);
    ref<Texture> opacityTexture = loadTexture(spec.opacity);
    ref<Texture> roughnessTexture = loadTexture(spec.roughness);
    ref<Texture> metallicTexture = loadTexture(spec.metallic);
    ref<Texture> normalTexture = loadTexture(spec.normal);
    ref<Texture> emissionTexture = loadTexture(spec.emission);
    ref<Texture> displacementTexture = loadTexture(spec.disp);

    pMaterial->setIndexOfRefraction(spec.ior);

    // If there is either a roughness or metallic texture, convert texture(s) and constant (if any) to an ORM texture.
    if (metallicTexture || roughnessTexture)
    {
        ref<Texture> pSpecularTex = createSpecularTexture(spec.roughness, roughnessTexture, spec.metallic, metallicTexture, pRenderContext);
        pMaterial->setSpecularTexture(pSpecularTex);
    }
    else
    {
        pMaterial->setSpecularParams(float4(0.f, spec.roughness.uniformValue.r, spec.metallic.uniformValue.r, 1.f));
    }

    if (spec.opacity.uniformValue.r < 1.f || opacityTexture)
    {
        pMaterial->setAlphaThreshold(spec.opacityThreshold);
        if (spec.opacityThreshold > 0.f)
        {
            // Opacity encodes cutout values
            // Pack opacity into the alpha channel
            if (baseColorTexture || opacityTexture)
            {
                baseColorTexture = packBaseColorAlpha(spec.baseColor, baseColorTexture, spec.opacity, opacityTexture, pRenderContext);
            }
            else
            {
                spec.baseColor.uniformValue = float4(spec.baseColor.uniformValue.xyz(), spec.opacity.uniformValue.r);
            }
        }
        else if (opacityTexture)
        {
            // Opacity encodes (1 - specular-transmission)
            // Create a greyscale specular transmission color texture using (1-opacity), as a slightly hacky means of
            // supporting textured specular transmission.
            logDebug(
                "UsdPreviewSurface '{}' has texture-mapped opacity. Converting to textured specular transmission.",
                shader.GetPath().GetString()
            );
            ref<Texture> transmissionTexture = createSpecularTransmissionTexture(spec.opacity, opacityTexture, pRenderContext);
            pMaterial->setTransmissionTexture(transmissionTexture);
            pMaterial->setSpecularTransmission(1.f);
        }
        else
        {
            logDebug(
                "UsdPreviewSurface '{}' has uniform opacity. Converting to uniform specular transmission.", shader.GetPath().GetString()
            );
            pMaterial->setSpecularTransmission(1.f - spec.opacity.uniformValue.r);
        }
    }

    // Base color
    if (baseColorTexture)
    {
        pMaterial->setBaseColorTexture(baseColorTexture);
    }
    else
    {
        pMaterial->setBaseColor(spec.baseColor.uniformValue);
    }

    // Normal
    if (normalTexture)
    {
        pMaterial->setNormalMap(normalTexture);
    }

    // Emission
    if (emissionTexture)
    {
        pMaterial->setEmissiveTexture(emissionTexture);
    }
    else
    {
        pMaterial->setEmissiveColor(spec.emission.uniformValue.xyz());
    }

    if (any(spec.emission.textureScale != float4(1.f, 1.f, 1.f, 1.f)))
    {
        if (spec.emission.textureScale.x != spec.emission.textureScale.y || spec.emission.textureScale.x != spec.emission.textureScale.z ||
            spec.emission.textureScale.x != spec.emission.textureScale.a)
        {
            logWarning(
                "UsdPreviewSurface '{}' input 'emissiveColor' specifies a vector texture value scale. Applying red "
                "component to all channels.",
                shader.GetPath().GetString()
            );
        }
        pMaterial->setEmissiveFactor(spec.emission.textureScale.x);
    }

    if (displacementTexture)
    {
        pMaterial->setDisplacementMap(displacementTexture);
    }

    // Here, we apply a y-flip transformation to the texture, along with any USD-specified transform.
    // We perform a flip because textures are stored in top-down order, but we address them assuming
    // bottom-up order, per the USD spec.
    Transform texTrans;
    if (spec.texTransform.transform.getMatrix() != float4x4::identity())
    {
        // Compute the inverse of the texture transform, as Falcor assumes
        // the transform applies to the texture, rather than the texture coordinates,
        // as UsdPreviewSurface does.
        FALCOR_ASSERT(spec.texTransform.transform.getCompositionOrder() == Transform::CompositionOrder::ScaleRotateTranslate);
        texTrans.setTranslation(-spec.texTransform.transform.getTranslation());
        float3 scale = spec.texTransform.transform.getScaling();
        float3 rot = spec.texTransform.transform.getRotationEuler();
        texTrans.setRotationEuler(-rot);
        // Negate y scale to perform y-flip
        texTrans.setScaling(float3(1.f / scale.x, -1.f / scale.y, 1.f));
        texTrans.setCompositionOrder(Transform::CompositionOrder::TranslateRotateScale);
    }
    else
    {
        // Apply y-flip
        texTrans.setScaling(float3(1.f, -1.f, 1.f));
    }
    pMaterial->setTextureTransform(texTrans);

    if (any(spec.volumeAbsorption.uniformValue > 0.f))
    {
        pMaterial->setVolumeAbsorption(spec.volumeAbsorption.uniformValue.xyz());
    }

    if (any(spec.volumeScattering.uniformValue > 0.f))
    {
        pMaterial->setVolumeScattering(spec.volumeScattering.uniformValue.xyz());
    }

    // Cache the result of the conversion.
    mMaterialCache.add(shader, spec, pMaterial);

    return pMaterial;
}
} // namespace Falcor
