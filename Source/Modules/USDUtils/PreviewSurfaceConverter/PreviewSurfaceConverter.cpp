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

// Return the ultimate source of the given input.
UsdShadeInput getSourceInput(UsdShadeInput input, UsdShadeConnectableAPI& source, TfToken& sourceName, UsdShadeAttributeType& sourceType)
{
    if (input && input.HasConnectedSource())
    {
        if (input.GetConnectedSource(&source, &sourceName, &sourceType))
        {
            auto inputs = source.GetInputs();
            if (!inputs.empty())
            {
                // If there's a connected source of type asset, return it.
                for (uint32_t i = 0; i < inputs.size(); ++i)
                {
                    logDebug("Input '{}' has source '{}'.", input.GetBaseName().GetString(), inputs[i].GetBaseName().GetString());
                    SdfValueTypeName typeName(inputs[i].GetTypeName());
                    if (typeName == SdfValueTypeNames->Asset)
                    {
                        return inputs[i];
                    }
                }
                // Otherwise, if there's no input of type asset, use the first connected source.
                return inputs[0];
            }
        }
    }
    return input;
}

// Get the layer in which the given authored attribute is defined, if any.
SdfLayerHandle getLayerHandle(const UsdAttribute& attr, const UsdTimeCode& time)
{
    for (auto& spec : attr.GetPropertyStack(time))
    {
        if (spec->HasDefaultValue() || spec->GetLayer()->GetNumTimeSamplesForPath(spec->GetPath()) > 0)
        {
            return spec->GetLayer();
        }
    }
    return SdfLayerHandle();
}
} // namespace

ref<StandardMaterial> PreviewSurfaceConverter::getCachedMaterial(const UsdShadeShader& shader)
{
    std::unique_lock lock(mCacheMutex);

    std::string shaderName = shader.GetPath().GetString();

    while (true)
    {
        auto iter = mPrimMaterialCache.find(shader.GetPrim());
        if (iter == mPrimMaterialCache.end())
        {
            // No entry for this shader. Add one, with a nullptr Falcor material instance, to indicate that work is
            // underway.
            logDebug("No cached material prim for UsdPreviewSurface '{}'; starting conversion.", shaderName);
            mPrimMaterialCache.insert(std::make_pair(shader.GetPrim(), nullptr));
            return nullptr;
        }
        if (iter->second != nullptr)
        {
            // Found a valid entry
            logDebug("Found a cached material prim for UsdPreviewSurface '{}'.", shaderName);
            return iter->second;
        }
        // There is an cache entry, but it is null, indicating that another thread is current performing conversion.
        // Wait until we are signaled that a new entry has been added to the cache, and loop again to check for
        // existence in the cache. Note that this mechanism requires that conversion never fail to create an entry in
        // the cache after calling this function.
        logDebug("In-progress cached entry for UsdPreviewSurface '{}' detected. Waiting.", shaderName);
        mPrimCacheUpdated.wait(lock);
    }
    FALCOR_UNREACHABLE();
    return nullptr;
}

ref<StandardMaterial> PreviewSurfaceConverter::getCachedMaterial(const StandardMaterialSpec& spec)
{
    std::unique_lock lock(mCacheMutex);

    const std::string shaderName(spec.name);

    while (true)
    {
        auto iter = mSpecMaterialCache.find(spec);
        if (iter == mSpecMaterialCache.end())
        {
            // No entry for this shader. Add one, with a nullptr Falcor material instance, to indicate that work is
            // underway.
            logDebug("No cached material for spec '{}'; starting conversion.", shaderName);
            mSpecMaterialCache.insert(std::make_pair(spec, nullptr));
            return nullptr;
        }
        if (iter->second != nullptr)
        {
            // Found a valid entry
            logDebug("Found a cached material spec for UsdPreviewSurface '{}'.", shaderName);
            return iter->second;
        }
        // There is an cache entry, but it is null, indicating that another thread is current performing conversion.
        // Wait until we are signaled that a new entry has been added to the cache, and loop again to check for
        // existence in the cache. Note that this mechanism requires that conversion never fail to create an entry in
        // the cache after calling this function.
        logDebug("In-progress cached entry for spec '{}' detected. Waiting.", shaderName);
        mSpecCacheUpdated.wait(lock);
    }
    FALCOR_UNREACHABLE();
    return nullptr;
}

void PreviewSurfaceConverter::cacheMaterial(const UsdShadeShader& shader, ref<StandardMaterial> pMaterial)
{
    {
        std::unique_lock lock(mCacheMutex);
        if (mPrimMaterialCache.find(shader.GetPrim()) == mPrimMaterialCache.end())
        {
            FALCOR_THROW("Expected PreviewSurfaceConverter cache entry for '{}' not found.", shader.GetPath().GetString());
        }
        mPrimMaterialCache[shader.GetPrim()] = pMaterial;
    }
    mPrimCacheUpdated.notify_all();
}

void PreviewSurfaceConverter::cacheMaterial(const StandardMaterialSpec& spec, ref<StandardMaterial> pMaterial)
{
    {
        std::unique_lock lock(mCacheMutex);
        if (mSpecMaterialCache.find(spec) == mSpecMaterialCache.end())
        {
            FALCOR_THROW("Expected PreviewSurfaceConverter spec cache entry for '{}' not found.", spec.name);
        }
        mSpecMaterialCache[spec] = pMaterial;
    }
    mSpecCacheUpdated.notify_all();
}

bool getFloat2Value(const UsdShadeInput& input, float2& val)
{
    SdfValueTypeName typeName(input.GetTypeName());
    if (typeName == SdfValueTypeNames->Float2)
    {
        GfVec2f value;
        input.Get<GfVec2f>(&value, UsdTimeCode::EarliestTime());
        val = float2(value[0], value[1]);
        return true;
    }
    else if (typeName == SdfValueTypeNames->Float)
    {
        float value;
        input.Get<float>(&value, UsdTimeCode::EarliestTime());
        val = float2(value, value);
        return true;
    }
    return false;
}

bool getFloat4Value(const UsdShadeInput& input, float4& val)
{
    SdfValueTypeName typeName(input.GetTypeName());
    if (typeName == SdfValueTypeNames->Float4)
    {
        GfVec4f value;
        input.Get<GfVec4f>(&value, UsdTimeCode::EarliestTime());
        val = float4(value[0], value[1], value[2], value[3]);
        return true;
    }
    else if (typeName == SdfValueTypeNames->Float3)
    {
        GfVec3f value;
        input.Get<GfVec3f>(&value, UsdTimeCode::EarliestTime());
        val = float4(value[0], value[1], value[2], 1.f);
        return true;
    }
    else if (typeName == SdfValueTypeNames->Float)
    {
        float value;
        input.Get<float>(&value, UsdTimeCode::EarliestTime());
        val = float4(value, value, value, value);
        return true;
    }
    return false;
}

StandardMaterialSpec::ConvertedInput PreviewSurfaceConverter::convertTexture(
    const UsdShadeInput& input,
    const TfToken& outputName,
    bool assumeSrgb,
    bool scaleSupported
) const
{
    const TfToken fallbackToken("fallback");
    const TfToken fileToken("file");
    const TfToken float2PrimvarReaderToken("UsdPrimvarReader_float2");
    const TfToken transform2dToken("UsdTransform2d");
    const TfToken sRGBToken("sRGB");
    const TfToken rawToken("raw");
    const TfToken sourceColorSpaceToken("sourceColorSpace");
    const TfToken stToken("st");
    const TfToken wrapSToken("wrapS");
    const TfToken wrapTToken("wrapT");
    const TfToken biasToken("bias");
    const TfToken scaleToken("scale");
    const TfToken rotationToken("rotation");
    const TfToken translationToken("translation");

    StandardMaterialSpec::ConvertedInput ret;

    // Note that the wrapS, wrapT, scale, and bias inputs are unsupported, and are ignored.

    UsdPrim prim(input.GetPrim());
    if (!prim.IsA<UsdShadeShader>())
    {
        logWarning("Expected UsdUVTexture node '{}' is not a UsdShadeShader.", prim.GetPath().GetString());
        return ret;
    }

    UsdShadeShader shader(prim);

    TfToken id = getAttribute(prim.GetAttribute(UsdShadeTokens->infoId), TfToken());
    if (id != TfToken("UsdUVTexture"))
    {
        logWarning("Expected UsdUVTexture node '{}' is not a UsdUVTexture.", prim.GetPath().GetString());
        return ret;
    }

    if (shader.GetInput(wrapSToken) || shader.GetInput(wrapTToken))
    {
        // Issue a low-priorty message, under the assumption that wrap modes most often don't matter.
        logInfo("UsdUvTexture node '{}' specifies a wrap mode, which is not supported.", prim.GetPath().GetString());
    }

    if (shader.GetInput(biasToken))
    {
        float4 value;
        if (!getFloat4Value(input, value) || any(value != float4(0.f, 0.f, 0.f, 0.f)))
        {
            logWarning("UsdUvTexture node '{}' specifies a non-zero bias, which is not supported.", prim.GetPath().GetString());
        }
    }

    if (UsdShadeInput scaleInput = shader.GetInput(scaleToken); scaleInput)
    {
        if (!getFloat4Value(scaleInput, ret.textureScale))
        {
            logWarning("UsdUvTexture node '{}' specifies value scale of an unsupported type.", prim.GetPath().GetString());
        }
        else if (!scaleSupported && any(ret.textureScale != float4(1.f, 1.f, 1.f, 1.f)))
        {
            logWarning("UsdUvTexture node '{}' specifies a value scale, which is not supported.", prim.GetPath().GetString());
        }
    }

    UsdShadeConnectableAPI source;
    TfToken sourceName;
    UsdShadeAttributeType sourceType;
    UsdShadeInput stInput(getSourceInput(shader.GetInput(stToken), source, sourceName, sourceType));
    if (stInput)
    {
        id = getAttribute(source.GetPrim().GetAttribute(UsdShadeTokens->infoId), TfToken());
        if (id == transform2dToken)
        {
            if (UsdShadeInput scaleInput = source.GetInput(scaleToken); scaleInput)
            {
                float2 scaleVec;
                if (getFloat2Value(scaleInput, scaleVec))
                {
                    ret.texTransform.setScaling(float3(scaleVec.x, scaleVec.y, 1.f));
                }
            }

            if (UsdShadeInput rotateInput = source.GetInput(rotationToken); rotateInput)
            {
                float degrees;
                if (rotateInput.Get<float>(&degrees, UsdTimeCode::EarliestTime()))
                {
                    ret.texTransform.setRotationEulerDeg(float3(0.f, 0.f, degrees));
                }
            }

            if (UsdShadeInput translateInput = source.GetInput(translationToken); translateInput)
            {
                float2 translateVec;
                if (getFloat2Value(translateInput, translateVec))
                {
                    ret.texTransform.setTranslation(float3(translateVec.x, translateVec.y, 0.f));
                }
            }
            ret.texTransform.setCompositionOrder(Transform::CompositionOrder::ScaleRotateTranslate);
        }
        else if (id != float2PrimvarReaderToken)
        {
            logWarning(
                "UsdUVTexture node '{}' defines an st primvar reader '{}' of an unexpected type: '{}'.",
                prim.GetPath().GetString(),
                source.GetPrim().GetPath().GetString(),
                id.GetString()
            );
        }
    }
    else
    {
        // Issue a lower priority message if there is simply no st input defined, under the assumption that the default
        // behavior is expected.
        logInfo("UsdUVTexture node '{}' does not define an st input.", prim.GetPath().GetString());
    }

    // Initialize the uniform converted value using the fallback value, if any.
    GfVec4f fallbackValue = getAttribute(prim.GetAttribute(fallbackToken), GfVec4f(0.f, 0.f, 0.f, 1.f));
    ret.uniformValue = float4(fallbackValue[0], fallbackValue[1], fallbackValue[2], fallbackValue[3]);

    // Color space may be specified in USD either as an attribute of the input connection, which we check here, or
    // directly on the the asset, which we check below, and which takes precedence.
    ret.loadSRGB = assumeSrgb;
    TfToken colorSpace = getAttribute(prim.GetAttribute(sourceColorSpaceToken), TfToken());
    if (colorSpace == sRGBToken)
        ret.loadSRGB = true;
    else if (colorSpace == rawToken)
        ret.loadSRGB = false;

    // Convert output specification to texture channel flag.
    ret.channels = TextureChannelFlags::None;
    if (outputName == TfToken("r"))
    {
        ret.channels |= TextureChannelFlags::Red;
    }
    if (outputName == TfToken("g"))
    {
        ret.channels |= TextureChannelFlags::Green;
    }
    if (outputName == TfToken("b"))
    {
        ret.channels |= TextureChannelFlags::Blue;
    }
    if (outputName == TfToken("a"))
    {
        ret.channels |= TextureChannelFlags::Alpha;
    }
    if (outputName == TfToken("rg"))
    {
        // Note: not legal per the UsdPreviewSurface spec
        ret.channels |= TextureChannelFlags::Red;
        ret.channels |= TextureChannelFlags::Green;
    }
    if (outputName == TfToken("rgb"))
    {
        ret.channels |= TextureChannelFlags::Red;
        ret.channels |= TextureChannelFlags::Green;
        ret.channels |= TextureChannelFlags::Blue;
    }

    if (ret.channels == TextureChannelFlags::None)
    {
        logWarning("No valid output specified by UsdUVTexture '{}'.", prim.GetName().GetString());
        return ret;
    }

    UsdShadeInput fileInput(getSourceInput(shader.GetInput(fileToken), source, sourceName, sourceType));
    if (fileInput)
    {
        SdfAssetPath path;
        UsdPrim filePrim(fileInput.GetPrim());
        UsdAttribute fileAttrib = filePrim.GetAttribute(TfToken("inputs:file"));

        TfToken fileColorSpace = fileAttrib.GetColorSpace();
        if (fileColorSpace == sRGBToken)
            ret.loadSRGB = true;
        else if (fileColorSpace == rawToken)
            ret.loadSRGB = false;

        fileInput.Get<SdfAssetPath>(&path, UsdTimeCode::EarliestTime());
        ret.texturePath = path.GetResolvedPath().empty() ? path.GetAssetPath() : path.GetResolvedPath();

        // If the filename contains <UDIM>, the asset refers to a UDIM texture set, which Falcor does not support.
        // Replace <UDIM> with 1001 in an attempt to at least load one texture from the set.
        auto udimPos = ret.texturePath.find("<UDIM>");
        std::string replacedFilename = ret.texturePath;
        if (udimPos != std::string::npos)
        {
            ret.texturePath.replace(udimPos, 6, "1001");
            // This hoop-jumping is required because we need to resolve the given path relative to the
            // layer in which it is defined, to ensure that e.g., "../../Textures/foo.dds" resolves properly.
            SdfLayerHandle layerHandle(getLayerHandle(fileAttrib, UsdTimeCode::EarliestTime()));
            ret.texturePath = SdfComputeAssetPathRelativeToLayer(layerHandle, ret.texturePath);
            if (ret.texturePath.empty())
            {
                ret.texturePath = path.GetAssetPath();
            }
        }
    }
    else
    {
        logWarning("UsdUVTexture '{}' does not specify a file input.", prim.GetName().GetString());
    }
    return ret;
}

StandardMaterialSpec::ConvertedInput PreviewSurfaceConverter::convertFloat(const UsdShadeInput& input, const TfToken& sourceName) const
{
    StandardMaterialSpec::ConvertedInput ret;
    SdfValueTypeName typeName(input.GetTypeName());
    if (typeName == SdfValueTypeNames->Asset)
    {
        ret = convertTexture(input, sourceName);
    }
    else if (typeName == SdfValueTypeNames->Float)
    {
        input.Get<float>(&ret.uniformValue.r, UsdTimeCode::EarliestTime());
    }
    else
    {
        logWarning("Unexpected value type when converting float input: '{}'.", typeName.GetAsToken().GetString());
    }
    return ret;
}

StandardMaterialSpec::ConvertedInput PreviewSurfaceConverter::convertColor(
    const UsdShadeInput& input,
    const TfToken& sourceName,
    bool assumeSrgb,
    bool scaleSupported
) const
{
    StandardMaterialSpec::ConvertedInput ret;

    SdfValueTypeName typeName(input.GetTypeName());
    ret.uniformValue = float4(0.f, 0.f, 0.f, 1.f);
    if (typeName == SdfValueTypeNames->Color3f)
    {
        GfVec3f v;
        input.Get<GfVec3f>(&v, UsdTimeCode::EarliestTime());
        ret.uniformValue = float4(v[0], v[1], v[2], 1.f);
    }
    else if (typeName == SdfValueTypeNames->Asset)
    {
        ret = convertTexture(input, sourceName, assumeSrgb, scaleSupported);
    }
    else
    {
        logWarning("Unexpected value type when converting color input: '{}'.", typeName.GetAsToken().GetString());
    }
    return ret;
}

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
    StandardMaterialSpec::ConvertedInput& opacity,
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
    StandardMaterialSpec::ConvertedInput& baseColor,
    ref<Texture> baseColorTexture,
    StandardMaterialSpec::ConvertedInput& opacity,
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
    StandardMaterialSpec::ConvertedInput& roughness,
    ref<Texture> roughnessTexture,
    StandardMaterialSpec::ConvertedInput& metallic,
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

ref<Texture> PreviewSurfaceConverter::loadTexture(const StandardMaterialSpec::ConvertedInput& ci)
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
        Bitmap::UniqueConstPtr pBitmap(Bitmap::createFromFile(ci.texturePath, false));
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

    for (auto& curInput : shader.GetInputs())
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
            spec.baseColor = convertColor(input, sourceName, true);
            spec.updateTexTransform(spec.baseColor);
        }
        else if (inputName == "emissivecolor")
        {
            // Get color value(s), assume that textures are sRGB by default.
            spec.emission = convertColor(input, sourceName, true);
        }
        else if (inputName == "usespecularworkflow")
        {
            if (typeName == SdfValueTypeNames->Int)
            {
                int specularWorkflow;
                input.Get<int>(&specularWorkflow, UsdTimeCode::EarliestTime());
                if (specularWorkflow)
                {
                    logWarning("Specular workflow is not supported.");
                }
            }
            else
            {
                logWarning("Unsupported specular workflow value type: '{}'.", typeName.GetAsToken().GetString());
            }
        }
        else if (inputName == "metallic")
        {
            spec.metallic = convertFloat(input, sourceName);
            spec.updateTexTransform(spec.metallic);
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
            spec.roughness = convertFloat(input, sourceName);
            spec.updateTexTransform(spec.roughness);
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
            spec.opacity = convertFloat(input, sourceName);
            spec.updateTexTransform(spec.opacity);
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
        else if (inputName == "normal")
        {
            if (typeName == SdfValueTypeNames->Normal3f)
            {
                logWarning("Falcor's standard material does not support uniform normal values.");
            }
            else if (typeName == SdfValueTypeNames->Asset)
            {
                spec.normal = convertTexture(input, sourceName);
                spec.updateTexTransform(spec.normal);
            }
            else
            {
                logWarning("Unsupported normal value type: '{}'.", typeName.GetAsToken().GetString());
            }
        }
        else if (inputName == "displacement")
        {
            spec.disp = convertFloat(input, sourceName);
            if (!spec.disp.isTextured())
            {
                logWarning("Falcor's standard material does not support uniform displacement.");
            }
            else
            {
                spec.updateTexTransform(spec.disp);
            }
        }
        else if (inputName == "occlusion")
        {
            logWarning("Falcor's standard material does not support an occlusion parameter.");
        }
        else
        {
            logWarning("Unsupported UsdPreviewSurface input '{}'.", inputName);
        }
    }
    return spec;
}

ref<Material> PreviewSurfaceConverter::convert(const UsdShadeMaterial& material, const std::string& primName, RenderContext* pRenderContext)
{
    TfToken renderContext(""); // Blank implies universal context. Use e.g. "falcor" for a renderer- or
                               // material-specific context.
    UsdShadeOutput surface(material.GetSurfaceOutput(renderContext));

    if (!surface.IsDefined())
    {
        logDebug("Material '{}' does not define a universal surface output.", primName);
        return nullptr;
    }

    // Get the UsdShadeShader that provides the surface output
    UsdShadeShader shader;
    UsdPrim surfacePrim(surface.GetPrim());
    if (surfacePrim.IsDefined())
    {
        UsdShadeConnectableAPI source;
        TfToken sourceName;
        UsdShadeAttributeType sourceType;
        if (UsdShadeConnectableAPI::GetConnectedSource(surface, &source, &sourceName, &sourceType))
        {
            UsdPrim prim(source.GetPrim());
            if (prim.IsA<UsdShadeShader>())
            {
                shader = UsdShadeShader(prim);
            }
        }
    }
    if (!shader)
    {
        logDebug("Material '{}' surface output is not a UsdShadeShader.", primName);
        return nullptr;
    }

    TfToken id = getAttribute(shader.GetPrim().GetAttribute(UsdShadeTokens->infoId), TfToken());
    if (id != TfToken("UsdPreviewSurface"))
    {
        logDebug("Material '{}' has a surface output node of type '{}', not UsdPreviewSurface.", primName, id.GetString());
        return nullptr;
    }

    logDebug("Material '{}' has UsdPreviewSurface output '{}'.", primName, shader.GetPath().GetString());

    // Is there a valid cached instance for this material prim?  If so, simply return it.
    // Note that this call will block if another thread is concurrently converting the same material.
    ref<StandardMaterial> pMaterial = getCachedMaterial(shader);
    if (pMaterial)
    {
        return pMaterial;
    }

    // Due to the fact that threads may be waiting for material conversion to complete, the following code must always
    // create non-null by-prim and by-spec material cache entries.

    std::string materialName(material.GetPrim().GetPath().GetString());

    // Construct a StandardMaterialSpec before creating the StandardMaterial itself from it in order to
    // avoid creating duplicate material instances.
    StandardMaterialSpec spec = createSpec(materialName, shader);

    // Does there already exist a material matching this spec? If so, return it.
    pMaterial = getCachedMaterial(spec);
    if (pMaterial)
    {
        // There is an entry for this material in the by-spec cache. Ensure that there is also an entry in the by-prim
        // cache.
        cacheMaterial(shader, pMaterial);
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
        // Handle non-unit opacity
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
            pMaterial->setAlphaThreshold(spec.opacityThreshold);
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

    if (spec.texTransform.getMatrix() != float4x4::identity())
    {
        // Compute the inverse of the texture transform, as Falcor assumes
        // the transform applies to the texture, rather than the texture coordinates,
        // as UsdPreviewSurface does.
        FALCOR_ASSERT(spec.texTransform.getCompositionOrder() == Transform::CompositionOrder::ScaleRotateTranslate);
        Transform inv;
        inv.setTranslation(-spec.texTransform.getTranslation());
        float3 scale = spec.texTransform.getScaling();
        inv.setScaling(float3(1.f / scale.x, 1.f / scale.y, 1.f));
        float3 rot = spec.texTransform.getRotationEuler();
        inv.setRotationEuler(-rot);
        inv.setCompositionOrder(Transform::CompositionOrder::TranslateRotateScale);
        pMaterial->setTextureTransform(inv);
    }

    // Cache the result of the conversion, both by shader node and by spec.
    cacheMaterial(shader, pMaterial);
    cacheMaterial(spec, pMaterial);

    return pMaterial;
}
} // namespace Falcor
