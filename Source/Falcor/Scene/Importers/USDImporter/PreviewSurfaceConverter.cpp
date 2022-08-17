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
#include "PreviewSurfaceConverter.h"
#include "Utils.h"
#include "USDHelpers.h"
#include "Core/API/RenderContext.h"

BEGIN_DISABLE_USD_WARNINGS
#include <pxr/usd/usdGeom/xformCommonAPI.h>
#include <pxr/usd/usdRender/tokens.h>
#include <pxr/usdImaging/usdImaging/tokens.h>
#include <pxr/usd/ar/resolver.h>
#include <pxr/usd/sdf/layerUtils.h>
END_DISABLE_USD_WARNINGS

using namespace pxr;

namespace Falcor
{
    namespace
    {
        const char kSpecTransShaderFile[]("Scene/Importers/USDImporter/CreateSpecularTransmissionTexture.cs.slang");
        const char kSpecTransShaderEntry[]("createSpecularTransmissionTexture");

        const char kPackAlphaShaderFile[]("Scene/Importers/USDImporter/PackBaseColorAlpha.cs.slang");
        const char kPackAlphaShaderEntry[]("packBaseColorAlpha");

        const char kSpecularShaderFile[]("Scene/Importers/USDImporter/CreateSpecularTexture.cs.slang");
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
        UsdShadeInput getSourceInput(UsdShadeInput input)
        {
            if (input && input.HasConnectedSource())
            {
                UsdShadeConnectableAPI source;
                TfToken sourceName;
                UsdShadeAttributeType sourceType;
                if (input.GetConnectedSource(&source, &sourceName, &sourceType))
                {
                    auto inputs = source.GetInputs();
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
    }

    StandardMaterial::SharedPtr PreviewSurfaceConverter::getCachedMaterial(const UsdShadeShader& shader)
    {
        std::unique_lock lock(mCacheMutex);

        std::string shaderName = shader.GetPath().GetString();

        while (true)
        {
            auto iter = mMaterialCache.find(shader.GetPrim());
            if (iter == mMaterialCache.end())
            {
                // No entry for this shader. Add one, with a nullptr Falcor material instance, to indicate that work is underway.
                logDebug("No cached material for UsdPreviewSurface '{}'; starting conversion.", shaderName);
                mMaterialCache.insert(std::make_pair(shader.GetPrim(), nullptr));
                return nullptr;
            }
            if (iter->second != nullptr)
            {
                // Found a valid entry
                logDebug("Found a cached material for UsdPreviewSurface '{}'.", shaderName);
                return iter->second;
            }
            // There is an dictionary entry, but it is null, indicating that another thread is current performing conversion.
            // Wait until we are signaled that a new entry has been added to the dictionary, and check again.
            // Note that this mechanism requires that conversion never fail to create an entry in the dictionary after calling this function.
            logDebug("In-progressed cached entry for UsdPreviewSurface '{}' detected. Waiting.", shaderName);
            mCacheUpdated.wait(lock);
        }
        FALCOR_UNREACHABLE();
        return nullptr;
    }

    void PreviewSurfaceConverter::cacheMaterial(const UsdShadeShader& shader, StandardMaterial::SharedPtr pMaterial)
    {
        {
            std::unique_lock lock(mCacheMutex);
            if (mMaterialCache.find(shader.GetPrim()) == mMaterialCache.end())
            {
                throw RuntimeError("Expected PreviewSurfaceConverter cache entry for '{}' not found.", shader.GetPath().GetString());
            }
            mMaterialCache[shader.GetPrim()] = pMaterial;
        }
        mCacheUpdated.notify_all();
    }

    bool isUniformFloat4(const UsdShadeInput& input, const float4& uniformValue)
    {
        auto typeName = input.GetTypeName();
        if (typeName == SdfValueTypeNames->Float4)
        {
            GfVec4f v;
            input.Get<GfVec4f>(&v, UsdTimeCode::EarliestTime());
            float4 value(v[0], v[1], v[2], v[3]);
            return value == uniformValue;
        }
        return false;
    }

    PreviewSurfaceConverter::ConvertedInput PreviewSurfaceConverter::convertTexture(const UsdShadeInput& input, bool assumeSrgb)
    {
        const TfToken fallbackToken("fallback");
        const TfToken fileToken("file");
        const TfToken float2PrimvarReaderToken("UsdPrimvarReader_float2");
        const TfToken sRGBToken("sRGB");
        const TfToken sourceColorSpaceToken("sourceColorSpace");
        const TfToken stToken("st");
        const TfToken wrapSToken("wrapS");
        const TfToken wrapTToken("wrapT");
        const TfToken biasToken("bias");
        const TfToken scaleToken("scale");

        ConvertedInput ret;

        // Note that the wrapS, wrapT, scale, and bias inputs are unsupported, and are ignored.

        UsdPrim prim(input.GetPrim());
        if (!prim.IsA<UsdShadeShader>())
        {
            logWarning("Expected UsdUVTexture node '{}' is not a UsdShadeShader.", prim.GetPath().GetString());
            return ret;
        }

        UsdShadeShader shader(prim);

        TfToken id = getAttribute(prim.GetAttribute(UsdShadeTokens->infoId), TfToken());
        if (id != UsdImagingTokens->UsdUVTexture)
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
            if (!isUniformFloat4(shader.GetInput(biasToken), float4(0.f, 0.f, 0.f, 0.f)))
            {
                logWarning("UsdUvTexture node '{}' specifies a non-zero bias, which is not supported.", prim.GetPath().GetString());
            }
        }

        if (shader.GetInput(scaleToken))
        {
            if (!isUniformFloat4(shader.GetInput(scaleToken), float4(1.f, 1.f, 1.f, 1.f)))
            {
                logWarning("UsdUvTexture node '{}' specifies a non-unity scale, which is not supported.", prim.GetPath().GetString());
            }
        }

        UsdShadeInput stInput(getSourceInput(shader.GetInput(stToken)));
        if (stInput)
        {
            // Check the primvar reader associated with the st input to ensure it's of the expected type.
            // Note that we only issue a warning here, and use the associated geometry's texcoords for lookup in any case.
            UsdPrim stPrim(stInput.GetPrim());
            id = getAttribute(stPrim.GetAttribute(UsdShadeTokens->infoId), TfToken());
            if (id != float2PrimvarReaderToken)
            {
                logWarning("UsdUVTexture node '{}' defines an st primvar reader of an unexpected type: '{}'.", prim.GetPath().GetString(), id.GetString());
            }
        }
        else
        {
            // Issue a lower priority message if there is simply no st input defined, under the assumption that the default behavior is expected.
            logInfo("UsdUVTexture node '{}' does not define an st input.", prim.GetPath().GetString());
        }

        // Initialize the uniform converted value using the fallback value, if any.
        GfVec4f fallbackValue = getAttribute(prim.GetAttribute(fallbackToken), GfVec4f(0.f, 0.f, 0.f, 1.f));
        ret.uniformValue = float4(fallbackValue[0], fallbackValue[1], fallbackValue[2], fallbackValue[3]);

        // Color space may be specified in USD either as an attribute of the input connection, which we check here, or directly on the the asset,
        // which we check below, and which takes precedence.
        bool loadSRGB = assumeSrgb;
        TfToken colorSpace = getAttribute(prim.GetAttribute(sourceColorSpaceToken), TfToken());
        if (colorSpace == sRGBToken) loadSRGB = true;

        // Convert output specification to texture channel flag.
        ret.channels = TextureChannelFlags::None;
        for (auto& output : shader.GetOutputs())
        {
            if (output.GetBaseName() == TfToken("r"))
            {
                ret.channels |= TextureChannelFlags::Red;
            }
            if (output.GetBaseName() == TfToken("g"))
            {
                ret.channels |= TextureChannelFlags::Green;
            }
            if (output.GetBaseName() == TfToken("b"))
            {
                ret.channels |= TextureChannelFlags::Blue;
            }
            if (output.GetBaseName() == TfToken("a"))
            {
                ret.channels |= TextureChannelFlags::Alpha;
            }
            if (output.GetBaseName() == TfToken("rgb"))
            {
                ret.channels |= TextureChannelFlags::Red;
                ret.channels |= TextureChannelFlags::Green;
                ret.channels |= TextureChannelFlags::Blue;
            }
        }
        if (ret.channels == TextureChannelFlags::None)
        {
            logWarning("No valid output specified by UsdUVTexture '{}'.", prim.GetName().GetString());
            return ret;
        }

        UsdShadeInput fileInput(getSourceInput(shader.GetInput(fileToken)));
        if (fileInput)
        {
            SdfAssetPath path;
            UsdPrim prim(fileInput.GetPrim());
            UsdAttribute fileAttrib = prim.GetAttribute(TfToken("inputs:file"));

            TfToken colorSpace = fileAttrib.GetColorSpace();
            if (colorSpace == sRGBToken) loadSRGB = true;

            fileInput.Get<SdfAssetPath>(&path, UsdTimeCode::EarliestTime());
            std::string filename = path.GetResolvedPath().empty() ? path.GetAssetPath() : path.GetResolvedPath();

            // If the filename contains <UDIM>, the asset refers to a UDIM texture set, which Falcor does not support.
            // Replace <UDIM> with 1001 in an attempt to at least load one texture from the set.
            auto udimPos = filename.find("<UDIM>");
            std::string replacedFilename = filename;
            if (udimPos != std::string::npos)
            {
                filename.replace(udimPos, 6, "1001");
                // This hoop-jumping is required because we need to resolve the given path relative to the
                // layer in which it is defined, to ensure that e.g., "../../Textures/foo.dds" resolves properly.
                SdfLayerHandle layerHandle(getLayerHandle(fileAttrib, UsdTimeCode::EarliestTime()));
                filename = SdfComputeAssetPathRelativeToLayer(layerHandle, filename);
                if (filename.empty())
                {
                    filename = path.GetAssetPath();
                }
            }

            // Create the texture by first reading the image (which is relatively slow) outside of the mutex,
            // and then creating the texture itself inside it.
            Bitmap::UniqueConstPtr pBitmap(Bitmap::createFromFile(filename, false));
            if (pBitmap)
            {
                ResourceFormat format = loadSRGB ? linearToSrgbFormat(pBitmap->getFormat()) : pBitmap->getFormat();
                {
                    std::scoped_lock lock(mMutex);
                    ret.pTexture = Texture::create2D(pBitmap->getWidth(), pBitmap->getHeight(), format, 1, Texture::kMaxPossible, pBitmap->getData());
                }
            }
            // Else, a warning will have been emitted by Bitmap::createFromFile(), and the fallback value will be used.
        }
        else
        {
            logWarning("UsdUVTexture '{}' does not specify a file input.", prim.GetName().GetString());
        }
        return ret;
    }

    PreviewSurfaceConverter::ConvertedInput PreviewSurfaceConverter::convertFloat(const UsdShadeInput& input)
    {
        ConvertedInput ret;
        SdfValueTypeName typeName(input.GetTypeName());
        if (typeName == SdfValueTypeNames->Asset)
        {
            ret = convertTexture(input);
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

    PreviewSurfaceConverter::ConvertedInput PreviewSurfaceConverter::convertColor(const UsdShadeInput& input, bool assumeSrgb)
    {
        ConvertedInput ret;

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
            ret = convertTexture(input, assumeSrgb);
        }
        else
        {
            logWarning("Unexpected value type when converting color input: '{}'.", typeName.GetAsToken().GetString());
        }
        return ret;
    }

    PreviewSurfaceConverter::PreviewSurfaceConverter()
    {
        mpSpecTransPass = ComputePass::create(kSpecTransShaderFile, kSpecTransShaderEntry);

        mpPackAlphaPass = ComputePass::create(kPackAlphaShaderFile, kPackAlphaShaderEntry);

        mpSpecularPass = ComputePass::create(kSpecularShaderFile, kSpecularShaderEntry);

        Sampler::Desc samplerDesc;
        samplerDesc.setFilterMode(Sampler::Filter::Linear, Sampler::Filter::Linear, Sampler::Filter::Point);
        samplerDesc.setAddressingMode(Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp, Sampler::AddressMode::Clamp);

        mpSampler = Sampler::create(samplerDesc);
    }

    // Convert textured opacity to textured specular transparency.
    Texture::SharedPtr PreviewSurfaceConverter::createSpecularTransmissionTexture(ConvertedInput& opacity, RenderContext* pRenderContext)
    {
        std::scoped_lock lock(mMutex);

        if (popcount((uint32_t)opacity.channels) > 1)
        {
            logWarning("Cannot create transmission texture; opacity texture provides more than one channel of data.");
            return nullptr;
        }

        uint2 resolution(opacity.pTexture->getWidth(), opacity.pTexture->getHeight());
        Texture::SharedPtr pTexture = Texture::create2D(resolution.x, resolution.y, ResourceFormat::RGBA8Unorm, 1, Texture::kMaxPossible, nullptr, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess | Resource::BindFlags::RenderTarget);

        mpSpecTransPass["CB"]["outDim"] = resolution;
        mpSpecTransPass["CB"]["opacityChannel"] = getChannelIndex(opacity.channels);
        mpSpecTransPass["opacityTexture"] = opacity.pTexture;
        mpSpecTransPass["outputTexture"] = pTexture;

        mpSpecTransPass->execute(pRenderContext, resolution.x, resolution.y);

        pTexture->generateMips(pRenderContext);

        return pTexture;
    }

    // Combine base color and alpha, one or both of which may be textured, into a single texture.
    // If both are textured, they may be of different resolutions.
    // This is only performed when a material makes use of cutout opacity.
    Texture::SharedPtr PreviewSurfaceConverter::packBaseColorAlpha(ConvertedInput& baseColor, ConvertedInput& opacity, RenderContext* pRenderContext)
    {
        std::scoped_lock lock(mMutex);

        if (opacity.pTexture && popcount((uint32_t)opacity.channels) > 1)
        {
            logWarning("Cannot set alpha channel; opacity texture provides more than one channel.");
            return nullptr;
        }

        // Set output resolution to the maxium of the input dimensions
        uint32_t width  = std::max<uint32_t>((opacity.pTexture ? opacity.pTexture->getWidth() : 0), (baseColor.pTexture ? baseColor.pTexture->getWidth() : 0));
        uint32_t height = std::max<uint32_t>((opacity.pTexture ? opacity.pTexture->getHeight() : 0), (baseColor.pTexture ? baseColor.pTexture->getHeight() : 0));
        uint2 resolution(width, height);

        // sRGB format textures can't be bound as UAVs. Render to an RGBA32Float intermediate texture, and then blit to RGBA8UnormSrgb to preserve precision near zero.
        Texture::SharedPtr pTexture = Texture::create2D(resolution.x, resolution.y, ResourceFormat::RGBA32Float, 1, 1, nullptr, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess);

        mpPackAlphaPass["CB"]["opacityChannel"] = opacity.pTexture ? getChannelIndex(opacity.channels) : -1;
        mpPackAlphaPass["CB"]["opacityValue"] = opacity.uniformValue.r;
        mpPackAlphaPass["CB"]["baseColorTextureValid"] = baseColor.pTexture ? 1 : 0;
        mpPackAlphaPass["CB"]["baseColorValue"] = baseColor.uniformValue;
        mpPackAlphaPass["CB"]["outDim"] = resolution;
        mpPackAlphaPass["opacityTexture"] = opacity.pTexture;
        mpPackAlphaPass["baseColorTexture"] = baseColor.pTexture;
        mpPackAlphaPass["sampler"] = mpSampler;
        mpPackAlphaPass["outputTexture"] = pTexture;

        mpPackAlphaPass->execute(pRenderContext, resolution.x, resolution.y);

        // Create the output mipmapped sRGB texture
        Texture::SharedPtr pFinal = Texture::create2D(resolution.x, resolution.y, ResourceFormat::RGBA8UnormSrgb, 1, Texture::kMaxPossible, nullptr, Resource::BindFlags::ShaderResource | Resource::BindFlags::RenderTarget);

        // Blit the intermediate texture to the output texture to perform format conversion
        pRenderContext->blit(pTexture->getSRV(), pFinal->getRTV());

        pFinal->generateMips(pRenderContext);

        return pFinal;
    }

    // Combine roughness and metallic parameters, one or both of which are textured, into a specular/ORM texture.
    // If both are textured, they may be of different resolutions.
    Texture::SharedPtr PreviewSurfaceConverter::createSpecularTexture(ConvertedInput& roughness, ConvertedInput& metallic, RenderContext* pRenderContext)
    {
        std::scoped_lock lock(mMutex);

        if (roughness.pTexture && popcount((uint32_t)roughness.channels) > 1)
        {
            logWarning("Cannot create specular texture; roughness texture provides more than one channel.");
            return nullptr;
        }
        if (metallic.pTexture && popcount((uint32_t)metallic.channels) > 1)
        {
            logWarning("Cannot create specular texture; metallic texture provides more than one channel.");
            return nullptr;
        }

        // Set output resolution to the maxium of the input dimensions
        uint32_t width  = std::max<uint32_t>((roughness.pTexture ? roughness.pTexture->getWidth() : 0), (metallic.pTexture ? metallic.pTexture->getWidth() : 0));
        uint32_t height = std::max<uint32_t>((roughness.pTexture ? roughness.pTexture->getHeight() : 0), (metallic.pTexture ? metallic.pTexture->getHeight() : 0));
        uint2 resolution(width, height);

        Texture::SharedPtr pTexture = Texture::create2D(width, height, ResourceFormat::RGBA8Unorm, 1, Texture::kMaxPossible, nullptr, Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess | Resource::BindFlags::RenderTarget);

        mpSpecularPass["CB"]["roughnessChannel"] = roughness.pTexture ? getChannelIndex(roughness.channels) : -1;
        mpSpecularPass["CB"]["roughnessValue"] = roughness.uniformValue.r;
        mpSpecularPass["CB"]["metallicChannel"] = metallic.pTexture ? getChannelIndex(metallic.channels) : -1;
        mpSpecularPass["CB"]["metallicValue"] = metallic.uniformValue.r;
        mpSpecularPass["CB"]["outDim"] = resolution;
        mpSpecularPass["metallicTexture"] = metallic.pTexture;
        mpSpecularPass["roughnessTexture"] = roughness.pTexture;
        mpSpecularPass["sampler"] = mpSampler;
        mpSpecularPass["outputTexture"] = pTexture;

        mpSpecularPass->execute(pRenderContext, resolution.x, resolution.y);

        pTexture->generateMips(pRenderContext);

        return pTexture;
    }

    Material::SharedPtr PreviewSurfaceConverter::convert(const UsdShadeMaterial& material, const std::string& primName, RenderContext* pRenderContext)
    {
        UsdShadeOutput surface(material.GetOutput(TfToken("surface")));

        if (!surface.IsDefined())
        {
            logDebug("Material '{}' does not define a surface output.", primName);
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
        if (id != UsdImagingTokens->UsdPreviewSurface)
        {
            logDebug("Material '{}' has a surface output node of type '{}', not UsdPreviewSurface.", primName, id.GetString());
            return nullptr;
        }

        logDebug("Material '{}' has UsdPreviewSurface output '{}'.", primName, shader.GetPath().GetString());

        // Is there a valid cached instance for this UsdPreviewSurface?  If so, simply return it.
        // Note that this call will block if another thread is concurrently converting the same shader.
        StandardMaterial::SharedPtr pMaterial = getCachedMaterial(shader);
        if (pMaterial)
        {
            return pMaterial;
        }

        // Get material name.
        std::string materialName(material.GetPrim().GetPath().GetString());

        // Create the output material
        pMaterial = StandardMaterial::create(materialName);

        ConvertedInput baseColor;
        ConvertedInput metallic;
        ConvertedInput roughness(0.5f);
        ConvertedInput opacity(1.f);
        float opacityThreshold = 0.f;
        float ior = 1.5f;

        // Traverse the UsdPreviewSurface node inputs
        for (auto& curInput : shader.GetInputs())
        {
            logDebug("UsdPreviewSurface '{}' has input: {}", shader.GetPath().GetString(), curInput.GetBaseName().GetString());

            // Convert the input name to lowercase to allow for mixed capitalization
            std::string inputName = curInput.GetBaseName();
            std::transform(inputName.begin(), inputName.end(), inputName.begin(), ::tolower);

            // Get the most-upstream source of the input.
            UsdShadeInput input = getSourceInput(curInput);

            SdfValueTypeName typeName(input.GetTypeName());
            logDebug("UsdPreviewSurface '{}' input '{}' has type '{}'.", shader.GetPath().GetString(), inputName, typeName.GetAsToken().GetString());

            if (inputName == "diffusecolor")
            {
                // Get diffuse color value(s), assume that textures are sRGB by default.
                baseColor = convertColor(input, true);
            }
            else if (inputName == "emissivecolor")
            {
                // Get color value(s), assume that textures are sRGB by default.
                ConvertedInput emissive = convertColor(input, true);
                if (emissive.pTexture)
                {
                    pMaterial->setEmissiveTexture(emissive.pTexture);
                }
                else
                {
                    pMaterial->setEmissiveColor(emissive.uniformValue);
                }
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
                metallic = convertFloat(input);
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
                roughness = convertFloat(input);
            }
            else if (inputName == "clearcoat")
            {
                logWarning("Falcor does not support clearcoat.");
            }
            else if (inputName == "clearcoatroughness")
            {
                logWarning("Falcor does not support clearcoat roughness.");
            }
            else if (inputName == "opacity")
            {
                opacity = convertFloat(input);
            }
            else if (inputName == "opacitythreshold")
            {
                if (typeName == SdfValueTypeNames->Float)
                {
                    input.Get<float>(&opacityThreshold, UsdTimeCode::EarliestTime());
                }
                else if (typeName == SdfValueTypeNames->Asset)
                {
                    logWarning("Falcor does not support textured opacity threshold.");
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
                    input.Get<float>(&ior, UsdTimeCode::EarliestTime());
                }
                else if (typeName == SdfValueTypeNames->Asset)
                {
                    logWarning("Falcor does not support textured IoR.");
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
                    logWarning("Falcor does not support uniform normal values.");
                }
                else if (typeName == SdfValueTypeNames->Asset)
                {
                    ConvertedInput norm = convertTexture(input);
                    if (norm.pTexture)
                    {
                        pMaterial->setNormalMap(norm.pTexture);
                    }
                }
                else
                {
                    logWarning("Unsupported normal value type: '{}'.", typeName.GetAsToken().GetString());
                }
            }
            else if (inputName == "displacement")
            {
                ConvertedInput disp = convertFloat(input);
                if (disp.channels == TextureChannelFlags::None)
                {
                    logWarning("Falcor does not support uniform displacement.");
                }
                else if (disp.pTexture)
                {
                    pMaterial->setDisplacementMap(disp.pTexture);
                }
            }
            else if (inputName == "occlusion")
            {
                logWarning("Falcor does not support occlusion material parameter.");
            }
            else
            {
                logWarning("Unsupported UsdPreviewSurface input '{}'.", inputName);
            }
        }

        pMaterial->setIndexOfRefraction(ior);

        // If there is either a roughness or metallic texture, convert texture(s) and constant (if any) to an ORM texture.
        if (metallic.pTexture || roughness.pTexture)
        {
            Texture::SharedPtr pSpecularTex = createSpecularTexture(roughness, metallic, pRenderContext);
            pMaterial->setSpecularTexture(pSpecularTex);
        }
        else
        {
            pMaterial->setSpecularParams(float4(0.f, roughness.uniformValue.r, metallic.uniformValue.r, 1.f));
        }

        if (opacity.uniformValue.r < 1.f || opacity.pTexture)
        {
            // Handle non-unit opacity
            if (opacityThreshold > 0.f)
            {
                // Opacity encodes cutout values
                // Pack opacity into the alpha channel
                if (baseColor.pTexture|| opacity.pTexture)
                {
                    baseColor.pTexture = packBaseColorAlpha(baseColor, opacity, pRenderContext);
                }
                else
                {
                    baseColor.uniformValue = float4(baseColor.uniformValue.rgb, opacity.uniformValue.r);
                }
                pMaterial->setAlphaThreshold(opacityThreshold);
            }
            else if (opacity.pTexture)
            {
                // Opacity encodes (1 - specular-transmission)
                // Create a greyscale specular transmission color texture using (1-opacity), as a slightly hacky means of supporting textured specular transmission.
                Texture::SharedPtr pTransmissionTexture = createSpecularTransmissionTexture(opacity, pRenderContext);
                pMaterial->setTransmissionTexture(pTransmissionTexture);
                pMaterial->setSpecularTransmission(1.f);
            }
            else
            {
                pMaterial->setSpecularTransmission(1.f - opacity.uniformValue.r);
            }
        }
        if (baseColor.pTexture)
        {
            pMaterial->setBaseColorTexture(baseColor.pTexture);
        }
        else
        {
            pMaterial->setBaseColor(baseColor.uniformValue);
        }

        // Cache the result of the conversion
        cacheMaterial(shader, pMaterial);

        return pMaterial;
    }
}
