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
#include "ConvertedInput.h"
#include <pxr/usd/sdf/layerUtils.h>

using namespace pxr;
namespace Falcor
{
namespace
{

static const TfToken gStToken("st");
static const TfToken gTransform2dToken("UsdTransform2d");
static const TfToken gScaleToken("scale");
static const TfToken gRotationToken("rotation");
static const TfToken gTranslationToken("translation");
static const TfToken gFloat2PrimvarReaderToken("UsdPrimvarReader_float2");
static const TfToken gFallbackToken("fallback");
static const TfToken gFileToken("file");
static const TfToken gSRGBToken("sRGB");
static const TfToken gRawToken("raw");
static const TfToken gSourceColorSpaceToken("sourceColorSpace");
static const TfToken gWrapSToken("wrapS");
static const TfToken gWrapTToken("wrapT");
static const TfToken gBiasToken("bias");
static const TfToken gUVTextureToken("UsdUVTexture");
static const TfToken gRToken("r");
static const TfToken gGToken("g");
static const TfToken gBToken("b");
static const TfToken gAToken("a");
static const TfToken gRGToken("rg");
static const TfToken gRGBToken("rgb");
static const TfToken gInputsFileToken("inputs:file");

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
} // namespace

bool ConvertedInput::convertTextureCoords(const UsdShadeShader& shader, Falcor::Transform& xform)
{
    UsdShadeConnectableAPI source;
    TfToken sourceName;
    UsdShadeAttributeType sourceType;
    UsdShadeInput stInput(getSourceInput(shader.GetInput(gStToken), source, sourceName, sourceType));
    if (!stInput)
    {
        return false;
    }

    TfToken id = getAttribute(source.GetPrim().GetAttribute(UsdShadeTokens->infoId), TfToken());
    if (id == gTransform2dToken)
    {
        if (UsdShadeInput scaleInput = source.GetInput(gScaleToken); scaleInput)
        {
            float2 scaleVec;
            if (getFloat2Value(scaleInput, scaleVec))
            {
                xform.setScaling(float3(scaleVec.x, scaleVec.y, 1.f));
            }
        }

        if (UsdShadeInput rotateInput = source.GetInput(gRotationToken); rotateInput)
        {
            float degrees;
            if (rotateInput.Get<float>(&degrees, UsdTimeCode::EarliestTime()))
            {
                xform.setRotationEulerDeg(float3(0.f, 0.f, degrees));
            }
        }

        if (UsdShadeInput translateInput = source.GetInput(gTranslationToken); translateInput)
        {
            float2 translateVec;
            if (getFloat2Value(translateInput, translateVec))
            {
                xform.setTranslation(float3(translateVec.x, translateVec.y, 0.f));
            }
        }
        xform.setCompositionOrder(Transform::CompositionOrder::ScaleRotateTranslate);
    }
    else if (id != gFloat2PrimvarReaderToken)
    {
        logWarning(
            "UsdUVTexture node '{}' defines an st primvar reader '{}' of an unexpected type: '{}'.",
            shader.GetPrim().GetPath().GetString(),
            source.GetPrim().GetPath().GetString(),
            id.GetString()
        );
    }
    return true;
}

ConvertedInput ConvertedInput::convertTexture(
    const UsdShadeInput& input,
    const TfToken& outputName,
    const TextureEncoding& texEncoding,
    ConvertedTexTransform& texTransform
)
{
    ConvertedInput ret;

    ret.shadeInput = input;

    // Note that the wrapS, wrapT, scale, and bias inputs are unsupported, and are ignored.

    UsdPrim prim(input.GetPrim());
    if (!prim.IsA<UsdShadeShader>())
    {
        logWarning("Expected UsdUVTexture node '{}' is not a UsdShadeShader.", prim.GetPath().GetString());
        return ret;
    }

    UsdShadeShader shader(prim);

    TfToken id = getAttribute(prim.GetAttribute(UsdShadeTokens->infoId), TfToken());
    if (id != gUVTextureToken)
    {
        logWarning("Expected UsdUVTexture node '{}' is not a UsdUVTexture.", prim.GetPath().GetString());
        return ret;
    }

    if (shader.GetInput(gWrapSToken) || shader.GetInput(gWrapTToken))
    {
        // Issue a low-priorty message, under the assumption that wrap modes most often don't matter.
        logInfo("UsdUvTexture node '{}' specifies a wrap mode, which is not supported.", prim.GetPath().GetString());
    }

    if (UsdShadeInput biasInput = shader.GetInput(gBiasToken); biasInput)
    {
        float4 value;
        if (!getFloat4Value(biasInput, value))
        {
            logWarning(
                "UsdUvTexture node '{}' specifies value scale of an unsupported type: '{}'.",
                prim.GetPath().GetString(),
                biasInput.GetTypeName().GetAsToken().GetString()
            );
        }
        else if (texEncoding == TextureEncoding::Normal)
        {
            if (any(value.xy() != float2(-1.f, -1.f)))
            {
                logWarning(
                    "UsdUvTexture normalMap node '{}' specifies a bias other than (-1, -1), which is not supported.",
                    prim.GetPath().GetString()
                );
            }
        }
        else if (any(value != float4(0.f, 0.f, 0.f, 0.f)))
        {
            logWarning("UsdUvTexture node '{}' specifies a non-zero bias, which is not supported.", prim.GetPath().GetString());
        }
    }

    if (UsdShadeInput scaleInput = shader.GetInput(gScaleToken); scaleInput)
    {
        if (!getFloat4Value(scaleInput, ret.textureScale))
        {
            logWarning(
                "UsdUvTexture node '{}' specifies value scale of an unsupported type: '{}'.",
                prim.GetPath().GetString(),
                scaleInput.GetTypeName().GetAsToken().GetString()
            );
        }
        else if (texEncoding == TextureEncoding::Normal)
        {
            if (any(ret.textureScale.xy() != float2(2.f, 2.f)))
            {
                logWarning(
                    "UsdUvTexture normallMap '{}' has scale other than (2.0, 2.0), which is not supported.", prim.GetPath().GetString()
                );
            }
        }
        else if (any(ret.textureScale != float4(1.f, 1.f, 1.f, 1.f)))
        {
            logWarning("UsdUvTexture node '{}' specifies a value scale, which is not supported.", prim.GetPath().GetString());
        }
    }

    if (convertTextureCoords(shader, ret.texTransform))
    {
        texTransform.update(ret);
    }
    else
    {
        // Issue a lower priority message if there is simply no st input defined, under the assumption that the default
        // behavior is expected.
        logInfo("UsdUVTexture node '{}' does not define an st input.", prim.GetPath().GetString());
    }

    // Initialize the uniform converted value using the fallback value, if any.
    GfVec4f fallbackValue = getAttribute(prim.GetAttribute(gFallbackToken), GfVec4f(0.f, 0.f, 0.f, 1.f));
    ret.uniformValue = float4(fallbackValue[0], fallbackValue[1], fallbackValue[2], fallbackValue[3]);

    // Color space may be specified in USD either as an attribute of the input connection, which we check here, or
    // directly on the the asset, which we check below, and which takes precedence.
    ret.loadSRGB = texEncoding == TextureEncoding::Srgb;
    TfToken colorSpace = getAttribute(prim.GetAttribute(gSourceColorSpaceToken), TfToken());
    if (colorSpace == gSRGBToken)
        ret.loadSRGB = true;
    else if (colorSpace == gRawToken)
        ret.loadSRGB = false;

    // Convert output specification to texture channel flag.
    ret.channels = TextureChannelFlags::None;
    if (outputName == gRToken)
    {
        ret.channels |= TextureChannelFlags::Red;
    }
    if (outputName == gGToken)
    {
        ret.channels |= TextureChannelFlags::Green;
    }
    if (outputName == gBToken)
    {
        ret.channels |= TextureChannelFlags::Blue;
    }
    if (outputName == gAToken)
    {
        ret.channels |= TextureChannelFlags::Alpha;
    }
    if (outputName == gRGToken)
    {
        ret.channels |= TextureChannelFlags::Red;
        ret.channels |= TextureChannelFlags::Green;
    }
    if (outputName == gRGBToken)
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

    UsdShadeConnectableAPI source;
    TfToken sourceName;
    UsdShadeAttributeType sourceType;
    UsdShadeInput fileInput(getSourceInput(shader.GetInput(gFileToken), source, sourceName, sourceType));
    if (fileInput)
    {
        SdfAssetPath path;
        UsdPrim filePrim(fileInput.GetPrim());
        UsdAttribute fileAttrib = filePrim.GetAttribute(gInputsFileToken);

        TfToken fileColorSpace = fileAttrib.GetColorSpace();
        if (fileColorSpace == gSRGBToken)
            ret.loadSRGB = true;
        else if (fileColorSpace == gRawToken)
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

ConvertedInput ConvertedInput::convertFloat(const UsdShadeInput& input, const TfToken& sourceName, ConvertedTexTransform& texTransform)
{
    // Get the source attribute
    ConvertedInput ret;
    ret.shadeInput = input;

    SdfValueTypeName typeName(input.GetTypeName());
    if (typeName == SdfValueTypeNames->Asset)
    {
        ret = convertTexture(input, sourceName, TextureEncoding::Linear, texTransform);
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

ConvertedInput ConvertedInput::convertColor(
    const UsdShadeInput& input,
    const TfToken& sourceName,
    const TextureEncoding& texEncoding,
    ConvertedTexTransform& texTransform
)
{
    ConvertedInput ret;
    ret.shadeInput = input;

    SdfValueTypeName typeName(input.GetTypeName());
    ret.uniformValue = float4(0.f, 0.f, 0.f, 1.f);
    if (typeName == SdfValueTypeNames->Color3f || typeName == SdfValueTypeNames->Float3)
    {
        GfVec3f v;
        if (input.Get<GfVec3f>(&v, UsdTimeCode::EarliestTime()))
        {
            ret.uniformValue = float4(v[0], v[1], v[2], 1.f);
        }
    }
    else if (typeName == SdfValueTypeNames->Asset)
    {
        ret = convertTexture(input, sourceName, texEncoding, texTransform);
    }
    else
    {
        logWarning(
            "Unexpected value type when converting color input: '{}' for '{}'.",
            typeName.GetAsToken().GetString(),
            input.GetFullName().GetString()
        );
    }
    return ret;
}

void ConvertedTexTransform::update(const ConvertedInput& input)
{
    if (!input.isTextured())
        return;
    const Falcor::Transform& newTransform = input.texTransform;
    if (newTransform.getMatrix() != Falcor::float4x4::identity() && transform.getMatrix() == Falcor::float4x4::identity())
    {
        transform = newTransform;
        return;
    }
    if (newTransform.getMatrix() != transform.getMatrix())
    {
        Falcor::logWarning(
            "Shader input '{}' specifies a texture transform that differs from that used on another texture, which is not supported. "
            "Applying the first "
            "encountered non-idenity transform to all textures.",
            input.shadeInput.GetPrim().GetPath().GetString()
        );
    }
}
} // namespace Falcor
