/***************************************************************************
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
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
***************************************************************************/
#include "Framework.h"
#include "Material.h"
#include "API/ConstantBuffer.h"
#include "API/Texture.h"
#include "API/Buffer.h"
#include "Utils/Platform/OS.h"
#include "Utils/Math/FalcorMath.h"
#include "Graphics/Program/ProgramVars.h"
#include "Graphics/Program/GraphicsProgram.h"
#include <sstream>

namespace Falcor
{
    uint32_t Material::sMaterialCounter = 0;
    static const char* kMaterialVarName = "materialBlock";

    Material::Material(const std::string& name) : mName(name)
    {
        mData.id = sMaterialCounter;
        sMaterialCounter++;
    }

    Material::SharedPtr Material::create(const std::string& name)
    {
        Material* pMaterial = new Material(name);
        return SharedPtr(pMaterial);
    }

    Material::~Material() = default;
    
    void Material::resetGlobalIdCounter()
    {
        sMaterialCounter = 0;
    }

    void Material::setID(int32_t id)
    {
        mParamBlockDirty = mParamBlockDirty || (mData.id != id);
        mData.id = id;
    }

    void Material::setShadingModel(uint32_t model) 
    { 
        mParamBlockDirty = mParamBlockDirty || (EXTRACT_SHADING_MODEL(mData.flags) != model);
        mData.flags = PACK_SHADING_MODEL(mData.flags, model); 
    }

    void Material::setAlphaTestMode(AlphaTestMode alphaMode) 
    { 
        mParamBlockDirty = mParamBlockDirty || (alphaTestMode != alphaMode);
        alphaTestMode = alphaMode;
    }

    void Material::setDoubleSided(bool doubleSided) 
    { 
        mParamBlockDirty = mParamBlockDirty || (EXTRACT_DOUBLE_SIDED(mData.flags) != doubleSided);
        mData.flags = PACK_DOUBLE_SIDED(mData.flags, doubleSided ? 1 : 0); 
    }

    void Material::setAlphaThreshold(float alpha)
    { 
        mParamBlockDirty = mParamBlockDirty || (mData.alphaThreshold != alpha);
        mData.alphaThreshold = alpha; 
    }

    void Material::setHashedAlphaTestScale(float scale)
    {
        mParamBlockDirty = mParamBlockDirty || (hashedAlphaScale != scale);
        hashedAlphaScale = scale;
    }

    void Material::setDiffuseBrdf(uint32_t brdf)
    {
        mParamBlockDirty = mParamBlockDirty || brdf != diffuseBrdf;
        diffuseBrdf = brdf;
    }
    
    void Material::setHeightScaleOffset(float scale, float offset) 
    { 
        mParamBlockDirty = mParamBlockDirty || (mData.heightScaleOffset.x != scale) || (mData.heightScaleOffset.y != offset);
        mData.heightScaleOffset = vec2(scale, offset);
    }

    void Material::setIndexOfRefraction(float IoR) 
    { 
        mParamBlockDirty = mParamBlockDirty || (mData.IoR != IoR);
        mData.IoR = IoR; 
    }

    void Material::setSampler(Sampler::SharedPtr pSampler)
    {
        mParamBlockDirty = mParamBlockDirty || (pSampler != mSampler);
        mSampler = pSampler;
    }

    void Material::setBaseColorTexture(Texture::SharedPtr& pBaseColor)
    {
        mParamBlockDirty = mParamBlockDirty || (diffuseChannel != pBaseColor);
        diffuseChannel = pBaseColor;
        bool hasAlpha = pBaseColor && doesFormatHasAlpha(pBaseColor->getFormat());
        setAlphaTestMode(hasAlpha ? AlphaTestMode::HashedIsotropic : AlphaTestMode::Disabled);
    }

    void Material::setSpecularTexture(Texture::SharedPtr pSpecular)
    {
        mParamBlockDirty = mParamBlockDirty || (specularChannel != pSpecular);
        specularChannel = pSpecular;
    }

    void Material::setEmissiveTexture(const Texture::SharedPtr& pEmissive)
    {
        mParamBlockDirty = mParamBlockDirty || (emissiveChannel != pEmissive);
        emissiveChannel = pEmissive;
    }

    void Material::setBaseColor(const vec4& color)
    {
        mParamBlockDirty = mParamBlockDirty || (diffuseChannel != color);
        diffuseChannel = color;
    }

    void Material::setSpecularParams(const vec4& color)
    {
        mParamBlockDirty = mParamBlockDirty || (specularChannel != color);
        specularChannel = color;
    }

    void Material::setEmissiveColor(const vec3& color)
    {
        mParamBlockDirty = mParamBlockDirty || (emissiveChannel != vec4(color, 1.0f));
        emissiveChannel = vec4(color, 1.0f);
    }

    void Material::updateOcclusionFlag()
    {
        bool hasMap = false;
        switch (EXTRACT_SHADING_MODEL(mData.flags))
        {
        case ShadingModelMetalRough:
            hasMap = (specularChannel.type != MaterialChannel::Type::Unused);
            break;
        case ShadingModelSpecGloss:
            hasMap = (occlusionChannel.type != MaterialChannel::Type::Unused);
            break;
        default:
            should_not_get_here();
        }
        bool shouldEnable = mOcclusionMapEnabled && hasMap;
        mData.flags = PACK_OCCLUSION_MAP(mData.flags, shouldEnable ? 1 : 0);
    }
    uint32_t Material::getNormalMode(Texture::SharedPtr pNormalMap) const
    {
        uint32_t normalMode = NormalMapUnused;
        if (pNormalMap)
        {
            switch (getFormatChannelCount(pNormalMap->getFormat()))
            {
            case 2:
                normalMode = NormalMapRG;
                break;
            case 3:
            case 4: // Some texture formats don't support RGB, only RGBA. We have no use for the alpha channel in the normal map.
                normalMode = NormalMapRGB;
                break;
            default:
                should_not_get_here();
                logWarning("Unsupported normal map format for material " + mName);
            }
        }
        return normalMode;
    }

    void Material::setNormalMap(Texture::SharedPtr pNormalMap)
    {
        mParamBlockDirty = mParamBlockDirty || (normalMap != MaterialChannel(pNormalMap));
        normalMap = pNormalMap;
        uint32_t normalMode = getNormalMode(pNormalMap);
        mData.flags = PACK_NORMAL_MAP_TYPE(mData.flags, normalMode);
    }

    void Material::setOcclusionMap(Texture::SharedPtr pOcclusionMap)
    {
        mParamBlockDirty = mParamBlockDirty || (occlusionChannel != pOcclusionMap);
        occlusionChannel = pOcclusionMap;
        updateOcclusionFlag();
    }

    void Material::setLightMap(Texture::SharedPtr pLightMap)
    {
        mParamBlockDirty = mParamBlockDirty || (lightmapChannel != pLightMap);
        lightmapChannel = pLightMap;
        mParamBlockDirty = true;
    }

    void Material::setHeightMap(Texture::SharedPtr pHeightMap)
    {
        mParamBlockDirty = mParamBlockDirty || (heightChannel != pHeightMap);
        heightChannel = pHeightMap;
        mParamBlockDirty = true;
    }

    bool Material::operator==(const Material& other) const 
    {
#define compare_field(_a) if (_a != other._a) return false
        compare_field(diffuseChannel);
        compare_field(specularChannel);
        compare_field(emissiveChannel);
        compare_field(mData.alphaThreshold);
        compare_field(mData.IoR);
        compare_field(mData.flags);
        compare_field(mData.heightScaleOffset);
        compare_field(normalMap);
        compare_field(occlusionChannel);
        compare_field(lightmapChannel);
        compare_field(heightChannel);
#undef compare_field

        if (mSampler != other.mSampler) return false;
        return true;
    }
    
    #if _LOG_ENABLED
#define check_offset(_a) assert(pCB->getVariableOffset(std::string(varName) + "materialData." #_a) == (offsetof(MaterialData, _a) + offset))
#else
#define check_offset(_a)
#endif

    void setMaterialChannelIntoParameterBlock(ParameterBlock* pBlock, ConstantBuffer* pCB, const std::string& varName, const MaterialChannel& channel)
    {
        if (channel.type == MaterialChannel::Type::Texture)
        {
            pBlock->setTexture(varName + ".tex", channel.texture);
        }
        else if (channel.type == MaterialChannel::Type::Constant)
        {
            pCB->setVariable(varName + ".val", channel.constantValue);
        }
    }

    void Material::setMaterialIntoBlockCommon(ParameterBlock* pBlock, ConstantBuffer* pCB, size_t offset, const std::string& varName) const
    {
        // OPTME:
        // First set the desc and the values
        static const size_t dataSize = sizeof(MaterialData);
        static_assert(dataSize % sizeof(glm::vec4) == 0, "Material::MaterialData size should be a multiple of 16");

        check_offset(heightScaleOffset);
        assert(offset + dataSize <= pCB->getSize());

        pCB->setBlob(&mData, offset, dataSize);

        // Now set the channels
#define set_channel(channelName) setMaterialChannelIntoParameterBlock(pBlock, pCB, varName + #channelName, channelName)
        set_channel(diffuseChannel);
        set_channel(specularChannel);
        set_channel(emissiveChannel);
        set_channel(occlusionChannel);
        set_channel(lightmapChannel);
        set_channel(heightChannel);
        set_channel(normalMap);
#undef set_texture
        if (alphaTestMode == AlphaTestMode::HashedAnisotropic ||
            alphaTestMode == AlphaTestMode::HashedIsotropic)
            pCB->setVariable(varName + "alphaTest.hashedAlphaScale", hashedAlphaScale);
        pBlock->setSampler(varName + "samplerState", mSampler);
    }

    void Material::setIntoParameterBlock(ParameterBlock* pBlock, const std::string& varName) const
    {
        ConstantBuffer* pCB = pBlock->getConstantBuffer(pBlock->getReflection()->getName()).get();
        setMaterialIntoBlockCommon(pBlock, pCB, 0, varName);
        pBlock->setTypeName(shaderTypeName);
        pBlock->genericTypeParamName = "TMaterial";
        pBlock->genericTypeArgumentName = shaderTypeName;
    }

    // SLANG-INTEGRATION: forward declare
    ReflectionType::SharedPtr reflectType(slang::TypeLayoutReflection* pSlangType);

    void buildTypeStr(std::stringstream & sb, const MaterialChannel& channel)
    {
        switch (channel.type)
        {
        case MaterialChannel::Type::Unused:
            sb << "UnusedChannel";
            return;
        case MaterialChannel::Type::Texture:
            sb << "TextureChannel";
            return;
        case MaterialChannel::Type::Constant:
            sb << "ConstantChannel";
            return;
        default:
            should_not_get_here();
            logWarning("Unsupported channel type.");
        }
    }

    const char * getAlphaTestShaderType(AlphaTestMode mode)
    {
        switch (mode)
        {
        case AlphaTestMode::Basic:
            return "BasicAlphaTest";
        case AlphaTestMode::HashedAnisotropic:
            return "HashedAlphaTest<1>";
        case AlphaTestMode::HashedIsotropic:
            return "HashedAlphaTest<0>";
        default:
            return "NoAlphaTest";
        }
    }

    const char * getNormalModeShaderType(uint32_t normalMode)
    {
        switch (normalMode)
        {
        case NormalMapRGB:
            return "NormalMap<1>";
        case NormalMapRG:
            return "NormalMap<0>";
        default:
            return "NoNormalMap";
        }
    }

    ParameterBlock::SharedConstPtr Material::getParameterBlock() const
    {
        if (mParamBlockDirty)
        {
            mParamBlockDirty = false;
            // build type name
            std::stringstream sb;
            sb << "StandardMaterial<";
            buildTypeStr(sb, diffuseChannel); sb << ", ";
            buildTypeStr(sb, specularChannel); sb << ", ";
            buildTypeStr(sb, emissiveChannel); sb << ", ";
            buildTypeStr(sb, occlusionChannel); sb << ", ";
            buildTypeStr(sb, lightmapChannel); sb << ", ";
            buildTypeStr(sb, heightChannel);  sb << ", ";
            sb << getNormalModeShaderType(getNormalMode(normalMap.texture)) << ", ";
            sb << getAlphaTestShaderType(alphaTestMode) << ", ";
            sb << diffuseBrdf;
            sb << ">";
            shaderTypeName = sb.str();
            if (mpParameterBlock == nullptr || spBlockReflection == nullptr 
                || reflectionTypeName != shaderTypeName)
            {
                GraphicsProgram::SharedPtr pProgram = GraphicsProgram::createFromFile("Framework/Shaders/MaterialBlock.slang", "", "main");
                ProgramReflection::SharedConstPtr pReflection = pProgram->getActiveVersion()->getReflector();
                auto slangReq = pProgram->getActiveVersion()->slangRequest;
                auto reflection = spGetReflection(slangReq);

                auto materialType = spReflection_FindTypeByName(reflection, shaderTypeName.c_str());
                auto layout = spReflection_GetTypeLayout(reflection, materialType, SLANG_LAYOUT_RULES_DEFAULT);
                auto blockType = reflectType((slang::TypeLayoutReflection*)layout);
                auto blockReflection = ParameterBlockReflection::create(blockType);
                spBlockReflection = blockReflection;
                reflectionTypeName = shaderTypeName;
                assert(spBlockReflection);
                mpParameterBlock = ParameterBlock::create(spBlockReflection, true);
            }
            setIntoParameterBlock(mpParameterBlock.get(), "");
        }
        return mpParameterBlock;
    }
}
