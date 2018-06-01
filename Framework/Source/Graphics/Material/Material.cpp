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

namespace Falcor
{
    uint32_t Material::sMaterialCounter = 0;
    ParameterBlockReflection::SharedConstPtr Material::spBlockReflection;
    static const char* kMaterialVarName = "materialBlock";

    Material::Material(const std::string& name) : mName(name)
    {
        mData.id = sMaterialCounter;
        sMaterialCounter++;
        if (spBlockReflection == nullptr)
        {
            GraphicsProgram::SharedPtr pProgram = GraphicsProgram::createFromFile("Framework/Shaders/MaterialBlock.slang", "", "main");
            ProgramReflection::SharedConstPtr pReflection = pProgram->getReflector();
            spBlockReflection = pReflection->getParameterBlock(kMaterialVarName);
            assert(spBlockReflection);
        }
        mpParameterBlock = ParameterBlock::create(spBlockReflection, true);
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

    void Material::setAlphaMode(uint32_t alphaMode) 
    { 
        mParamBlockDirty = mParamBlockDirty || (EXTRACT_ALPHA_MODE(mData.flags) != alphaMode);
        mData.flags = PACK_ALPHA_MODE(mData.flags, alphaMode);
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
        mParamBlockDirty = mParamBlockDirty || (pSampler != mData.resources.samplerState);
        mData.resources.samplerState = pSampler;
    }

    void Material::setBaseColorTexture(Texture::SharedPtr& pBaseColor)
    {
        mParamBlockDirty = mParamBlockDirty || (mData.resources.baseColor != pBaseColor);
        mData.resources.baseColor = pBaseColor;
        updateBaseColorType();
        bool hasAlpha = pBaseColor && doesFormatHasAlpha(pBaseColor->getFormat());
        setAlphaMode(hasAlpha ? AlphaModeMask : AlphaModeOpaque);
    }

    void Material::setSpecularTexture(Texture::SharedPtr pSpecular)
    {
        mParamBlockDirty = mParamBlockDirty || (mData.resources.specular != pSpecular);
        mData.resources.specular = pSpecular;
        updateSpecularType();
    }

    void Material::setEmissiveTexture(const Texture::SharedPtr& pEmissive)
    {
        mParamBlockDirty = mParamBlockDirty || (mData.resources.emissive != pEmissive);
        mData.resources.emissive = pEmissive;
        updateEmissiveType();
    }

    void Material::setBaseColor(const vec4& color)
    {
        mParamBlockDirty = mParamBlockDirty || (mData.baseColor != color);
        mData.baseColor = color;
        updateBaseColorType();
    }

    void Material::setSpecularParams(const vec4& color)
    {
        mParamBlockDirty = mParamBlockDirty || (mData.specular != color);
        mData.specular = color;
        updateSpecularType();
    }

    void Material::setEmissiveColor(const vec3& color)
    {
        mParamBlockDirty = mParamBlockDirty || (mData.emissive != color);
        mData.emissive = color;
        updateEmissiveType();
    }

    template<typename vec>
    static uint32_t getChannelMode(bool hasTexture, const vec& color)
    {
        if (hasTexture) return ChannelTypeTexture;
        if (luminance(color) == 0) return ChannelTypeUnused;
        return ChannelTypeConst;
    }

    void Material::updateBaseColorType()
    {
        mData.flags = PACK_DIFFUSE_TYPE(mData.flags, getChannelMode(mData.resources.baseColor != nullptr, mData.baseColor));
    }

    void Material::updateSpecularType()
    {
        mData.flags = PACK_SPECULAR_TYPE(mData.flags, getChannelMode(mData.resources.specular != nullptr, mData.specular));
    }

    void Material::updateEmissiveType()
    {
        mData.flags = PACK_EMISSIVE_TYPE(mData.flags, getChannelMode(mData.resources.emissive != nullptr, mData.emissive));
    }

    void Material::updateOcclusionFlag()
    {
        bool hasMap = false;
        switch (EXTRACT_SHADING_MODEL(mData.flags))
        {
        case ShadingModelMetalRough:
            hasMap = (mData.resources.specular != nullptr);
            break;
        case ShadingModelSpecGloss:
            hasMap = (mData.resources.occlusionMap != nullptr);
            break;
        default:
            should_not_get_here();
        }
        bool shouldEnable = mOcclusionMapEnabled && hasMap;
        mData.flags = PACK_OCCLUSION_MAP(mData.flags, shouldEnable ? 1 : 0);
    }

    void Material::setNormalMap(Texture::SharedPtr pNormalMap)
    {
        mParamBlockDirty = mParamBlockDirty || (mData.resources.normalMap != pNormalMap);
        mData.resources.normalMap = pNormalMap;
        uint32_t normalMode = NormalMapUnused;
        if (pNormalMap)
        {
            switch(getFormatChannelCount(pNormalMap->getFormat()))
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
        mData.flags = PACK_NORMAL_MAP_TYPE(mData.flags, normalMode);
    }

    void Material::setOcclusionMap(Texture::SharedPtr pOcclusionMap)
    {
        mParamBlockDirty = mParamBlockDirty || (mData.resources.occlusionMap != pOcclusionMap);
        mData.resources.occlusionMap = pOcclusionMap;
        mParamBlockDirty = true;
        updateOcclusionFlag();
    }

    void Material::setLightMap(Texture::SharedPtr pLightMap)
    {
        mParamBlockDirty = mParamBlockDirty || (mData.resources.lightMap != pLightMap);
        mData.resources.lightMap = pLightMap;
        mData.flags = PACK_LIGHT_MAP(mData.flags, pLightMap ? 1 : 0);
        mParamBlockDirty = true;
    }

    void Material::setHeightMap(Texture::SharedPtr pHeightMap)
    {
        mParamBlockDirty = mParamBlockDirty || (mData.resources.heightMap != pHeightMap);
        mData.resources.heightMap = pHeightMap;
        mData.flags = PACK_HEIGHT_MAP(mData.flags, pHeightMap ? 1 : 0);
        mParamBlockDirty = true;
    }

    bool Material::operator==(const Material& other) const 
    {
#define compare_field(_a) if (mData._a != other.mData._a) return false
        compare_field(baseColor);
        compare_field(specular);
        compare_field(emissive);
        compare_field(alphaThreshold);
        compare_field(IoR);
        compare_field(flags);
        compare_field(heightScaleOffset);
#undef compare_field

#define compare_texture(_a) if (mData.resources._a != other.mData.resources._a) return false
        compare_texture(baseColor);
        compare_texture(specular);
        compare_texture(emissive);
        compare_texture(normalMap);
        compare_texture(occlusionMap);
        compare_texture(lightMap);
        compare_texture(heightMap);
#undef compare_texture
        if (mData.resources.samplerState != other.mData.resources.samplerState) return false;
        return true;
    }
    
    #if _LOG_ENABLED
#define check_offset(_a) assert(pCB->getVariableOffset(std::string(varName) + #_a) == (offsetof(MaterialData, _a) + offset))
#else
#define check_offset(_a)
#endif

    static void setMaterialIntoBlockCommon(ParameterBlock* pBlock, ConstantBuffer* pCB, size_t offset, const std::string& varName, const MaterialData& data)
    {
        // OPTME:
        // First set the desc and the values
        static const size_t dataSize = sizeof(MaterialData) - sizeof(MaterialResources);
        static_assert(dataSize % sizeof(glm::vec4) == 0, "Material::MaterialData size should be a multiple of 16");

        check_offset(emissive);
        check_offset(heightScaleOffset);
        assert(offset + dataSize <= pCB->getSize());

        pCB->setBlob(&data, offset, dataSize);

        // Now set the textures
#define set_texture(texName) pBlock->setTexture(varName + "resources." #texName, data.resources.texName)
        set_texture(baseColor);
        set_texture(specular);
        set_texture(emissive);
        set_texture(normalMap);
        set_texture(occlusionMap);
        set_texture(lightMap);
        set_texture(heightMap);
#undef set_texture
        pBlock->setSampler(varName + "resources.samplerState", data.resources.samplerState);
    }

    void Material::setIntoParameterBlock(ParameterBlock* pBlock) const
    {
        ConstantBuffer* pCB = pBlock->getDefaultConstantBuffer().get();
        setMaterialIntoBlockCommon(pBlock, pCB, 0, "", mData);
    }

    void Material::setIntoProgramVars(ProgramVars* pVars, ConstantBuffer* pCb, const char varName[]) const
    {
        size_t offset = pCb->getVariableOffset(varName);

        if (offset == ConstantBuffer::kInvalidOffset)
        {
            logError(std::string("Material::setIntoProgramVars() - variable \"") + varName + "\" not found in constant buffer\n");
            return;
        }
        setMaterialIntoBlockCommon(pVars->getDefaultBlock().get(), pCb, offset, std::string(varName) + '.', mData);
    }

    ParameterBlock::SharedConstPtr Material::getParameterBlock() const
    {
        if (mParamBlockDirty)
        {
            mParamBlockDirty = false;
            setIntoParameterBlock(mpParameterBlock.get());
        }
        return mpParameterBlock;
    }
}
