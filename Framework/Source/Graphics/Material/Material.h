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
#pragma once
#include "glm/vec3.hpp"
#include <map>
#include <vector>
#include "glm/common.hpp"
#include "glm/geometric.hpp"
#include "API/Texture.h"
#include "glm/mat4x4.hpp"
#include "API/Sampler.h"
#include "Data/HostDeviceData.h"
#include "Graphics/Program/ParameterBlock.h"

namespace Falcor
{
    class Texture;
    class ProgramVars;
    class ConstantBuffer;

    class MaterialChannel
    {
    public:
        enum class Type
        {
            Unused, Constant, Texture
        };
        Type type = Type::Unused;
        Texture::SharedPtr texture;
        float4 constantValue;
        MaterialChannel() = default;
        MaterialChannel(Texture::SharedPtr tex)
        {
            type = Type::Texture;
            texture = tex;
        }
        MaterialChannel(float4 val)
        {
            constantValue = val;
            type = Type::Constant;
        }
        bool operator == (const MaterialChannel & other) const
        {
            if (type != other.type) return false;
            if (type == Type::Texture)
                return texture == other.texture;
            else
                return constantValue == other.constantValue;
        }
        bool operator != (const MaterialChannel & other) const
        {
            return !(*this == other);
        }
    };

    class Material : public std::enable_shared_from_this<Material>
    {
    public:
        using SharedPtr = std::shared_ptr<Material>;
        using SharedConstPtr = std::shared_ptr<const Material>;

        /** Create a new material.
            \param[in] name The material name
        */
        static SharedPtr create(const std::string& name);

        ~Material();

        /** Set the material name.
        */
        void setName(const std::string& name) { mName = name; }

        /** Get the material name.
        */
        const std::string& getName() const { return mName; }

        /** Get the material ID.
        */
        const int32_t getId() const { return mData.id; }

        /** Set the material ID
        */
        void setID(int32_t id);

        /** Reset all global id counter of model, mesh and material
        */
        static void resetGlobalIdCounter();

        /** Set the diffuse texture
        */
        void setDiffuseTexture(Texture::SharedPtr& pDiffuse);

        /** Get the diffuse texture
        */
        Texture::SharedPtr getDiffuseTexture() const { return diffuseChannel.texture; }

        /** Set the specular texture
        */
        void setSpecularTexture(Texture::SharedPtr pSpecular);

        /** Get the specular texture
        */
        Texture::SharedPtr getSpecularTexture() const { return specularChannel.texture; }

        /** Set the emissive texture
        */
        void setEmissiveTexture(const Texture::SharedPtr& pEmissive);

        /** Get the emissive texture
        */
        Texture::SharedPtr getEmissiveTexture() const { return emissiveChannel.texture; }

        /** Set the shading model
        */
        void setShadingModel(uint32_t model);

        /** Get the shading model
        */
        uint32_t getShadingModel() const { return EXTRACT_SHADING_MODEL(mData.flags); }

        /** Set the normal map
        */
        void setNormalMap(Texture::SharedPtr pNormalMap);

        /** Get the normal map
        */
        Texture::SharedPtr getNormalMap() const { return normalChannel.texture; }

        /** Set the occlusion map
        */
        void setOcclusionMap(Texture::SharedPtr pOcclusionMap);

        /** Get the occlusion map
        */
        Texture::SharedPtr getOcclusionMap() const { return occlusionChannel.texture; }

        /** Set the light map
        */
        void setLightMap(Texture::SharedPtr pLightMap);

        /** Get the light map
        */
        Texture::SharedPtr getLightMap() const { return lightmapChannel.texture; }

        /** Set the height map
        */
        void setHeightMap(Texture::SharedPtr pHeightMap);

        /** Get the height map
        */
        Texture::SharedPtr getHeightMap() const { return heightChannel.texture; }

        /** Set the diffuse color
        */
        void setDiffuseColor(const vec4& color);

        /** Get the diffuse color
        */
        const vec4& getDiffuseColor() const { return diffuseChannel.constantValue; }

        /** Set the specular color
        */
        void setSpecularColor(const vec4& color);

        /** Get the specular color
        */
        const vec4& getSpecularColor() const { return specularChannel.constantValue; }

        /** Set the emissive color
        */
        void setEmissiveColor(const vec3& color);

        /** Get the emissive color
        */
        const vec3& getEmissiveColor() const { return *(vec3*)&(emissiveChannel.constantValue); }

        /** Set the alpha mode
        */
        void setAlphaMode(uint32_t alphaMode);

        /** Get the alpha mode
        */
        uint32_t getAlphaMode() const { return EXTRACT_ALPHA_MODE(mData.flags); }

        /** Set the double-sided flag. This flag doesn't affect the rasterizer state, just the shading
        */
        void setDoubleSided(bool doubleSided);

        /** Returns true if the material is double-sided
        */
        bool getDoubleSided() const { return EXTRACT_DOUBLE_SIDED(mData.flags); }

        /** Set the alpha threshold. The threshold is only used if the alpha mode is `AlphaModeMask`
        */
        void setAlphaThreshold(float alpha);

        /** Get the alpha threshold
        */
        float getAlphaThreshold() const { return mData.alphaThreshold; }

        /** Get the flags
        */
        uint32_t getFlags() const { return mData.flags; }

        /** Set the height scale and offset
        */
        void setHeightScaleOffset(float scale, float offset);

        /** Get the height scale
        */
        float getHeightScale() const { return mData.heightScaleOffset.x; }

        /** Get the height offset
        */
        float getHeightOffset() const { return mData.heightScaleOffset.y; }

        /** Set the index of refraction
        */
        void setIndexOfRefraction(float IoR);

        /** Get the index of refraction
        */
        float getIndexOfRefraction() const { return mData.IoR; }

        /** Comparison operator
        */
        bool operator==(const Material& other) const;

        /** Bind a sampler to the material
        */
        void setSampler(Sampler::SharedPtr pSampler);

        /** Get the sampler attached to the material
        */
        Sampler::SharedPtr getSampler() const { return mSampler; }

        /** Get the ParameterBlock object for the material. Each material is created with a parameter-block. Using it is more efficient than assigning data to a custom constant-buffer.
        */
        ParameterBlock::SharedConstPtr getParameterBlock() const;

    private:        
        Material(const std::string& name);
        std::string mName;
        MaterialChannel diffuseChannel, specularChannel, emissiveChannel;
        MaterialChannel normalChannel, occlusionChannel, lightmapChannel, heightChannel;
        MaterialData mData;
        Sampler::SharedPtr mSampler;
        mutable bool mParamBlockDirty = true;
        mutable ParameterBlockReflection::SharedConstPtr spBlockReflection;
        mutable std::string reflectionTypeName; // type name of current reflection data
        mutable std::string shaderTypeName; // type name of parameter block
        mutable ParameterBlock::SharedPtr mpParameterBlock;
        static uint32_t sMaterialCounter;
        void setMaterialIntoBlockCommon(ParameterBlock* pBlock, ConstantBuffer* pCB, size_t offset, const std::string& varName) const;
        void setIntoParameterBlock(ParameterBlock* pBlock, const std::string& varName) const;
    };

#undef Texture2D
}
