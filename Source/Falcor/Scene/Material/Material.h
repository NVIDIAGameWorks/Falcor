/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
#include "MaterialData.slang"
#include "MaterialDefines.slangh"
#include "Scene/Transform.h"

namespace Falcor
{
    /** Channel Layout For Different Shading Models
        (Options listed in MaterialDefines.slangh)

        ShadingModelMetalRough
            BaseColor
                - RGB - Base Color
                - A   - Transparency
            Specular
                - R - Occlusion
                - G - Roughness
                - B - Metallic
                - A - Reserved

        ShadingModelSpecGloss
            BaseColor
                - RGB - Diffuse Color
                - A   - Transparency
            Specular
                - RGB - Specular Color
                - A   - Gloss

        ShadingModelHairChiang16
            BaseColor
                - RGB - Absorption coefficient, sigmaA
                - A   - Unused
            Specular
                - R   - Longitudinal roughness, betaM
                - G   - Azimuthal roughness, betaN
                - B   - The angle that the small scales on the surface of hair are offset from the base cylinder (in degrees).
                - A   - Unused

        Common for all shading models
            Emissive
                - RGB - Emissive Color
                - A   - Unused
            Normal
                - 3-Channel standard normal map, or 2-Channel BC5 format
    */

    class dlldecl Material : public std::enable_shared_from_this<Material>
    {
    public:
        using SharedPtr = std::shared_ptr<Material>;
        using SharedConstPtr = std::shared_ptr<const Material>;

        /** Flags indicating if and what was updated in the material
        */
        enum class UpdateFlags
        {
            None                = 0x0,  ///< Nothing updated
            DataChanged         = 0x1,  ///< Material data (properties) changed
            ResourcesChanged    = 0x2,  ///< Material resources (textures, sampler) changed
        };

        /** Texture slots available in the material
        */
        enum class TextureSlot
        {
            BaseColor,
            Specular,
            Emissive,
            Normal,
            Occlusion,
            SpecularTransmission,
            Displacement,

            Count // Must be last
        };

        /** Create a new material.
            \param[in] name The material name
        */
        static SharedPtr create(const std::string& name);

        ~Material();

        /** Render the UI.
            \return True if the material was modified.
        */
        bool renderUI(Gui::Widgets& widget);

        /** Returns the updates since the last call to clearUpdates.
        */
        UpdateFlags getUpdates() const { return mUpdates; }

        /** Clears the updates.
        */
        void clearUpdates() { mUpdates = UpdateFlags::None; }

        /** Returns the global updates (across all materials) since the last call to clearGlobalUpdates.
        */
        static UpdateFlags getGlobalUpdates() { return sGlobalUpdates; }

        /** Clears the global updates.
        */
        static void clearGlobalUpdates() { sGlobalUpdates = UpdateFlags::None; }

        /** Set the material name.
        */
        void setName(const std::string& name) { mName = name; }

        /** Get the material name.
        */
        const std::string& getName() const { return mName; }

        /** Set one of the available texture slots.
        */
        void setTexture(TextureSlot slot, Texture::SharedPtr pTexture);

        /** Load one of the available texture slots.
        */
        void loadTexture(TextureSlot slot, const std::string& filename, bool useSrgb = true);

        /** Clear one of the available texture slots.
        */
        void clearTexture(TextureSlot slot);

        /** Get one of the available texture slots.
        */
        Texture::SharedPtr getTexture(TextureSlot slot) const;

        /** Return the maximum dimensions of the bound textures.
        */
        uint2 getMaxTextureDimensions() const;

        /** Check if a texture is required to be in sRGB format
            Note: This depends on the shading model being used for the material.
        */
        bool isSrgbTextureRequired(TextureSlot slot);

        /** Set the base color texture
        */
        void setBaseColorTexture(Texture::SharedPtr pBaseColor) { setTexture(TextureSlot::BaseColor, pBaseColor); }

        /** Get the base color texture
        */
        Texture::SharedPtr getBaseColorTexture() const { return getTexture(TextureSlot::BaseColor); }

        /** Set the specular texture
        */
        void setSpecularTexture(Texture::SharedPtr pSpecular) { setTexture(TextureSlot::Specular, pSpecular); }

        /** Get the specular texture
        */
        Texture::SharedPtr getSpecularTexture() const { return getTexture(TextureSlot::Specular); }

        /** Set the emissive texture
        */
        void setEmissiveTexture(const Texture::SharedPtr& pEmissive) { setTexture(TextureSlot::Emissive, pEmissive); }

        /** Get the emissive texture
        */
        Texture::SharedPtr getEmissiveTexture() const { return getTexture(TextureSlot::Emissive); }

        /** Set the specular transmission texture
        */
        void setSpecularTransmissionTexture(const Texture::SharedPtr& pTransmission) { setTexture(TextureSlot::SpecularTransmission, pTransmission); }

        /** Get the specular transmission texture
        */
        Texture::SharedPtr getSpecularTransmissionTexture() const { return getTexture(TextureSlot::SpecularTransmission); }

        /** Set the shading model
        */
        void setShadingModel(uint32_t model);

        /** Get the shading model
        */
        uint32_t getShadingModel() const { return EXTRACT_SHADING_MODEL(mData.flags); }

        /** Set the normal map
        */
        void setNormalMap(Texture::SharedPtr pNormalMap) { setTexture(TextureSlot::Normal, pNormalMap); }

        /** Get the normal map
        */
        Texture::SharedPtr getNormalMap() const { return getTexture(TextureSlot::Normal); }

        /** Set the occlusion map
        */
        void setOcclusionMap(Texture::SharedPtr pOcclusionMap) { setTexture(TextureSlot::Occlusion, pOcclusionMap); }

        /** Get the occlusion map
        */
        Texture::SharedPtr getOcclusionMap() const { return getTexture(TextureSlot::Occlusion); }

        /** Set the displacement map
        */
        void setDisplacementMap(Texture::SharedPtr pDisplacementMap) { setTexture(TextureSlot::Displacement, pDisplacementMap); }

        /** Get the displacement map
        */
        Texture::SharedPtr getDisplacementMap() const { return getTexture(TextureSlot::Displacement); }

        /** Set the base color
        */
        void setBaseColor(const float4& color);

        /** Get the base color
        */
        const float4& getBaseColor() const { return mData.baseColor; }

        /** Set the specular parameters
        */
        void setSpecularParams(const float4& color);

        /** Get the specular parameters
        */
        const float4& getSpecularParams() const { return mData.specular; }

        /** Set the roughness
            Only available for metallic/roughness shading model.
        */
        void setRoughness(float roughness);

        /** Get the roughness
            Only available for metallic/roughness shading model.
        */
        float getRoughness() const { return getShadingModel() == ShadingModelMetalRough ? mData.specular.g : 0.f; }

        /** Set the metallic value
            Only available for metallic/roughness shading model.
        */
        void setMetallic(float metallic);

        /** Get the metallic value
            Only available for metallic/roughness shading model.
        */
        float getMetallic() const { return getShadingModel() == ShadingModelMetalRough ? mData.specular.b : 0.f; }

        /** Set the specular transmission
        */
        void setSpecularTransmission(float specularTransmission);

        /** Get the specular transmission
        */
        float getSpecularTransmission() const { return mData.specularTransmission; }

        /** Set the volume absorption (absorption coefficient).
        */
        void setVolumeAbsorption(const float3& volumeAbsorption);

        /** Get the volume absorption (absorption coefficient).
        */
        const float3& getVolumeAbsorption() const { return mData.volumeAbsorption; }

        /** Set the emissive color
        */
        void setEmissiveColor(const float3& color);

        /** Set the emissive factor
        */
        void setEmissiveFactor(float factor);

        /** Get the emissive color
        */
        const float3& getEmissiveColor() const { return mData.emissive; }

        /** Get the emissive factor
        */
        float getEmissiveFactor() const { return mData.emissiveFactor; }

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
        bool isDoubleSided() const { return EXTRACT_DOUBLE_SIDED(mData.flags); }

        /** Set the alpha threshold. The threshold is only used if the alpha mode is `AlphaModeMask`
        */
        void setAlphaThreshold(float alpha);

        /** Get the alpha threshold
        */
        float getAlphaThreshold() const { return mData.alphaThreshold; }

        /** Get the flags
        */
        uint32_t getFlags() const { return mData.flags; }

        /** Set the index of refraction
        */
        void setIndexOfRefraction(float IoR);

        /** Get the index of refraction
        */
        float getIndexOfRefraction() const { return mData.IoR; }

        /** Set the nested priority used for nested dielectrics
        */
        void setNestedPriority(uint32_t priority);

        /** Get the nested priority used for nested dielectrics.
            \return Nested priority, with 0 reserved for the highest possible priority.
        */
        uint32_t getNestedPriority() const { return EXTRACT_NESTED_PRIORITY(mData.flags); }

        /** Returns true if material is emissive.
        */
        bool isEmissive() const { return EXTRACT_EMISSIVE_TYPE(mData.flags) != ChannelTypeUnused; }

        /** Comparison operator.
            \return True if all materials properties *except* the name are identical.
        */
        bool operator==(const Material& other) const;

        /** Bind a sampler to the material
        */
        void setSampler(Sampler::SharedPtr pSampler);

        /** Get the sampler attached to the material
        */
        Sampler::SharedPtr getSampler() const { return mResources.samplerState; }

        /** Returns the material data struct.
        */
        const MaterialData& getData() const { return mData; }

        /** Returns the material resources struct.
        */
        const MaterialResources& getResources() const { return mResources; }

        /** Set the material texture transform.
        */
        void setTextureTransform(const Transform& texTransform);

        /** Get a reference to the material texture transform.
        */
        Transform& getTextureTransform() { return mTextureTransform; }

        /** Get the material texture transform.
        */
        const Transform& getTextureTransform() const { return mTextureTransform; }

    private:
        void markUpdates(UpdateFlags updates);

        void setFlags(uint32_t flags);
        void updateBaseColorType();
        void updateSpecularType();
        void updateEmissiveType();
        void updateSpecularTransmissionType();
        void updateAlphaMode();
        void updateNormalMapMode();
        void updateOcclusionFlag();
        void updateDisplacementFlag();

        Material(const std::string& name);
        std::string mName;
        MaterialData mData;
        MaterialResources mResources;
        Transform mTextureTransform;
        bool mOcclusionMapEnabled = false;
        mutable UpdateFlags mUpdates = UpdateFlags::None;
        static UpdateFlags sGlobalUpdates;
    };

    enum_class_operators(Material::UpdateFlags);
}
