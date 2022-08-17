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
#include "Material.h"
#include "BasicMaterialData.slang"

namespace Falcor
{
    /** Base class for basic non-layered materials.

        Texture channel layout:

            Displacement
                - RGB - Displacement data
                - A   - Unused

        See additional texture channels defined in derived classes.
    */
    class FALCOR_API BasicMaterial : public Material
    {
    public:
        using SharedPtr = std::shared_ptr<BasicMaterial>;
        using SharedConstPtr = std::shared_ptr<const BasicMaterial>;

        /** Render the UI.
            \return True if the material was modified.
        */
        virtual bool renderUI(Gui::Widgets& widget) override;

        /** Update material. This prepares the material for rendering.
            \param[in] pOwner The material system that this material is used with.
            \return Updates since last call to update().
        */
        Material::UpdateFlags update(MaterialSystem* pOwner) override;

        /** Returns true if the material has a displacement map.
        */
        bool isDisplaced() const override;

        /** Compares material to another material.
            \param[in] pOther Other material.
            \return true if all materials properties *except* the name are identical.
        */
        bool isEqual(const Material::SharedPtr& pOther) const override;

        /** Set the alpha mode.
        */
        void setAlphaMode(AlphaMode alphaMode) override;

        /** Set the alpha threshold. The threshold is only used when alpha mode != AlphaMode::Opaque.
        */
        void setAlphaThreshold(float alphaThreshold) override;

        /** Set one of the available texture slots.
            The call is ignored if the slot doesn't exist.
            \param[in] slot The texture slot.
            \param[in] pTexture The texture.
            \return True if the texture slot was changed, false otherwise.
        */
        bool setTexture(const TextureSlot slot, const Texture::SharedPtr& pTexture) override;

        /** Optimize texture usage for the given texture slot.
            This function may replace constant textures by uniform material parameters etc.
            \param[in] slot The texture slot.
            \param[in] texInfo Information about the texture bound to this slot.
            \param[out] stats Optimization stats passed back to the caller.
        */
        void optimizeTexture(const TextureSlot slot, const TextureAnalyzer::Result& texInfo, TextureOptimizationStats& stats) override;

        /** Set the default texture sampler for the material.
        */
        void setDefaultTextureSampler(const Sampler::SharedPtr& pSampler) override;

        /** Get the default texture sampler for the material.
        */
        Sampler::SharedPtr getDefaultTextureSampler() const override { return mpDefaultSampler; }

        /** Get the material data blob for uploading to the GPU.
        */
        virtual MaterialDataBlob getDataBlob() const override { return prepareDataBlob(mData); }


        // Additional member functions for BasicMaterial

        /** Set the base color texture.
        */
        void setBaseColorTexture(const Texture::SharedPtr& pBaseColor) { setTexture(TextureSlot::BaseColor, pBaseColor); }

        /** Get the base color texture.
        */
        Texture::SharedPtr getBaseColorTexture() const { return getTexture(TextureSlot::BaseColor); }

        /** Set the specular texture.
        */
        void setSpecularTexture(const Texture::SharedPtr& pSpecular) { setTexture(TextureSlot::Specular, pSpecular); }

        /** Get the specular texture.
        */
        Texture::SharedPtr getSpecularTexture() const { return getTexture(TextureSlot::Specular); }

        /** Set the emissive texture.
        */
        void setEmissiveTexture(const Texture::SharedPtr& pEmissive) { setTexture(TextureSlot::Emissive, pEmissive); }

        /** Get the emissive texture.
        */
        Texture::SharedPtr getEmissiveTexture() const { return getTexture(TextureSlot::Emissive); }

        /** Set the specular transmission texture.
        */
        void setTransmissionTexture(const Texture::SharedPtr& pTransmission) { setTexture(TextureSlot::Transmission, pTransmission); }

        /** Get the specular transmission texture.
        */
        Texture::SharedPtr getTransmissionTexture() const { return getTexture(TextureSlot::Transmission); }

        /** Set the normal map.
        */
        void setNormalMap(const Texture::SharedPtr& pNormalMap) { setTexture(TextureSlot::Normal, pNormalMap); }

        /** Get the normal map.
        */
        Texture::SharedPtr getNormalMap() const { return getTexture(TextureSlot::Normal); }

        /** Set the displacement map.
        */
        void setDisplacementMap(const Texture::SharedPtr& pDisplacementMap) { setTexture(TextureSlot::Displacement, pDisplacementMap); }

        /** Get the displacement map.
        */
        Texture::SharedPtr getDisplacementMap() const { return getTexture(TextureSlot::Displacement); }

        /** Set the displacement scale.
        */
        void setDisplacementScale(float scale);

        /** Get the displacement scale.
        */
        float getDisplacementScale() const { return mData.displacementScale; }

        /** Set the displacement offset.
        */
        void setDisplacementOffset(float offset);

        /** Get the displacement offset.
        */
        float getDisplacementOffset() const { return mData.displacementOffset; }

        /** Set the base color.
        */
        void setBaseColor(const float4& color);

        /** Get the base color.
        */
        float4 getBaseColor() const { return (float4)mData.baseColor; }

        /** Set the specular parameters.
        */
        void setSpecularParams(const float4& color);

        /** Get the specular parameters.
        */
        float4 getSpecularParams() const { return (float4)mData.specular; }

        /** Set the transmission color.
        */
        void setTransmissionColor(const float3& transmissionColor);

        /** Get the transmission color.
        */
        float3 getTransmissionColor() const { return (float3)mData.transmission; }

        /** Set the diffuse transmission.
        */
        void setDiffuseTransmission(float diffuseTransmission);

        /** Get the diffuse transmission.
        */
        float getDiffuseTransmission() const { return (float)mData.diffuseTransmission; }

        /** Set the specular transmission.
        */
        void setSpecularTransmission(float specularTransmission);

        /** Get the specular transmission.
        */
        float getSpecularTransmission() const { return (float)mData.specularTransmission; }

        /** Set the volume absorption (absorption coefficient).
        */
        void setVolumeAbsorption(const float3& volumeAbsorption);

        /** Get the volume absorption (absorption coefficient).
        */
        float3 getVolumeAbsorption() const { return (float3)mData.volumeAbsorption; }

        /** Set the volume scattering (scattering coefficient).
        */
        void setVolumeScattering(const float3& volumeScattering);

        /** Get the volume scattering (scattering coefficient).
        */
        float3 getVolumeScattering() const { return (float3)mData.volumeScattering; }

        /** Set the volume phase function anisotropy (g).
        */
        void setVolumeAnisotropy(float volumeAnisotropy);

        /** Get the volume phase function anisotropy (g).
        */
        float getVolumeAnisotropy() const { return (float)mData.volumeAnisotropy; }

        /** Get the normal map type.
        */
        NormalMapType getNormalMapType() const { return mData.getNormalMapType(); }

        /** Set the index of refraction.
        */
        void setIndexOfRefraction(float IoR);

        /** Get the index of refraction.
        */
        float getIndexOfRefraction() const { return (float)mData.IoR; }

        /** Returns the material data struct.
        */
        const BasicMaterialData& getData() const { return mData; }

        /** Comparison operator.
            \return True if all materials properties *except* the name are identical.
        */
        bool operator==(const BasicMaterial& other) const;

    protected:
        BasicMaterial(const std::string& name, MaterialType type);

        bool isAlphaSupported() const;
        void prepareDisplacementMapForRendering();
        void adjustDoubleSidedFlag();
        void updateAlphaMode();
        void updateNormalMapType();
        void updateEmissiveFlag();
        virtual void updateDeltaSpecularFlag() {}

        virtual void renderSpecularUI(Gui::Widgets& widget) {}
        virtual void setEmissiveColor(const float3& color) {}

        BasicMaterialData mData;                    ///< Material parameters.

        Sampler::SharedPtr mpDefaultSampler;
        Sampler::SharedPtr mpDisplacementMinSampler;
        Sampler::SharedPtr mpDisplacementMaxSampler;

        // Additional data for texture usage.
        float2 mAlphaRange = float2(0.f, 1.f);      ///< Conservative range of opacity (alpha) values for the material.
        bool mIsTexturedBaseColorConstant = false;  ///< Flag indicating if the color channels of the base color texture are constant.
        bool mIsTexturedAlphaConstant = false;      ///< Flag indicating if the alpha channel of the base color texture is constant.
        bool mDisplacementMapChanged = false;       ///< Flag indicating of displacement map has changed.

        friend class SceneCache;
    };

    inline std::string to_string(ShadingModel model)
    {
        switch (model)
        {
#define tostr(t_) case ShadingModel::t_: return #t_;
            tostr(MetalRough);
            tostr(SpecGloss);
#undef tostr
        default:
            FALCOR_UNREACHABLE();
            return "";
        }
    }
}
