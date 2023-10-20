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
#pragma once
#include "MaterialData.slang"
#include "TextureHandle.slang"
#include "MaterialTypeRegistry.h"
#include "MaterialParamLayout.h"
#include "SerializedMaterialParams.h"
#include "Core/Macros.h"
#include "Core/Error.h"
#include "Core/Object.h"
#include "Core/API/Formats.h"
#include "Core/API/Texture.h"
#include "Core/API/Sampler.h"
#include "Utils/Image/TextureAnalyzer.h"
#include "Utils/UI/Gui.h"
#include "Scene/Transform.h"
#include "MaterialTypeRegistry.h"
#include <array>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>

namespace Falcor
{
    class MaterialSystem;
    class BasicMaterial;

    /** Abstract base class for materials.
    */
    class FALCOR_API Material : public Object
    {
        FALCOR_OBJECT(Material)
    public:
        /** Flags indicating if and what was updated in the material.
        */
        enum class UpdateFlags : uint32_t
        {
            None                = 0x0,  ///< Nothing updated.
            CodeChanged         = 0x1,  ///< Material shader code changed.
            DataChanged         = 0x2,  ///< Material data (parameters) changed.
            ResourcesChanged    = 0x4,  ///< Material resources (textures, buffers, samplers) changed.
            DisplacementChanged = 0x8,  ///< Displacement mapping parameters changed (only for materials that support displacement).
            EmissiveChanged     = 0x10, ///< Material emissive properties changed.
        };

        /** Texture slots available for use.
            A material does not need to expose/bind all slots.
        */
        enum class TextureSlot
        {
            BaseColor,
            Specular,
            Emissive,
            Normal,
            Transmission,
            Displacement,
            Index, // For MERLMix material

            Count // Must be last
        };

        struct TextureSlotInfo
        {
            std::string         name;                               ///< Name of texture slot.
            TextureChannelFlags mask = TextureChannelFlags::None;   ///< Mask of enabled texture channels.
            bool                srgb = false;                       ///< True if texture should be loaded in sRGB space.

            bool isEnabled() const { return mask != TextureChannelFlags::None; }
            bool hasChannel(TextureChannelFlags channel) const { return is_set(mask, channel); }
            bool operator==(const TextureSlotInfo& rhs) const { return name == rhs.name && mask == rhs.mask && srgb == rhs.srgb; }
            bool operator!=(const TextureSlotInfo& rhs) const { return !((*this) == rhs); }
        };

        struct TextureSlotData
        {
            ref<Texture>  pTexture;                           ///< Texture bound to texture slot.

            bool hasData() const { return pTexture != nullptr; }
            bool operator==(const TextureSlotData& rhs) const { return pTexture == rhs.pTexture; }
            bool operator!=(const TextureSlotData& rhs) const { return !((*this) == rhs); }
        };

        struct TextureOptimizationStats
        {
            std::array<size_t, (size_t)TextureSlot::Count> texturesRemoved = {};
            size_t disabledAlpha = 0;
            size_t constantBaseColor = 0;
            size_t constantNormalMaps = 0;
        };

        virtual ~Material() = default;

        /** Render the UI.
            \return True if the material was modified.
        */
        virtual bool renderUI(Gui::Widgets& widget);

        /** Update material. This prepares the material for rendering.
            \param[in] pOwner The material system that this material is used with.
            \return Updates since last call to update().
        */
        virtual Material::UpdateFlags update(MaterialSystem* pOwner) = 0;

        /** Set the material name.
        */
        virtual void setName(const std::string& name) { mName = name; }

        /** Get the material name.
        */
        virtual const std::string& getName() const { return mName; }

        /** Get the material type.
        */
        virtual MaterialType getType() const { return mHeader.getMaterialType(); }

        /** Returns true if the material is opaque.
        */
        virtual bool isOpaque() const { return getAlphaMode() == AlphaMode::Opaque; }

        /** Returns true if the material has a displacement map.
        */
        virtual bool isDisplaced() const { return false; }

        /** Returns true if the material is emissive.
        */
        virtual bool isEmissive() const { return mHeader.isEmissive(); }

        /** Returns true if the material is dynamic.
            Dynamic materials are updated every frame, otherwise `update()` is called reactively upon changes.
        */
        virtual bool isDynamic() const { return false; }

        /** Compares material to another material.
            \param[in] pOther Other material.
            \return true if all materials properties *except* the name are identical.
        */
        virtual bool isEqual(const ref<Material>& pOther) const = 0;

        /** Set the double-sided flag. This flag doesn't affect the cull state, just the shading.
        */
        virtual void setDoubleSided(bool doubleSided);

        /** Returns true if the material is double-sided.
        */
        virtual bool isDoubleSided() const { return mHeader.isDoubleSided(); }

        /** Set the thin surface flag.
        */
        virtual void setThinSurface(bool thinSurface);

        /** Returns true if the material is a thin surface.
        */
        virtual bool isThinSurface() const { return mHeader.isThinSurface(); }

        /** Set the alpha mode.
        */
        virtual void setAlphaMode(AlphaMode alphaMode);

        /** Get the alpha mode.
        */
        virtual AlphaMode getAlphaMode() const { return mHeader.getAlphaMode(); }

        /** Set the alpha threshold. The threshold is only used when alpha mode != AlphaMode::Opaque.
        */
        virtual void setAlphaThreshold(float alphaThreshold);

        /** Get the alpha threshold.
        */
        virtual float getAlphaThreshold() const { return (float)mHeader.getAlphaThreshold(); }

        /** Get the alpha mask texture handle.
        */
        virtual TextureHandle getAlphaTextureHandle() const { return mHeader.getAlphaTextureHandle(); }

        /** Set the nested priority used for nested dielectrics.
        */
        virtual void setNestedPriority(uint32_t priority);

        /** Get the nested priority used for nested dielectrics.
            \return Nested priority, with 0 reserved for the highest possible priority.
        */
        virtual uint32_t getNestedPriority() const { return mHeader.getNestedPriority(); }

        /** Set the index of refraction.
        */
        virtual void setIndexOfRefraction(float IoR);

        /** Get the index of refraction.
        */
        virtual float getIndexOfRefraction() const { return (float)mHeader.getIoR(); }

        /** Get information about a texture slot.
            \param[in] slot The texture slot.
            \return Info about the slot. If the slot doesn't exist isEnabled() returns false.
        */
        virtual const TextureSlotInfo& getTextureSlotInfo(const TextureSlot slot) const;

        /** Check if material has a given texture slot.
            \param[in] slot The texture slot.
            \return True if the texture slot exists. Use getTexture() to check if a texture is bound.
        */
        virtual bool hasTextureSlot(const TextureSlot slot) const { return getTextureSlotInfo(slot).isEnabled(); }

        /** Set one of the available texture slots.
            The call is ignored with a warning if the slot doesn't exist.
            \param[in] slot The texture slot.
            \param[in] pTexture The texture.
            \return True if the texture slot was changed, false otherwise.
        */
        virtual bool setTexture(const TextureSlot slot, const ref<Texture>& pTexture);

        /** Load one of the available texture slots.
            The call is ignored with a warning if the slot doesn't exist.
            \param[in] The texture slot.
            \param[in] path Path to load texture from.
            \param[in] useSrgb Load texture as sRGB format.
            \return True if the texture was successfully loaded, false otherwise.
        */
        virtual bool loadTexture(const TextureSlot slot, const std::filesystem::path& path, bool useSrgb = true);

        /** Clear one of the available texture slots.
            The call is ignored with a warning if the slot doesn't exist.
            \param[in] The texture slot.
        */
        virtual void clearTexture(const TextureSlot slot) { setTexture(slot, nullptr); }

        /** Get one of the available texture slots.
            \param[in] The texture slot.
            \return Texture object if bound, or nullptr if unbound or slot doesn't exist.
        */
        virtual ref<Texture> getTexture(const TextureSlot slot) const;

        /** Optimize texture usage for the given texture slot.
            This function may replace constant textures by uniform material parameters etc.
            \param[in] slot The texture slot.
            \param[in] texInfo Information about the texture bound to this slot.
            \param[out] stats Optimization stats passed back to the caller.
        */
        virtual void optimizeTexture(const TextureSlot slot, const TextureAnalyzer::Result& texInfo, TextureOptimizationStats& stats) {}

        /** Return the maximum dimensions of the bound textures.
        */
        virtual uint2 getMaxTextureDimensions() const;

        /** Set the default texture sampler for the material.
        */
        virtual void setDefaultTextureSampler(const ref<Sampler>& pSampler) {}

        /** Get the default texture sampler for the material.
        */
        virtual ref<Sampler> getDefaultTextureSampler() const { return nullptr; }

        /** Set the material texture transform.
        */
        virtual void setTextureTransform(const Transform& texTransform);

        /** Get a reference to the material texture transform.
        */
        virtual Transform& getTextureTransform() { return mTextureTransform; }

        /** Get the material texture transform.
        */
        virtual const Transform& getTextureTransform() const { return mTextureTransform; }

        /** Get a reference to the material header data.
        */
        virtual const MaterialHeader& getHeader() const { return mHeader; }

        /** Get the material data blob for uploading to the GPU.
        */
        virtual MaterialDataBlob getDataBlob() const = 0;

        /** Get shader modules for the material.
            The shader modules must be added to any program using the material.
            \return List of shader modules.
        */
        virtual ProgramDesc::ShaderModuleList getShaderModules() const = 0;

        /** Get type conformances for the material.
            The type conformances must be set on any program using the material.
        */
        virtual TypeConformanceList getTypeConformances() const = 0;

        /** Get shader defines for the material.
            The defines must be set on any program using the material.
        */
        virtual DefineList getDefines() const { return {}; }

        /** Get the number of buffers used by this material.
        */
        virtual size_t getMaxBufferCount() const { return 0; }

        /** Returns the maximum number of textures this material will use.
            By default we use the number of texture slots. The reason for this is that,
            for now, once the MaterialSystem has been set up with some number of texture slots,
            it is not possible to allocate more. This limitation will be lifted in the future.
        */
        virtual size_t getMaxTextureCount() const { return (size_t)Material::TextureSlot::Count; }

        /** Get the number of 3D textures used by this material.
        */
        virtual size_t getMaxTexture3DCount() const { return 0; }

        // Temporary convenience function to downcast Material to BasicMaterial.
        // This is because a large portion of the interface hasn't been ported to the Material base class yet.
        // TODO: Remove this helper later
        ref<BasicMaterial> toBasicMaterial();

        /** Size of the material instance the material produces.
            Used to set `anyValueSize` on `IMaterialInstance` above the default (128B), for exceptionally large materials.
            Large material instances can have a singificant performance impact.
        */
        virtual size_t getMaterialInstanceByteSize() const { return 128; }

        virtual const MaterialParamLayout& getParamLayout() const { FALCOR_THROW("Material does not have a parameter layout."); }
        virtual SerializedMaterialParams serializeParams() const { FALCOR_THROW("Material does not support serializing parameters."); }
        virtual void deserializeParams(const SerializedMaterialParams& params) { FALCOR_THROW("Material does not support deserializing parameters."); }

    protected:
        Material(ref<Device> pDevice, const std::string& name, MaterialType type);

        using UpdateCallback = std::function<void(Material::UpdateFlags)>;
        void registerUpdateCallback(const UpdateCallback& updateCallback) { mUpdateCallback = updateCallback; }
        void markUpdates(UpdateFlags updates);
        bool hasTextureSlotData(const TextureSlot slot) const;
        void updateTextureHandle(MaterialSystem* pOwner, const ref<Texture>& pTexture, TextureHandle& handle);
        void updateTextureHandle(MaterialSystem* pOwner, const TextureSlot slot, TextureHandle& handle);
        void updateDefaultTextureSamplerID(MaterialSystem* pOwner, const ref<Sampler>& pSampler);
        bool isBaseEqual(const Material& other) const;

        static NormalMapType detectNormalMapType(const ref<Texture>& pNormalMap);

        template<typename T>
        MaterialDataBlob prepareDataBlob(const T& data) const
        {
            MaterialDataBlob blob = {};
            blob.header = mHeader;
            static_assert(sizeof(mHeader) + sizeof(data) <= sizeof(blob));
            static_assert(offsetof(MaterialDataBlob, payload) == sizeof(MaterialHeader));
            std::memcpy(&blob.payload, &data, sizeof(data));
            return blob;
        }

        ref<Device> mpDevice;

        std::string mName;                          ///< Name of the material.
        MaterialHeader mHeader;                     ///< Material header data available in all material types.
        Transform mTextureTransform;                ///< Texture transform. This is currently applied at load time by pre-transforming the texture coordinates.

        std::array<TextureSlotInfo, (size_t)TextureSlot::Count> mTextureSlotInfo;   ///< Information about texture slots.
        std::array<TextureSlotData, (size_t)TextureSlot::Count> mTextureSlotData;   ///< Data bound to texture slots. Only enabled slots can have any data.

        mutable UpdateFlags mUpdates = UpdateFlags::None;
        UpdateCallback mUpdateCallback;             ///< Callback to track updates with the material system this material is used with.

        friend class MaterialSystem;
        friend class SceneCache;
    };

    inline std::string to_string(Material::TextureSlot slot)
    {
        switch (slot)
        {
#define tostr(a) case Material::TextureSlot::a: return #a;
            tostr(BaseColor);
            tostr(Specular);
            tostr(Emissive);
            tostr(Normal);
            tostr(Transmission);
            tostr(Displacement);
            tostr(Index);
#undef tostr
        default:
            FALCOR_THROW("Invalid texture slot");
        }
    }

    FALCOR_ENUM_CLASS_OPERATORS(Material::UpdateFlags);
}
