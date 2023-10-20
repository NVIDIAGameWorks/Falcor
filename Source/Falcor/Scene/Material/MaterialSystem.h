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
#include "Material.h"
#include "Core/Macros.h"
#include "Core/API/fwd.h"
#include "Core/API/ParameterBlock.h"
#include "Core/API/Buffer.h"
#include "Core/API/Sampler.h"
#include "Core/Program/DefineList.h"
#include "Core/Program/Program.h"
#include "Utils/Image/TextureManager.h"
#include "Utils/UI/Gui.h"
#include <memory>
#include <vector>
#include <set>

namespace Falcor
{
    /** This class represents a material system.

        It holds a collection of materials and their associated resources.
        The host side has interfaces for managing the materials, rendering
        the UI, and uploading/updating GPU buffers, etc.

        The matching shader side Slang module holds all GPU resources and
        has interfaces for preparing the material at a shading point.

        The update() function must be called before using the material system.
        It ensures all GPU data is uploaded and ready for use.
    */
    class FALCOR_API MaterialSystem
    {
    public:
        struct MaterialStats
        {
            uint64_t materialTypeCount = 0;             ///< Number of material types.
            uint64_t materialCount = 0;                 ///< Number of materials.
            uint64_t materialOpaqueCount = 0;           ///< Number of materials that are opaque.
            uint64_t materialMemoryInBytes = 0;         ///< Total memory in bytes used by the material data.
            uint64_t textureCount = 0;                  ///< Number of unique textures. A texture can be referenced by multiple materials.
            uint64_t textureCompressedCount = 0;        ///< Number of unique compressed textures.
            uint64_t textureTexelCount = 0;             ///< Total number of texels in all textures.
            uint64_t textureTexelChannelCount = 0;      ///< Total number of texel channels in all textures.
            uint64_t textureMemoryInBytes = 0;          ///< Total memory in bytes used by the textures.
        };

        /** Constructor. Throws an exception if creation failed.
        */
        MaterialSystem(ref<Device> pDevice);

        /** Render the UI.
        */
        void renderUI(Gui::Widgets& widget);

        /** Update material system. This prepares all resources for rendering.
        */
        Material::UpdateFlags update(bool forceUpdate);

        /** Get shader defines.
            These need to be set before binding the material system parameter block.
            Adds defines to an existing list, rather than creating a new list.
            \param[in,out] defines List of type defines.
        */
        void getDefines(DefineList& defines) const;
        DefineList getDefines() const
        {
            DefineList result;
            getDefines(result);
            return result;
        }

        /** Get type conformances for all material types used.
            These need to be set on a program before using the material system in shaders
            that need to create a material of *any* type, such as compute or raygen shaders.
            \param[in,out] conformances List of type conformances.
        */
        void getTypeConformances(TypeConformanceList& conformances) const;
        TypeConformanceList getTypeConformances() const
        {
            TypeConformanceList typeConformances;
            getTypeConformances(typeConformances);
            return typeConformances;
        }

        /** Get type conformances for a given material type in use.
            \param[in] type Material type.
            \return List of type conformances.
        */
        TypeConformanceList getTypeConformances(const MaterialType type) const;

        /** Get shader modules for all materials in use.
            The shader modules must be added to any program using the material system.
            \param[in,out] shaderModuleList List of shader modules.
        */
        void getShaderModules(ProgramDesc::ShaderModuleList& shaderModuleList) const;

        /** Get shader modules for all materials in use.
            The shader modules must be added to any program using the material system.
            \return List of shader modules.
        */
        ProgramDesc::ShaderModuleList getShaderModules() const;

        /** Bind the material system to a shader var.
        */
        void bindShaderData(const ShaderVar& var) const;

        /** Set a default texture sampler to use for all materials.
        */
        void setDefaultTextureSampler(const ref<Sampler>& pSampler);

        /** Add a texture sampler.
            If an identical sampler already exists, the sampler is not added and the existing ID returned.
            \param[in] pSampler The sampler.
            \return The ID of the sampler.
        */
        uint32_t addTextureSampler(const ref<Sampler>& pSampler);

        /** Get the total number of texture samplers.
        */
        uint32_t getTextureSamplerCount() const { return (uint32_t)mTextureSamplers.size(); }

        /** Get a texture sampler by ID.
        */
        const ref<Sampler>& getTextureSampler(const uint32_t samplerID) const { return mTextureSamplers[samplerID]; }

        /** Add a buffer resource to be managed.
            \param[in] pBuffer The buffer.
            \return The ID of the buffer.
        */
        uint32_t addBuffer(const ref<Buffer>& pBuffer);

        /** Replace a previously managed buffer by a new buffer.
            \param[in] id The ID of the buffer.
            \param[in] pBuffer The buffer.
        */
        void replaceBuffer(uint32_t id, const ref<Buffer>& pBuffer);

        /** Get the total number of managed buffers.
        */
        uint32_t getBufferCount() const { return (uint32_t)mBuffers.size(); }

        /** Add a 3D texture resource to be managed.
            \param[in] pTexture The texture.
            \return The ID of the texture.
        */
        uint32_t addTexture3D(const ref<Texture>& pTexture);

        /** Get the total number of 3D textures.
        */
        uint32_t getTexture3DCount() const { return (uint32_t)mTextures3D.size(); }

        /** Add a material.
            If an identical material already exists, the material is not added and the existing ID returned.
            \param[in] pMaterial The material.
            \return The ID of the material.
        */
        MaterialID addMaterial(const ref<Material>& pMaterial);

        /** Replace a material.
            \param materialID The ID of the material to replace.
            \param pReplacement The material to replace it with.
        */
        void replaceMaterial(const MaterialID materialID, const ref<Material>& pReplacement);
        void replaceMaterial(const ref<Material>& pMaterial, const ref<Material>& pReplacement);

        /** Get a list of all materials.
        */
        const std::vector<ref<Material>>& getMaterials() const { return mMaterials; }

        /** Get the total number of materials.
        */
        uint32_t getMaterialCount() const { return (uint32_t)mMaterials.size(); }

        /** Get the number of materials of the given type.
        */
        uint32_t getMaterialCountByType(const MaterialType type) const;

        /** Get the set of all material types used.
        */
        std::set<MaterialType> getMaterialTypes() const;

        /** Check if material of the given type is used.
        */
        bool hasMaterialType(MaterialType type) const;

        /** Check if a material with the given ID exists.
            \param[in] materialID The material ID.
            \return True if the material exists.
        */
        bool hasMaterial(const MaterialID materialID) const;

        /** Get a material by ID.
            \param[in] materialID The material ID.
            \return The material, or throws if the material doesn't exist.
        */
        const ref<Material>& getMaterial(const MaterialID materialID) const;

        /** Get a material by name.
            \return The material, or nullptr if material doesn't exist.
        */
        ref<Material> getMaterialByName(const std::string& name) const;

        /** Remove all duplicate materials.
            \param[in] idMap Vector that holds for each material the ID of the material that replaces it.
            \return The number of materials removed.
        */
        size_t removeDuplicateMaterials(std::vector<MaterialID>& idMap);

        /** Optimize materials.
            This function analyzes textures and replaces constant textures by uniform material parameters.
        */
        void optimizeMaterials();

        /** Get stats for the material system. This can be a slow operation.
        */
        MaterialStats getStats() const;

        /** Get texture manager. This holds all textures.
        */
        TextureManager& getTextureManager() { return *mpTextureManager; }


    private:
        void updateMetadata();
        void updateUI();
        void createParameterBlock();
        void uploadMaterial(const uint32_t materialID);

        ref<Device> mpDevice;

        std::vector<ref<Material>> mMaterials;                      ///< List of all materials.
        std::vector<Material::UpdateFlags> mMaterialsUpdateFlags;   ///< List of all material update flags, after the update() calls
        std::unique_ptr<TextureManager> mpTextureManager;           ///< Texture manager holding all material textures.
        ProgramDesc::ShaderModuleList mShaderModules;                   ///< Shader modules for all materials in use.
        std::map<MaterialType, TypeConformanceList> mTypeConformances; ///< Type conformances for each material type in use.


        // Metadata
        size_t mTextureDescCount = 0;                               ///< Number of texture descriptors in GPU descriptor array. This variable is for book-keeping until unbounded descriptor arrays are supported (see #1321).
        size_t mBufferDescCount = 0;                                ///< Number of buffer descriptors in GPU descriptor array. This variable is for book-keeping until unbounded descriptor arrays are supported (see #1321).
        size_t mTexture3DDescCount = 0;                             ///< Number of 3D texture descriptors in GPU descriptor array. This variable is for book-keeping until unbounded descriptor arrays are supported (see #1321).
        size_t mReservedTextureDescCount = 0;                       ///< Number of reserved texture descriptors in GPU descriptor array. This variable is for book-keeping until unbounded descriptor arrays are supported (see #1321).
        size_t mReservedBufferDescCount = 0;                        ///< Number of reserved buffer descriptors in GPU descriptor array. This variable is for book-keeping until unbounded descriptor arrays are supported (see #1321).
        size_t mReservedTexture3DDescCount = 0;                     ///< Number of reserved 3D texture descriptors in GPU descriptor array. This variable is for book-keeping until unbounded descriptor arrays are supported (see #1321).
        std::vector<uint32_t> mMaterialCountByType;                 ///< Number of materials of each type, indexed by MaterialType.
        std::set<MaterialType> mMaterialTypes;                      ///< Set of all material types used.
        bool mHasSpecGlossStandardMaterial = false;                 ///< True if standard materials using the SpecGloss shading model exist.
        std::vector<MaterialID> mDynamicMaterialIDs;                ///< Material IDs for all dynamic materials.

        bool mSamplersChanged = false;                              ///< Flag indicating if samplers were added/removed since last update.
        bool mBuffersChanged = false;                               ///< Flag indicating if buffers were added/removed since last update.
        bool mTextures3DChanged = false;                            ///< Flag indicating if 3D textures were added/removed since last update.
        bool mMaterialsChanged = false;                             ///< Flag indicating if materials were added/removed since last update. Per-material updates are tracked by each material's update flags.

        Material::UpdateFlags mMaterialUpdates = Material::UpdateFlags::None; ///< Material updates across all materials since last update.

        // GPU resources
        ref<Fence> mpFence;
        ref<ParameterBlock> mpMaterialsBlock;                       ///< Parameter block for binding all material resources.
        ref<Buffer> mpMaterialDataBuffer;                           ///< GPU buffer holding all material data.
        ref<Sampler> mpDefaultTextureSampler;                       ///< Default texture sampler to use for all materials.
        std::vector<ref<Sampler>> mTextureSamplers;                 ///< Texture sampler states. These are indexed by ID in the materials.
        std::vector<ref<Buffer>> mBuffers;                          ///< Buffers used by the materials. These are indexed by ID in the materials.
        std::vector<ref<Texture>> mTextures3D;                      ///< 3D textures used by the materials. These are indexed by ID in the materials.

        // UI variables
        std::vector<uint32_t> mSortedMaterialIndices;               ///< Indices of materials, sorted alphabetically by case-insensitive name.
        bool mSortMaterialsByName = false;                          ///< If true, display materials sorted by name, rather than by ID.

        friend class Material;
        friend class SceneCache;
    };
};
