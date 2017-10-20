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

namespace Falcor
{
    class Texture;
    class ProgramVars;
    class ConstantBuffer;

    /** A surface material object.
        The core part of material is the 'MaterialData' data structure which consists of multiple layers and additional modifiers.
        Modifiers can modify shading attributes *before* performing the actual shading. So far only two modifiers are supported natively:
            - Alpha Map                     - Alpha test. Decides whether to shade at all.
                                                Invalidates the shading hit based on the opacity sampled from the alpha texture.
                                                'MaterialValues#alphaThreshold' is used as alpha threshold for alpha testing (from 0 to 1).
            - Normal Map                    - Normal mapping. Rotates the shading frame before performing shading.
            - Height Map [Unimplemented]    - Moves the shading hit point along the shading normal by the amount fetched from height map texture.
                                                Can be used in parallax and relief mapping in pixel shader, or as a displacement map during tessellation stage.

        After modifiers are applied, the shading is performed with 'MatMaxLayers' number of layers (MatMaxLayers can be adjusted, by default is 3). Each layer can be one of the following types:
            - MatLambert        - The simplest Lambertian diffuse BRDF.
                                    'MaterialLayerValues#albedo' is used as albedo color if no texture is applied.
                                    Other MaterialLayerValues parameters are not used for this Type of layer.

            - MatConductor      - The conductor BRDF. Has metallic glossy look, designed for metals and compute corresponding conductor Fresnel.
                                    Normal Distribution Function (NDF) Type, which is responsible for rough look of the material and takes roughness as an input. Can take on the following values:
                                    'NDFGGX'        - Modern distribution model, characterized by a more smooth highlight.
                                    'NDFBeckmann'   - Beckmann normal distribution.

                                    The layer's texture is used as specular reflection color if available. Otherwise 'MaterialLayerValues#albedo' is used.
                                    'MaterialLayerValues#roughness' is used as material roughness, takes values from 0 (perfect smooth specular) to 1 (highest roughness, diffuse-like).
                                        If a texture is set in the layer, the alpha channel of the texture will be used as roughness.
                                    'MaterialLayerValues#extraParam' is used to store Fresnel parameters of the conductor.
                                        X - real part of complex index of refraction for conductors (actual values are a common knowledge for many metals).
                                        Y - imaginary part of complex index of refraction for conductors.
                                    Other MaterialLayerValues parameters are not used for this type of layer.

            - MatDielectric     - The dielectric BSDF. Has plastic specular/glossy transparent look, designed for transparent dielectrics (like glass, water, coatings etc.) and compute corresponding dielectric Fresnel.
                                    'MaterialLayerValues#albedo' is used as albedo color if no texture is applied.
                                    'MaterialLayerValues#roughness' is used as material roughness, takes values from 0 (perfect smooth specular) to 1 (highest roughness, diffuse-like).
                                        If a texture is set in the layer, the alpha channel of the texture will be used as roughness.
                                    'MaterialLayerValues#extraParam' is used to store index of refraction for dielectric Fresnel parameter (common knowledge, e.g. ~1.3 for unsalted water).
                                    Other MaterialLayerValues parameters are not used for this Type of layer.

        Besides that, each layer defines how it will compose its shading results with the results of previous layers.
        The shading happens in a last-to-first order, so each next layer with smaller index composites its shading results with the previous results according to the value of 'Blending' variable of this layer:
            - BlendFresnel  - Composite the results of this layer according to the Fresnel value of the layer. Available only for layers with Fresnel (conductor/dielectric).
                                Useful for describing a material with e.g. a dielectric coating, where the underlying material layer is blended according to the dielectric Fresnel-based transmission.
            - BlendConstant - Constant-weight blending between this layer and the previous result. The constant value is taken from albedo.a component of the current layer.
                                Useful for mixing multiple BSDFs, e.g. diffuse+conductor, or a multi-lobe conductor.
            - BlendAdd      - Just add the shading results of this layer to the results of previous shading.
    */
    class Material : public std::enable_shared_from_this<Material>
    {
    public:
        using SharedPtr = std::shared_ptr<Material>;
        using SharedConstPtr = std::shared_ptr<const Material>;

        struct Layer
        {
            /** Layer type
            */
            enum class Type
            {
                Lambert = MatLambert,           ///< Lambert layer
                Conductor = MatConductor,       ///< Conductor layer
                Dielectric = MatDielectric,     ///< Dialectric layer
                Emissive = MatEmissive,         ///< Emissive layer
                User = MatUser                  ///< User-defined shading layer
            };

            /** Normal Distribution Functions
            */
            enum class NDF
            {
                Beckmann = NDFBeckmann, ///< Beckmann function
                GGX = NDFGGX,           ///< GGX function
                User = NDFUser          ///< User-defined function
            };

            /** Blend modes for blending between layers
            */
            enum class Blend
            {
                Fresnel = BlendFresnel,     ///< Blend based on fresnel value. Available for conductor and dialectric layers.
                Additive = BlendAdd,        ///< Regular additive blending. Add results for this layer to the previous.
                Constant = BlendConstant    ///< Constant blending using value from the layer texture's alpha channel, or the albedo's alpha value if no texture is bound.
            };

            Type type = Type::Lambert;
            NDF ndf = NDF::GGX;
            Blend blend = Blend::Fresnel;
            glm::vec4 albedo;               ///< Base albedo color. Used when no texture is specified.
            glm::vec4 roughness;            ///< Material layer roughness.
            glm::vec4 extraParam;           ///< Extra parameters interpreted differently depending on layer type.
            Texture::SharedPtr pTexture;    ///< Color texture map for a layer.
            float pmf = 1;
        };

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
        const int32_t getId() const { return mData.values.id; }

        /** Set the material ID
        */
        void setID(int32_t id) { mData.values.id = id; }

        /** Reset all global id counter of model, mesh and material
        */
        static void resetGlobalIdCounter();

        /** Returns the number of active layers in the material
        */
        uint32_t getNumLayers() const;

        /** Returns a material layer descriptor. If the index is out-of-bound, will return a default layer
            \param[in] layerIdx Material layer index
        */
        Layer getLayer(uint32_t layerIdx) const;

        /** Adds a layer to the material. Returns true if succeeded
            \param[in] layer The material layer to add
        */
        bool addLayer(const Layer& layer);

        /** Removes a layer of the material.
            \param[in] layerIdx Material layer index
        */
        void removeLayer(uint32_t layerIdx);

        /** Set a layer's type.
            \param[in] layerId Material layer index
            \param[in] type Layer type
        */
        void setLayerType(uint32_t layerId, Layer::Type type) { mData.desc.layers[layerId].type = (uint32_t)type; mDescDirty = true; }

        /** Set a layer's NDF
            \param[in] layerId Material layer index
            \param[in] ndf NDF type
        */
        void setLayerNdf(uint32_t layerId, Layer::NDF ndf) { mData.desc.layers[layerId].ndf = (uint32_t)ndf; mDescDirty = true; }

        /** Set a layer's blend
            \param[in] layerId Material layer index
            \param[in] blend Layer blend mode
        */
        void setLayerBlend(uint32_t layerId, Layer::Blend blend) { mData.desc.layers[layerId].blending = (uint32_t)blend; mDescDirty = true; }

        /** Set a layer's albedo color
            \param[in] layerId Material layer index
            \param[in] albedo RGBA color
        */
        void setLayerAlbedo(uint32_t layerId, const glm::vec4& albedo) { mData.values.layers[layerId].albedo = albedo; }

        /** Set a layer's roughness
            \param[in] layerId Material layer index
        */
        void setLayerRoughness(uint32_t layerId, const glm::vec4& roughness) { mData.values.layers[layerId].roughness = roughness; }

        /** Set extra parameters on a layer interpreted based on layer type (IoR, etc.)
            \param[in] layerId Material layer index
            \param[in] data Extra parameter data
        */
        void setLayerUserParam(uint32_t layerId, const glm::vec4& data) { mData.values.layers[layerId].extraParam = data; }

        /** Set a layer's texture
            \param[in] layerId Material layer index
            \param[in] pTexture Texture
        */
        void setLayerTexture(uint32_t layerId, const Texture::SharedPtr& pTexture);

        /** Gets number of textures in the material
        */
        uint32_t getTextureCount() const { return mTextureCount; }

        /** Set the normal map
            \param[in] pNormalMap Normal map texture
        */
        void setNormalMap(Texture::SharedPtr& pNormalMap);

        /** Get the normal map texture
        */
        Texture::SharedPtr getNormalMap() const { return mData.textures.normalMap; }

        /** Set the alpha map used for alpha testing.
            \param[in] pAlphaMap Alpha map texture
        */
        void setAlphaMap(const Texture::SharedPtr& pAlphaMap);

        /** Get the alpha map texture
        */
        Texture::SharedPtr getAlphaMap() const { return mData.textures.alphaMap; }

        /** Set the alpha threshold value
        */
        void setAlphaThreshold(float threshold) { mData.values.alphaThreshold = threshold; }

        /** Get the alpha threshold value
        */
        float getAlphaThreshold() const { return mData.values.alphaThreshold; }

        /** Set the ambient occlusion map
            \param[in] pAoMap Ambient occlusion map texture
        */
        void setAmbientOcclusionMap(const Texture::SharedPtr& pAoMap);

        /** Gets the ambient occlusion texture
        */
        Texture::SharedPtr getAmbientOcclusionMap() const { return mData.textures.ambientMap; }

        /** Set the height map value
        */
        void setHeightMap(const Texture::SharedPtr& pHeightMap);

        /** Returns the height map value
        */
        Texture::SharedPtr getHeightMap() const { return mData.textures.heightMap; }

        /** Set the height scale values
            \param[in] mod Two modifier values for the height map. X = scale, Y = offset.
        */
        void setHeightModifiers(const glm::vec2& mod) { mData.values.height = mod; }

        /** Get the height scale values
        */
        glm::vec2 getHeightModifiers() const { return mData.values.height; }

        /** Check if this is a double-sided material. Meshes with double sided materials should be drawn without culling, and for backfacing polygons, the normal has to be inverted.
        */
        bool isDoubleSided() const { return mDoubleSided; }

        /** Set the material as double-sided. Meshes with double sided materials should be drawn without culling, and for backfacing polygons, the normal has to be inverted.
        */
        void setDoubleSided(bool doubleSided) { mDoubleSided = doubleSided; mDescDirty = true; }

        /** Set the material parameters into a constant buffer. To use this you need to include/import 'ShaderCommon' inside your shader.
            \param[in] pVars The graphics vars of the shader to set material into.
            \param[in] pCB The constant buffer to set the parameters into.
            \param[in] varName The name of the MaterialData member in the buffer.
        */
        void setIntoProgramVars(ProgramVars* pVars, ConstantBuffer* pCB, const char varName[]) const;

        /** Set the sampler used when rendering this material.
        */
        void setSampler(const Sampler::SharedPtr& pSampler) { mData.samplerState = pSampler; }

        /** Get this material's sampler.
        */
        Sampler::SharedPtr getSampler() const { return mData.samplerState; }

        /** Comparison operator. Compares Materials by their data values.
        */
        bool operator==(const Material& other) const;

        /** The a string for a MaterialDesc string which can be patched into the shader. It can be used to statically compile the material into a program, resulting in better generated code
        */
        const std::string& getMaterialDescStr() const { finalize(); return mDescString; }

        /** Get an identifier for material desc. This can be used for fast comparison of desc objects. 2 materials will have the same identifier if and only if their desc is exactly the same.
        */
        uint64_t getDescIdentifier() const;

    private:
        void finalize() const;
        void normalize() const;
        void updateTextureCount() const;

        static const uint32_t kTexCount = MatMaxLayers + 4;
        static_assert(sizeof(MaterialTextures) == (sizeof(Texture::SharedPtr) * kTexCount), "Wrong number of textures in Material::mTextures");

        Material(const std::string& name);
        mutable MaterialData mData; ///< Material data shared between host and device
        bool mDoubleSided = false;  ///< Used for culling
        std::string mName;
        mutable uint32_t mTextureCount = 0;

        // The next functions and fields are used for material compilation into shaders.
        // We only compile based on the material descriptor, so as an optimization we minimize the number of shader permutations based on the desc
        struct DescId
        {
            MaterialDesc desc;
            uint64_t id;
            uint32_t refCount;
        };
        mutable bool mDescDirty = true;
        mutable std::string mDescString;
        mutable size_t mDescIdentifier;
        void updateDescIdentifier() const;
        void removeDescIdentifier() const;
        void updateDescString() const;
        static uint32_t sMaterialCounter;
        static std::vector<DescId> sDescIdentifier; // vector is slower then map, but map requires 'less' operator. This vector is only being used when the material is dirty, which shouldn't happen often
    };
}
