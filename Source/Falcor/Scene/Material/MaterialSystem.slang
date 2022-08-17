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
import Scene.Material.ShadingUtils;
import Scene.Material.TextureHandle;
import Scene.Material.TextureSampler;
__exported import Scene.SceneTypes;
__exported import Scene.Displacement.DisplacementData;
__exported import Scene.Material.BasicMaterialData;
__exported import Scene.Material.HomogeneousVolumeData;

#if !defined(MATERIAL_SYSTEM_TEXTURE_DESC_COUNT) || !defined(MATERIAL_SYSTEM_SAMPLER_DESC_COUNT) || !defined(MATERIAL_SYSTEM_BUFFER_DESC_COUNT)
// This error occurs when the module is imported without setting the defines returned by getDefines().
// This check is necessary until unbounded descriptor arrays are supported (see #1321).
#error "MaterialSystem defines are not set!"
#endif

/** This struct represents the GPU-side material system.
*/
struct MaterialSystem
{
    uint materialCount;                                                         ///< Total number of materials.
    StructuredBuffer<MaterialDataBlob> materialData;                            ///< Material parameters. The format of the data blob depends on the material type.
    SamplerState materialSamplers[MATERIAL_SYSTEM_SAMPLER_DESC_COUNT];          ///< Sampler states for all materials. TODO: Make this an unbounded array (see #1321).

#if MATERIAL_SYSTEM_TEXTURE_DESC_COUNT > 0
    Texture2D<float4> materialTextures[MATERIAL_SYSTEM_TEXTURE_DESC_COUNT];     ///< Texture resources for all materials. TODO: Make this an unbounded array (see #1321).
#else
    Texture2D<float4> materialTextures[1]; // Zero-length arrays are not supported.
#endif

#if MATERIAL_SYSTEM_BUFFER_DESC_COUNT > 0
    ByteAddressBuffer materialBuffers[MATERIAL_SYSTEM_BUFFER_DESC_COUNT];       ///< Buffer resources for all materials. TODO: Make this an unbounded array (see #1321).
#else
    ByteAddressBuffer materialBuffers[1]; // Zero-length arrays are not supported.
#endif

    /** Get the total number of materials.
    */
    uint getMaterialCount()
    {
        return materialCount;
    }

    /** Get the type of the given material.
    */
    MaterialType getMaterialType(const uint materialID)
    {
        return materialData[materialID].header.getMaterialType();
    }

    /** Get the material header for the given material.
    */
    MaterialHeader getMaterialHeader(const uint materialID)
    {
        return materialData[materialID].header;
    }

    /** Get the material data blob for the given material.
        The format of the data blob depends on the material type.
    */
    MaterialDataBlob getMaterialDataBlob(const uint materialID)
    {
        return materialData[materialID];
    }

    /** Returns true if the given material is a basic material.
        This function is currently used by legacy code that doesn't all material types, but may be removed in the future.
    */
    bool isBasicMaterial(const uint materialID)
    {
        return materialData[materialID].header.isBasicMaterial();
    }

    /** Get the material data for a basic material.
        Note that no type checking is done. The caller is responsible for ensuring that the given material is a basic material.
        This function is currently used by legacy code that doesn't all material types, but may be removed in the future.
    */
    BasicMaterialData getBasicMaterialData(const uint materialID)
    {
        return reinterpret<BasicMaterialData, MaterialPayload>(materialData[materialID].payload);
    }

    /** Get a texture sampler by ID.
        Note: the use of [__unsafeForceInlineEarly] causes Slang to inline this function in its output source.
        Currently there is a bug in glslang/spv-out that generates invalid SPIRV code for functions that returns a SamplerState.
        We workaround this issue by always inlining this function in our generated glsl code.
    */
#ifdef FALCOR_VK
    [__unsafeForceInlineEarly]
#endif
    SamplerState getTextureSampler(const uint samplerID)
    {
        return materialSamplers[samplerID];
    }

    /** Get the default texture sampler for a material.
        Note: the use of [__unsafeForceInlineEarly] causes Slang to inline this function in its output source.
        Currently there is a bug in glslang/spv-out that generates invalid SPIRV code for functions that returns a SamplerState.
        We workaround this issue by always inlining this function in our generated glsl code.
    */
#ifdef FALCOR_VK
    [__unsafeForceInlineEarly]
#endif
    SamplerState getDefaultTextureSampler(const uint materialID)
    {
        uint samplerID = getMaterialHeader(materialID).getDefaultTextureSamplerID();
        return getTextureSampler(samplerID);
    }

    /** Get a buffer by ID.
    */
    ByteAddressBuffer getBuffer(const uint bufferID)
    {
        return materialBuffers[bufferID];
    }

    /** Check if a material has an emissive texture.
        \param[in] materialID Material ID
        \return Returns true if material has emissive texture.
    */
    bool isEmissiveTextured(const uint materialID)
    {
        // This function only handle basic materials for now. Other material types return false.
        // TODO: Generalize to support arbitrary materials (#1314).
        if (isBasicMaterial(materialID))
        {
            BasicMaterialData md = getBasicMaterialData(materialID);
            return md.texEmissive.getMode() == TextureHandle::Mode::Texture;
        }
        return false;
    }

    /** Evaluates the outgoing radiance on a given material.
        This function samples the emissive texture if its textured.
        Note the caller check the facing status themselves to discard back-facing hits.
        \param[in] materialID Material ID
        \param[in] uv Texture coordinates
        \return Evaluated outgoing radiance on the front-facing side.
    */
    float3 evalEmissive(const uint materialID, const float2 uv)
    {
        // This function only handle basic materials for now. Other material types return zero.
        // TODO: Generalize to support arbitrary materials (#1314).
        if (isBasicMaterial(materialID))
        {
            BasicMaterialData md = getBasicMaterialData(materialID);
            SamplerState s = getDefaultTextureSampler(materialID);
            ExplicitLodTextureSampler lod = { 0.f };
            return sampleTexture(md.texEmissive, s, uv, float4(md.emissive, 1.f), lod).rgb * md.emissiveFactor;
        }
        return float3(0.f);
    }

    /** Evaluates the index of refraction of the medium inside the given material.
        \param[in] materialID Material ID
        \return Index of refraction.
    */
    float evalIoR(const uint materialID)
    {
        // TODO: Move IoR field into MaterialHeader so it's available for all material types.
        if (isBasicMaterial(materialID))
        {
            BasicMaterialData md = getBasicMaterialData(materialID);
            return md.IoR;
        }
        return 1.f;
    }

    /** Get information about a texture.
        \param[in] handle Texture handle.
        \return Texture info or zero initialized struct if no texture.
    */
    TextureInfo getTextureInfo(const TextureHandle handle)
    {
        TextureInfo info = {};
        switch (handle.getMode())
        {
        case TextureHandle::Mode::Texture:
            materialTextures[handle.getTextureID()].GetDimensions(0, info.width, info.height, info.mipLevels);
            info.depth = 1;
        default:
        }
        return info;
    }

    /** Sample data from a texture at a fixed level of detail.
        This is a convenience function for common texture sampling. If the handle is not referring to a texture, zero is returned.
        \param[in] handle Texture handle.
        \param[in] s Texture sampler state.
        \param[in] uv Texture coordinate.
        \param[in] lod Fixed texture level-of-detail.
        \return Sampled color.
    */
    float4 sampleTexture(const TextureHandle handle, SamplerState s, const float2 uv, const float lod)
    {
        ExplicitLodTextureSampler explicitLOD = { lod };
        return sampleTexture(handle, s, uv, float4(0.f), explicitLOD);
    }

    /** Sample data from a texture.
        \param[in] handle Texture handle.
        \param[in] s Sampler state.
        \param[in] uv Texture coordinate.
        \param[in] uniformValue Value to return if texture handle is referring to a uniform value.
        \param[in] lod Method for computing texture level-of-detail, must implement the `ITextureSampler` interface.
        \return Sampled color.
    */
    float4 sampleTexture<L:ITextureSampler>(const TextureHandle handle, SamplerState s, const float2 uv, const float4 uniformValue, L lod)
    {
        switch (handle.getMode())
        {
        case TextureHandle::Mode::Uniform:
            return uniformValue;
        case TextureHandle::Mode::Texture:
            return lod.sampleTexture(materialTextures[handle.getTextureID()], s, uv);
        default:
            return float4(0.f);
        }
    }

    /** Load displacement mapping data for the given material.
    */
    void loadDisplacementData(const uint materialID, out DisplacementData displacementData)
    {
        // Note we can't zero initialize struct containing resources.
        displacementData.scale = 0.f;
        displacementData.bias = 0.f;
        displacementData.size = {};

        uint textureID = 0;
        uint samplerID = 0;
        uint displacementMinSamplerID = 0;
        uint displacementMaxSamplerID = 0;

        // Cast the material data to a BasicMaterialData struct.
        // TODO: This is a workaround until all materials support displacement (#1310).
        if (isBasicMaterial(materialID))
        {
            BasicMaterialData md = getBasicMaterialData(materialID);

            displacementData.scale = md.displacementScale;
            displacementData.bias = md.displacementOffset;

            if (md.texDisplacementMap.getMode() == TextureHandle::Mode::Texture)
            {
                textureID = md.texDisplacementMap.getTextureID();
                samplerID = getMaterialHeader(materialID).getDefaultTextureSamplerID();
                displacementMinSamplerID = md.getDisplacementMinSamplerID();
                displacementMaxSamplerID = md.getDisplacementMaxSamplerID();

                materialTextures[textureID].GetDimensions(displacementData.size.x, displacementData.size.y);
            }
        }

        // If the material isn't displaced, return dummy resources at index 0 instead.
        displacementData.texture = materialTextures[textureID];
        displacementData.samplerState = materialSamplers[samplerID];
        displacementData.samplerStateMin = materialSamplers[displacementMinSamplerID];
        displacementData.samplerStateMax = materialSamplers[displacementMaxSamplerID];
    }

    /** Get the homogeneous medium properties of a material.
        \param[in] materialID Material ID.
        \return Returns the homogeneous volume data.
    */
    HomogeneousVolumeData getHomogeneousVolumeData(const uint materialID)
    {
        // This function fetches the volume parameters from a basic material, where these fields are readily available.
        // Other materials are assigned default values of all zeros.
        // TODO: Refactor volume support to handle arbitrary materials (#1313).
        BasicMaterialData md = {};
        if (isBasicMaterial(materialID))
        {
            md = getBasicMaterialData(materialID);
        }
        return { md.volumeAbsorption, md.volumeScattering, md.volumeAnisotropy };
    }
};

#ifdef MATERIAL_SYSTEM_PARAMETER_BLOCK
// Declare parameter block used for shader reflection.
ParameterBlock<MaterialSystem> gMaterialsBlock;
void main() {};
#endif
