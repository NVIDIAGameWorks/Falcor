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
import Scene.Material.TextureHandle;
import Scene.Material.TextureSampler;
import Scene.Material.AlphaTest;
import Rendering.Volumes.PhaseFunction;
import Utils.SlangUtils;
__exported import Scene.SceneTypes;
__exported import Scene.Displacement.DisplacementData;
__exported import Scene.Material.BasicMaterialData;
__exported import Scene.Material.VolumeProperties;

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

    /// Texture resources for all materials. TODO: Make this an unbounded array (see #1321).
    Texture2D<float4> materialTextures[ArrayMax<1, MATERIAL_SYSTEM_TEXTURE_DESC_COUNT>.value];

    /// Buffer resources for all materials. TODO: Make this an unbounded array (see #1321).
    ByteAddressBuffer materialBuffers[ArrayMax<1, MATERIAL_SYSTEM_BUFFER_DESC_COUNT>.value];

    Texture3D<float4> materialTextures3D[ArrayMax<1, MATERIAL_SYSTEM_TEXTURE_3D_DESC_COUNT>.value];


    /**
     * When UDIMs are used, this array contains indirection from <UDIM>-1001 to materialTextures.
     *
     * TextureHandle in an UDIM mode points to this array rather than to the materialTextures,
     * and the index is the base offset. The specific texture index is found as handle's
     * texID + int(u) + 10 * int(v). If the index is -1, then the texture does not exist
     * and the texture handle is uniform instead.
     */
    StructuredBuffer<int> udimIndirection;

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
        Note: the use of [ForceInline] causes Slang to inline this function in its output source.
        Currently there is a bug in glslang/spv-out that generates invalid SPIRV code for functions that returns a SamplerState.
        We workaround this issue by always inlining this function in our generated glsl code.
    */
#ifdef FALCOR_VULKAN
    [ForceInline]
#endif
    SamplerState getTextureSampler(const uint samplerID)
    {
        return materialSamplers[samplerID];
    }

    /** Get the default texture sampler for a material.
        Note: the use of [ForceInline] causes Slang to inline this function in its output source.
        Currently there is a bug in glslang/spv-out that generates invalid SPIRV code for functions that returns a SamplerState.
        We workaround this issue by always inlining this function in our generated glsl code.
    */
#ifdef FALCOR_VULKAN
    [ForceInline]
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

    Texture3D<float4> getTexture3D(const uint textureID)
    {
        return materialTextures3D[textureID];
    }

    /** Check if a material has displacement.
        \param[in] materialID Material ID
        \return Returns true if material has displacement.
    */
    bool isDisplaced(const uint materialID)
    {
        // This function only handle basic materials for now. Other material types return false.
        // TODO: Generalize to support arbitrary materials (#1314).
        if (isBasicMaterial(materialID))
        {
            BasicMaterialData md = getBasicMaterialData(materialID);
            return getResolvedTextureHandle(md.texDisplacementMap).getMode() == TextureHandle::Mode::Texture;
        }
        return false;
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
            return getResolvedTextureHandle(md.texEmissive).getMode() == TextureHandle::Mode::Texture;
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
        return getMaterialHeader(materialID).getIoR();
    }

    /** Resolves TextureHandle from possibly-UDIM to standard TextureHandle
        \param[in] handle Texture handle with possibly UDIM flag
        \param[in, out] uv Texture coordinates, used to determine UDIM, and modified into 0-1 range if UDIMs are used
        \return Texture handle with UDIM flag resolved away
     */
    TextureHandle getResolvedTextureHandle(const TextureHandle handle, inout float2 uv)
    {
#if MATERIAL_SYSTEM_UDIM_INDIRECTION_ENABLED == 0 // we have no UDIMs
        return handle;
#else
        if (!handle.getUdimEnabled())
            return handle;
        uint udimBase = handle.getTextureID();
        uint udimID = udimBase + uint(uv[0]) + 10 * uint(uv[1]);
        uv = frac(uv);
        int texID = -1;
        uint numStructs, stride;
        udimIndirection.GetDimensions(numStructs, stride);
        if (udimID < numStructs)
            texID = udimIndirection[udimID];

        TextureHandle result = handle;
        if (texID == -1)
        {
            result.setMode(TextureHandle::Mode::Uniform);
            return result;
        }
        else
        {
            result.setTextureID(texID);
            result.setUdimEnabled(false);
        }
        return result;
#endif
    }

    /** Resolves TextureHandle for the first UDIM, to get generic texture info
     */
    TextureHandle getResolvedTextureHandle(const TextureHandle handle)
    {
        float2 uv00 = { 0.f, 0.f };
        return getResolvedTextureHandle(handle, uv00);
    }

    /** Get information about a texture.
        \param[in] handle Texture handle.
        \return Texture info or zero initialized struct if no texture.
    */
    TextureInfo getTextureInfo(TextureHandle handle)
    {
        handle = getResolvedTextureHandle(handle);
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
    float4 sampleTexture<L:ITextureSampler>(TextureHandle handle, SamplerState s, float2 uv, const float4 uniformValue, L lod)
    {
        handle = getResolvedTextureHandle(handle, uv);
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

            TextureHandle displacementHandle = getResolvedTextureHandle(md.texDisplacementMap);
            if (displacementHandle.getMode() == TextureHandle::Mode::Texture)
            {
                textureID = displacementHandle.getTextureID();
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

    /** Perform alpha test at a hit point and return the result.
        \param[in] v Interpolated vertex data.
        \param[in] materialID Material ID at the hit.
        \param[in] lod Method for computing texture level-of-detail, must implement the `ITextureSampler` interface.
        \return True if hit should be ignored/discarded.
    */
    bool alphaTest<L : ITextureSampler>(const VertexData v, const uint materialID, L lod)
    {
        // Load the material header first and early out if alpha test not supported.
        const MaterialHeader header = materialData[materialID].header;

        if (header.getAlphaMode() != AlphaMode::Mask) return false;

        SamplerState s = getTextureSampler(header.getDefaultTextureSamplerID());
        float alpha = sampleTexture(header.getAlphaTextureHandle(), s, v.texC, float4(1.f), lod).a;
        float alphaThreshold = header.getAlphaThreshold();

        return evalAlphaTest(alpha, alphaThreshold, v.posW);
    }

    /** Perform alpha test at a hit point and return the result.
        This version samples alpha at a fixed level of detail.
        \param[in] v Interpolated vertex data.
        \param[in] materialID Material ID at the hit.
        \param[in] lod Fixed texture level of detail.
        \return True if hit should be ignored/discarded.
    */
    bool alphaTest(const VertexData v, const uint materialID, const float lod)
    {
        ExplicitLodTextureSampler explicitLOD = { lod };
        return alphaTest(v, materialID, explicitLOD);
    }
};

#ifdef MATERIAL_SYSTEM_PARAMETER_BLOCK
// Declare parameter block used for shader reflection.
ParameterBlock<MaterialSystem> gMaterialsBlock;
void main() {};
#endif
