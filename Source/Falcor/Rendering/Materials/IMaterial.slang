/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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
__exported import Rendering.Materials.IBSDF;
__exported import Scene.Material.MaterialSystem;
__exported import Scene.Material.TextureSampler;
__exported import Scene.ShadingData;

/** Interface for materials.

    Implementations of this interface hold all data describing the material properties.
    Material instances are created and initialized with data by `MaterialSystem`.

    When querying a material for the BSDF at a given shading point, a BSDF instance implementing
    the IBSDF interface is returned. This has all material properties evaluated at the shading point.
    The process of creating this instance is referred to as "pattern generation". The returned
    instance can be used for evaluation and sampling of the BSDF, and associated functionality.
    
    Note:
     - The [anyValueSize(n)] attribute specifies the maximum size in bytes an implementation type
       to `IMaterial` can be. Slang compiler will check the implementation types and emits an error
       if an implementation exceeds this size.
     - The value set here should correspond to the size of `MaterialDataBlob` (128B).
*/
[anyValueSize(128)]
interface IMaterial
{
    // Requires an implementation to define a type named `BSDF` that
    // conforms to the `IBSDF` interface.
    associatedtype BSDF : IBSDF;

    /** Return a BSDF instance for the given shading point.
        This operation performs pattern generation and may sample textures etc.
        \param[in] ms Material system. This provides the resources for pattern generation.
        \param[in] sd Shading data.
        \param[in] lod Method for computing texture level-of-detail, must implement the `ITextureSampler` interface.
        \return BSDF instance implementing the `IBSDF` interface.
    */
    BSDF setupBSDF(const MaterialSystem ms, const ShadingData sd, const ITextureSampler lod);

    /** Modify the tangent space at the given shading point.
        This function is called by the material system when preparing the `ShadingData` struct.
        It allows materials to implement normal mapping and other techniques that modify the local surface geometry.
        Materials that do not need to modify the default tangent frame should leave the shading data unchanged.
        \param[in] ms Material system. This provides the resources for pattern generation.
        \param[in] tangent Tangent in world space (xyz) and bitangent sign (w). The tangent is *only* valid when w != 0.
        \param[in] lod Method to use for computing texture level of detail, must implement the `ITextureSampler` interface.
        \param[in,out] sd Shading data.
    */
    void modifyTangentSpace(const MaterialSystem ms, const float4 tangentW, const ITextureSampler lod, inout ShadingData sd);

    /** Evaluate opacity at the given shading point.
        \param[in] ms Material system. This provides the resources for opacity evaluation.
        \param[in] v Interpolated attributes at the shading point.
        \param[in] lod Method to use for computing texture level of detail, must implement the `ITextureSampler` interface.
        \return Opacity value in range 0..1.
    */
    float evalOpacity(const MaterialSystem ms, const VertexData v, const ITextureSampler lod);
}
