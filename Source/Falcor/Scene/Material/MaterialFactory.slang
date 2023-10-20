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
__exported import Rendering.Materials.IMaterial;
import Scene.Material.ShadingUtils;
import Utils.Math.ShadingFrame;

/** Extension to add shading and material factory to MaterialSystem.
    Making this an extension avoids a circular dependency on MaterialSystem.

    All functions that sample textures take an `ITextureSampler` instance as
    input parameter, which determines how texture level-of-detail is computed.
    There are overloaded functions for some common cases like fixed LOD or
    explicit gradients. Refer to the TextureSampler module for more choices.
    For HW computed texture gradients, pass in `ImplicitLodTextureSampler`.
*/
extension MaterialSystem
{
    /** Create an `IMaterial` instance for the given material.
        Note the type conformances must have been set on the host-side for any
        program that calls this function. See MaterialSystem::getTypeConformances().
        \param[in] materialID The material ID.
        \return A material instance representing the material.
    */
    IMaterial getMaterial(const uint materialID)
    {
        MaterialType type = getMaterialType(materialID);
        return createDynamicObject<IMaterial, MaterialDataBlob>(int(type), materialData[materialID]);
    }

    /** Create a `IMaterialInstance` instance for the given shading location.
        This operation first creates a material for the shading location.
        Then the material is queried to create the material instance, in which step
        pattern generation is performed to pre-compute all material parameters.
        This may involve sampling textures, executing procedural functions, etc.
        This step also computes the final shading frame used by the material instance.
        \param[in] sd Shading data.
        \param[in] lod Method for computing texture level-of-detail, must implement the `ITextureSampler` interface.
        \param[in] hints Optional hints to control the behavior of the material instance. Combination of `MaterialInstanceHints` flags.
        \return A material instance.
    */
    IMaterialInstance getMaterialInstance<L:ITextureSampler>(const ShadingData sd, L lod, const uint hints = (uint)MaterialInstanceHints::None)
    {
        let material = getMaterial(sd.materialID);
        return material.setupMaterialInstance(this, sd, lod, hints);
    }

    /** Prepares shading data at a hit point.
        \param[in] v Interpolated vertex data.
        \param[in] materialID Material ID at the hit.
        \param[in] viewDir View direction.
        \param[in] lod Method for computing texture level-of-detail, must implement the `ITextureSampler` interface.
        \return Shading data struct.
    */
    ShadingData prepareShadingData<L:ITextureSampler>(const VertexData v, const uint materialID, const float3 viewDir, L lod)
    {
        ShadingData sd = {};

        sd.posW = v.posW;
        sd.uv = v.texC;
        sd.V = viewDir;

        // Compute shading frame based on the interpolated normal and given tangent (xyz) and sign (w).
        // We keep around the original tangent for later orthogonalization operations if valid, otherwise the invented tangent.
        bool valid;
        sd.frame = ShadingFrame::createSafe(v.normalW, v.tangentW, valid);
        sd.tangentW = valid ? v.tangentW : float4(sd.frame.T, 1.f);

        // Primitive data
        sd.faceN = v.faceNormalW;
        sd.frontFacing = dot(sd.V, sd.faceN) >= 0.f;
        sd.curveRadius = v.curveRadius;

        // Material data
        sd.mtl = materialData[materialID].header;
        sd.materialID = materialID;

        // Assume the default IoR for vacuum on the front-facing side.
        // The renderer may override this for nested dielectrics.
        sd.IoR = 1.f;

        return sd;
    }
}
