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
import LoadShadingData;
import Utils.Color.ColorHelpers;
import Rendering.RTXDI.RTXDI;

/** Prepares RTXDI surface data.
*/
struct PrepareSurfaceData
{
    Texture2D<PackedHitInfo> vbuffer;
    Texture2D<float2> texGrads;

    uint2 frameDim;

    void execute(const uint2 pixel)
    {
        if (any(pixel >= frameDim)) return;

        ShadingData sd;
        let lod = ExplicitLodTextureSampler(0.f); // TODO: Implement texture level-of-detail.
        bool isValidSurface = loadShadingData(pixel, frameDim, gScene.camera, vbuffer, lod, sd);

        if (isValidSurface)
        {
            // Create BSDF instance and query its properties.
            let bsdf = gScene.materials.getBSDF(sd, lod);
            let bsdfProperties = bsdf.getProperties(sd);

            // RTXDI uses a simple material model with only diffuse and specular reflection lobes.
            // We query the BSDF for the diffuse albedo and specular reflectance, and use their luminances as weights.
            // Note: Final shading uses the full material model, the simplified model is only used for resampling purposes.
            gRTXDI.setSurfaceData(pixel, sd.computeNewRayOrigin(), sd.N, bsdfProperties.diffuseReflectionAlbedo, bsdfProperties.specularReflectance, bsdfProperties.roughness);
        }
        else
        {
            gRTXDI.setInvalidSurfaceData(pixel);
        }
    }
};

ParameterBlock<PrepareSurfaceData> gPrepareSurfaceData;

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gPrepareSurfaceData.execute(dispatchThreadId.xy);
}
