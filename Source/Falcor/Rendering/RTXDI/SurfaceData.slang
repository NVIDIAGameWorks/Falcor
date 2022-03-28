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
import Utils.Math.PackedFormats;
import Utils.Math.FormatConversion;

__exported import PackedTypes;

/** Stores data about primary hit points.
    To reduce register pressure, some of the fields are stored in packed format
    and decoded as being accessed through Slang properties.
*/
struct SurfaceData
{
    // The following fields are stored in PackedSurfaceData.

    float3 position;        ///< Position in world-space.
    float viewDepth;        ///< Distance from viewer to surface (negative for invalid surface).
    uint packedNormal;      ///< Packed surface normal in world-space (octahedral mapping).
    uint packedWeights;     ///< Packed diffuse/specular reflectance and roughness.

    // The following fields are *NOT* stored in PackedSurfaceData.

    float3 viewDir;         ///< Direction to the viewer in world-space.

    // True if surface data is valid.
    property bool valid
    {
        get { return viewDepth >= 0.f; }
    }

    // Shading normal.
    property float3 normal
    {
        get { return decodeNormal2x16(packedNormal); }
        set { packedNormal = encodeNormal2x16(newValue); }
    }

    // Diffuse reflectance.
    property float diffuse
    {
        get { return unpackUnorm8(packedWeights); }
        set { packedWeights = (packedWeights & ~0xff) | packUnorm8(newValue); }
    }

    // Specular reflectance.
    property float specular
    {
        get { return unpackUnorm8(packedWeights >> 8); }
        set { packedWeights = (packedWeights & ~0xff00) | (packUnorm8(newValue) << 8); }
    }

    // Specular roughness.
    property float roughness
    {
        get { return unpackUnorm8(packedWeights >> 16); }
        set { packedWeights = (packedWeights & ~0xff0000) | (packUnorm8(newValue) << 16); }
    }

    // Diffuse lobe probability.
    property float diffuseProb
    {
        get { return unpackUnorm8(packedWeights >> 24); }
        set { packedWeights = (packedWeights & ~0xff000000) | (packUnorm8(newValue) << 24); }
    }

    /** Initialize an invalid surface.
    */
    __init()
    {
        this = {};
        viewDepth = -1.f;
    }

    /** Initialize a surface.
        \param[in] packed Packed surface data.
        \param[in] viewDir View direction.
    */
    __init(const PackedSurfaceData packed, const float3 viewDir)
    {
        position = asfloat(packed.position);
        viewDepth = asfloat(packed.viewDepth);
        packedNormal = packed.normal;
        packedWeights = packed.weights;
        this.viewDir = viewDir;
    }

    /** Pack surface data.
    */
    PackedSurfaceData pack()
    {
        PackedSurfaceData packed = {};
        packed.position = asuint(position);
        packed.viewDepth = asuint(viewDepth);
        packed.normal = packedNormal;
        packed.weights = packedWeights;
        return packed;
    }
};
