/***************************************************************************
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
#include "Data/HostDeviceData.h"

BEGIN_NAMESPACE_FALCOR

static const uint kInvalidIndex = 0xffffffff;

/** Describes a mesh area light.

    Note that due to mesh merging etc. it's not guaranteed that a mesh light
    represents a single localized light source in the scene.
    We should not make any assumptions of the extents or geometry of a mesh light.
*/
struct MeshLightData
{
    uint        meshInstanceID      DEFAULTS(kInvalidIndex);    ///< Mesh instance ID in the scene (= getGlobalHitID()).
    uint        triangleOffset      DEFAULTS(kInvalidIndex);    ///< Offset into LightCollection's global list of emissive triangles.
    uint        triangleCount       DEFAULTS(0);                ///< Number of triangles in mesh light.
    uint        flags               DEFAULTS(0);                ///< Flags for the material system.

    float3      emissiveColor       DEFAULTS(float3(0.f));      ///< Material emissive color. This will be overridden by the emissive texture if it exists.
    float       emissiveFactor      DEFAULTS(1.0f);             ///< Material emissive multiplication factor. The emitted radiance is emissiveColor * emissiveFactor.

#ifdef HOST_CODE
    MeshLightData() { init(); }
#endif

    /** Initialize struct to all default values.
    */
    SETTER_DECL void init()
    {
        meshInstanceID = kInvalidIndex;
        triangleOffset = kInvalidIndex;
        triangleCount = 0;
        flags = 0;
        emissiveColor = {};
        emissiveFactor = 1.f;
    }

    /** Returns true if light uses an emissive texture.
    */
    bool isTextured() CONST_FUNCTION
    {
        return EXTRACT_EMISSIVE_TYPE(flags) == ChannelTypeTexture;
    }
};

END_NAMESPACE_FALCOR
