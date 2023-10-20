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
#include "GBuffer.h"

namespace Falcor
{
// Update 'mtlData' channel format if size changes.
// Note: Currently, we only store the first 8 bytes of the material header.
static_assert(sizeof(MaterialHeader) == 16);
} // namespace Falcor

// List of primary GBuffer channels. These correspond to the render targets
// used in the GBufferRaster pixel shader. Note that channel order should
// correspond to SV_TARGET index order.
const ChannelList GBuffer::kGBufferChannels = {
    // clang-format off
    { "posW",           "gPosW",            "Position in world space",                           true /* optional */, ResourceFormat::RGBA32Float },
    { "normW",          "gNormW",           "Shading normal in world space",                     true /* optional */, ResourceFormat::RGBA32Float },
    { "tangentW",       "gTangentW",        "Shading tangent in world space (xyz) and sign (w)", true /* optional */, ResourceFormat::RGBA32Float },
    { "faceNormalW",    "gFaceNormalW",     "Face normal in world space",                        true /* optional */, ResourceFormat::RGBA32Float },
    { "texC",           "gTexC",            "Texture coordinate",                                true /* optional */, ResourceFormat::RG32Float   },
    { "texGrads",       "gTexGrads",        "Texture gradients (ddx, ddy)",                      true /* optional */, ResourceFormat::RGBA16Float },
    { "mvec",           "gMotionVector",    "Motion vector",                                     true /* optional */, ResourceFormat::RG32Float   },
    { "mtlData",        "gMaterialData",    "Material data (ID, header.x, header.y, lobes)",     true /* optional */, ResourceFormat::RGBA32Uint  },
    // clang-format on
};

GBuffer::GBuffer(ref<Device> pDevice) : GBufferBase(pDevice)
{
    // The list of primary GBuffer channels should contain 8 entries, corresponding to the 8 render targets.
    FALCOR_ASSERT(kGBufferChannels.size() == 8);
}
