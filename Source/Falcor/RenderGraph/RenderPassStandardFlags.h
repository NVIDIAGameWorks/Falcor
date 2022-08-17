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
#pragma once
#include "Core/Macros.h"
#include <cstdint>

namespace Falcor
{
    /** Flags to indicate what have changed since last frame.
        One or more flags can be OR'ed together.
    */
    enum class RenderPassRefreshFlags : uint32_t
    {
        None                    = 0x0,
        LightingChanged         = 0x1,      ///< Lighting has changed.
        RenderOptionsChanged    = 0x2,      ///< Options that affect the rendering have changed.
    };

    /** The refresh flags above are passed to RenderPass::execute() via a
        field with this name in the dictionary.
    */
    static const char kRenderPassRefreshFlags[] = "_refreshFlags";

    /** First available preudorandom number generator dimension.
    */
    static const char kRenderPassPRNGDimension[] = "_prngDimension";

    /** Adjust shading normals on primary hits.
    */
    static const char kRenderPassGBufferAdjustShadingNormals[] = "_gbufferAdjustShadingNormals";

    FALCOR_ENUM_CLASS_OPERATORS(RenderPassRefreshFlags);
}
