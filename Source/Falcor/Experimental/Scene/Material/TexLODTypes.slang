/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

/** Enums for texture level-of-detail -- see TexLODHelpers.slang
*/

#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

/** This enum is shared between CPU/GPU.
    It enumerates the different the different texture LOD modes.
*/
enum class TexLODMode
    // TODO: Remove the ifdefs and the include when Slang supports enum type specifiers.
#ifdef HOST_CODE
    : uint32_t
#endif
{
    Mip0 = 0,
    RayCones = 1,
    RayDiffsIsotropic = 2,
    RayDiffsAnisotropic = 3,
};

/** This enum is shared between CPU/GPU.
    It enumerates the different the ray cone modes. RayTracingGems1 uses the G-buffer to store the surface spread angle. Combo and Unified avoid the G-buffer.
    Combo uses ray differntials to compute the spread at the first hit and then average edge curvature for all hits after that.
    Unified uses interpolated edge curvatures for the first hit and then average edge curvatures for all hits after that.
*/
enum class RayConeMode
    // TODO: Remove the ifdefs and the include when Slang supports enum type specifiers.
#ifdef HOST_CODE
    : uint32_t
#endif
{
    RayTracingGems1 = 0,
    Combo = 1,
    Unified = 2,
};

END_NAMESPACE_FALCOR
