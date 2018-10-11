/***************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

namespace Falcor
{
    enum class RtBuildFlags
    {
        None = 0,
        AllowUpdate         = 0x1,
        AllowCompaction     = 0x2,
        FastTrace           = 0x4,
        FastBuild           = 0x8,
        MinimizeMemory      = 0x10,
        PerformUpdate       = 0x20,
    };
    enum_class_operators(RtBuildFlags);

#define rt_flags(a) case RtBuildFlags::a: return #a
    inline std::string to_string(RtBuildFlags flags)
    {
        switch (flags)
        {
            rt_flags(None);
            rt_flags(AllowUpdate);
            rt_flags(AllowCompaction);
            rt_flags(FastTrace);
            rt_flags(FastBuild);
            rt_flags(MinimizeMemory);
            rt_flags(PerformUpdate);
        default:
            should_not_get_here();
            return "";
        }
    }
#undef rt_flags

    inline D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS getDxrBuildFlags(RtBuildFlags buildFlags)
    {
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAGS dxr = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_NONE;

        if (is_set(buildFlags, RtBuildFlags::AllowUpdate)) dxr |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;
        if (is_set(buildFlags, RtBuildFlags::AllowCompaction)) dxr |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION;
        if (is_set(buildFlags, RtBuildFlags::FastTrace)) dxr |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;
        if (is_set(buildFlags, RtBuildFlags::FastBuild)) dxr |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD;
        if (is_set(buildFlags, RtBuildFlags::MinimizeMemory)) dxr |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_MINIMIZE_MEMORY;
        if (is_set(buildFlags, RtBuildFlags::PerformUpdate)) dxr |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;

        return dxr;
    }
}

// The max scalars supported by our driver
#define FALCOR_RT_MAX_PAYLOAD_SIZE_IN_BYTES (14 * sizeof(float))

namespace Falcor
{
    class RenderContext;
    class RtProgramVars;
    class RtState;

}
