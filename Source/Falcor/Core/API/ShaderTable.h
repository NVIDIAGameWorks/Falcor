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
#pragma once

#include "Device.h"
#include <slang-gfx.h>

namespace Falcor
{
class Scene;
class Program;
class RtStateObject;
class RtProgramVars;
class RenderContext;

// clang-format off
/**
 * This class represents the GPU shader table for raytracing programs.
 * We are using the following layout for the shader table:
 *
 * +------------+--------+--------+-----+--------+---------+--------+-----+--------+--------+-----+--------+-----+---------+---------+-----+---------+
 * |            |        |        | ... |        |         |        | ... |        |        | ... |        | ... |         |         | ... |         |
 * |   RayGen   |  Miss  |  Miss  | ... |  Miss  |  Hit    |  Hit   | ... |  Hit   |  Hit   | ... |  Hit   | ... |  Hit    |  Hit    | ... |  Hit    |
 * |   Entry    |  Idx0  |  Idx1  | ... | IdxM-1 |  Ray0   |  Ray1  | ... | RayK-1 |  Ray0  | ... | RayK-1 | ... |  Ray0   |  Ray1   | ... | RayK-1  |
 * |            |        |        | ... |        |  Geom0  |  Geom0 | ... |  Geom0 |  Geom1 | ... |  Geom1 | ... | GeomN-1 | GeomN-1 | ... | GeomN-1 |
 * +------------+--------+--------+-----+--------+---------+--------+-----+--------+--------+-----+--------+-----+---------+---------+-----+---------+
 *
 * The first record is the ray gen record, followed by the M miss records, followed by the geometry hit group records.
 * For each of the N geometries in the scene we have K hit group records, where K is the number of ray types (the same for all geometries).
 * The size of each record is based on the requirements of the local root signatures. By default, raygen, miss, and hit group records
 * contain only the program identifier (32B).
 *
 * User provided local root signatures are currently not supported for performance reasons. Managing and updating data for custom root
 * signatures results in significant overhead. To get the root signature that matches this table, call the static function
 * getRootSignature().
 */
// clang-format on

// In GFX, we use gfx::IShaderTable directly. We wrap
// the ComPtr with `ShaderTablePtr` class so it will be freed
// with the deferred release mechanism.
class ShaderTablePtr
{
public:
    ShaderTablePtr(ref<Device> pDevice) : mpDevice(pDevice) {}

    gfx::IShaderTable& operator*() { return *mTable; }

    gfx::IShaderTable* operator->() { return mTable; }

    gfx::IShaderTable* get() { return mTable.get(); }

    gfx::IShaderTable** writeRef() { return mTable.writeRef(); }

    operator gfx::IShaderTable*() { return mTable.get(); }

    ~ShaderTablePtr() { mpDevice->releaseResource(mTable); }

private:
    ref<Device> mpDevice;
    Slang::ComPtr<gfx::IShaderTable> mTable;
};
} // namespace Falcor
