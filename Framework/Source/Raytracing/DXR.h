/************************************************************************************************************************************\
|*                                                                                                                                    *|
|*     Copyright © 2017 NVIDIA Corporation.  All rights reserved.                                                                     *|
|*                                                                                                                                    *|
|*  NOTICE TO USER:                                                                                                                   *|
|*                                                                                                                                    *|
|*  This software is subject to NVIDIA ownership rights under U.S. and international Copyright laws.                                  *|
|*                                                                                                                                    *|
|*  This software and the information contained herein are PROPRIETARY and CONFIDENTIAL to NVIDIA                                     *|
|*  and are being provided solely under the terms and conditions of an NVIDIA software license agreement                              *|
|*  and / or non-disclosure agreement.  Otherwise, you have no rights to use or access this software in any manner.                   *|
|*                                                                                                                                    *|
|*  If not covered by the applicable NVIDIA software license agreement:                                                               *|
|*  NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOFTWARE FOR ANY PURPOSE.                                            *|
|*  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.                                                           *|
|*  NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE,                                                                     *|
|*  INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.                       *|
|*  IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES,                               *|
|*  OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT,                         *|
|*  NEGLIGENCE OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOURCE CODE.            *|
|*                                                                                                                                    *|
|*  U.S. Government End Users.                                                                                                        *|
|*  This software is a "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT 1995),                                       *|
|*  consisting  of "commercial computer  software"  and "commercial computer software documentation"                                  *|
|*  as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government only as a commercial end item.     *|
|*  Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995),                                          *|
|*  all U.S. Government End Users acquire the software with only those rights set forth herein.                                       *|
|*                                                                                                                                    *|
|*  Any use of this software in individual and commercial software must include,                                                      *|
|*  in the user documentation and internal comments to the code,                                                                      *|
|*  the above Disclaimer (as applicable) and U.S. Government End Users Notice.                                                        *|
|*                                                                                                                                    *|
 \************************************************************************************************************************************/
#pragma once
#ifdef FALCOR_DXR
#include "d3d12_1.h"

MAKE_SMART_COM_PTR(ID3D12DeviceRaytracingPrototype);
MAKE_SMART_COM_PTR(ID3D12CommandListRaytracingPrototype);
MAKE_SMART_COM_PTR(ID3D12StateObjectPrototype);

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

#include "RtModel.h"
#include "RtScene.h"
#include "RtShader.h"
#include "RtProgram/RtProgram.h"
#include "RtProgram/RtProgramVersion.h"
#include "RtProgram/SingleShaderProgram.h"
#include "RtProgram/HitProgram.h"
#include "RtProgramVars.h"
#include "RtState.h"
#include "RtStateObject.h"
#include "API/Device.h"
#include "API/RenderContext.h"
#include "RtSample.h"
#include "RtSceneRenderer.h"

namespace Falcor
{
    class RenderContext;
    void raytrace(RenderContext* pContext, RtProgramVars::SharedPtr pVars, RtState::SharedPtr pState, uint32_t width, uint32_t height);
}

#endif