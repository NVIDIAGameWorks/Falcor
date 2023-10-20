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
#include "Handles.h"
#include "Core/Macros.h"
#include "Utils/Logger.h"

#include <slang.h>
#include <slang-gfx.h>
#include <slang-com-ptr.h>

#define FALCOR_GFX_CALL(call)                      \
    {                                              \
        gfx::Result result_ = call;                \
        if (SLANG_FAILED(result_))                 \
        {                                          \
            gfxReportError("GFX", #call, result_); \
        }                                          \
    }

#if FALCOR_HAS_D3D12
#define FALCOR_D3D_CALL(call)                      \
    {                                              \
        HRESULT result_ = call;                    \
        if (FAILED(result_))                       \
        {                                          \
            gfxReportError("D3D", #call, result_); \
        }                                          \
    }
#define FALCOR_GET_COM_INTERFACE(_base, _type, _var) \
    FALCOR_MAKE_SMART_COM_PTR(_type);                \
    FALCOR_CONCAT_STRINGS(_type, Ptr) _var;          \
    FALCOR_D3D_CALL(_base->QueryInterface(IID_PPV_ARGS(&_var)));
#define FALCOR_MAKE_SMART_COM_PTR(_a) _COM_SMARTPTR_TYPEDEF(_a, __uuidof(_a))
#endif

namespace Falcor
{
/**
 * Report a GFX or D3D error.
 * This will throw a RuntimeError exception.
 */
FALCOR_API void gfxReportError(const char* api, const char* call, gfx::Result result);

} // namespace Falcor
