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
#include "GFXHandles.h"
#include "Utils/Logger.h"

#include <slang.h>
#include <slang-gfx.h>
#include <slang-com-ptr.h>

#define FALCOR_GFX_CALL(a) {auto hr_ = a; if(SLANG_FAILED(hr_)) { reportError(#a); }}

#if FALCOR_GFX_D3D12
#define FALCOR_D3D_CALL(_a) {HRESULT hr_ = _a; if (FAILED(hr_)) { reportError( #_a); }}
#define FALCOR_GET_COM_INTERFACE(_base, _type, _var) FALCOR_MAKE_SMART_COM_PTR(_type); FALCOR_CONCAT_STRINGS(_type, Ptr) _var; FALCOR_D3D_CALL(_base->QueryInterface(IID_PPV_ARGS(&_var)));
#define FALCOR_MAKE_SMART_COM_PTR(_a) _COM_SMARTPTR_TYPEDEF(_a, __uuidof(_a))
#endif

#define UNSUPPORTED_IN_GFX(msg_) {logWarning("{} is not supported in GFX. Ignoring call.", msg_);}
