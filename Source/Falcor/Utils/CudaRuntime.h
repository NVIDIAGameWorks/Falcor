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

/**
 * CUDA runtime defines vector types in the global namespace. Some of these
 * types clash with the vector types in Falcor, which live in the Falcor::math
 * and Falcor namespace. To avoid this clash, we rename the CUDA types here.
 * Falcor code should includle this header instead of <cuda_runtime.h>.
 */

#define int1 cuda_int1
#define int2 cuda_int2
#define int3 cuda_int3
#define int4 cuda_int4
#define uint1 cuda_uint1
#define uint2 cuda_uint2
#define uint3 cuda_uint3
#define uint4 cuda_uint4
#define float1 cuda_float1
#define float2 cuda_float2
#define float3 cuda_float3
#define float4 cuda_float4

#include <cuda_runtime.h>

#undef int1
#undef int2
#undef int3
#undef int4
#undef uint1
#undef uint2
#undef uint3
#undef uint4
#undef float1
#undef float2
#undef float3
#undef float4
