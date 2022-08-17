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
#include "ErrorHandling.h"
#include "Errors.h" // TODO: This is only used for RuntimeError, we should consider removing it again.
#include <fmt/format.h>
#include <string>

#ifdef _DEBUG

#define FALCOR_ASSERT(a)\
    if (!(a)) {\
        std::string str = fmt::format("assertion failed( {} )\n{}({})", #a, __FILE__, __LINE__); \
        Falcor::reportFatalError(str);\
    }
#define FALCOR_ASSERT_MSG(a, msg)\
    if (!(a)) {\
        std::string str = fmt::format("assertion failed( {} ): {}\n{}({})", #a, msg, __FILE__, __LINE__); \
        Falcor::reportFatalError(str); \
    }
#define FALCOR_ASSERT_OP(a, b, OP)\
    if (!(a OP b)) {\
        std::string str = fmt::format("assertion failed( {} {} {} ({} {} {}) )\n{}({})", #a, #OP, #b, a, #OP, b, __FILE__, __LINE__); \
        Falcor::reportFatalError(str); \
    }
#define FALCOR_ASSERT_EQ(a, b) FALCOR_ASSERT_OP(a, b, == )
#define FALCOR_ASSERT_NE(a, b) FALCOR_ASSERT_OP(a, b, != )
#define FALCOR_ASSERT_GE(a, b) FALCOR_ASSERT_OP(a, b, >= )
#define FALCOR_ASSERT_GT(a, b) FALCOR_ASSERT_OP(a, b, > )
#define FALCOR_ASSERT_LE(a, b) FALCOR_ASSERT_OP(a, b, <= )
#define FALCOR_ASSERT_LT(a, b) FALCOR_ASSERT_OP(a, b, < )

#else // _DEBUG

#define FALCOR_ASSERT(a) {}
#define FALCOR_ASSERT_MSG(a, msg) {}
#define FALCOR_ASSERT_OP(a, b, OP) {}
#define FALCOR_ASSERT_EQ(a, b) FALCOR_ASSERT_OP(a, b, == )
#define FALCOR_ASSERT_NE(a, b) FALCOR_ASSERT_OP(a, b, != )
#define FALCOR_ASSERT_GE(a, b) FALCOR_ASSERT_OP(a, b, >= )
#define FALCOR_ASSERT_GT(a, b) FALCOR_ASSERT_OP(a, b, > )
#define FALCOR_ASSERT_LE(a, b) FALCOR_ASSERT_OP(a, b, <= )
#define FALCOR_ASSERT_LT(a, b) FALCOR_ASSERT_OP(a, b, < )

#endif // _DEBUG

#define FALCOR_UNIMPLEMENTED() do{ FALCOR_ASSERT_MSG(false, "Not implemented"); throw Falcor::RuntimeError("Not implemented"); } while(0)

#define FALCOR_UNREACHABLE() FALCOR_ASSERT(false)

#define FALCOR_PRINT( x ) do { Falcor::logInfo("{} = {}", #x, x); } while(0)
