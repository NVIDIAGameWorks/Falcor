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

#if defined(_MSC_VER)
#define BEGIN_DISABLE_USD_WARNINGS                                                                   \
    __pragma(warning(push)) __pragma(warning(disable : 4003)) /* Not enough macro arguments */       \
        __pragma(warning(disable : 4244))                     /* Conversion possible loss of data */ \
        __pragma(warning(disable : 4267))                     /* Conversion possible loss of data */ \
        __pragma(warning(disable : 4305))                     /* Truncation double to float */       \
        __pragma(warning(disable : 5033))                     /* 'register' storage class specifier deprecated */
#define END_DISABLE_USD_WARNINGS __pragma(warning(pop))
#elif defined(__clang__)
#define BEGIN_DISABLE_USD_WARNINGS __pragma(clang diagnostic push) __pragma(clang diagnostic ignored "-Wignored-attributes")
#define END_DISABLE_USD_WARNINGS __pragma(clang diagnostic pop)
#elif defined(__GNUC__)
#define BEGIN_DISABLE_USD_WARNINGS _Pragma("GCC diagnostic push") _Pragma("GCC diagnostic ignored \"-Wignored-attributes\"")
_Pragma("GCC diagnostic ignored \"-Wparentheses\"")
#define END_DISABLE_USD_WARNINGS _Pragma("GCC diagnostic pop")
#else
#define BEGIN_DISABLE_USD_WARNINGS
#define END_DISABLE_USD_WARNINGS
#endif
