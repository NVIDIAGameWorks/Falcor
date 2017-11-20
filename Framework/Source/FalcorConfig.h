/***************************************************************************
# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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

#ifdef _DEBUG
#define _LOG_ENABLED 1
#else
#define _LOG_ENABLED 0 // Set this to 1 to enable log messages in release builds
#endif 

#define _PROFILING_ENABLED 1                // Set this to 1 to enable CPU/GPU profiling
#define _PROFILING_LOG 0                    // Set this to 1 to dump profiling data while profiler is active.
#define _PROFILING_LOG_BATCH_SIZE 1024 * 1  // This can be used to control how many samples are accumulated before they are dumped to file.

#define _ENABLE_NVAPI false // Controls NVIDIA specific DX extensions. If it is set to true, make sure you have the NVAPI package in your 'Externals' directory. View the readme for more information.

#define FALCOR_BUILD_SLANG                  1 // Set this to 1 to enable Slang compiler to be built into Falcor
#define FALCOR_USE_SLANG_AS_PREPROCESSOR    0 // Set this to 1 to use Slang as a source-to-source preprocessor

#if (FALCOR_USE_SLANG_AS_PREPROCESSOR) && !FALCOR_BUILD_SLANG
#error Trying to use Slang without building it
#endif

#define FALCOR_USE_PYTHON                   0 // Set to 1 to build Python embedding API and samples.  See README.txt in "LearningWithEmbeddedPython" sample for more information.