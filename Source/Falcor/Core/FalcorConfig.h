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

#define FALCOR_ENABLE_LOGGER            1 // Set this to 1 to enable logging.
#define FALCOR_ENABLE_PROFILER          1 // Set this to 1 to enable CPU/GPU profiling.

#define FALCOR_ENABLE_NVAPI             0 // Set this to 1 to enable NVIDIA specific DX extensions. Make sure you have the NVAPI package in your 'Externals' directory. View the readme for more information.
#define FALCOR_ENABLE_CUDA              0 // Set this to 1 to enable CUDA use and CUDA/DX interoperation. Make sure you have the CUDA SDK package in your 'Externals' directory. View the readme for more information.
#define FALCOR_ENABLE_OPTIX             0 // Set this to 1 to enable OptiX. Make sure you have the OptiX SDK package in your 'Externals' directory. View the readme for more information.
#define FALCOR_ENABLE_D3D12_AGILITY_SDK 1 // Set this to 1 to enable D3D12 Agility SDK. Make sure you have the Agility SDK package in your `Externals` directory. View the readme for more information.
#define FALCOR_ENABLE_NRD               1 // Set this to 1 to enable NRD. Make sure you have the NRD SDK package in your `Externals` directory. View the readme for more information.
#define FALCOR_ENABLE_DLSS              1 // Set this to 1 to enable DLSS. Make sure you have the DLSS SDK package in your `Externals` directory. View the readme for more information.
#define FALCOR_ENABLE_RTXDI             1 // Set this to 1 to enable RTXDI. Make sure you have the RTXDI SDK package in your `Externals` directory. View the readme for more information.
