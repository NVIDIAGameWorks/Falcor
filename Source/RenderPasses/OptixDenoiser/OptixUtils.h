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

#include "Falcor.h"
#include "Utils/CudaRuntime.h"
#include "Utils/CudaUtils.h"

#include <optix.h>
#include <optix_stubs.h>

/**
 * Utility class for a GPU/device buffer for use with CUDA.
 * Adapted from Ingo Wald's SIGGRAPH 2019 tutorial code for OptiX 7.
 */
class CudaBuffer
{
public:
    CudaBuffer() {}

    CUdeviceptr getDevicePtr() { return (CUdeviceptr)mpDevicePtr; }
    size_t getSize() { return mSizeBytes; }

    void allocate(size_t size)
    {
        if (mpDevicePtr)
            free();
        mSizeBytes = size;
        FALCOR_CUDA_CHECK(cudaMalloc((void**)&mpDevicePtr, mSizeBytes));
    }

    void resize(size_t size) { allocate(size); }

    void free()
    {
        FALCOR_CUDA_CHECK(cudaFree(mpDevicePtr));
        mpDevicePtr = nullptr;
        mSizeBytes = 0;
    }

private:
    size_t mSizeBytes = 0;
    void* mpDevicePtr = nullptr;
};

// Initializes OptiX context. Throws an exception if initialization fails.
OptixDeviceContext initOptix(Falcor::Device* pDevice);
