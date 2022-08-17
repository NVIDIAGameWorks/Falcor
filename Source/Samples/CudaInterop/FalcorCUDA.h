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
#include "Core/API/Texture.h"
#include "Core/API/RenderContext.h"

#include <cuda_runtime.h>

namespace FalcorCUDA
{
    /** Initializes the CUDA driver API. Returns true if successful, false otherwise.
    */
    bool initCUDA();

    /** Imports the texture into a CUDA mipmapped array and returns the array in mipmappedArray. This method should only be called once per
        texture resource.
        \param pTex Pointer to the texture being imported
        \param mipmappedArray Reference to the array to import to
        \param usageFlags The requested flags to be bound to the mipmapped array
        \return True if successful, false otherwise
    */
    bool importTextureToMipmappedArray(Falcor::Texture::SharedPtr pTex, cudaMipmappedArray_t& mipmappedArray, uint32_t cudaUsageFlags);

    /** Maps a texture to a surface object which can be read and written within a CUDA kernel.
        This method should only be called once per texture on initial load. Store the returned surface object for repeated use.
        \param pTex Pointer to the texture being mapped
        \param usageFlags The requested flags to be bound to the underlying mipmapped array that will be used to create the surface object
        \return The surface object that the input texture is bound to
    */
    cudaSurfaceObject_t mapTextureToSurface(Falcor::Texture::SharedPtr pTex, uint32_t usageFlags);
};
