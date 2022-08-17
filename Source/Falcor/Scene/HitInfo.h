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
#include "Core/Macros.h"
#include "Core/API/Formats.h"
#include "Core/API/Shader.h"

namespace Falcor
{
    class Scene;

    /** Host side utility to setup the bit allocations for device side HitInfo.

        By default, HitInfo is encoded in 128 bits. There is a compression mode
        where HitInfo is encoded in 64 bits. This mode is only available in
        scenes that exclusively use triangle meshes and are small enough so
        the header information fits in 32 bits. In compression mode,
        barycentrics are quantized to 16 bit unorms.

        See HitInfo.slang for more information.
    */
    class FALCOR_API HitInfo
    {
    public:
        static const uint32_t kMaxPackedSizeInBytes = 16;
        static const ResourceFormat kDefaultFormat = ResourceFormat::RGBA32Uint;

        HitInfo() = default;
        HitInfo(const Scene& scene, bool useCompression = false) { init(scene, useCompression); }
        void init(const Scene& scene, bool useCompression);

        /** Returns defines needed packing/unpacking a HitInfo struct.
        */
        Shader::DefineList getDefines() const;

        /** Returns the resource format required for encoding packed hit information.
        */
        ResourceFormat getFormat() const;

    private:
        bool mUseCompression = false;       ///< Store in compressed format (64 bits instead of 128 bits).

        uint32_t mTypeBits = 0;             ///< Number of bits to store hit type.
        uint32_t mInstanceIDBits = 0;       ///< Number of bits to store instance ID.
        uint32_t mPrimitiveIndexBits = 0;   ///< Number of bits to store primitive index.
    };
}
