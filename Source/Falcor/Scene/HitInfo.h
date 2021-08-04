/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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

namespace Falcor
{
    class Scene;

    class dlldecl HitInfo
    {
    public:
        static const uint32_t kMaxPackedSizeInBytes = 12;
        static const ResourceFormat kDefaultFormat = ResourceFormat::RG32Uint;

        HitInfo() = default;
        HitInfo(const Scene & scene) { init(scene); }
        void init(const Scene& scene);

        /** Returns defines needed packing/unpacking a HitInfo struct.
        */
        Shader::DefineList getDefines() const;

        /** Returns the resource format required for encoding packed hit information.
        */
        ResourceFormat getFormat() const;

    private:
        uint32_t mTypeBits = 0;             ///< Number of bits to store hit type.
        uint32_t mInstanceIndexBits = 0;    ///< Number of bits to store instance index.
        uint32_t mPrimitiveIndexBits = 0;   ///< Number of bits to store primitive index.

        uint32_t mDataSize = 0;             ///< Number of uints to store unpacked hit information.
        uint32_t mPackedDataSize = 0;       ///< Number of uints to store packed hit information.
    };
}
