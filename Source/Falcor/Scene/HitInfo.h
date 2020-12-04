/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

    class HitInfo
    {
    public:
        static const uint32_t kInvalidIndex = 0xffffffff;
        static const uint32_t kMaxPackedSizeInBytes = 12;
        static const ResourceFormat kDefaultFormat = ResourceFormat::RG32Uint;

        /** Returns defines needed packing/unpacking a HitInfo struct.
        */
        Shader::DefineList getDefines() const
        {
            assert((mInstanceTypeBits + mInstanceIndexBits) <= 32 && mPrimitiveIndexBits <= 32);
            Shader::DefineList defines;
            defines.add("HIT_INSTANCE_TYPE_BITS", std::to_string(mInstanceTypeBits));
            defines.add("HIT_INSTANCE_INDEX_BITS", std::to_string(mInstanceIndexBits));
            defines.add("HIT_PRIMITIVE_INDEX_BITS", std::to_string(mPrimitiveIndexBits));
            return defines;
        }

        /** Returns the resource format required for encoding packed hit information.
        */
        ResourceFormat getFormat() const
        {
            assert((mInstanceTypeBits + mInstanceIndexBits) <= 32 && mPrimitiveIndexBits <= 32);
            if (mInstanceTypeBits + mInstanceIndexBits + mPrimitiveIndexBits <= 32) return ResourceFormat::RG32Uint;
            else return ResourceFormat::RGBA32Uint; // RGB32Uint can't be used for UAV writes
        }

        HitInfo() = default;
        HitInfo(const Scene & scene) { init(scene); }
        void init(const Scene& scene);

    private:
        uint32_t mInstanceTypeBits = 0;
        uint32_t mInstanceIndexBits = 0;
        uint32_t mPrimitiveIndexBits = 0;
    };
}
