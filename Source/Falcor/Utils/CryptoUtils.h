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
#include <array>
#include <cstdint>
#include <cstdlib>

namespace Falcor
{
    /** Helper to compute SHA-1 hash.
    */
    class FALCOR_API SHA1
    {
    public:
        using MD = std::array<uint8_t, 20>; ///< Message digest.

        SHA1();

        /** Update hash by adding one byte.
            \param[in] value Value to hash.
        */
        void update(uint8_t value);

        /** Update hash by adding the given data.
            \param[in] data Data to hash.
            \param[in] len Length of data in bytes.
        */
        void update(const void* data, size_t len);

        /** Return final message digest.
            \return Returns the SHA-1 message digest.
        */
        MD finalize();

        /** Compute SHA-1 hash over the given data.
            \param[in] data Data to hash.
            \param[in] len Length of data in bytes.
            \return Returns the SHA-1 message digest.
        */
        static MD compute(const void* data, size_t len);

    private:
        void addByte(uint8_t x);
        void processBlock(const uint8_t* ptr);

        uint32_t mIndex;
        uint64_t mBits;
        uint32_t mState[5];
        uint8_t mBuf[64];
    };
};
