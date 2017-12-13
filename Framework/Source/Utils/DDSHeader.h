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
#include "Utils/Platform/OS.h"
#include "Utils/DXHeader.h"

namespace Falcor
{
    namespace DdsHelper
    {
        struct DdsHeader
        {
            struct PixelFormat
            {
                uint32_t structSize;
                uint32_t flags;
                uint32_t fourCC;
                uint32_t bitcount;
                uint32_t rMask;
                uint32_t gMask;
                uint32_t bMask;
                uint32_t aMask;

                // flags
                static const uint32_t kAlphaPixelsMask = 0x1;
                static const uint32_t kAlphaMask = 0x2;
                static const uint32_t kFourCCFlag = 0x4;
                static const uint32_t kRgbMask = 0x40;
                static const uint32_t kYuvMask = 0x200;
                static const uint32_t kLuminanceMask = 0x20000;
                static const uint32_t kBumpMask = 0x00080000;
            };

            uint32_t headerSize;
            uint32_t flags;
            uint32_t height;
            uint32_t width;
            union
            {
                uint32_t pitch;
                uint32_t linearSize;
            };

            uint32_t depth;
            uint32_t mipCount;
            uint32_t reserved[11];
            PixelFormat	pixelFormat;
            uint32_t caps[4];
            uint32_t reserved2;

            // Flags
            static const uint32_t kCapsMask = 0x1;
            static const uint32_t kHeightMask = 0x2;
            static const uint32_t kWidthMask = 0x4;
            static const uint32_t kPitchMask = 0x8;
            static const uint32_t kPixelFormatMask = 0x1000;
            static const uint32_t kMipCountMask = 0x20000;
            static const uint32_t kLinearSizeMask = 0x80000;
            static const uint32_t kDepthMask = 0x800000;

            // Caps[0]
            static const uint32_t kCapsComplexMask = 0x8;
            static const uint32_t kCapsMipMapMask = 0x400000;
            static const uint32_t kCapsTextureMask = 0x1000;

            // Caps[1]
            static const uint32_t kCaps2CubeMapMask = 0x200;
            static const uint32_t kCaps2CubeMapPosXMask = 0x400;
            static const uint32_t kCaps2CubeMapNegXMask = 0x800;
            static const uint32_t kCaps2CubeMapPosYMask = 0x1000;
            static const uint32_t kCaps2CubeMapNegYMask = 0x2000;
            static const uint32_t kCaps2CubeMapPosZMask = 0x4000;
            static const uint32_t kCaps2CubeMapNegZMask = 0x8000;
            static const uint32_t kCaps2VolumeMask = 0x200000;
        };

        struct DdsHeaderDX10
        {
            DXFormat            dxgiFormat;
            DXResourceDimension resourceDimension;
            uint32_t            miscFlag;
            uint32_t            arraySize;
            uint32_t            miscFlags2;

            static const uint32_t kCubeMapMask = 0x4;
        };

        struct DdsData
        {
            DdsHeader header;
            DdsHeaderDX10 dx10Header;
            bool hasDX10Header;
            std::vector<uint8_t> data;
        };
    }
}
