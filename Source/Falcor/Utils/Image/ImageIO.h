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
#include "Bitmap.h"
#include "Core/Macros.h"
#include "Core/API/Texture.h"
#include <filesystem>

namespace Falcor
{
    class CopyContext;

    class FALCOR_API ImageIO
    {
    public:
        enum class CompressionMode
        {
            /** Stores RGB data with 1 bit of alpha.
                8 bytes per block.
            */
            BC1,

            /** Stores RGBA data. Combines BC1 for RGB with 4 bits of alpha.
                16 bytes per block.
            */
            BC2,

            /** Stores RGBA data. Combines BC1 for RGB and BC4 for alpha.
                16 bytes per block.
            */
            BC3,

            /** Stores a single grayscale channel.
                8 bytes per block.
            */
            BC4,

            /** Stores two channels using BC4 for each channel.
                16 bytes per block.
            */
            BC5,

            /** Stores RGB 16-bit floating point data.
                16 bytes per block.
            */
            BC6,

            /** Stores 8-bit RGB or RGBA data.
                16 bytes per block.
            */
            BC7,

            /** No compression mode specified.
            */
            None
        };

        /** Load a DDS file to a Bitmap. If the file contains an image array and/or mips, only the first image will be loaded.
            Throws an exception if the DDS file is malformed.
            \param[in] path Path of file to load.
            \return Bitmap object containing image data if loading was successful. Otherwise, nullptr.
        */
        static Bitmap::UniqueConstPtr loadBitmapFromDDS(const std::filesystem::path& path); // top down = true

        /** Load a DDS file to a Texture.
            Throws an exception if the DDS file is malformed.
            \param[in] path Path of file to load.
            \param[in] loadAsSrgb If true, convert the image format property to a corresponding sRGB format if available. Image data is not changed.
            \return Texture object containing image data if loading was successful. Otherwise, nullptr.
        */
        static Texture::SharedPtr loadTextureFromDDS(const std::filesystem::path& path, bool loadAsSrgb);

        /** Saves a bitmap to a DDS file.
            Throws an exception if path is invalid or the image cannot be saved.
            \param[in] path Path to save to.
            \param[in] bitmap Bitmap object to save.
            \param[in] mode Block compression mode. By default, will save data as-is and will not decompress if already compressed.
            \param[in] if true, generate and save full mipmap chain; requires the caller to have initialized COM.
        */
        static void saveToDDS(const std::filesystem::path& path, const Bitmap& bitmap, CompressionMode mode = CompressionMode::None, bool generateMips = false);

        /** Saves a Texture to a DDS file. All mips and array images are saved.
            Throws an exception if the path is invalid or the image cannot be saved.

            TODO: Support exporting single subresource. Options for one or all are probably enough?

            \param[in] pContext Copy context used to read texture data from the GPU.
            \param[in] path Path to save to.
            \param[in] pBitmap Bitmap object to save.
            \param[in] mode Block compression mode. By default, will save data as-is and will not decompress if already compressed.
            \param[in] if true, generate and save full mipmap chain; requires the caller to have initialized COM.
        */
        static void saveToDDS(CopyContext* pContext, const std::filesystem::path& path, const Texture::SharedPtr& pTexture, CompressionMode mode = CompressionMode::None, bool generateMips = false);
    };
}
