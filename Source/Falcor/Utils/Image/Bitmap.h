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
#include "Core/Platform/OS.h"
#include "Core/API/Formats.h"
#include <memory>
#include <filesystem>

namespace Falcor
{
    class Texture;

    /** A class representing a memory bitmap
    */
    class FALCOR_API Bitmap
    {
    public:
        enum class ExportFlags : uint32_t
        {
            None = 0u,              //< Default
            ExportAlpha = 1u << 0,  //< Save alpha channel as well
            Lossy = 1u << 1,        //< Try to store in a lossy format
            Uncompressed = 1u << 2, //< Prefer faster load to a more compact file size
        };

        enum class FileFormat
        {
            PngFile,    //< PNG file for lossless compressed 8-bits images with optional alpha
            JpegFile,   //< JPEG file for lossy compressed 8-bits images without alpha
            TgaFile,    //< TGA file for lossless uncompressed 8-bits images with optional alpha
            BmpFile,    //< BMP file for lossless uncompressed 8-bits images with optional alpha
            PfmFile,    //< PFM file for floating point HDR images with 32-bit float per channel
            ExrFile,    //< EXR file for floating point HDR images with 16-bit float per channel
            DdsFile,    //< DDS file for storing GPU resource formats, including block compressed formats
                        //< See ImageIO. TODO: Remove(?) Bitmap IO implementation when ImageIO supports other formats
        };

        using UniquePtr = std::unique_ptr<Bitmap>;
        using UniqueConstPtr = std::unique_ptr<const Bitmap>;

        /** Create from memory.
            \param[in] width Width in pixels.
            \param[in] height Height in pixels
            \param[in] format Resource format.
            \param[in] pData Pointer to data. Data will be copied internally during creation and does not need to be managed by the caller.
            \return A new bitmap object.
        */
        static UniqueConstPtr create(uint32_t width, uint32_t height, ResourceFormat format, const uint8_t* pData);

        /** Create a new object from file.
            \param[in] path Path to load from. If the file can't be found relative to the current directory, Falcor will search for it in the common directories.
            \param[in] isTopDown Control the memory layout of the image. If true, the top-left pixel is the first pixel in the buffer, otherwise the bottom-left pixel is first.
            \return If loading was successful, a new object. Otherwise, nullptr.
        */
        static UniqueConstPtr createFromFile(const std::filesystem::path& path, bool isTopDown);

        /** Store a memory buffer to a file.
            \param[in] path Path to write to.
            \param[in] width The width of the image.
            \param[in] height The height of the image.
            \param[in] fileFormat The destination file format. See FileFormat enum above.
            \param[in] exportFlags The flags to export the file. See ExportFlags above.
            \param[in] ResourceFormat the format of the resource data
            \param[in] isTopDown Control the memory layout of the image. If true, the top-left pixel will be stored first, otherwise the bottom-left pixel will be stored first
            \param[in] pData Pointer to the buffer containing the image
        */
        static void saveImage(const std::filesystem::path& path, uint32_t width, uint32_t height, FileFormat fileFormat, ExportFlags exportFlags, ResourceFormat resourceFormat, bool isTopDown, void* pData);

        /**  Open dialog to save image to a file
            \param[in] pTexture Texture to save to file
        */
        static void saveImageDialog(Texture* pTexture);

        /** Get a pointer to the bitmap's data store
        */
        uint8_t* getData() const { return mpData.get(); }

        /** Get the width of the bitmap
        */
        uint32_t getWidth() const { return mWidth; }

        /** Get the height of the bitmap
        */
        uint32_t getHeight() const { return mHeight; }

        /** Get the data format
        */
        ResourceFormat getFormat() const { return mFormat; }

        /** Get the row pitch in bytes. For compressed formats this corresponds to one row of blocks, not pixels.
        */
        uint32_t getRowPitch() const { return mRowPitch; }

        /** Get the data size in bytes
        */
        uint32_t getSize() const { return mSize; }

        /** Get the file dialog filter vec for images.
            \param[in] format If set to ResourceFormat::Unknown, will return all the supported image file formats. If set to something else, will only return file types which support this format.
        */
        static FileDialogFilterVec getFileDialogFilters(ResourceFormat format = ResourceFormat::Unknown);

        /** Get a file extension from a resource format
        */
        static std::string getFileExtFromResourceFormat(ResourceFormat format);

        /** Get the file format flags for the image extension
            \param[in] ext The image file extension to get the
        */
        static FileFormat getFormatFromFileExtension(const std::string& ext);

    protected:
        Bitmap() = default;
        Bitmap(uint32_t width, uint32_t height, ResourceFormat format);
        Bitmap(uint32_t width, uint32_t height, ResourceFormat format, const uint8_t* pData);

        std::unique_ptr<uint8_t[]> mpData;
        uint32_t mWidth = 0;
        uint32_t mHeight = 0;
        uint32_t mRowPitch = 0;
        uint32_t mSize = 0;
        ResourceFormat mFormat = ResourceFormat::Unknown;
    };

    FALCOR_ENUM_CLASS_OPERATORS(Bitmap::ExportFlags);
}
