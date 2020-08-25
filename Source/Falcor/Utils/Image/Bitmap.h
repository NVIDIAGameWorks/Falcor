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

namespace Falcor
{
    class Texture;

    /** A class representing a memory bitmap
    */
    class dlldecl Bitmap : public std::enable_shared_from_this<Bitmap>
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
        };

        using UniquePtr = std::unique_ptr<Bitmap>;
        using UniqueConstPtr = std::unique_ptr<const Bitmap>;

        /** Create a new object from file.
            \param[in] filename Filename, including a path. If the file can't be found relative to the current directory, Falcor will search for it in the common directories.
            \param[in] isTopDown Control the memory layout of the image. If true, the top-left pixel is the first pixel in the buffer, otherwise the bottom-left pixel is first.
            \return If loading was successful, a new object. Otherwise, nullptr.
        */
        static UniqueConstPtr createFromFile(const std::string& filename, bool isTopDown);

        /** Store a memory buffer to a PNG file.
            \param[in] filename Output filename. Can include a path - absolute or relative to the executable directory.
            \param[in] width The width of the image.
            \param[in] height The height of the image.
            \param[in] fileFormat The destination file format. See FileFormat enum above.
            \param[in] exportFlags The flags to export the file. See ExportFlags above.
            \param[in] ResourceFormat the format of the resource data
            \param[in] isTopDown Control the memory layout of the image. If true, the top-left pixel will be stored first, otherwise the bottom-left pixel will be stored first
            \param[in] pData Pointer to the buffer containing the image
        */
        static void saveImage(const std::string& filename, uint32_t width, uint32_t height, FileFormat fileFormat, ExportFlags exportFlags, ResourceFormat resourceFormat, bool isTopDown, void* pData);

        /**  Open dialog to save image to a file
            \param[in] pTexture Texture to save to file
             
        */
        static void saveImageDialog(Texture* pTexture);

        ~Bitmap();

        /** Get a pointer to the bitmap's data store
        */
        uint8_t* getData() const { return mpData; }

        /** Get the width of the bitmap
        */
        uint32_t getWidth() const { return mWidth; }

        /** Get the height of the bitmap
        */
        uint32_t getHeight() const { return mHeight; }

        /** Get the number of bytes per pixel
        */
        ResourceFormat getFormat() const { return mFormat; }

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

    private:
        Bitmap() = default;
        uint8_t* mpData = nullptr;
        uint32_t mWidth = 0;
        uint32_t mHeight = 0;
        ResourceFormat mFormat;
    };

    enum_class_operators(Bitmap::ExportFlags);
}
