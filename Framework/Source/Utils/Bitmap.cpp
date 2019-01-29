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
#include "Framework.h"
#include "Bitmap.h"
#include "FreeImage.h"
#include "Utils/Platform/OS.h"
#include "API/Device.h"
#include <cstring>
#include "StringUtils.h"
#include "API/Texture.h"

namespace Falcor
{
#ifdef FALCOR_VK
    static bool isRGB32fSupported() 
    { 
        VkFormatProperties p;
        vkGetPhysicalDeviceFormatProperties(gpDevice->getApiHandle(), VK_FORMAT_R32G32B32_SFLOAT, &p);
        return p.optimalTilingFeatures != 0;
    }
#else
    static bool isRGB32fSupported() { return false; } // FIX THIS
#endif
    const Bitmap* genError(const std::string& errMsg, const std::string& filename)
    {
        std::string err = "Error when loading image file " + filename + '\n' + errMsg + '.';
        logError(err);
        return nullptr;
    }

    Bitmap::UniqueConstPtr Bitmap::createFromFile(const std::string& filename, bool isTopDown)
    {
        std::string fullpath;
        if(findFileInDataDirectories(filename, fullpath) == false)
        {
            msgBox("Error when loading image file " + filename + "\n. Can't find the file");
            return nullptr;
        }

        FREE_IMAGE_FORMAT fifFormat = FIF_UNKNOWN;
        
        fifFormat = FreeImage_GetFileType(fullpath.c_str(), 0);
        if(fifFormat == FIF_UNKNOWN)
        {
            // Can't get the format from the file. Use file extension
            fifFormat = FreeImage_GetFIFFromFilename(fullpath.c_str());

            if(fifFormat == FIF_UNKNOWN)
            {
                return UniqueConstPtr(genError("Image Type unknown", filename));
            }
        }

        // Check the the library supports loading this image Type
        if(FreeImage_FIFSupportsReading(fifFormat) == false)
        {
            return UniqueConstPtr(genError("Library doesn't support the file format", filename));
        }

        // Read the DIB
        FIBITMAP* pDib = FreeImage_Load(fifFormat, fullpath.c_str());
        if(pDib == nullptr)
        {
            return UniqueConstPtr(genError("Can't read image file", filename));
        }

        // create the bitmap
        auto pBmp = new Bitmap;
        pBmp->mHeight = FreeImage_GetHeight(pDib);
        pBmp->mWidth = FreeImage_GetWidth(pDib);

        if(pBmp->mHeight == 0 || pBmp->mWidth == 0 || FreeImage_GetBits(pDib) == nullptr)
        {
            return UniqueConstPtr(genError("Invalid image", filename));
        }

        uint32_t bpp = FreeImage_GetBPP(pDib);
        switch(bpp)
        {
        case 128:
            pBmp->mFormat = ResourceFormat::RGBA32Float;    // 4xfloat32 HDR format
            break;
        case 96:
            pBmp->mFormat = isRGB32fSupported() ? ResourceFormat::RGB32Float : ResourceFormat::RGBA32Float;     // 3xfloat32 HDR format
            break;
        case 64:
            pBmp->mFormat = ResourceFormat::RGBA16Float;    // 4xfloat16 HDR format
            break;
        case 48:
            pBmp->mFormat = ResourceFormat::RGB16Float;     // 3xfloat16 HDR format
            break;
        case 32:
            pBmp->mFormat = ResourceFormat::BGRA8Unorm;
            break;
        case 24:
            pBmp->mFormat = ResourceFormat::BGRX8Unorm;
            break;
        case 16:
            pBmp->mFormat = ResourceFormat::RG8Unorm;
            break;
        case 8:
            pBmp->mFormat = ResourceFormat::R8Unorm;
            break;
        default:
            genError("Unknown bits-per-pixel", filename);
            return nullptr;
        }

        // Convert the image to RGBX image
        if(bpp == 24)
        {
            logWarning("Converting 24-bit texture to 32-bit");
            bpp = 32;
            auto pNew = FreeImage_ConvertTo32Bits(pDib);
            FreeImage_Unload(pDib);
            pDib = pNew;
        }
        else if (bpp == 96 && (isRGB32fSupported() == false))
        {
            logWarning("Converting 96-bit texture to 128-bit");
            bpp = 128;
            auto pNew = FreeImage_ConvertToRGBAF(pDib);
            FreeImage_Unload(pDib);
            pDib = pNew;
        }

        uint32_t bytesPerPixel = bpp / 8;

        pBmp->mpData = new uint8_t[pBmp->mHeight * pBmp->mWidth * bytesPerPixel];
        FreeImage_ConvertToRawBits(pBmp->mpData, pDib, pBmp->mWidth * bytesPerPixel, bpp, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, isTopDown);

        FreeImage_Unload(pDib);
        return UniqueConstPtr(pBmp);
    }

    Bitmap::~Bitmap()
    {
        delete[] mpData;
        mpData = nullptr;
    }

    static FREE_IMAGE_FORMAT toFreeImageFormat(Bitmap::FileFormat fmt)
    {
        switch(fmt)
        {
        case Bitmap::FileFormat::PngFile:
            return FIF_PNG;
        case Bitmap::FileFormat::JpegFile:
            return FIF_JPEG;
        case Bitmap::FileFormat::TgaFile:
            return FIF_TARGA;
        case Bitmap::FileFormat::BmpFile:
            return FIF_BMP;
        case Bitmap::FileFormat::PfmFile:
            return FIF_PFM;
        case Bitmap::FileFormat::ExrFile:
            return FIF_EXR;
        default:
            should_not_get_here();
        }
        return FIF_PNG;
    }

    static FREE_IMAGE_TYPE getImageType(uint32_t bytesPerPixel)
    {
        switch(bytesPerPixel)
        {
        case 4:
            return FIT_BITMAP;
        case 12:
            return FIT_RGBF;
        case 16:
            return FIT_RGBAF;
        default:
            should_not_get_here();
        }
        return FIT_BITMAP;
    }
    
    Bitmap::FileFormat Bitmap::getFormatFromFileExtension(const std::string& ext)
    {
        // This array is in the order of the enum
        static const char* kExtensions[] = {
            /* PngFile */ "png",
            /*JpegFile */ "jpg",
            /* TgaFile */ "tga",
            /* BmpFile */ "bmp",
            /* PfmFile */ "pfm",
            /* ExrFile */ "exr"
        };

        for (uint32_t i = 0 ; i < arraysize(kExtensions) ; i++)
        {
            if (kExtensions[i] == ext) return Bitmap::FileFormat(i);
        }
        logError("Can't find a matching format for file extension `" + ext + "`");
        return Bitmap::FileFormat(-1);
    }

    FileDialogFilterVec Bitmap::getFileDialogFilters(ResourceFormat format)
    {
        FileDialogFilterVec filters;
        bool showHdr = true;
        bool showLdr = true;

        if(format != ResourceFormat::Unknown)
        {
            FormatType type = getFormatType(format);
            uint32_t bitsPerTexel = getFormatBytesPerBlock(format);

            showHdr = type == FormatType::Float && (bitsPerTexel == 16 || bitsPerTexel == 12);
            showLdr = !showHdr;
        }

        if (showHdr)
        {
            filters.push_back({"hdr", "High Dynamic Range"});
            filters.push_back({"exr", "High Dynamic Range"});
            filters.push_back({"pfm", "Portable Float Map"});
        }

        if(showLdr)
        {
            filters.push_back({ "png", "Portable Network Graphics" });
            filters.push_back({ "jpg", "JPEG" });
            filters.push_back({ "bmp", "Bitmap Image File" });
            filters.push_back({ "tga", "Truevision Graphics Adapter" });
        }
        return filters;
    }

    std::string Bitmap::getFilExtFromResourceFormat(ResourceFormat format)
    {
        auto filters = getFileDialogFilters(format);
        return filters.front().ext;
    }

    void Bitmap::saveImageDialog(const Texture::SharedPtr& pTexture)
    {
        std::string filePath;
        auto supportExtensions = getFileDialogFilters(pTexture->getFormat());

        if (saveFileDialog(supportExtensions, filePath))
        {
            std::string ext = getExtensionFromFile(filePath);
            auto format = getFormatFromFileExtension(ext);
            pTexture->captureToFile(0, 0, filePath, format);
        }
    }

    void Bitmap::saveImage(const std::string& filename, uint32_t width, uint32_t height, FileFormat fileFormat, ExportFlags exportFlags, ResourceFormat resourceFormat, bool isTopDown, void* pData)
    {
        if(pData == nullptr)
        {
            logError("Bitmap::saveImage provided no data to save.");
            return;
        }
        
        if(is_set(exportFlags, ExportFlags::Uncompressed) && is_set(exportFlags, ExportFlags::Lossy))
        {
            logError("Bitmap::saveImage incompatible flags: lossy cannot be combined with uncompressed.");
            return;
        }

        int flags = 0;
        FIBITMAP* pImage = nullptr;
        uint32_t bytesPerPixel = getFormatBytesPerBlock(resourceFormat);

        //TODO replace this code for swapping channels. Can't use freeimage masks b/c they only care about 16 bpp images
        //issue #74 in gitlab
        if (resourceFormat == ResourceFormat::RGBA8Unorm || resourceFormat == ResourceFormat::RGBA8Snorm || resourceFormat == ResourceFormat::RGBA8UnormSrgb)
        {
            for (uint32_t a = 0; a < width*height; a++)
            {
                uint32_t* pPixel = (uint32_t*)pData;
                pPixel += a;
                uint8_t* ch = (uint8_t*)pPixel;
                std::swap(ch[0], ch[2]);
                if (is_set(exportFlags, ExportFlags::ExportAlpha) == false)
                {
                    ch[3] = 0xff;
                }
            }
        }

        if (fileFormat == Bitmap::FileFormat::PfmFile || fileFormat == Bitmap::FileFormat::ExrFile)
        {
            if(bytesPerPixel != 16 && bytesPerPixel != 12)
            {
                logError("Bitmap::saveImage supports only 32-bit/channel RGB/RGBA images as PFM/EXR files.");
                return;
            }

            const bool exportAlpha = is_set(exportFlags, ExportFlags::ExportAlpha);

            if(fileFormat == Bitmap::FileFormat::PfmFile)
            {
                if (is_set(exportFlags, ExportFlags::Lossy))
                {
                    logError("Bitmap::saveImage: PFM does not support lossy compression mode.");
                    return;
                }
                if (exportAlpha)
                {
                    logError("Bitmap::saveImage: PFM does not support alpha channel.");
                    return;
                }
            }

            if (exportAlpha && bytesPerPixel != 16)
            {
                logError("Bitmap::saveImage requesting to export alpha-channel to EXR file, but the resource doesn't have an alpha-channel");
                return;
            }

            // Upload the image manually and flip it vertically
            bool scanlineCopy = exportAlpha ? bytesPerPixel == 16 : bytesPerPixel == 12;

            pImage = FreeImage_AllocateT(exportAlpha ? FIT_RGBAF : FIT_RGBF, width, height);
            BYTE* head = (BYTE*)pData;
            for(unsigned y = 0; y < height; y++) 
            {
                float* dstBits = (float*)FreeImage_GetScanLine(pImage, height - y - 1);
                if(scanlineCopy)
                {
                    std::memcpy(dstBits, head, bytesPerPixel * width);
                }
                else
                {
                    assert(exportAlpha == false);
                    for(unsigned x = 0; x < width; x++) 
                    {
                        dstBits[x*3 + 0] = (((float*)head)[x*4 + 0]);
                        dstBits[x*3 + 1] = (((float*)head)[x*4 + 1]);
                        dstBits[x*3 + 2] = (((float*)head)[x*4 + 2]);
                    }
                }
                head += bytesPerPixel * width;
            }

            if(fileFormat == Bitmap::FileFormat::ExrFile)
            {
                flags = 0;
                if (is_set(exportFlags, ExportFlags::Uncompressed))
                {
                    flags |= EXR_NONE | EXR_FLOAT;
                }
                else if (is_set(exportFlags, ExportFlags::Lossy))
                {
                    flags |= EXR_B44 | EXR_ZIP;
                }
            }
        }
        else
        {
            FIBITMAP* pTemp = FreeImage_ConvertFromRawBits((BYTE*)pData, width, height, bytesPerPixel * width, bytesPerPixel * 8, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, isTopDown);
            if(is_set(exportFlags, ExportFlags::ExportAlpha) == false || fileFormat == Bitmap::FileFormat::JpegFile)
            {
                pImage = FreeImage_ConvertTo24Bits(pTemp);
                FreeImage_Unload(pTemp);
            }
            else
            {
                pImage = pTemp;
            }

            std::vector<std::string> warnings;
            switch(fileFormat)
            {
            case FileFormat::JpegFile:
                if (is_set(exportFlags, ExportFlags::Lossy) == false || is_set(exportFlags, ExportFlags::Uncompressed))
                {
                    flags = JPEG_QUALITYSUPERB | JPEG_SUBSAMPLING_444;
                }
                if (is_set(exportFlags, ExportFlags::ExportAlpha))
                {
                    warnings.push_back("JPEG format does not support alpha channel.");
                }
                break;

            // Lossless formats
            case FileFormat::PngFile:
                flags = is_set(exportFlags, ExportFlags::Uncompressed) ? PNG_Z_NO_COMPRESSION : PNG_Z_BEST_COMPRESSION;

                if (is_set(exportFlags, ExportFlags::Lossy))
                {
                    warnings.push_back("PNG format does not support lossy compression mode.");
                }
                break;

            case FileFormat::TgaFile:
                if (is_set(exportFlags, ExportFlags::Lossy))
                {
                    warnings.push_back("TGA format does not support lossy compression mode.");
                }
                break;

            case FileFormat::BmpFile:
                if (is_set(exportFlags, ExportFlags::Lossy))
                {
                    warnings.push_back("BMP format does not support lossy compression mode.");
                }
                if (is_set(exportFlags, ExportFlags::ExportAlpha))
                {
                    warnings.push_back("BMP format does not support alpha channel.");
                }
                break;

            default:
                should_not_get_here();
            }

            if(warnings.empty() == false)
            {
                logWarning("Bitmap::saveImage: " + joinStrings(warnings, " "));
            }
        }

        FreeImage_Save(toFreeImageFormat(fileFormat), pImage, filename.c_str(), flags);
        FreeImage_Unload(pImage);
    }
}
