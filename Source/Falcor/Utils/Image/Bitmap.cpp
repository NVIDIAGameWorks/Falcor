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
#include "Bitmap.h"
#include "Core/API/Texture.h"
#include "Utils/Logger.h"
#include "Utils/StringUtils.h"

#include <FreeImage.h>

namespace Falcor
{
    static bool isRGB32fSupported() { return false; } // FIX THIS

    static void genWarning(const std::string& errMsg, const std::filesystem::path& path)
    {
        logWarning("Error when loading image file from '{}': {}", path, errMsg);
    }

    static bool isConvertibleToRGBA32Float(ResourceFormat format)
    {
        FormatType type = getFormatType(format);
        bool isHalfFormat = (type == FormatType::Float && getNumChannelBits(format, 0) == 16);
        bool isLargeIntFormat = ((type == FormatType::Uint || type == FormatType::Sint) && getNumChannelBits(format, 0) >= 16);
        return isHalfFormat || isLargeIntFormat;
    }

    /** Converts half float image to RGBA float image.
    */
    static std::vector<float> convertHalfToRGBA32Float(uint32_t width, uint32_t height, uint32_t channelCount, const void* pData)
    {
        std::vector<float> newData(width * height * 4u, 0.f);
        const glm::detail::hdata* pSrc = reinterpret_cast<const glm::detail::hdata*>(pData);
        float* pDst = newData.data();

        for (uint32_t i = 0; i < width * height; ++i)
        {
            for (uint32_t c = 0; c < channelCount; ++c)
            {
                *pDst++ = glm::detail::toFloat32(*pSrc++);
            }
            pDst += (4 - channelCount);
        }

        return newData;
    }

    /** Converts integer image to RGBA float image.
        Unsigned integers are normalized to [0,1], signed integers to [-1,1].
    */
    template<typename SrcT>
    static std::vector<float> convertIntToRGBA32Float(uint32_t width, uint32_t height, uint32_t channelCount, const void* pData)
    {
        std::vector<float> newData(width * height * 4u, 0.f);
        const SrcT* pSrc = reinterpret_cast<const SrcT*>(pData);
        float* pDst = newData.data();

        for (uint32_t i = 0; i < width * height; ++i)
        {
            for (uint32_t c = 0; c < channelCount; ++c)
            {
                *pDst++ = float(*pSrc++) / float(std::numeric_limits<SrcT>::max());
            }
            pDst += (4 - channelCount);
        }

        return newData;
    }

    /** Converts an image of the given format to an RGBA float image.
    */
    static std::vector<float> convertToRGBA32Float(ResourceFormat format, uint32_t width, uint32_t height, const void* pData)
    {
        FALCOR_ASSERT(isConvertibleToRGBA32Float(format));

        FormatType type = getFormatType(format);
        uint32_t channelCount = getFormatChannelCount(format);
        uint32_t channelBits = getNumChannelBits(format, 0);

        std::vector<float> floatData;

        if (type == FormatType::Float && channelBits == 16)
        {
            floatData = convertHalfToRGBA32Float(width, height, channelCount, pData);
        }
        else if (type == FormatType::Uint && channelBits == 16)
        {
            floatData = convertIntToRGBA32Float<uint16_t>(width, height, channelCount, pData);
        }
        else if (type == FormatType::Uint && channelBits == 32)
        {
            floatData = convertIntToRGBA32Float<uint32_t>(width, height, channelCount, pData);
        }
        else if (type == FormatType::Sint && channelBits == 16)
        {
            floatData = convertIntToRGBA32Float<int16_t>(width, height, channelCount, pData);
        }
        else if (type == FormatType::Sint && channelBits == 32)
        {
            floatData = convertIntToRGBA32Float<int32_t>(width, height, channelCount, pData);
        }
        else
        {
            FALCOR_UNREACHABLE();
        }

        // Default alpha channel to 1.
        if (channelCount < 4)
        {
            for (uint32_t i = 0; i < width * height; ++i) floatData[i * 4 + 3] = 1.f;
        }

        return floatData;
    }

    /** Converts 96bpp to 128bpp RGBA without clamping.
        Note that we can't use FreeImage_ConvertToRGBAF() as it clamps to [0,1].
    */
    static FIBITMAP* convertToRGBAF(FIBITMAP* pDib)
    {
        const unsigned width = FreeImage_GetWidth(pDib);
        const unsigned height = FreeImage_GetHeight(pDib);

        auto pNew = FreeImage_AllocateT(FIT_RGBAF, width, height);
        FreeImage_CloneMetadata(pNew, pDib);

        const unsigned src_pitch = FreeImage_GetPitch(pDib);
        const unsigned dst_pitch = FreeImage_GetPitch(pNew);

        const BYTE *src_bits = (BYTE*)FreeImage_GetBits(pDib);
        BYTE* dst_bits = (BYTE*)FreeImage_GetBits(pNew);

        for (unsigned y = 0; y < height; y++)
        {
            const FIRGBF *src_pixel = (FIRGBF*)src_bits;
            FIRGBAF* dst_pixel = (FIRGBAF*)dst_bits;

            for (unsigned x = 0; x < width; x++)
            {
                // Convert pixels directly, while adding a "dummy" alpha of 1.0
                dst_pixel[x].red = src_pixel[x].red;
                dst_pixel[x].green = src_pixel[x].green;
                dst_pixel[x].blue = src_pixel[x].blue;
                dst_pixel[x].alpha = 1.0F;

            }
            src_bits += src_pitch;
            dst_bits += dst_pitch;
        }
        return pNew;
    }

    Bitmap::UniqueConstPtr Bitmap::create(uint32_t width, uint32_t height, ResourceFormat format, const uint8_t* pData)
    {
        return Bitmap::UniqueConstPtr(new Bitmap(width, height, format, pData));
    }

    Bitmap::UniqueConstPtr Bitmap::createFromFile(const std::filesystem::path& path, bool isTopDown)
    {
        std::filesystem::path fullPath;
        if (!findFileInDataDirectories(path, fullPath))
        {
            logWarning("Error when loading image file. Can't find image file '{}'.", path);
            return nullptr;
        }

        FREE_IMAGE_FORMAT fifFormat = FIF_UNKNOWN;

        fifFormat = FreeImage_GetFileType(fullPath.string().c_str(), 0);
        if (fifFormat == FIF_UNKNOWN)
        {
            // Can't get the format from the file. Use file extension
            fifFormat = FreeImage_GetFIFFromFilename(fullPath.string().c_str());

            if (fifFormat == FIF_UNKNOWN)
            {
                genWarning("Image type unknown", path);
                return nullptr;
            }
        }

        // Check the library supports loading this image type
        if (FreeImage_FIFSupportsReading(fifFormat) == false)
        {
            genWarning("Library doesn't support the file format", path);
            return nullptr;
        }

        // Read the DIB
        FIBITMAP* pDib = FreeImage_Load(fifFormat, fullPath.string().c_str());
        if (pDib == nullptr)
        {
            genWarning("Can't read image file", path);
            return nullptr;
        }

        // Create the bitmap
        const uint32_t height = FreeImage_GetHeight(pDib);
        const uint32_t width = FreeImage_GetWidth(pDib);

        if (height == 0 || width == 0 || FreeImage_GetBits(pDib) == nullptr)
        {
            genWarning("Invalid image", path);
            return nullptr;
        }

        // Convert palettized images to RGBA.
        FREE_IMAGE_COLOR_TYPE colorType = FreeImage_GetColorType(pDib);
        if (colorType == FIC_PALETTE)
        {
            auto pNew = FreeImage_ConvertTo32Bits(pDib);
            FreeImage_Unload(pDib);
            pDib = pNew;

            if (pDib == nullptr)
            {
                genWarning("Failed to convert palettized image to RGBA format", path);
                return nullptr;
            }
        }

        ResourceFormat format = ResourceFormat::Unknown;
        uint32_t bpp = FreeImage_GetBPP(pDib);
        switch(bpp)
        {
        case 128:
            format = ResourceFormat::RGBA32Float;    // 4xfloat32 HDR format
            break;
        case 96:
            format = isRGB32fSupported() ? ResourceFormat::RGB32Float : ResourceFormat::RGBA32Float;     // 3xfloat32 HDR format
            break;
        case 64:
            format = ResourceFormat::RGBA16Float;    // 4xfloat16 HDR format
            break;
        case 48:
            format = ResourceFormat::RGB16Float;     // 3xfloat16 HDR format
            break;
        case 32:
            format = ResourceFormat::BGRA8Unorm;
            break;
        case 24:
            format = ResourceFormat::BGRX8Unorm;
            break;
        case 16:
            format = (FreeImage_GetImageType(pDib) == FIT_UINT16) ? ResourceFormat::R16Unorm : ResourceFormat::RG8Unorm;
            break;
        case 8:
            format = ResourceFormat::R8Unorm;
            break;
        default:
            genWarning("Unknown bits-per-pixel", path);
            return nullptr;
        }

        // Convert the image to RGBX image
        if (bpp == 24)
        {
            bpp = 32;
            auto pNew = FreeImage_ConvertTo32Bits(pDib);
            FreeImage_Unload(pDib);
            pDib = pNew;
        }
        else if (bpp == 96 && (isRGB32fSupported() == false))
        {
            bpp = 128;
            auto pNew = convertToRGBAF(pDib);
            FreeImage_Unload(pDib);
            pDib = pNew;
        }

        // PFM images are loaded y-flipped, fix this by inverting the isTopDown flag.
        if (fifFormat == FIF_PFM) isTopDown = !isTopDown;

        UniqueConstPtr pBmp = UniqueConstPtr(new Bitmap(width, height, format));
        FreeImage_ConvertToRawBits(pBmp->getData(), pDib, pBmp->getRowPitch(), bpp, FI_RGBA_RED_MASK, FI_RGBA_GREEN_MASK, FI_RGBA_BLUE_MASK, isTopDown);
        FreeImage_Unload(pDib);
        return pBmp;
    }

    Bitmap::Bitmap(uint32_t width, uint32_t height, ResourceFormat format)
        : mWidth(width)
        , mHeight(height)
        , mFormat(format)
        , mRowPitch(getFormatRowPitch(format, width))
    {
        if (isCompressedFormat(format))
        {
            uint32_t blockSizeY = getFormatHeightCompressionRatio(format);
            FALCOR_ASSERT(height % blockSizeY == 0); // Should divide evenly
            mSize = mRowPitch * (height / blockSizeY);
        }
        else
        {
            mSize = height * mRowPitch;
        }

        mpData = std::unique_ptr<uint8_t[]>(new uint8_t[mSize]);
    }

    Bitmap::Bitmap(uint32_t width, uint32_t height, ResourceFormat format, const uint8_t* pData)
        : Bitmap(width, height, format)
    {
        std::memcpy(mpData.get(), pData, mSize);
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
            FALCOR_UNREACHABLE();
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
            FALCOR_UNREACHABLE();
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
            /* ExrFile */ "exr",
            /* DdsFile */ "dds"
        };

        for (size_t i = 0 ; i < std::size(kExtensions) ; i++)
        {
            if (kExtensions[i] == ext) return Bitmap::FileFormat(i);
        }
        reportError("Can't find a matching format for file extension '" + ext + "'");
        return Bitmap::FileFormat(-1);
    }

    FileDialogFilterVec Bitmap::getFileDialogFilters(ResourceFormat format)
    {
        FileDialogFilterVec filters;
        bool showHdr = true;
        bool showLdr = true;

        if (format != ResourceFormat::Unknown)
        {
            // Save float, half and large integer (16/32 bit) formats as HDR.
            showHdr = getFormatType(format) == FormatType::Float || isConvertibleToRGBA32Float(format);
            showLdr = !showHdr;
        }

        if (showHdr)
        {
            filters.push_back({ "exr", "High Dynamic Range" });
            filters.push_back({ "pfm", "Portable Float Map" });
            filters.push_back({ "hdr", "Radiance HDR" });
        }

        if (showLdr)
        {
            filters.push_back({ "png", "Portable Network Graphics" });
            filters.push_back({ "jpg", "JPEG" });
            filters.push_back({ "bmp", "Bitmap Image File" });
            filters.push_back({ "tga", "Truevision Graphics Adapter" });
        }

        // DDS can store all formats
        filters.push_back({ "dds", "DirectDraw Surface" });

        // List of formats we can only load from
        if (format == ResourceFormat::Unknown)
        {
            filters.push_back({ "hdr", "High Dynamic Range" });
        }
        return filters;
    }

    std::string Bitmap::getFileExtFromResourceFormat(ResourceFormat format)
    {
        auto filters = getFileDialogFilters(format);
        return filters.front().ext;
    }

    void Bitmap::saveImageDialog(Texture* pTexture)
    {
        std::filesystem::path path;
        auto supportExtensions = getFileDialogFilters(pTexture->getFormat());

        if (saveFileDialog(supportExtensions, path))
        {
            std::string ext = getExtensionFromPath(path);
            auto format = getFormatFromFileExtension(ext);
            pTexture->captureToFile(0, 0, path, format);
        }
    }

    void Bitmap::saveImage(const std::filesystem::path& path, uint32_t width, uint32_t height, FileFormat fileFormat, ExportFlags exportFlags, ResourceFormat resourceFormat, bool isTopDown, void* pData)
    {
        if (pData == nullptr)
        {
            reportError("Bitmap::saveImage provided no data to save.");
            return;
        }

        if (is_set(exportFlags, ExportFlags::Uncompressed) && is_set(exportFlags, ExportFlags::Lossy))
        {
            reportError("Bitmap::saveImage incompatible flags: lossy cannot be combined with uncompressed.");
            return;
        }

        if (fileFormat == FileFormat::DdsFile)
        {
            reportError("Bitmap::saveImage cannot save DDS files. Use ImageIO instead.");
            return;
        }

        int flags = 0;
        FIBITMAP* pImage = nullptr;
        uint32_t bytesPerPixel = getFormatBytesPerBlock(resourceFormat);

        // Convert 8-bit RGBA to BGRA byte order.
        // TODO: Replace this code for swapping channels. Can't use FreeImage masks b/c they only care about 16 bpp images.
        if (resourceFormat == ResourceFormat::RGBA8Unorm || resourceFormat == ResourceFormat::RGBA8Snorm || resourceFormat == ResourceFormat::RGBA8UnormSrgb)
        {
            for (uint32_t a = 0; a < width * height; a++)
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
            std::vector<float> floatData;
            if (isConvertibleToRGBA32Float(resourceFormat))
            {
                floatData = convertToRGBA32Float(resourceFormat, width, height, pData);
                pData = floatData.data();
                resourceFormat = ResourceFormat::RGBA32Float;
                bytesPerPixel = 16;
            }
            else if (bytesPerPixel != 16 && bytesPerPixel != 12)
            {
                reportError("Bitmap::saveImage supports only 32-bit/channel RGB/RGBA or 16-bit RGBA images as PFM/EXR files.");
                return;
            }

            const bool exportAlpha = is_set(exportFlags, ExportFlags::ExportAlpha);

            if (fileFormat == Bitmap::FileFormat::PfmFile)
            {
                if (is_set(exportFlags, ExportFlags::Lossy))
                {
                    reportError("Bitmap::saveImage: PFM does not support lossy compression mode.");
                    return;
                }
                if (exportAlpha)
                {
                    reportError("Bitmap::saveImage: PFM does not support alpha channel.");
                    return;
                }
            }

            if (exportAlpha && bytesPerPixel != 16)
            {
                reportError("Bitmap::saveImage requesting to export alpha-channel to EXR file, but the resource doesn't have an alpha-channel");
                return;
            }

            // Upload the image manually and flip it vertically
            bool scanlineCopy = exportAlpha ? bytesPerPixel == 16 : bytesPerPixel == 12;

            pImage = FreeImage_AllocateT(exportAlpha ? FIT_RGBAF : FIT_RGBF, width, height);
            BYTE* head = (BYTE*)pData;
            for (unsigned y = 0; y < height; y++)
            {
                float* dstBits = (float*)FreeImage_GetScanLine(pImage, height - y - 1);
                if (scanlineCopy)
                {
                    std::memcpy(dstBits, head, bytesPerPixel * width);
                }
                else
                {
                    FALCOR_ASSERT(exportAlpha == false);
                    for (unsigned x = 0; x < width; x++)
                    {
                        dstBits[x*3 + 0] = (((float*)head)[x*4 + 0]);
                        dstBits[x*3 + 1] = (((float*)head)[x*4 + 1]);
                        dstBits[x*3 + 2] = (((float*)head)[x*4 + 2]);
                    }
                }
                head += bytesPerPixel * width;
            }

            if (fileFormat == Bitmap::FileFormat::ExrFile)
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
            if (is_set(exportFlags, ExportFlags::ExportAlpha) == false || fileFormat == Bitmap::FileFormat::JpegFile)
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
                FALCOR_UNREACHABLE();
            }

            if (warnings.empty() == false)
            {
                logWarning("Bitmap::saveImage: {}", joinStrings(warnings, " "));
            }
        }

        if (!FreeImage_Save(toFreeImageFormat(fileFormat), pImage, path.string().c_str(), flags))
        {
            reportError("Bitmap::saveImage: FreeImage failed to save image");
        }
        FreeImage_Unload(pImage);
    }
}
