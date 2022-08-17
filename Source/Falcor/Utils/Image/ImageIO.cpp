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
#include "ImageIO.h"
#include "Core/Errors.h"
#include "Core/API/CopyContext.h"
#include "Utils/Logger.h"

#include <dds_header/DDSHeader.h>
#include <nvtt/nvtt.h>

#include <filesystem>
#include <fstream>

namespace Falcor
{
    namespace
    {
        struct ImportData
        {
            // Commonly used values converted or casted for cleaner access
            ResourceFormat format;
            Resource::Type type;
            uint32_t width;
            uint32_t height;
            uint32_t depth;
            uint32_t arraySize;
            uint32_t mipLevels;
            bool hasDX10Header = false;

            // Data to be imported
            std::vector<uint8_t> imageData;
        };

        struct ExportData
        {
            // Commonly used values converted or casted for cleaner access
            nvtt::TextureType type;
            ResourceFormat format;
            uint32_t width;
            uint32_t height;
            uint32_t depth;
            uint32_t faceCount;
            uint32_t mipLevels;

            // Data to be exported
            std::vector<nvtt::Surface> images;
        };

        ImageIO::CompressionMode convertFormatToMode(ResourceFormat format)
        {
            switch (format)
            {
            case ResourceFormat::BC1Unorm:
            case ResourceFormat::BC1UnormSrgb:
                return ImageIO::CompressionMode::BC1;
            case ResourceFormat::BC2Unorm:
            case ResourceFormat::BC2UnormSrgb:
                return ImageIO::CompressionMode::BC2;
            case ResourceFormat::BC3Unorm:
            case ResourceFormat::BC3UnormSrgb:
                return ImageIO::CompressionMode::BC3;
            case ResourceFormat::BC4Unorm:
                return ImageIO::CompressionMode::BC4;
            case ResourceFormat::BC5Snorm:
            case ResourceFormat::BC5Unorm:
                return ImageIO::CompressionMode::BC5;
            case ResourceFormat::BC6HS16:
                return ImageIO::CompressionMode::BC6;
            case ResourceFormat::BC7Unorm:
            case ResourceFormat::BC7UnormSrgb:
                return ImageIO::CompressionMode::BC7;
            default:
                throw RuntimeError("No corresponding compression mode for the provided ResourceFormat.");
            }
        }

        // Returns the corresponding NVTT compression format for the provided compression mode.
        nvtt::Format convertModeToNvttFormat(ImageIO::CompressionMode mode)
        {
            switch (mode)
            {
            case ImageIO::CompressionMode::None:
                return nvtt::Format::Format_RGBA;
            case ImageIO::CompressionMode::BC1:
                return nvtt::Format::Format_BC1;
            case ImageIO::CompressionMode::BC2:
                return nvtt::Format::Format_BC2;
            case ImageIO::CompressionMode::BC3:
                return nvtt::Format::Format_BC3;
            case ImageIO::CompressionMode::BC4:
                return nvtt::Format::Format_BC4;
            case ImageIO::CompressionMode::BC5:
                return nvtt::Format::Format_BC5;
            case ImageIO::CompressionMode::BC6:
                return nvtt::Format::Format_BC6S;
            case ImageIO::CompressionMode::BC7:
                return nvtt::Format::Format_BC7;
            default:
                throw RuntimeError("Invalid compression mode.");
            }
        }

        // Returns the corresponding NVTT compression format for the provided ResourceFormat. This conversion function should be used to convert formats for compressed textures.
        nvtt::Format convertFormatToNvttFormat(ResourceFormat format)
        {
            switch (format)
            {
            case ResourceFormat::BC1Unorm:
            case ResourceFormat::BC1UnormSrgb:
                return nvtt::Format::Format_BC1;
            case ResourceFormat::BC2Unorm:
            case ResourceFormat::BC2UnormSrgb:
                return nvtt::Format::Format_BC2;
            case ResourceFormat::BC3Unorm:
            case ResourceFormat::BC3UnormSrgb:
                return nvtt::Format::Format_BC3;
            case ResourceFormat::BC4Unorm:
                return nvtt::Format::Format_BC4;
            case ResourceFormat::BC5Snorm:
            case ResourceFormat::BC5Unorm:
                return nvtt::Format::Format_BC5;
            case ResourceFormat::BC6HS16:
                return nvtt::Format::Format_BC6S;
            case ResourceFormat::BC7Unorm:
            case ResourceFormat::BC7UnormSrgb:
                return nvtt::Format::Format_BC7;
            default:
                throw RuntimeError("No corresponding NVTT compression format for the specified ResourceFormat.");
            }
        }

        // Returns the corresponding NVTT input format for the provided ResourceFormat. Should only be used to convert formats for non-compressed textures.
        nvtt::InputFormat convertToNvttInputFormat(ResourceFormat format)
        {
            // Special case for R32FloatX32 as this is otherwise indistinguishable from RG32Float
            if (format == ResourceFormat::R32FloatX32)
            {
                throw RuntimeError("Image is in an unsupported ResourceFormat.");
            }

            uint32_t channelCount = getFormatChannelCount(format);
            uint32_t xBits = getNumChannelBits(format, 0);
            uint32_t yBits = getNumChannelBits(format, 1);
            uint32_t zBits = getNumChannelBits(format, 2);
            uint32_t wBits = getNumChannelBits(format, 3);

            bool isR32Float = channelCount == 1 && xBits == 32;
            bool isSupportedTwoChannel = channelCount == 2 && xBits == yBits; // all RG formats
            bool isSupportedThreeChannel = channelCount == 3 && xBits == yBits && yBits == zBits; // all RGB formats
            bool isSupportedFourChannel = xBits == yBits && yBits == zBits && zBits == wBits;

            // These are fairly broadly sorted into the five NVTT input formats. Most resource formats will require
            // modifications to the data before being passed to NVTT for exporting; this is done later on in setImage().
            if (isR32Float || isSupportedTwoChannel || isSupportedThreeChannel || isSupportedFourChannel)
            {
                if (isSupportedThreeChannel)
                {
                    logWarning("NVTT is incompatible with three channel images. This image will be padded with a solid alpha channel.");
                }

                if (isR32Float)
                {
                    return nvtt::InputFormat::InputFormat_R_32F;
                }

                if (xBits == 8)
                {
                    if (getFormatType(format) == FormatType::Uint || getFormatType(format) == FormatType::Unorm)
                    {
                        return nvtt::InputFormat::InputFormat_BGRA_8UB;
                    }
                    else return nvtt::InputFormat::InputFormat_BGRA_8SB;
                }
                else if (xBits == 16) return nvtt::InputFormat::InputFormat_RGBA_16F;
                else if (xBits == 32) return nvtt::InputFormat::InputFormat_RGBA_32F;
            }

            throw RuntimeError("Image is in an unsupported ResourceFormat.");
        }

        // Check if any of base image dimensions need to be clamped to a multiple of 4.
        // This function should only be called if the image is being compressed and mipmaps are being automatically generated.
        bool clampIfNeeded(ExportData& image)
        {
            bool clamped = false;
            if (image.width > 1u && image.width % 4 != 0)
            {
                image.width = std::max(1u, image.width - image.width % 4);
                clamped = true;
            }
            if (image.height > 1u && image.height % 4 != 0)
            {
                image.height = std::max(1u, image.height - image.height % 4);
                clamped = true;
            }
            if (image.depth > 1u && image.depth % 4 != 0)
            {
                image.depth = std::max(1u, image.depth - image.depth % 4);
                clamped = true;
            }

            return clamped;
        }

        // Fill the alpha channel with 1's.
        void fillAlphaChannel(nvtt::Surface& image)
        {
            // Create a dummy Surface and fill with 1's then copy the alpha channel. DirectXTex fills the alpha channel
            // with 0's for images that do not have an alpha, but NVTT does not have an equivalent alpha-less InputFormat.
            // The alpha channel must thus be manually filled with 1's otherwise the resulting image may not display
            // properly. BGRX8 is a unique case that it is a four channel format with no alpha.
            nvtt::Surface alpha(image);
            alpha.fill(1.0, 1.0, 1.0, 1.0);
            image.copyChannel(alpha, 3, 3);
        }

        // Prepare the original image data for being passed to NVTT for exporting. Certain image formats will also need
        // the data to be modified to include empty blue and/or solid alpha channels. This is because NVTT only supports
        // five specific input formats: 8-bit unsigned BGRA, 8-bit signed BGRA, 16-bit floating point RGBA,
        // 32-bit floating point RGBA, and single channel 32-bit floating point.
        //
        // NVTT's Surface always holds a single image's worth of UNCOMPRESSED data. Re-compression is necessary
        // if image compression needs to be maintained.
        template <typename T>
        void setImage(const void* subresourceData, nvtt::Surface& surface, ExportData image, uint32_t srcWidth, uint32_t srcHeight, uint32_t srcDepth)
        {
            std::vector<T> modified;
            uint32_t pixelCount = srcWidth * srcHeight * srcDepth;
            uint32_t channelCount = getFormatChannelCount(image.format);
            T alpha = 0;

            // Need to flip red and blue channels for all 8 bit formats that aren't BGRA/BGRX as NVTT only supports BGRA inputs for these cases
            bool reverseRB = getNumChannelBits(image.format, 0) == 8 && image.format != ResourceFormat::BGRA8Unorm && image.format != ResourceFormat::BGRA8UnormSrgb
                && image.format != ResourceFormat::BGRX8Unorm && image.format != ResourceFormat::BGRX8UnormSrgb;
            // Need to fill the alpha channel with 1's for all formats that do not have an alpha channel
            bool fillAlpha = channelCount == 2 || channelCount == 3 || image.format == ResourceFormat::BGRX8Unorm || image.format == ResourceFormat::BGRX8UnormSrgb;

            modified.resize(4 * pixelCount);

            T* src = (T*)subresourceData;
            T* dst = (T*)modified.data();
            for (uint32_t h = 0; h < image.height; ++h)
            {
                for (uint32_t w = 0; w < image.width; ++w)
                {
                    uint32_t i = h * srcWidth + w; // Source data index
                    uint32_t j = h * image.width + w; // Destination data index - Same as source index if no clamping is involved
                    if (channelCount == 1)
                    {
                        dst[j] = src[i];
                    }
                    else if (channelCount == 2)
                    {
                        dst[4 * j] = reverseRB ? 0 : src[2 * i];
                        dst[4 * j + 1] = src[2 * i + 1];
                        dst[4 * j + 2] = reverseRB ? src[2 * i] : 0;
                        dst[4 * j + 3] = 0;
                    }
                    else if (channelCount == 3)
                    {
                        dst[4 * j] = reverseRB ? src[3 * i + 2] : src[3 * i];
                        dst[4 * j + 1] = src[3 * i + 1];
                        dst[4 * j + 2] = reverseRB ? src[3 * i] : src[3 * i + 2];
                        dst[4 * j + 3] = 0;
                    }
                    else if (channelCount == 4)
                    {
                        dst[4 * j] = reverseRB ? src[4 * i + 2] : src[4 * i];
                        dst[4 * j + 1] = src[4 * i + 1];
                        dst[4 * j + 2] = reverseRB ? src[4 * i] : src[4 * i + 2];
                        dst[4 * j + 3] = src[4 * i + 3];
                    }
                }
            }

            if (isCompressedFormat(image.format))
            {
                nvtt::Format compressionFormat = convertFormatToNvttFormat(image.format);
                if (!surface.setImage3D(compressionFormat, (int)image.width, (int)image.height, (int)image.depth, modified.data()))
                {
                    throw RuntimeError("Failed to set image data.");
                }
            }
            else
            {
                nvtt::InputFormat inputFormat = convertToNvttInputFormat(image.format);
                if (!surface.setImage(inputFormat, (int)image.width, (int)image.height, (int)image.depth, modified.data()))
                {
                    throw RuntimeError("Failed to set image data.");
                }
            }

            if (fillAlpha) fillAlphaChannel(surface);
        }

        // Saves image data to a DDS file using the specified compression mode. Optionally generates mips.
        void exportDDS(const std::filesystem::path& path, ExportData& image, ImageIO::CompressionMode mode, bool generateMips)
        {
            nvtt::CompressionOptions compressionOptions;
            nvtt::Format format = convertModeToNvttFormat(mode);
            compressionOptions.setFormat(format);
            if (format == nvtt::Format::Format_RGBA && !isCompressedFormat(image.format))
            {
                if (getFormatType(image.format) == FormatType::Float)
                {
                    compressionOptions.setPixelType(nvtt::PixelType::PixelType_Float);
                    if (image.format == ResourceFormat::R32Float)
                    {
                        compressionOptions.setPixelFormat(32, 0, 0, 0);
                    }
                    else
                    {
                        uint32_t bits = getNumChannelBits(image.format, 0);
                        compressionOptions.setPixelFormat(bits, bits, bits, bits);
                    }
                }
            }
            else if (format == nvtt::Format::Format_BC6S)
            {
                compressionOptions.setPixelType(nvtt::PixelType::PixelType_Float);
            }

            nvtt::OutputOptions outputOptions;
            std::string pathStr = path.string();
            outputOptions.setFileName(pathStr.c_str());
            if (format == nvtt::Format::Format_BC6S || format == nvtt::Format::Format_BC7)
            {
                outputOptions.setContainer(nvtt::Container::Container_DDS10);
            }
            outputOptions.setSrgbFlag(isSrgbFormat(image.format));

            nvtt::Context context;
            if (!context.outputHeader(image.type, image.width, image.height, image.depth, image.mipLevels, image.images[0].isNormalMap(), compressionOptions, outputOptions))
            {
                throw RuntimeError("Failed to output file header.");
            }

            for (uint32_t f = 0; f < image.faceCount; ++f)
            {
                size_t faceIndex = f * image.mipLevels;
                nvtt::Surface tmp = image.images[faceIndex];
                if (!context.compress(tmp, f, 0, compressionOptions, outputOptions))
                {
                    throw RuntimeError("Failed to compress file.");
                }
                for (uint32_t m = 1; m < image.mipLevels; ++m)
                {
                    if (generateMips)
                    {
                        tmp.buildNextMipmap(nvtt::MipmapFilter::MipmapFilter_Box);
                    }
                    else
                    {
                        tmp = image.images[faceIndex + m];
                    }

                    if (!context.compress(tmp, f, m, compressionOptions, outputOptions))
                    {
                        throw RuntimeError("Failed to compress file.");
                    }
                }
            }
        }

        // Reads image information from the DDS header data contained in pHeaderData.
        void readDDSHeader(ImportData& data, const void* pHeaderData, size_t& headerSize, bool loadAsSrgb)
        {
            // Check magic number
            auto magic = *static_cast<const uint32_t*>(pHeaderData);
            if (magic != DDS_MAGIC)
            {
                throw RuntimeError("Unexpected magic number for a DDS file.");
            }

            // Check size fields for both the DDS_HEADER and DDS_PIXELFORMAT structs
            auto pHeader = reinterpret_cast<const DDS_HEADER*>(static_cast<const uint8_t*>(pHeaderData) + sizeof(uint32_t));
            if (pHeader->size != sizeof(DDS_HEADER) || pHeader->ddspf.size != sizeof(DDS_PIXELFORMAT))
            {
                throw RuntimeError("DDS header size mismatch.");
            }

            // Check for the presence of the extended DX10 header and fill in ImportData fields with their corresponding values
            data.mipLevels = (pHeader->mipMapCount == 0) ? 1 : pHeader->mipMapCount;
            auto pixelFormat = pHeader->ddspf;
            auto fourCC = pixelFormat.fourCC;
            if (fourCC == MAKEFOURCC('D', 'X', '1', '0'))
            {
                // DX10 header extension is present
                data.hasDX10Header = true;
                if (headerSize != sizeof(uint32_t) + sizeof(DDS_HEADER) + sizeof(DDS_HEADER_DXT10))
                {
                    throw RuntimeError("DX10 header extension size mismatch.");
                }
                auto pDX10Header = reinterpret_cast<const DDS_HEADER_DXT10*>(static_cast<const uint8_t*>(pHeaderData) + sizeof(uint32_t) + sizeof(DDS_HEADER));
                data.arraySize = pDX10Header->arraySize;
                if (data.arraySize == 0)
                {
                    throw RuntimeError("Array size cannot be zero.");
                }
                data.format = getResourceFormat(pDX10Header->dxgiFormat);
                switch (pDX10Header->resourceDimension)
                {
                case DDS_DIMENSION_TEXTURE1D:
                    data.width = pHeader->width;
                    data.height = 1;
                    data.depth = 1;
                    data.type = Resource::Type::Texture1D;
                    break;
                case DDS_DIMENSION_TEXTURE2D:
                    data.width = pHeader->width;
                    data.height = pHeader->height;
                    data.depth = 1;
                    if (pDX10Header->miscFlag && DDS_RESOURCE_MISC_TEXTURECUBE)
                    {
                        data.type = Resource::Type::TextureCube;
                        data.arraySize *= 6;
                    }
                    else
                    {
                        data.type = Resource::Type::Texture2D;
                    }
                    break;
                case DDS_DIMENSION_TEXTURE3D:
                    data.width = pHeader->width;
                    data.height = pHeader->height;
                    data.depth = pHeader->depth;
                    data.type = Resource::Type::Texture3D;
                    break;
                default:
                    throw RuntimeError("Unsupported texture dimension.");
                }
            }
            else
            {
                // DX10 header extension is not present
                headerSize -= sizeof(DDS_HEADER_DXT10);
                data.arraySize = 1;

                if (pHeader->flags & DDS_HEADER_FLAGS_VOLUME)
                {
                    data.width = pHeader->width;
                    data.height = pHeader->height;
                    data.depth = pHeader->depth;
                    data.type = Resource::Type::Texture3D;
                }
                else
                {
                    data.width = pHeader->width;
                    data.height = pHeader->height;
                    data.depth = 1;
                    if (pHeader->caps2 & DDS_CUBEMAP)
                    {
                        if (!(pHeader->caps2 & DDS_CUBEMAP_ALLFACES))
                        {
                            throw RuntimeError("All six faces must be defined for a legacy D3D9 DDS texture cube.");
                        }
                        data.arraySize *= 6;
                        data.type = Resource::Type::TextureCube;
                    }
                    else
                    {
                        data.type = Resource::Type::Texture2D;
                    }
                }

                data.format = getResourceFormat(GetDXGIFormat(pixelFormat));
            }

            if (loadAsSrgb)
            {
                data.format = linearToSrgbFormat(data.format);
            }
        }

        // Loads the information and data for the specified image. This function does not handle creation of the texture for the image.
        void loadDDS(const std::filesystem::path& path, bool loadAsSrgb, ImportData& data)
        {
            std::ifstream file(path, std::ios::in | std::ios::binary | std::ios::ate);
            if (!file)
            {
                throw RuntimeError("Failed to open file.");
            }

            // Check image file length
            std::streampos fileLen = file.tellg();
            if (file.fail())
            {
                throw RuntimeError("Failed to read stream position.");
            }
            file.seekg(0, std::ios::beg);
            if (file.fail())
            {
                throw RuntimeError("Failed to set stream position.");
            }
            size_t filesize = fileLen;
            if (filesize < (sizeof(uint32_t) + sizeof(DDS_HEADER)))
            {
                throw RuntimeError("Failed to read DDS header (file too small).");
            }

            // Read the DDS header
            const size_t maxHeaderSize = sizeof(uint32_t) + sizeof(DDS_HEADER) + sizeof(DDS_HEADER_DXT10);
            uint8_t header[maxHeaderSize] = {};
            size_t headerSize = maxHeaderSize;

            // The actual header size may be smaller than the max size; be sure not to read past the end of the file.
            file.read(reinterpret_cast<char*>(header), std::min<size_t>(filesize, headerSize));
            if (file.fail())
            {
                throw RuntimeError("Failed to read DDS header.");
            }

            readDDSHeader(data, header, headerSize, loadAsSrgb);

            // Save the rest of the data after the header
            if (filesize <= headerSize)
            {
                throw RuntimeError("No image data after DDS header.");
            }

            size_t imageSize = filesize - headerSize;
            data.imageData.resize(imageSize);
            file.seekg(headerSize, std::ios::beg);
            if (file.fail())
            {
                throw RuntimeError("Failed to set stream position.");
            }
            file.read(reinterpret_cast<char*>(data.imageData.data()), imageSize);
            if (file.fail())
            {
                throw RuntimeError("Failed to read image data.");
            }
        }
    }

    Bitmap::UniqueConstPtr ImageIO::loadBitmapFromDDS(const std::filesystem::path& path)
    {
        std::filesystem::path fullPath;
        if (!findFileInDataDirectories(path, fullPath))
        {
            logWarning("Failed to load DDS image from '{}': Can't find file.", path);
            return nullptr;
        }

        ImportData data;
        try
        {
            loadDDS(fullPath, false, data);
        }
        catch (const RuntimeError& e)
        {
            logWarning("Failed to load DDS image from '{}': {}", path, e.what());
            return nullptr;
        }

        if (data.type == Resource::Type::Texture3D || data.type == Resource::Type::TextureCube)
        {
            logWarning("Failed to load DDS image from '{}': Invalid resource type {}.", path, to_string(data.type));
            return nullptr;
        }

        // Create from first image
        return Bitmap::create(data.width, data.height, data.format, data.imageData.data());
    }

    Texture::SharedPtr ImageIO::loadTextureFromDDS(const std::filesystem::path& path, bool loadAsSrgb)
    {
        std::filesystem::path fullPath;
        if (!findFileInDataDirectories(path, fullPath))
        {
            logWarning("Failed to load DDS image from '{}': Can't find file.", path);
            return nullptr;
        }

        ImportData data;
        try
        {
            loadDDS(fullPath, loadAsSrgb, data);
        }
        catch (const RuntimeError& e)
        {
            logWarning("Failed to load DDS image from '{}': {}", path, e.what());
            return nullptr;
        }

        Texture::SharedPtr pTex;
        // TODO: Automatic mip generation
        switch (data.type)
        {
        case Resource::Type::Texture1D:
            pTex = Texture::create1D(data.width, data.format, data.arraySize, data.mipLevels, data.imageData.data());
            break;
        case Resource::Type::Texture2D:
            pTex = Texture::create2D(data.width, data.height, data.format, data.arraySize, data.mipLevels, data.imageData.data());
            break;
        case Resource::Type::TextureCube:
            pTex = Texture::createCube(data.width, data.height, data.format, data.arraySize / 6, data.mipLevels, data.imageData.data());
            break;
        case Resource::Type::Texture3D:
            pTex = Texture::create3D(data.width, data.height, data.depth, data.format, data.mipLevels, data.imageData.data());
            break;
        default:
            logWarning("Failed to load DDS image from '{}': Unrecognized texture type.", path);
            return nullptr;
        }

        if (pTex != nullptr)
        {
            pTex->setSourcePath(fullPath);
        }

        return pTex;
    }

    void ImageIO::saveToDDS(const std::filesystem::path& path, const Bitmap& bitmap, CompressionMode mode, bool generateMips)
    {
        if (!hasExtension(path, "dds"))
        {
            logWarning("Saving DDS image to '{}' which does not have 'dds' file extension.", path);
        }

        try
        {
            ExportData image;
            image.type = nvtt::TextureType::TextureType_2D;
            image.width = bitmap.getWidth();
            image.height = bitmap.getHeight();
            image.depth = 1;
            image.format = bitmap.getFormat();
            image.faceCount = 1;
            image.mipLevels = generateMips ? nvtt::countMipmaps(image.width, image.height, image.depth) : 1;

            if (getFormatChannelCount(image.format) == 2 && mode != CompressionMode::BC5)
            {
                throw RuntimeError("Only BC5 compression is supported for two channel images.");
            }

            // The DX spec requires the dimensions of BC encoded textures to be a multiple of 4 at the base resolution.
            // If the texture has already been rescaled to meet this requirement, skip clamping.
            if (generateMips && (mode != CompressionMode::None))
            {
                bool clamped = clampIfNeeded(image);
                if (clamped)
                {
                    logWarning("Saving DDS image to '{}' with clamped image dimensions to accomodate mipmaps and compression.", path);
                }
            }

            uint32_t srcWidth = bitmap.getWidth();
            uint32_t srcHeight = bitmap.getHeight();

            nvtt::Surface surface;
            FormatType type = getFormatType(image.format);
            if (type == FormatType::Sint || type == FormatType::Snorm)
            {
                setImage<int8_t>(bitmap.getData(), surface, image, srcWidth, srcHeight, image.depth);
            }
            else if (type == FormatType::Uint || type == FormatType::Unorm || type == FormatType::UnormSrgb)
            {
                setImage<uint8_t>(bitmap.getData(), surface, image, srcWidth, srcHeight, image.depth);
            }
            else if (type == FormatType::Float)
            {
                if (getNumChannelBits(image.format, 0) == 16)
                {
                    setImage<glm::detail::hdata>(bitmap.getData(), surface, image, srcWidth, srcHeight, image.depth);
                }
                else if (getNumChannelBits(image.format, 0) == 32)
                {
                    setImage<float>(bitmap.getData(), surface, image, srcWidth, srcHeight, image.depth);
                }
            }

            image.images.push_back(surface);

            // NVTT's Surface is designed to only hold uncompressed data, which means saving a compressed image as-is
            // requires the data be re-compressed. The selected compression mode is updated here to reflect this.
            if (isCompressedFormat(image.format) && mode == CompressionMode::None)
            {
                mode = convertFormatToMode(image.format);
            }

            exportDDS(path, image, mode, generateMips);
        }
        catch (const RuntimeError& e)
        {
            throw RuntimeError("Failed to save DDS image to '{}': {}", path, e.what());
        }
    }

    void ImageIO::saveToDDS(CopyContext* pContext, const std::filesystem::path& path, const Texture::SharedPtr& pTexture, CompressionMode mode, bool generateMips)
    {
        if (!hasExtension(path, "dds"))
        {
            logWarning("Saving DDS image to '{}' which does not have 'dds' file extension.", path);
        }

        try
        {
            ExportData image;
            image.width = pTexture->getWidth();
            image.height = pTexture->getHeight();
            image.depth = pTexture->getDepth();
            image.mipLevels = generateMips ? nvtt::countMipmaps(image.width, image.height, image.depth) : pTexture->getMipCount();
            image.format = pTexture->getFormat();

            if (getFormatChannelCount(image.format) == 2 && mode != CompressionMode::BC5)
            {
                throw RuntimeError("Only BC5 compression is supported for two channel images.");
            }

            // The DX spec requires the dimensions of BC encoded textures to be a multiple of 4 at the base resolution.
            // If the texture has already been rescaled to meet this requirement, skip clamping.
            if (generateMips && (mode != CompressionMode::None))
            {
                bool clamped = clampIfNeeded(image);
                if (clamped)
                {
                    logWarning("Saving DDS image to '{}' with clamped image dimensions to accomodate mipmaps and compression.", path);
                }
            }

            switch (pTexture->getType())
            {
            case Resource::Type::Texture2D:
                image.type = nvtt::TextureType::TextureType_2D;
                image.faceCount = 1;
                break;
            case Resource::Type::Texture3D:
                image.type = nvtt::TextureType::TextureType_3D;
                image.faceCount = 1;
                break;
            case Resource::Type::TextureCube:
                image.type = nvtt::TextureType::TextureType_Cube;
                image.faceCount = 6;
                break;
            default:
                throw RuntimeError("Invalid texture type. Only 2D, 3D, and Cube are currently supported.");
            }

            for (uint32_t f = 0; f < image.faceCount; ++f)
            {
                for (uint32_t m = 0; m < image.mipLevels; ++m)
                {
                    uint32_t subresource = pTexture->getSubresourceIndex(f, m);
                    std::vector<uint8_t> subresourceData = pContext->readTextureSubresource(pTexture.get(), subresource);

                    nvtt::Surface surface;
                    FormatType type = getFormatType(image.format);
                    uint32_t width = (uint32_t)pTexture->getWidth(m);
                    uint32_t height = (uint32_t)pTexture->getHeight(m);
                    uint32_t depth = (uint32_t)pTexture->getDepth(m);

                    if (type == FormatType::Sint || type == FormatType::Snorm)
                    {
                        setImage<int8_t>(subresourceData.data(), surface, image, width, height, depth);
                    }
                    else if (type == FormatType::Uint || type == FormatType::Unorm || type == FormatType::UnormSrgb)
                    {
                        setImage<uint8_t>(subresourceData.data(), surface, image, width, height, depth);
                    }
                    else if (type == FormatType::Float)
                    {
                        if (getNumChannelBits(image.format, 0) == 16)
                        {
                            setImage<glm::detail::hdata>(subresourceData.data(), surface, image, width, height, depth);
                        }
                        else if (getNumChannelBits(image.format, 0) == 32)
                        {
                            setImage<float>(subresourceData.data(), surface, image, width, height, depth);
                        }
                    }

                    image.images.push_back(surface);

                    // Only need the base image if mipmaps are being generated
                    if (generateMips) break;
                }
            }

            // NVTT's Surface is designed to only hold uncompressed data, which means saving a compressed image as-is
            // requires the data be re-compressed. The selected compression mode is updated here to reflect this.
            if (isCompressedFormat(image.format) && mode == CompressionMode::None)
            {
                mode = convertFormatToMode(image.format);
            }

            exportDDS(path, image, mode, generateMips);
        }
        catch (const RuntimeError& e)
        {
            throw RuntimeError("Failed to save DDS image to '{}': {}", path, e.what());
        }
    }
}
