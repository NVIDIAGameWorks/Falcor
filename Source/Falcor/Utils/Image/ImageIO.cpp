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
#include "stdafx.h"
#include "ImageIO.h"
#include "DirectXTex.h"
#include <filesystem>

namespace Falcor
{
    namespace
    {
        /** Wrapper around DirectXTex image containers because there are separate API calls for each type.
        */
        struct ApiImage
        {
            /** If true, Image is holding the data. Otherwise data is in ScratchImage.
            */
            bool isSingleImage() const
            {
                return image.pixels != nullptr;
            }

            /** Get the DXGI Format of the image.
            */
            DXGI_FORMAT getFormat() const
            {
                return isSingleImage() ? image.format : scratchImage.GetMetadata().format;
            }

            // Single Raw Bitmap
            DirectX::Image image = {};

            // One or more images with metadata and managed memory.
            DirectX::ScratchImage scratchImage;
        };

        struct ImportData
        {
            // The full path of the file that was found in data directories
            std::string fullpath;

            // Commonly used values converted or casted for cleaner access
            ResourceFormat format;
            uint32_t width;
            uint32_t height;
            uint32_t depth;
            uint32_t arraySize;
            uint32_t mipLevels;

            // Original API data
            ApiImage image;
        };

        ImportData loadDDS(const std::string& filename, bool loadAsSrgb)
        {
            assert(hasSuffix(filename, ".dds", false));

            ImportData data;
            if (findFileInDataDirectories(filename, data.fullpath) == false)
            {
                throw std::exception(("Can't find file: " + filename).c_str());
            }

            DirectX::DDS_FLAGS flags = DirectX::DDS_FLAGS_NONE;
            if (FAILED(DirectX::LoadFromDDSFile(string_2_wstring(data.fullpath).c_str(), flags, nullptr, data.image.scratchImage)))
            {
                throw std::exception(("Failed to load file: " + filename).c_str());
            }

            const auto& meta = data.image.scratchImage.GetMetadata();
            ResourceFormat format = getResourceFormat(meta.format);
            data.format = loadAsSrgb ? linearToSrgbFormat(format) : format;
            data.width = (uint32_t)meta.width;
            data.height = (uint32_t)meta.height;
            data.depth = (uint32_t)meta.depth;
            data.arraySize = (uint32_t)meta.arraySize;
            data.mipLevels = (uint32_t)meta.mipLevels;

            return data;
        }

        void validateSavePath(const std::string& filename)
        {
            if (std::filesystem::path(filename).is_absolute() == false)
            {
                throw std::exception((filename + " is not an absolute path.").c_str());
            }

            if (getExtensionFromFile(filename) != "dds")
            {
                throw std::exception((filename + " does not end in dds").c_str());
            }
        }

        DXGI_FORMAT asCompressedFormat(DXGI_FORMAT format, ImageIO::CompressionMode mode)
        {
            // 7 block compression formats, and 3 variants for each.
            // Most of them are UNORM, SRGB, and TYPELESS, but BC4 and BC5 are different
            static const DXGI_FORMAT kCompressedFormats[7][3] =
            {
                // Unorm                sRGB                        Typeless
                {DXGI_FORMAT_BC1_UNORM, DXGI_FORMAT_BC1_UNORM_SRGB, DXGI_FORMAT_BC1_TYPELESS},
                {DXGI_FORMAT_BC2_UNORM, DXGI_FORMAT_BC2_UNORM_SRGB, DXGI_FORMAT_BC2_TYPELESS},
                {DXGI_FORMAT_BC3_UNORM, DXGI_FORMAT_BC3_UNORM_SRGB, DXGI_FORMAT_BC3_TYPELESS},

                // Unsigned             Signed                      Typeless
                {DXGI_FORMAT_BC4_UNORM, DXGI_FORMAT_BC4_SNORM,      DXGI_FORMAT_BC4_TYPELESS},
                {DXGI_FORMAT_BC5_UNORM, DXGI_FORMAT_BC5_SNORM,      DXGI_FORMAT_BC5_TYPELESS},
                {DXGI_FORMAT_BC6H_UF16, DXGI_FORMAT_BC6H_SF16,      DXGI_FORMAT_BC6H_TYPELESS},

                // Unorm                sRGB                        Typeless
                {DXGI_FORMAT_BC7_UNORM, DXGI_FORMAT_BC7_UNORM_SRGB, DXGI_FORMAT_BC7_TYPELESS}
            };

            bool isSRGB = DirectX::IsSRGB(format); // Applicable for 8-bit per channel RGB color modes
            bool isTypeless = DirectX::IsTypeless(format);

            switch (mode)
            {
            case ImageIO::CompressionMode::BC1:
            case ImageIO::CompressionMode::BC2:
            case ImageIO::CompressionMode::BC3:
            case ImageIO::CompressionMode::BC7:
                return kCompressedFormats[uint32_t(mode)][isSRGB ? 1 : (isTypeless ? 2 : 0)];
            case ImageIO::CompressionMode::BC4:
            case ImageIO::CompressionMode::BC5:
                // Always Unorm for single/two-channel (BC4 grayscale, or BC5 normals)
                return kCompressedFormats[uint32_t(mode)][(isTypeless ? 2 : 0)];
            case ImageIO::CompressionMode::BC6:
                // Always use Snorm for float formats
                // TODO: Check if format is signed
                return kCompressedFormats[uint32_t(mode)][isTypeless ? 2 : 1];
            default:
                return format;
            }
        }

        void compress(ApiImage& image, ImageIO::CompressionMode mode)
        {
            if (isCompressedFormat(getResourceFormat(image.getFormat())))
            {
                throw std::exception("Image is already compressed.");
            }

            const DirectX::TEX_COMPRESS_FLAGS flags = mode == ImageIO::CompressionMode::BC7 ? DirectX::TEX_COMPRESS_BC7_QUICK : DirectX::TEX_COMPRESS_DEFAULT;

            HRESULT result = S_OK;
            if (image.isSingleImage())
            {
                result = DirectX::Compress(image.image, asCompressedFormat(image.getFormat(), mode), flags, DirectX::TEX_THRESHOLD_DEFAULT, image.scratchImage);

                // Clear bitmap since compression outputted to the scratchImage
                image.image = {};
            }
            else
            {
                // Compression will output "in place" back to ApiImage, so move the input out here
                DirectX::ScratchImage inputImage(std::move(image.scratchImage));
                const auto& meta = inputImage.GetMetadata();

                result = DirectX::Compress(inputImage.GetImages(), inputImage.GetImageCount(), meta, asCompressedFormat(meta.format, mode), flags, DirectX::TEX_THRESHOLD_DEFAULT, image.scratchImage);
            }

            if (FAILED(result))
            {
                throw std::exception("Failed to compress.");
            }
        }

        /** Saves image data to a DDS file. Optionally compresses image.
        */
        void exportDDS(const std::string& filename, ApiImage& image, ImageIO::CompressionMode mode)
        {
            validateSavePath(filename);

            // Compress
            try
            {
                if (mode != ImageIO::CompressionMode::None)
                {
                    compress(image, mode);
                }
            }
            catch (const std::exception& e)
            {
                // Forward exception along with filename for context
                throw std::exception((filename + ": " + e.what()).c_str());
            }

            // Save
            const DirectX::DDS_FLAGS saveFlags = DirectX::DDS_FLAGS_NONE;
            HRESULT result = S_OK;
            if (image.isSingleImage())
            {
                result = DirectX::SaveToDDSFile(image.image, saveFlags, string_2_wstring(filename).c_str());
            }
            else
            {
                const auto& scratchImage = image.scratchImage;
                result = DirectX::SaveToDDSFile(scratchImage.GetImages(), scratchImage.GetImageCount(), scratchImage.GetMetadata(), saveFlags, string_2_wstring(filename).c_str());
            }

            if (FAILED(result))
            {
                throw std::exception(("Failed to export " + filename).c_str());
            }
        }
    }

    Bitmap::UniqueConstPtr ImageIO::loadBitmapFromDDS(const std::string& filename)
    {
        ImportData data = loadDDS(filename, false);

        const auto& scratchImage = data.image.scratchImage;
        const auto& meta = scratchImage.GetMetadata();
        if (meta.IsCubemap() || meta.IsVolumemap())
        {
            throw std::exception(("Cannot load " + filename + " as a Bitmap. Invalid resource dimension.").c_str());
        }

        // Create from first image
        auto pImage = scratchImage.GetImage(0, 0, 0);
        return Bitmap::create((uint32_t)data.width, (uint32_t)data.height, data.format, pImage->pixels);
    }

    Texture::SharedPtr ImageIO::loadTextureFromDDS(const std::string& filename, bool loadAsSrgb)
    {
        ImportData data = loadDDS(filename, loadAsSrgb);

        const auto& scratchImage = data.image.scratchImage;
        const auto& meta = scratchImage.GetMetadata();

        Texture::SharedPtr pTex;
        switch (meta.dimension)
        {
        case DirectX::TEX_DIMENSION_TEXTURE1D:
            pTex = Texture::create1D(data.width, data.format, data.arraySize, data.mipLevels, scratchImage.GetPixels());
            break;
        case DirectX::TEX_DIMENSION_TEXTURE2D:
            if (meta.IsCubemap())
            {
                pTex = Texture::createCube(data.width, data.height, data.format, data.arraySize / 6, data.mipLevels, scratchImage.GetPixels());
            }
            else
            {
                pTex = Texture::create2D(data.width, data.height, data.format, data.arraySize, data.mipLevels, scratchImage.GetPixels());
            }
            break;
        case DirectX::TEX_DIMENSION_TEXTURE3D:
            pTex = Texture::create3D(data.width, data.height, data.depth, data.format, data.mipLevels, scratchImage.GetPixels());
            break;
        }

        if (pTex != nullptr)
        {
            pTex->setSourceFilename(data.fullpath);
        }

        return pTex;
    }

    void ImageIO::saveToDDS(const std::string& filename, const Bitmap& bitmap, CompressionMode mode)
    {
        ApiImage image;
        image.image.width = bitmap.getWidth();
        image.image.height = bitmap.getHeight();
        image.image.format = getDxgiFormat(bitmap.getFormat());
        image.image.rowPitch = bitmap.getRowPitch();
        image.image.slicePitch = bitmap.getSize();
        image.image.pixels = bitmap.getData();

        exportDDS(filename, image, mode);
    }

    void ImageIO::saveToDDS(CopyContext* pContext, const std::string& filename, const Texture::SharedPtr& pTexture, CompressionMode mode)
    {
        DirectX::TexMetadata meta = {};
        meta.width = pTexture->getWidth();
        meta.height = pTexture->getHeight();
        meta.depth = pTexture->getDepth();
        meta.arraySize = pTexture->getArraySize();
        meta.mipLevels = pTexture->getMipCount();
        meta.format = getDxgiFormat(pTexture->getFormat());

        switch (pTexture->getType())
        {
        case Resource::Type::Texture1D:
            meta.dimension = DirectX::TEX_DIMENSION_TEXTURE1D;
            break;
        case Resource::Type::TextureCube:
            meta.miscFlags |= DirectX::TEX_MISC_TEXTURECUBE;
            // No break, cubes are also Texture2Ds
        case Resource::Type::Texture2D:
            meta.dimension = DirectX::TEX_DIMENSION_TEXTURE2D;
            break;
        case Resource::Type::Texture3D:
            meta.dimension = DirectX::TEX_DIMENSION_TEXTURE3D;
            throw std::exception("saveToDDS: Saving 3D textures currently not supported.");
        default:
            throw std::exception("saveToDDS: Invalid resource dimension.");
        }

        ApiImage image;
        auto& scratchImage = image.scratchImage;
        HRESULT result = scratchImage.Initialize(meta);
        assert(SUCCEEDED(result));

        for (uint32_t i = 0; i < pTexture->getArraySize(); i++)
        {
            for (uint32_t m = 0; m < pTexture->getMipCount(); m++)
            {
                uint32_t subresource = pTexture->getSubresourceIndex(i, m);
                const DirectX::Image* pImage = scratchImage.GetImage(m, i, 0);

                std::vector<uint8_t> subresourceData = pContext->readTextureSubresource(pTexture.get(), subresource);
                assert(subresourceData.size() == pImage->slicePitch);
                std::memcpy(pImage->pixels, subresourceData.data(), subresourceData.size());
            }
        }

        exportDDS(filename, image, mode);
    }

}
