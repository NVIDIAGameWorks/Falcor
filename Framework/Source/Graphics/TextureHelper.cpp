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
#include "TextureHelper.h"
#include "API/Texture.h"
#include "Utils/Bitmap.h"
#include "Utils/DDSHeader.h"
#include "Utils/BinaryFileStream.h"
#include "Utils/StringUtils.h"
#include <cstring>

static const bool kTopDown = true;

namespace Falcor
{
    using namespace DdsHelper;

    static const uint32_t kDdsMagicNumber = 0x20534444;

    bool checkDdsChannelMask(const DdsHeader::PixelFormat& format, uint32_t r, uint32_t g, uint32_t b, uint32_t a)
    {
        return (format.rMask == r && format.gMask == g && format.bMask == b && format.aMask == a);
    }

    uint32_t makeFourCC(const char name[4])
    {
        uint32_t fourCC = 0;
        for(uint32_t i = 0; i < 4; i++)
        {
            uint32_t shift = i * 8;
            fourCC |= ((uint32_t)name[i]) << shift;
        }
        return fourCC;
    }

    ResourceFormat falcorFormatFromDXGIFormat(DXFormat fmt) 
    {
        switch (fmt)
        {
        case FORMAT_R32G8X24_TYPELESS:
        case FORMAT_R16G16B16A16_TYPELESS:
        case FORMAT_R32G32B32_TYPELESS:
        case FORMAT_R32G32B32A32_TYPELESS:
        case FORMAT_R16G16B16A16_SNORM:
        case FORMAT_R32G32_TYPELESS:
        case FORMAT_R32_FLOAT_X8X24_TYPELESS:
        case FORMAT_X32_TYPELESS_G8X24_UINT:
        case FORMAT_R10G10B10A2_TYPELESS:
        case FORMAT_Y416:
        case FORMAT_Y210:
        case FORMAT_Y216:
        case FORMAT_R8G8B8A8_TYPELESS:
        case FORMAT_R16G16_TYPELESS:
        case FORMAT_R32_TYPELESS:
        case FORMAT_R24G8_TYPELESS:
        case FORMAT_R24_UNORM_X8_TYPELESS:
        case FORMAT_X24_TYPELESS_G8_UINT:
        case FORMAT_R8G8_B8G8_UNORM:
        case FORMAT_G8R8_G8B8_UNORM:
        case FORMAT_R10G10B10_XR_BIAS_A2_UNORM:
        case FORMAT_B8G8R8A8_TYPELESS:
        case FORMAT_B8G8R8X8_TYPELESS:
        case FORMAT_AYUV:
        case FORMAT_Y410:
        case FORMAT_YUY2:
        case FORMAT_P010:
        case FORMAT_P016:
        case FORMAT_R8G8_TYPELESS:
        case FORMAT_R16_TYPELESS:
        case FORMAT_A8P8:
        case FORMAT_B4G4R4A4_UNORM:
        case FORMAT_NV12:
        case FORMAT_420_OPAQUE:
        case FORMAT_NV11:
        case FORMAT_R8_TYPELESS:
        case FORMAT_AI44:
        case FORMAT_IA44:
        case FORMAT_P8:
        case FORMAT_R1_UNORM:
        case FORMAT_BC1_TYPELESS:
        case FORMAT_BC4_TYPELESS:
        case FORMAT_BC2_TYPELESS:
        case FORMAT_BC3_TYPELESS:
        case FORMAT_BC5_TYPELESS:
        case FORMAT_BC6H_TYPELESS:
        case FORMAT_BC7_TYPELESS:
            return ResourceFormat::Unknown;
        case FORMAT_R32G32B32A32_FLOAT:
            return ResourceFormat::RGBA32Float;
        case FORMAT_R32G32B32A32_UINT:
            return ResourceFormat::RGBA32Uint;
        case FORMAT_R32G32B32A32_SINT:
            return ResourceFormat::RGBA32Int;
        case FORMAT_R32G32B32_FLOAT:
            return ResourceFormat::RGB32Float;
        case FORMAT_R32G32B32_UINT:
            return ResourceFormat::RGB32Uint;
        case FORMAT_R32G32B32_SINT:
            return ResourceFormat::RGB32Int;
        case FORMAT_R16G16B16A16_FLOAT:
            return ResourceFormat::RGBA16Float;
        case FORMAT_R16G16B16A16_UNORM:
            return ResourceFormat::RGBA16Unorm;
        case FORMAT_R16G16B16A16_UINT:
            return ResourceFormat::RGBA16Uint;
        case FORMAT_R16G16B16A16_SINT:
            return ResourceFormat::RGBA16Int;
        case FORMAT_R32G32_FLOAT:
            return ResourceFormat::RG32Float;
        case FORMAT_R32G32_UINT:
            return ResourceFormat::RG32Uint;
        case FORMAT_R32G32_SINT:
            return ResourceFormat::RG32Int;
        case FORMAT_D32_FLOAT_S8X24_UINT:
            return ResourceFormat::D32FloatS8X24;
        case FORMAT_R10G10B10A2_UNORM:
            return ResourceFormat::RGB10A2Unorm;
        case FORMAT_R10G10B10A2_UINT:
            return ResourceFormat::RGB10A2Uint;
        case FORMAT_R11G11B10_FLOAT:
            return ResourceFormat::R11G11B10Float;
        case FORMAT_R8G8B8A8_UNORM:
            return ResourceFormat::RGBA8Unorm;
        case FORMAT_R8G8B8A8_UNORM_SRGB:
            return ResourceFormat::RGBA8UnormSrgb;
        case FORMAT_R8G8B8A8_UINT:
            return ResourceFormat::RGBA8Uint;
        case FORMAT_R8G8B8A8_SNORM:
            return ResourceFormat::RGBA8Snorm;
        case FORMAT_R8G8B8A8_SINT:
            return ResourceFormat::RGBA8Int;
        case FORMAT_R16G16_FLOAT:
            return ResourceFormat::RG16Float;
        case FORMAT_R16G16_UNORM:
            return ResourceFormat::RG16Unorm;
        case FORMAT_R16G16_UINT:
            return ResourceFormat::RG16Uint;
        case FORMAT_R16G16_SNORM:
            return ResourceFormat::RG16Snorm;
        case FORMAT_R16G16_SINT:
            return ResourceFormat::RG16Int;
        case FORMAT_D32_FLOAT:
            return ResourceFormat::D32Float;
        case FORMAT_R32_FLOAT:
            return ResourceFormat::R32Float;
        case FORMAT_R32_UINT:
            return ResourceFormat::R32Uint;
        case FORMAT_R32_SINT:
            return ResourceFormat::R32Int;
        case FORMAT_D24_UNORM_S8_UINT:
            return ResourceFormat::D24UnormS8;
        case FORMAT_R9G9B9E5_SHAREDEXP:
            return ResourceFormat::RGB9E5Float;
        case FORMAT_B8G8R8A8_UNORM:
            return ResourceFormat::BGRA8Unorm;
        case FORMAT_B8G8R8X8_UNORM:
            return ResourceFormat::BGRX8Unorm;
        case FORMAT_B8G8R8A8_UNORM_SRGB:
            return ResourceFormat::BGRA8UnormSrgb;
        case FORMAT_B8G8R8X8_UNORM_SRGB:
            return ResourceFormat::BGRX8UnormSrgb;
        case FORMAT_R8G8_UNORM:
            return ResourceFormat::RG8Unorm;
        case FORMAT_R8G8_UINT:
            return ResourceFormat::RG8Uint;
        case FORMAT_R8G8_SNORM:
            return ResourceFormat::RG8Snorm;
        case FORMAT_R8G8_SINT:
            return ResourceFormat::RG8Int;
        case FORMAT_R16_FLOAT:
            return ResourceFormat::R16Float;
        case FORMAT_D16_UNORM:
            return ResourceFormat::D16Unorm;
        case FORMAT_R16_UNORM:
            return ResourceFormat::R16Unorm;
        case FORMAT_R16_UINT:
            return ResourceFormat::R16Uint;
        case FORMAT_R16_SNORM:
            return ResourceFormat::R16Snorm;
        case FORMAT_R16_SINT:
            return ResourceFormat::R16Int;
        case FORMAT_B5G6R5_UNORM:
            return ResourceFormat::R5G6B5Unorm;
        case FORMAT_B5G5R5A1_UNORM:
            return ResourceFormat::RGB5A1Unorm;
        case FORMAT_R8_UNORM:
            return ResourceFormat::R8Unorm;
        case FORMAT_R8_UINT:
            return ResourceFormat::R8Uint;
        case FORMAT_R8_SNORM:
            return ResourceFormat::R8Snorm;
        case FORMAT_R8_SINT:
            return ResourceFormat::R8Int;
        case FORMAT_A8_UNORM:
            return ResourceFormat::Alpha8Unorm;
        case FORMAT_BC1_UNORM:
            return ResourceFormat::BC1Unorm;
        case FORMAT_BC1_UNORM_SRGB:
            return ResourceFormat::BC1UnormSrgb;
        case FORMAT_BC4_UNORM:
            return ResourceFormat::BC4Unorm;
        case FORMAT_BC4_SNORM:
            return ResourceFormat::BC4Snorm;
        case FORMAT_BC2_UNORM:
            return ResourceFormat::BC2Unorm;
        case FORMAT_BC2_UNORM_SRGB:
            return ResourceFormat::BC2UnormSrgb;
        case FORMAT_BC3_UNORM:
            return ResourceFormat::BC3Unorm;
        case FORMAT_BC3_UNORM_SRGB:
            return ResourceFormat::BC3UnormSrgb;
        case FORMAT_BC5_UNORM:
            return ResourceFormat::BC5Unorm;
        case FORMAT_BC5_SNORM:
            return ResourceFormat::BC5Snorm;
        case FORMAT_BC6H_SF16:
            return ResourceFormat::BC6HS16;
        case FORMAT_BC6H_UF16:
            return ResourceFormat::BC6HU16;
        case FORMAT_BC7_UNORM:
            return ResourceFormat::BC7Unorm;
        case FORMAT_BC7_UNORM_SRGB:
            return ResourceFormat::BC7UnormSrgb;
        default:
            return ResourceFormat::Unknown;
        }
    }

    DXFormat getRgbDxgiFormat(const DdsHeader::PixelFormat& format)
    {
        switch(format.bitcount)
        {
        case 32:
            if(checkDdsChannelMask(format, 0x000000ff, 0x0000ff00, 0x00ff0000, 0xff000000))
            {
                return FORMAT_R8G8B8A8_UNORM;
            }

            if(checkDdsChannelMask(format, 0x00ff0000, 0x0000ff00, 0x000000ff, 0xff000000))
            {
                return FORMAT_B8G8R8A8_UNORM;
            }

            if(checkDdsChannelMask(format, 0x00ff0000, 0x0000ff00, 0x000000ff, 0x00000000))
            {
                return FORMAT_B8G8R8X8_UNORM;
            }

            if(checkDdsChannelMask(format, 0x3ff00000, 0x000ffc00, 0x000003ff, 0xc0000000))
            {
                return FORMAT_R10G10B10A2_UNORM;
            }

            if(checkDdsChannelMask(format, 0x0000ffff, 0xffff0000, 0x00000000, 0x00000000))
            {
                return FORMAT_R16G16_UNORM;
            }

            if(checkDdsChannelMask(format, 0xffffffff, 0x00000000, 0x00000000, 0x00000000))
            {
                return FORMAT_R32_FLOAT;
            }
            break;

        case 16:
            if(checkDdsChannelMask(format, 0x7c00, 0x03e0, 0x001f, 0x8000))
            {
                return FORMAT_B5G5R5A1_UNORM;
            }
            if(checkDdsChannelMask(format, 0xf800, 0x07e0, 0x001f, 0x0000))
            {
                return FORMAT_B5G6R5_UNORM;
            }

            if(checkDdsChannelMask(format, 0x0f00, 0x00f0, 0x000f, 0xf000))
            {
                return FORMAT_B4G4R4A4_UNORM;
            }
            break;
        }
        should_not_get_here();
        return FORMAT_UNKNOWN;
    }

    DXFormat getLuminanceDxgiFormat(const DdsHeader::PixelFormat& format)
    {
        switch(format.bitcount)
        {
        case 16:
            if(checkDdsChannelMask(format, 0x0000ffff, 0x00000000, 0x00000000, 0x00000000))
            {
                return FORMAT_R16_UNORM;
            }
            if(checkDdsChannelMask(format, 0x000000ff, 0x00000000, 0x00000000, 0x0000ff00))
            {
                return FORMAT_R8G8_UNORM;
            }
            break;
        case 8:
            if(checkDdsChannelMask(format, 0x000000ff, 0x00000000, 0x00000000, 0x00000000))
            {
                return FORMAT_R8_UNORM;
            }
            break;
        }
        should_not_get_here();
        return FORMAT_UNKNOWN;
    }

    DXFormat getDxgiAlphaFormat(const DdsHeader::PixelFormat& format)
    {
        switch(format.bitcount)
        {
        case 8:
            return FORMAT_A8_UNORM;
        default:
            should_not_get_here();
            return FORMAT_UNKNOWN;
        }
    }

    DXFormat getDxgiBumpFormat(const DdsHeader::PixelFormat& format)
    {
        switch(format.bitcount)
        {
        case 16:
            if(checkDdsChannelMask(format, 0x00ff, 0xff00, 0x0000, 0x0000))
            {
                return FORMAT_R8G8_SNORM;
            }
            break;
        case 32:
            if(checkDdsChannelMask(format, 0x000000ff, 0x0000ff00, 0x00ff0000, 0xff000000))
            {
                return FORMAT_R8G8B8A8_SNORM;
            }
            if(checkDdsChannelMask(format, 0x0000ffff, 0xffff0000, 0x00000000, 0x00000000))
            {
                return FORMAT_R16G16_SNORM;
            }
            break;
        }
        should_not_get_here();
        return FORMAT_UNKNOWN;
    }

    DXFormat getDxgiFormatFrom4CC(uint32_t fourCC)
    {
        if(fourCC == makeFourCC("DXT1"))
        {
            return FORMAT_BC1_UNORM;
        }
        if(fourCC == makeFourCC("DXT2"))
        {
            return FORMAT_BC2_UNORM;
        }
        if(fourCC == makeFourCC("DXT3"))
        {
            return FORMAT_BC2_UNORM;
        }
        if(fourCC == makeFourCC("DXT4"))
        {
            return FORMAT_BC3_UNORM;
        }
        if(fourCC == makeFourCC("DXT5"))
        {
            return FORMAT_BC3_UNORM;
        }

        if(fourCC == makeFourCC("ATI1"))
        {
            return FORMAT_BC4_UNORM;
        }
        if(fourCC == makeFourCC("BC4U"))
        {
            return FORMAT_BC4_UNORM;
        }
        if(fourCC == makeFourCC("BC4S"))
        {
            return FORMAT_BC4_SNORM;
        }

        if(fourCC == makeFourCC("ATI2"))
        {
            return FORMAT_BC5_UNORM;
        }
        if(fourCC == makeFourCC("BC5U"))
        {
            return FORMAT_BC5_UNORM;
        }
        if(fourCC == makeFourCC("BC5S"))
        {
            return FORMAT_BC5_SNORM;
        }

        if(fourCC == makeFourCC("RGBG"))
        {
            return FORMAT_R8G8_B8G8_UNORM;
        }
        if(fourCC == makeFourCC("GRGB"))
        {
            return FORMAT_G8R8_G8B8_UNORM;
        }

        if(fourCC == makeFourCC("YUY2"))
        {
            return FORMAT_YUY2;
        }

        switch(fourCC)
        {
        case 36:
            return FORMAT_R16G16B16A16_UNORM;
        case 110:
            return FORMAT_R16G16B16A16_SNORM;
        case 111:
            return FORMAT_R16_FLOAT;
        case 112:
            return FORMAT_R16G16_FLOAT;
        case 113:
            return FORMAT_R16G16B16A16_FLOAT;
        case 114:
            return FORMAT_R32_FLOAT;
        case 115:
            return FORMAT_R32G32_FLOAT;
        case 116:
            return FORMAT_R32G32B32A32_FLOAT;
        }

        should_not_get_here();
        return FORMAT_UNKNOWN;
    }

    DXFormat getDxgiFormatFromPixelFormat(const DdsHeader::PixelFormat& format)
    {
        if(format.flags & DdsHeader::PixelFormat::kRgbMask)
        {
            return getRgbDxgiFormat(format);
        }
        else if (format.flags & DdsHeader::PixelFormat::kLuminanceMask)
        {
            return getLuminanceDxgiFormat(format);
        }
        else if(format.flags & DdsHeader::PixelFormat::kAlphaMask)
        {
            return getDxgiAlphaFormat(format);
        }
        else if (format.flags & DdsHeader::PixelFormat::kBumpMask)
        {
            return getDxgiBumpFormat(format);
        }
        else if(format.flags & DdsHeader::PixelFormat::kFourCCFlag)
        {
            return getDxgiFormatFrom4CC(format.fourCC);
        }

        return FORMAT_UNKNOWN;
    }

    ResourceFormat getDdsResourceFormat(const DdsData& data)
    {
        if(data.hasDX10Header)
        {
            return falcorFormatFromDXGIFormat(data.dx10Header.dxgiFormat);
        }
        else
        {
            return falcorFormatFromDXGIFormat(getDxgiFormatFromPixelFormat(data.header.pixelFormat));
        }
    }

    //Flip the data so it follows opengl conventions
    void flipData(DdsData& ddsData, ResourceFormat format, uint32_t width, uint32_t height, uint32_t depth, uint32_t mipDepth, bool isCubemap = false)
    {
        if (!isCompressedFormat(format) && !kTopDown)
        {
            std::vector<uint8_t> oldData(ddsData.data.size());
            oldData.swap(ddsData.data);
            const uint8_t* currentTexture = oldData.data();
            const uint8_t* currentDepth = oldData.data();
            uint8_t* currentPos = ddsData.data.data();

            for (uint32_t mipCounter = 0; mipCounter < mipDepth; ++mipCounter)
            {
                uint32_t heightPitch = max(width >> mipCounter, 1U) * getFormatBytesPerBlock(format);
                uint32_t currentMipHeight = max(height >> mipCounter, 1U);
                uint32_t depthPitch = currentMipHeight * heightPitch;	

                for (uint32_t depthCounter = 0; depthCounter < depth; ++depthCounter)
                {
                    currentTexture = currentDepth + depthPitch * depthCounter;
                    
                    if (isCubemap)
                    {
                        if (depthCounter % 6 == 2)
                        {
                            currentTexture += depthPitch;
                        }
                        else if (depthCounter % 6 == 3)
                        {
                            currentTexture -= depthPitch;
                        }
                    }
                    
                    for (uint32_t heightCounter = 1; heightCounter <= currentMipHeight; ++heightCounter)
                    {
                        std::memcpy(currentPos, currentTexture + (currentMipHeight - heightCounter) * heightPitch, heightPitch);
                        currentPos += heightPitch;
                    }
                    
                }

                currentDepth += depthPitch * depth;
            }
        }
    }

    void loadDDSDataFromFile(const std::string filename, DdsData& ddsData)
    {
        std::string fullpath;
        if (findFileInDataDirectories(filename, fullpath) == false)
        {
            logError(std::string("Can't find texture file ") + filename);
            //could not find file
            return;
        }

        BinaryFileStream stream(fullpath, BinaryFileStream::Mode::Read);

        //check the dds identifier
        uint32_t ddsIdentifier;
        stream >> ddsIdentifier;
        if (ddsIdentifier != kDdsMagicNumber)
        {
            //not valid dds file apparently
            logError(std::string("The dds file ") + filename + std::string(" is not a valid dds file"));
            return;
        }

        stream >> ddsData.header;

        if((ddsData.header.pixelFormat.flags & DdsHeader::PixelFormat::kFourCCFlag) && (makeFourCC("DX10") == ddsData.header.pixelFormat.fourCC))
        {
            ddsData.hasDX10Header = true;
            stream >> ddsData.dx10Header;
        }
        else
        {
            ddsData.hasDX10Header = false;
        }

        uint32_t dataSize = stream.getRemainingStreamSize();
        ddsData.data.resize(dataSize);
        stream.read(ddsData.data.data(), dataSize);
    }

    static ResourceFormat convertBgrxFormatToBgra(DdsData& ddsData, ResourceFormat format)
    {
#ifdef FALCOR_VK
        switch (format)
        {
        case ResourceFormat::BGRX8Unorm:
            format = ResourceFormat::BGRA8Unorm;
            break;
        case ResourceFormat::BGRX8UnormSrgb:
            format = ResourceFormat::BGRA8UnormSrgb;
            break;
        default:
            return format;
        }

        for (size_t i = 3; i < ddsData.data.size(); i+=4)
        {
            ddsData.data[i] = 0xFF;
        }
#endif
        return format;
    }

    Texture::SharedPtr createTextureFromDx10Dds(DdsData& ddsData, const std::string& filename, ResourceFormat format, uint32_t mipLevels, Texture::BindFlags bindFlags)
    {
        format = convertBgrxFormatToBgra(ddsData, format);

        uint32_t arraySize = ddsData.dx10Header.arraySize;
        assert(arraySize > 0);

        switch(ddsData.dx10Header.resourceDimension)
        {
        case DXResourceDimension::RESOURCE_DIMENSION_TEXTURE1D:
            return Texture::create1D(ddsData.header.width, format, arraySize, mipLevels, ddsData.data.data(), bindFlags);
        case DXResourceDimension::RESOURCE_DIMENSION_TEXTURE2D:
            if(ddsData.dx10Header.miscFlag & DdsHeaderDX10::kCubeMapMask)
            {
                flipData(ddsData, format, ddsData.header.width, ddsData.header.height, 6 * arraySize, mipLevels == Texture::kMaxPossible ? 1 : mipLevels, true);
                return Texture::createCube(ddsData.header.width, ddsData.header.height, format, arraySize, mipLevels, ddsData.data.data(), bindFlags);
            }
            else
            {
                flipData(ddsData, format, ddsData.header.width, ddsData.header.height, arraySize, mipLevels == Texture::kMaxPossible ? 1 : mipLevels);
                return Texture::create2D(ddsData.header.width, ddsData.header.height, format, arraySize, mipLevels, ddsData.data.data(), bindFlags);
            }
        case DXResourceDimension::RESOURCE_DIMENSION_TEXTURE3D:
            flipData(ddsData, format, ddsData.header.width, ddsData.header.height, ddsData.header.depth, mipLevels == Texture::kMaxPossible ? 1 : mipLevels);
            return Texture::create3D(ddsData.header.width, ddsData.header.height, ddsData.header.depth, format, mipLevels, ddsData.data.data(), bindFlags);
        case DXResourceDimension::RESOURCE_DIMENSION_BUFFER:
        case DXResourceDimension::RESOURCE_DIMENSION_UNKNOWN:
            //these file formats are not supported 
            logError(std::string("the resource dimension specified in ") + filename + std::string(" is not supported by Falcor"));
        default:
            should_not_get_here();
            return nullptr;
        }
    }

    Texture::SharedPtr createTextureFromLegacyDds(DdsData& ddsData, const std::string& filename, ResourceFormat format, uint32_t mipLevels, Texture::BindFlags bindFlags)
    {
        format = convertBgrxFormatToBgra(ddsData, format);

        //load the volume or 3D texture
        if(ddsData.header.flags & DdsHeader::kDepthMask)
        {
            flipData(ddsData, format, ddsData.header.width, ddsData.header.height, ddsData.header.depth, mipLevels == Texture::kMaxPossible ? 1 : mipLevels);
            return Texture::create3D(ddsData.header.width, ddsData.header.height, ddsData.header.depth, format, mipLevels, ddsData.data.data(), bindFlags);
        }
        //load the cubemap texture
        else if(ddsData.header.caps[1] & DdsHeader::kCaps2CubeMapMask)
        {
            return Texture::createCube(ddsData.header.width, ddsData.header.height, format, 1, mipLevels, ddsData.data.data(), bindFlags);
        }
        //This is a 2D Texture
        else
        {
            flipData(ddsData, format, ddsData.header.width, ddsData.header.height, 1, mipLevels == Texture::kMaxPossible ? 1 : mipLevels);
            return Texture::create2D(ddsData.header.width, ddsData.header.height, format, 1, mipLevels, ddsData.data.data(), bindFlags);
        }

        should_not_get_here();
        return nullptr;
    }

    Texture::SharedPtr createTextureFromDDSFile(const std::string filename, bool generateMips, bool loadAsSrgb, Texture::BindFlags bindFlags)
    {
        DdsData ddsData;
        loadDDSDataFromFile(filename, ddsData);

        ResourceFormat format = getDdsResourceFormat(ddsData);
        assert(format != ResourceFormat::Unknown);

        if (loadAsSrgb)
        {
            format = linearToSrgbFormat(format);
        }

        uint32_t mipLevels;
        if (generateMips == false || isCompressedFormat(format))
        {
            mipLevels = (ddsData.header.flags & DdsHeader::kMipCountMask) ? max(ddsData.header.mipCount, 1U) : 1;
        }
        else
        {
            mipLevels = Texture::kMaxPossible;
        }

        if (ddsData.hasDX10Header)
        {
            return createTextureFromDx10Dds(ddsData, filename, format, mipLevels, bindFlags);
        }
        else
        {
            return createTextureFromLegacyDds(ddsData, filename, format, mipLevels, bindFlags);
        }
        return nullptr;
    }

    Texture::SharedPtr createTextureFromFile(const std::string& filename, bool generateMipLevels, bool loadAsSrgb, Texture::BindFlags bindFlags)
    {
#define no_srgb()   \
    if(loadAsSrgb)  \
    {               \
        logWarning("createTexture2DFromFile() warning. " + std::to_string(pBitmap->getBytesPerPixel()) + " channel images doesn't have a matching sRGB format. Loading in linear space.");  \
    }

        if (hasSuffix(filename, ".dds"))
        {
            return createTextureFromDDSFile(filename, generateMipLevels, loadAsSrgb, bindFlags);
        }

        Bitmap::UniqueConstPtr pBitmap = Bitmap::createFromFile(filename, kTopDown);
        Texture::SharedPtr pTex;

        if(pBitmap)
        {
            ResourceFormat texFormat = pBitmap->getFormat();
            if(loadAsSrgb)
            {
                texFormat = linearToSrgbFormat(texFormat);
            }

            pTex = Texture::create2D(pBitmap->getWidth(), pBitmap->getHeight(), texFormat, 1, generateMipLevels ? Texture::kMaxPossible : 1, pBitmap->getData(), bindFlags);
            pTex->setSourceFilename(stripDataDirectories(filename));
        }
        return pTex;
    }
#undef no_srgb
}
