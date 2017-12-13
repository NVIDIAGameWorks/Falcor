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
#include "BinaryImage.hpp"
//#include "gpu/CudaModule.hpp"
#include <cstring>

using namespace FW;

//------------------------------------------------------------------------

#define C8(TYPE, OFS)           { ChannelType_ ## TYPE, ChannelFormat_Clamp, OFS, 1, 0, 8 }
#define C16(TYPE, OFS, SIZE)    { ChannelType_ ## TYPE, ChannelFormat_Clamp, 0, 2, OFS, SIZE }
#define C32(TYPE, OFS)          { ChannelType_ ## TYPE, ChannelFormat_Clamp, 0, 4, OFS, 8 }
#define CF32(TYPE, OFS)         { ChannelType_ ## TYPE, ChannelFormat_Float, OFS, 4, 0, 32 }

const ImageFormat::StaticFormat ImageFormat::s_staticFormats[] =
{
    /* R8_G8_B8 */      { 3,  3, { C8(R,0), C8(G,1), C8(B,2) },                         /*GL_RGB8, GL_RGB, GL_UNSIGNED_BYTE, false */},
    /* R8_G8_B8_A8 */   { 4,  4, { C8(R,0), C8(G,1), C8(B,2), C8(A,3) },                /*GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, false */},
    /* A8 */            { 1,  1, { C8(A,0) },                                           /*GL_ALPHA8, GL_ALPHA, GL_UNSIGNED_BYTE, false */},
    /* XBGR_8888 */     { 4,  3, { C32(R,0), C32(G,8), C32(B,16) },                     /*GL_RGB8, GL_RGBA, GL_UNSIGNED_BYTE, true */},
    /* ABGR_8888 */     { 4,  4, { C32(R,0), C32(G,8), C32(B,16), C32(A,24) },          /*GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, true */},

    /* RGB_565 */       { 2,  3, { C16(R,11,5), C16(G,5,6), C16(B,0,5) },               /*GL_RGB5, GL_RGB, GL_UNSIGNED_SHORT_5_6_5, false */},
    /* RGBA_5551 */     { 2,  4, { C16(R,11,5), C16(G,6,5), C16(B,1,5), C16(A,0,1) },   /*GL_RGB5_A1, GL_RGBA, GL_UNSIGNED_SHORT_5_5_5_1, false */},

    /* RGB_Vec3f */     { 12, 3, { CF32(R,0), CF32(G,4), CF32(B,8) },                   /*GL_RGB32F, GL_RGB, GL_FLOAT, false */},
    /* RGBA_Vec4f */    { 16, 4, { CF32(R,0), CF32(G,4), CF32(B,8), CF32(A,12) },       /*GL_RGBA32F, GL_RGBA, GL_FLOAT, false */},
    /* A_F32 */         { 4,  1, { CF32(A,0) },                                         /*GL_ALPHA32F_ARB, GL_ALPHA, GL_FLOAT, false */},

    /*BGRA_8888*/       { 4,  4, { C8(R,0), C8(G,1), C8(B,2), C8(A, 3) },               /*GL_RGBA8, GL_BGRA, GL_UNSIGNED_BYTE, false */},
    /*BGR_888*/         { 3,  3, { C8(R,0), C8(G,1), C8(B,2) },                         /*GL_RGB8, GL_BGR, GL_UNSIGNED_BYTE, false */},
    /*RG_88*/           { 2,  2, { C8(R,0), C8(G,1), },                                 /*GL_RG8,  GL_RG,  GL_UNSIGNED_BYTE, false */},
    /*R8*/              { 1,  1, { C8(R,0)},                                            /*GL_R,    GL_RED, GL_UNSIGNED_BYTE, false */},

    /*S3TC_DXT1*/       { 0,  0, {},                                                    /*GL_COMPRESSED_RGB_S3TC_DXT1_EXT, GL_RGB, GL_NONE, false*/},
    /*S3TC_DXT3*/       { 0,  0, {},                                                    /*GL_COMPRESSED_RGBA_S3TC_DXT3_EXT, GL_RGBA, GL_NONE, false*/},
    /*S3TC_DXT5*/       { 0,  0, {},                                                    /*GL_COMPRESSED_RGBA_S3TC_DXT5_EXT, GL_RGBA, GL_NONE, false*/},
    /*RGTC_R*/          { 0,  0, {},                                                    /*GL_COMPRESSED_RED_RGTC1, GL_RED, GL_NONE, false*/},
    /*RGTC_RG*/         { 0,  0, {},                                                    /*GL_COMPRESSED_RG_RGTC2, GL_RG, GL_NONE, false*/},
};

#undef C8
#undef C16
#undef C32
#undef CF32

//------------------------------------------------------------------------

#define RGB_565_TO_ABGR_8888(V)     ((V >> 8) & 0x000000F8) | (V >> 13) | ((V << 5) & 0x0000FC00) | ((V >> 1) & 0x00000300) | ((V << 19) & 0x00F80000) | ((V >> 14) & 0x00070000) | 0xFF000000
#define ABGR_8888_TO_RGB_565(V)     (U16)(((V << 8) & 0xF800) | ((V >> 5) & 0x07E0) | ((V >> 19) & 0x001F))

#define RGBA_5551_TO_ABGR_8888(V)   ((V >> 8) & 0x000000F8) | (V >> 13) | ((V << 5) & 0x0000F800) | (V & 0x00000700) | ((V << 18) & 0x00F80000) | ((V >> 13) & 0x00070000) | ((S32)(V << 31) >> 7)
#define ABGR_8888_TO_RGBA_5551(V)   (U16)(((V << 8) & 0xF800) | ((V >> 5) & 0x07C0) | ((V >> 18) & 0x003E) | (V >> 31))

//------------------------------------------------------------------------

ImageFormat::ID ImageFormat::getID(void) const
{
    if (m_id != ID_Max)
        return m_id;

    FW_ASSERT(FW_ARRAY_SIZE(s_staticFormats) == ID_Generic);
    for (int i = 0; i < (int)ID_Generic; i++)
    {
        const StaticFormat& f = s_staticFormats[i];
        if (m_genericBPP == f.bpp &&
            m_genericChannels.size() == f.numChannels &&
            std::memcmp(m_genericChannels.data(), f.channels, m_genericChannels.size()*sizeof(m_genericChannels[0])) == 0)
        {
            m_id = (ID)i;
            return m_id;
        }
    }

    m_id = ID_Generic;
    return m_id;
}

//------------------------------------------------------------------------

const ImageFormat::StaticFormat* ImageFormat::getStaticFormat(void) const
{
    ID id = getID();
    FW_ASSERT(FW_ARRAY_SIZE(s_staticFormats) == ID_Generic);
    return (id < ID_Generic) ? &s_staticFormats[id] : NULL;
}

//------------------------------------------------------------------------

int ImageFormat::getBPP(void) const
{
    FW_ASSERT(FW_ARRAY_SIZE(s_staticFormats) == ID_Generic);
    return (m_id < ID_Generic) ? s_staticFormats[m_id].bpp : m_genericBPP;
}

//------------------------------------------------------------------------

int ImageFormat::getNumChannels(void) const
{
    FW_ASSERT(FW_ARRAY_SIZE(s_staticFormats) == ID_Generic);
    return (m_id < ID_Generic) ? s_staticFormats[m_id].numChannels : (int)m_genericChannels.size();
}

//------------------------------------------------------------------------

const ImageFormat::Channel& ImageFormat::getChannel(int idx) const
{
    FW_ASSERT(idx >= 0 && idx < getNumChannels());
    FW_ASSERT(FW_ARRAY_SIZE(s_staticFormats) == ID_Generic);
    return (m_id < ID_Generic) ? s_staticFormats[m_id].channels[idx] : m_genericChannels[idx];
}

//------------------------------------------------------------------------

int ImageFormat::findChannel(ChannelType Type) const
{
    int num = getNumChannels();
    for (int i = 0; i < num; i++)
        if (getChannel(i).Type == Type)
            return i;
    return -1;
}

//------------------------------------------------------------------------

void ImageFormat::set(const ImageFormat& other)
{
    m_id = other.m_id;
    if (m_id >= ID_Generic)
    {
        m_genericBPP = other.m_genericBPP;
        m_genericChannels = other.m_genericChannels;
    }
}

//------------------------------------------------------------------------

void ImageFormat::clear(void)
{
    m_id = ID_Generic;
    m_genericBPP = 0;
    m_genericChannels.clear();
}

//------------------------------------------------------------------------

void ImageFormat::addChannel(const Channel& channel)
{
    if (m_id < ID_Generic)
    {
        const StaticFormat& f = s_staticFormats[m_id];
        m_genericBPP = f.bpp;
        m_genericChannels.resize(f.numChannels);
        for(int32_t i = 0 ; i < f.numChannels ; i++)
        {
            m_genericChannels[i] = f.channels[i];
        }
    }

    m_id = ID_Max;
    m_genericBPP = max(m_genericBPP, channel.wordOfs + channel.wordSize);
    m_genericChannels.push_back(channel);
}

//------------------------------------------------------------------------

#if 0
ImageFormat::ID ImageFormat::getGLFormat(void) const
{
    const ImageFormat::StaticFormat* sf = getStaticFormat();

    // Requires little endian machine => check.

    if (sf && sf->glLittleEndian)
    {
        U32 tmp = 0x12345678;
        if (*(U8*)&tmp != 0x78)
            sf = NULL;
    }

    // Maps directly to a GL format => done.

   if (sf && sf->glInternalFormat != GL_NONE)
        return getID();

    // Otherwise => select the closest match.

    U32 channels = 0;
    bool isFloat = false;
    for (int i = 0; i < getNumChannels(); i++)
    {
        const ImageFormat::Channel& c = getChannel(i);
        if (c.Type > ImageFormat::ChannelType_A)
            continue;

        channels |= 1 < c.Type;
        if (c.format == ImageFormat::ChannelFormat_Float)
            isFloat = true;
    }

    if ((channels & 7) == 0)
        return (isFloat) ? ImageFormat::A_F32 : ImageFormat::A8;
    if ((channels & 8) == 0)
        return (isFloat) ? ImageFormat::RGB_Vec3f : ImageFormat::R8_G8_B8;
    return (isFloat) ? ImageFormat::RGBA_Vec4f : ImageFormat::R8_G8_B8_A8;
}

//------------------------------------------------------------------------

bool ImageFormat::operator==(const ImageFormat& other) const
{
    if (m_id < ID_Generic || other.m_id < ID_Generic)
        return (getID() == other.getID());

    return (
        m_genericBPP == other.m_genericBPP &&
        m_genericChannels.size() == other.m_genericChannels.size() &&
        memcmp(m_genericChannels.data(), other.m_genericChannels.data(), m_genericChannels.size() * sizeof(m_genericChannels[0]) == 0));
}

//------------------------------------------------------------------------
Image::Image(const Vec2i& size, const ImageFormat& format, void* ptr, S64 stride)
{
    init(size, format);
    FW_ASSERT(size.min() == 0 || ptr);

    S64 lo = 0;
    S64 hi = 0;
    if (size.min() != 0)
    {
        lo = min(stride * (size.y - 1), (S64)0);
        hi = max(stride * (size.y - 1), (S64)0) + size.x * format.getBPP();
    }

    m_stride    = stride;
    m_buffer    = new Buffer((U8*)ptr + lo, hi - lo);
    m_ownBuffer = true;
    m_offset    = -lo;
}

//------------------------------------------------------------------------

Image::Image(const Vec2i& size, const ImageFormat& format, Buffer& buffer, S64 ofs, S64 stride)
{
    init(size, format);
    FW_ASSERT(size.min() == 0 || ofs + min(stride * (size.y - 1), (S64)0) >= 0);
    FW_ASSERT(size.min() == 0 || ofs + max(stride * (size.y - 1), (S64)0) + size.x * format.getBPP() <= buffer.getSize());

    m_stride    = stride;
    m_buffer    = &buffer;
    m_ownBuffer = false;
    m_offset    = ofs;
}

//------------------------------------------------------------------------

Image::~Image(void)
{
    if (m_ownBuffer)
        delete m_buffer;
}

//------------------------------------------------------------------------

U32 Image::getABGR(const Vec2i& pos) const
{
    FW_ASSERT(contains(pos, 1));
    const U8* p = getPtr(pos);

    switch (m_format.getID())
    {
    case ImageFormat::R8_G8_B8:     return p[0] | (p[1] << 8) | (p[2] << 16) | 0xFF000000;
    case ImageFormat::R8_G8_B8_A8:  return p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24);
    case ImageFormat::A8:           return *p << 24;
    case ImageFormat::XBGR_8888:    return *(const U32*)p | 0xFF000000;
    case ImageFormat::ABGR_8888:    return *(const U32*)p;

    case ImageFormat::RGB_565:      { U16 v = *(const U16*)p; return RGB_565_TO_ABGR_8888(v); }
    case ImageFormat::RGBA_5551:    { U16 v = *(const U16*)p; return RGBA_5551_TO_ABGR_8888(v); }

    case ImageFormat::RGB_Vec3f:    return Vec4f(*(const Vec3f*)p, 1.0f).toABGR();
    case ImageFormat::RGBA_Vec4f:   return ((const Vec4f*)p)->toABGR();
    case ImageFormat::A_F32:        return clamp((int)(*(const F32*)p * 255.0f + 0.5f), 0x00, 0xFF) << 24;

    default:
        {
            getChannels(pos);
            bool hasAlpha = false;
            U32 value = 0;

            for (int i = 0; i < m_channelTmp.getSize(); i++)
            {
                U32 v = clamp((int)(m_channelTmp[i] * 255.0f + 0.5f), 0x00, 0xFF);
                switch (m_format.getChannel(i).Type)
                {
                case ImageFormat::ChannelType_R:    value |= v; break;
                case ImageFormat::ChannelType_G:    value |= v << 8; break;
                case ImageFormat::ChannelType_B:    value |= v << 16; break;
                case ImageFormat::ChannelType_A:    value |= v << 24; hasAlpha = true; break;
                }
            }

            if (!hasAlpha)
                value |= 0xFF000000;
            return value;
        }
    }
}

//------------------------------------------------------------------------

void Image::setABGR(const Vec2i& pos, U32 value)
{
    FW_ASSERT(contains(pos, 1));
    U8* p = getMutablePtr(pos);

    switch (m_format.getID())
    {
    case ImageFormat::R8_G8_B8:     p[0] = (U8)value; p[1] = (U8)(value >> 8); p[2] = (U8)(value >> 16); break;
    case ImageFormat::R8_G8_B8_A8:  p[0] = (U8)value; p[1] = (U8)(value >> 8); p[2] = (U8)(value >> 16); p[3] = (U8)(value >> 24); break;
    case ImageFormat::A8:           *p = (U8)(value >> 24); break;
    case ImageFormat::XBGR_8888:    *(U32*)p = value; break;
    case ImageFormat::ABGR_8888:    *(U32*)p = value; break;

    case ImageFormat::RGB_565:      *(U16*)p = ABGR_8888_TO_RGB_565(value); break;
    case ImageFormat::RGBA_5551:    *(U16*)p = ABGR_8888_TO_RGBA_5551(value); break;

    case ImageFormat::RGB_Vec3f:    *(Vec3f*)p = Vec4f::fromABGR(value).getXYZ(); break;
    case ImageFormat::RGBA_Vec4f:   *(Vec4f*)p = Vec4f::fromABGR(value); break;
    case ImageFormat::A_F32:        *(F32*)p = (F32)(value >> 24) / 255.0f; break;

    default:
        for (int i = 0; i < m_channelTmp.getSize(); i++)
        {
            F32& channel = m_channelTmp[i];
            switch (m_format.getChannel(i).Type)
            {
            case ImageFormat::ChannelType_R:    channel = (F32)(value & 0xFF) / 255.0f; break;
            case ImageFormat::ChannelType_G:    channel = (F32)((value >> 8) & 0xFF) / 255.0f; break;
            case ImageFormat::ChannelType_B:    channel = (F32)((value >> 16) & 0xFF) / 255.0f; break;
            case ImageFormat::ChannelType_A:    channel = (F32)(value >> 24) / 255.0f; break;
            default:                            channel = 0.0f; break;
            }
        }
        setChannels(pos, m_channelTmp.getPtr());
        break;
    }
}

//------------------------------------------------------------------------

Vec4f Image::getVec4f(const Vec2i& pos) const
{
    FW_ASSERT(contains(pos, 1));
    const U8* p = getPtr(pos);

    switch (m_format.getID())
    {
    case ImageFormat::A8:           return Vec4f(0.0f, 0.0f, 0.0f, (F32)(*p / 255.0f));
    case ImageFormat::XBGR_8888:    return Vec4f::fromABGR(*(const U32*)p | 0xFF000000);
    case ImageFormat::ABGR_8888:    return Vec4f::fromABGR(*(const U32*)p);
    case ImageFormat::RGB_Vec3f:    return Vec4f(*(const Vec3f*)p, 1.0f);
    case ImageFormat::RGBA_Vec4f:   return *(const Vec4f*)p;
    case ImageFormat::A_F32:        return Vec4f(0.0f, 0.0f, 0.0f, *(const F32*)p);

    case ImageFormat::R8_G8_B8:
    case ImageFormat::R8_G8_B8_A8:
    case ImageFormat::RGB_565:
    case ImageFormat::RGBA_5551:
        return Vec4f::fromABGR(getABGR(pos));

    default:
        {
            getChannels(pos);
            Vec4f value(0.0f, 0.0f, 0.0f, 1.0f);
            for (int i = 0; i < m_channelTmp.getSize(); i++)
            {
                ImageFormat::ChannelType t = m_format.getChannel(i).Type;
                if (t <= ImageFormat::ChannelType_A)
                    value[t] = m_channelTmp[i];
            }
            return value;
        }
    }
}

//------------------------------------------------------------------------

void Image::setVec4f(const Vec2i& pos, const Vec4f& value)
{
    FW_ASSERT(contains(pos, 1));
    U8* p = getMutablePtr(pos);

    switch (m_format.getID())
    {
    case ImageFormat::A8:           *p = (U8)clamp((int)(value.w * 255.0f + 0.5f), 0x00, 0xFF); break;
    case ImageFormat::XBGR_8888:    *(U32*)p = value.toABGR(); break;
    case ImageFormat::ABGR_8888:    *(U32*)p = value.toABGR(); break;
    case ImageFormat::RGB_Vec3f:    *(Vec3f*)p = value.getXYZ(); break;
    case ImageFormat::RGBA_Vec4f:   *(Vec4f*)p = value; break;
    case ImageFormat::A_F32:        *(F32*)p = value.w; break;

    case ImageFormat::R8_G8_B8:
    case ImageFormat::R8_G8_B8_A8:
    case ImageFormat::RGB_565:
    case ImageFormat::RGBA_5551:
        setABGR(pos, value.toABGR());
        break;

    default:
        for (int i = 0; i < m_channelTmp.getSize(); i++)
        {
            ImageFormat::ChannelType t = m_format.getChannel(i).Type;
            m_channelTmp[i] = (t <= ImageFormat::ChannelType_A) ? value[t] : 0.0f;
        }
        setChannels(pos, m_channelTmp.getPtr());
        break;
    }
}

//------------------------------------------------------------------------

void Image::flipX(void)
{
    int bpp = getBPP();
    for (int y = 0; y < m_size.y; y++)
    {
        U8* ptrA = getMutablePtr(Vec2i(0, y));
        U8* ptrB = getMutablePtr(Vec2i(m_size.x - 1, y));
        for (int x = (m_size.x >> 1); x > 0; x--)
        {
            for (int i = 0; i < bpp; i++)
                swap(ptrA[i], ptrB[i]);
            ptrA += bpp;
            ptrB -= bpp;
        }
    }
}

//------------------------------------------------------------------------

void Image::flipY(void)
{
    int scanBytes = m_size.x * getBPP();
    Array<U8> tmp(NULL, scanBytes);
    for (int y = (m_size.y >> 1) - 1; y >= 0; y--)
    {
        U8* ptrA = getMutablePtr(Vec2i(0, y));
        U8* ptrB = getMutablePtr(Vec2i(0, m_size.y - 1 - y));
        memcpy(tmp.getPtr(), ptrA, scanBytes);
        memcpy(ptrA, ptrB, scanBytes);
        memcpy(ptrB, tmp.getPtr(), scanBytes);
    }
}

//------------------------------------------------------------------------

GLuint Image::createGLTexture(ImageFormat::ID desiredFormat, bool generateMipmaps) const
{
    // Select format.

    ImageFormat::ID formatID;
    if (desiredFormat == ImageFormat::ID_Max)
        formatID = m_format.getGLFormat();
    else
        formatID = ImageFormat(desiredFormat).getGLFormat();

    const ImageFormat::StaticFormat* sf = ImageFormat(formatID).getStaticFormat();
    FW_ASSERT(sf);

    // Image data not usable directly => convert.

    Image* converted = NULL;
    const Image* img = this;
    if (m_size.min() == 0 || m_format.getID() != formatID || m_stride != getBPP() * m_size.x)
    {
        converted = new Image(max(m_size, 1), formatID);
        converted->set(*this);
        img = converted;
    }

    // create texture.

    GLContext::staticInit();

    GLint oldTex = 0;
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &oldTex);

    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, generateMipmaps);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, (generateMipmaps) ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Uncomment to enable anisotropic filtering:
//    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, FW_S32_MAX);

    glTexImage2D(GL_TEXTURE_2D, 0, sf->glInternalFormat,
        img->getSize().x, img->getSize().y,
        0, sf->glFormat, sf->glType, img->getPtr());

    glBindTexture(GL_TEXTURE_2D, oldTex);
    GLContext::checkErrors();

    // Clean up.

    delete converted;
    return tex;
}

//------------------------------------------------------------------------

ImageFormat Image::chooseCudaFormat(CUDA_ARRAY_DESCRIPTOR* desc, ImageFormat::ID desiredFormat) const
{
#if (!FW_USE_CUDA)

    FW_UNREF(desc);
    FW_UNREF(desiredFormat);
    fail("Image::chooseCudaFormat(): Built without FW_USE_CUDA!");
    return m_format;

#else

    // Gather requirements.

    ImageFormat refFormat = m_format;
    if (desiredFormat != ImageFormat::ID_Max)
        refFormat = desiredFormat;

    int numChannels = min(refFormat.getNumChannels(), 4);
    int channelBits = 0;
    bool isFloat = false;
    for (int i = 0; i < numChannels; i++)
    {
        const ImageFormat::Channel& chan = refFormat.getChannel(i);
        channelBits = max(channelBits, chan.fieldSize);
        isFloat = (chan.format == ImageFormat::ChannelFormat_Float);
    }

    // Select format.

    CUarray_format datatype;
    int wordSize;

    if (isFloat)                datatype = CU_AD_FORMAT_FLOAT,          wordSize = 4;
    else if (channelBits <= 8)  datatype = CU_AD_FORMAT_UNSIGNED_INT8,  wordSize = 1;
    else if (channelBits <= 16) datatype = CU_AD_FORMAT_UNSIGNED_INT16, wordSize = 2;
    else                        datatype = CU_AD_FORMAT_UNSIGNED_INT32, wordSize = 4;

    ImageFormat formatA; // word per channel
    ImageFormat formatB; // single word

    for (int i = 0; i < numChannels; i++)
    {
        const ImageFormat::Channel& ref = refFormat.getChannel(i);

        ImageFormat::Channel chan;
        chan.Type       = ref.Type;
        chan.format     = (isFloat) ? ImageFormat::ChannelFormat_Float : ref.format;

        chan.wordOfs    = i * wordSize;
        chan.wordSize   = wordSize;
        chan.fieldOfs   = 0;
        chan.fieldSize  = wordSize * 8;
        formatA.addChannel(chan);

        chan.wordOfs    = 0;
        chan.wordSize   = wordSize * numChannels;
        chan.fieldOfs   = i * wordSize * 8;
        chan.fieldSize  = wordSize * 8;
        formatB.addChannel(chan);
    }

    // Fill in the descriptor.

    if (desc)
    {
        memset(desc, 0, sizeof(CUDA_ARRAY_DESCRIPTOR));
        desc->Width         = m_size.x;
        desc->Height        = m_size.y;
        desc->Format        = datatype;
        desc->NumChannels   = numChannels;
    }
    return (formatB == refFormat) ? formatB : formatA;

#endif
}

//------------------------------------------------------------------------

CUarray Image::createCudaArray(ImageFormat::ID desiredFormat, ImageFormat* formatOut, CUDA_ARRAY_DESCRIPTOR* arrayDescOut) const
{
#if (!FW_USE_CUDA)

    FW_UNREF(desiredFormat);
    FW_UNREF(formatOut);
    FW_UNREF(arrayDescOut);
    fail("Image::createCudaArray(): Built without FW_USE_CUDA!");
    return NULL;

#else

    // Choose format.

    CUDA_ARRAY_DESCRIPTOR arrayDesc;
    ImageFormat cudaFormat = chooseCudaFormat(&arrayDesc, desiredFormat);

    // Image data not usable directly => convert.

    Image* converted = NULL;
    const Image* img = this;
    if (m_size.min() == 0 || m_format != cudaFormat)
    {
        converted = new Image(max(m_size, 1), cudaFormat);
        converted->set(*this);
        img = converted;
        arrayDesc.Width = img->getSize().x;
        arrayDesc.Height = img->getSize().y;
    }

    // create CUDA array.

    CudaModule::staticInit();

    CUarray cudaArray;
    CudaModule::checkError("cuArrayCreate", cuArrayCreate(&cudaArray, &arrayDesc));

    // Upload data.

    CUDA_MEMCPY2D copyDesc;
    memset(&copyDesc, 0, sizeof(CUDA_MEMCPY2D));

    copyDesc.srcXInBytes    = 0;
    copyDesc.srcY           = 0;
    copyDesc.srcMemoryType  = CU_MEMORYTYPE_HOST;
    copyDesc.srcHost        = img->getPtr();
    copyDesc.srcPitch       = img->getSize().x * img->getBPP();
    copyDesc.dstXInBytes    = 0;
    copyDesc.dstY           = 0;
    copyDesc.dstMemoryType  = CU_MEMORYTYPE_ARRAY;
    copyDesc.dstArray       = cudaArray;
    copyDesc.WidthInBytes   = img->getSize().x * img->getBPP();
    copyDesc.Height         = img->getSize().y;

    CudaModule::checkError("cuMemcpy2D", cuMemcpy2D(&copyDesc));

    // Set output parameters.

    if (formatOut)
        *formatOut = cudaFormat;

    if (arrayDescOut)
        *arrayDescOut = arrayDesc;

    delete converted;
    return cudaArray;

#endif
}

//------------------------------------------------------------------------
// TODO: Figure out a cleaner interface for this.
// TODO: Expose wrap mode, filter mode, and flags.
// TODO: Support mipmapping.

CUtexObject Image::createCudaTexObject(ImageFormat::ID desiredFormat, ImageFormat* formatOut, CUDA_ARRAY_DESCRIPTOR* arrayDescOut, CUarray* arrayOut) const
{
#if (!FW_USE_CUDA)

    FW_UNREF(desiredFormat);
    FW_UNREF(formatOut);
    FW_UNREF(arrayDescOut);
    FW_UNREF(arrayOut);
    fail("Image::createCudaTexObject(): Built without FW_USE_CUDA!");
    return NULL;

#else

    // create CUDA array.

    CUDA_ARRAY_DESCRIPTOR arrayDesc;
    CUarray cudaArray = createCudaArray(desiredFormat, formatOut, &arrayDesc);

    // Fill in the resource descriptor.

    CUDA_RESOURCE_DESC resDesc;
    memset(&resDesc, 0, sizeof(resDesc));

    resDesc.resType             = CU_RESOURCE_TYPE_ARRAY;
    resDesc.res.array.hArray    = cudaArray;
    resDesc.flags               = 0;

    // Fill in the texture descriptor.

    CUDA_TEXTURE_DESC texDesc;
    memset(&texDesc, 0, sizeof(texDesc));

    texDesc.addressMode[0]      = CU_TR_ADDRESS_MODE_WRAP;
    texDesc.addressMode[1]      = CU_TR_ADDRESS_MODE_WRAP;
    texDesc.addressMode[2]      = CU_TR_ADDRESS_MODE_WRAP;
    texDesc.filterMode          = CU_TR_FILTER_MODE_LINEAR;
    texDesc.flags               = CU_TRSF_NORMALIZED_COORDINATES;
    texDesc.maxAnisotropy       = 16;
    texDesc.mipmapFilterMode    = CU_TR_FILTER_MODE_LINEAR;
    texDesc.mipmapLevelBias     = 0.0f;
    texDesc.minMipmapLevelClamp = 0.0f;
    texDesc.maxMipmapLevelClamp = 16.0f;

    // create CUDA texture object.

    CudaModule::staticInit();

    CUtexObject cudaTexObject;
    CudaModule::checkError("cuTexObjectCreate", cuTexObjectCreate(&cudaTexObject, &resDesc, &texDesc, NULL));

    // Set output parameters.

    if (arrayDescOut)
        *arrayDescOut = arrayDesc;

    if (arrayOut)
        *arrayOut = cudaArray;

    return cudaTexObject;

#endif
}

//------------------------------------------------------------------------
// Implements a polyphase filter with round-down semantics from:
//
// Non-Power-of-Two Mipmapping
// (NVIDIA whitepaper)
// http://developer.nvidia.com/object/np2_mipmapping.html

Image* Image::downscale2x(void) const
{
    // 1x1 or smaller => Bail out.

    int area = m_size.x * m_size.y;
    if (area <= 1)
        return NULL;

    // Choose filter dimensions.

    int fw = (m_size.x == 1) ? 1 : ((m_size.x & 1) == 0) ? 2 : 3;
    int fh = (m_size.y == 1) ? 1 : ((m_size.y & 1) == 0) ? 2 : 3;
    Vec2i resSize = max(m_size >> 1, 1);
    int halfArea = area >> 1;

    // Allocate temporary scanline buffer and result image.

    Image tmp(Vec2i(m_size.x, fh), ImageFormat::ABGR_8888);
    Image* res = new Image(resSize, ImageFormat::ABGR_8888);
    U32* resPtr = (U32*)res->getMutablePtr();

    // Process each scanline in the result.

    for (int y = 0; y < resSize.y; y++)
    {
        // Copy source scanlines into the temporary buffer.

        tmp.set(0, *this, Vec2i(0, y * 2), Vec2i(m_size.x, fh));

        // Choose weights along the Y-axis.

        Vec3i wy(resSize.y);
        if (fh == 3)
            wy = Vec3i(resSize.y - y, resSize.y, y + 1);

        // Process each pixel in the result.

        for (int x = 0; x < resSize.x; x++)
        {
            // Choose weights along the X-axis.

            Vec3i wx(resSize.x);
            if (fw == 3)
                wx = Vec3i(resSize.x - x, resSize.x, x + 1);

            // Compute weighted average of pixel values.

            Vec4i sum = 0;
            const U32* tmpPtr = (const U32*)tmp.getPtr(Vec2i(x * 2, 0));

            for (int yy = 0; yy < fh; yy++)
            {
                for (int xx = 0; xx < fw; xx++)
                {
                    U32 abgr = tmpPtr[xx];
                    int weight = wx[xx] * wy[yy];
                    sum.x += (abgr & 0xFF) * weight;
                    sum.y += ((abgr >> 8) & 0xFF) * weight;
                    sum.z += ((abgr >> 16) & 0xFF) * weight;
                    sum.w += (abgr >> 24) * weight;
                }
                tmpPtr += m_size.x;
            }

            sum = (sum + halfArea) / area;
            *resPtr++ = sum.x | (sum.y << 8) | (sum.z << 16) | (sum.w << 24);
        }
    }
    return res;
}

//------------------------------------------------------------------------

void Image::init(const Vec2i& size, const ImageFormat& format)
{
    FW_ASSERT(size.min() >= 0);
    m_size = size;
    m_format = format;
    m_channelTmp.resize(m_format.getNumChannels());
}

//------------------------------------------------------------------------

void Image::createBuffer(void)
{
    m_stride    = m_size.x * m_format.getBPP();
    m_buffer    = new Buffer;
    m_buffer->resize(m_stride * m_size.y);
    m_ownBuffer = true;
    m_offset    = 0;
}

//------------------------------------------------------------------------

void Image::replicatePixel(void)
{
    if (m_size.min() == 0)
        return;

    int bpp = getBPP();
    U8* ptr = getMutablePtr();
    int scanBytes = m_size.x * bpp;

    for (int x = 1; x < m_size.x; x++)
        memcpy(ptr + x * bpp, ptr, bpp);
    for (int y = 1; y < m_size.y; y++)
        memcpy(ptr + y * m_stride, ptr, scanBytes);
}

//------------------------------------------------------------------------

bool Image::canBlitDirectly(const ImageFormat& format)
{
    switch (format.getID())
    {
    case ImageFormat::R8_G8_B8:     return true;
    case ImageFormat::R8_G8_B8_A8:  return true;
    case ImageFormat::A8:           return true;
    case ImageFormat::XBGR_8888:    return true;
    case ImageFormat::ABGR_8888:    return true;

    case ImageFormat::RGB_565:      return true;
    case ImageFormat::RGBA_5551:    return true;

    case ImageFormat::RGB_Vec3f:    return true;
    case ImageFormat::RGBA_Vec4f:   return true;
    case ImageFormat::A_F32:        return true;

    default:                        return false;
    }
}

//------------------------------------------------------------------------

bool Image::canBlitThruABGR(const ImageFormat& format)
{
    switch (format.getID())
    {
    case ImageFormat::R8_G8_B8:     return true;
    case ImageFormat::R8_G8_B8_A8:  return true;
    case ImageFormat::A8:           return true;
    case ImageFormat::XBGR_8888:    return true;
    case ImageFormat::ABGR_8888:    return true;

    case ImageFormat::RGB_565:      return true;
    case ImageFormat::RGBA_5551:    return true;

    case ImageFormat::RGB_Vec3f:    return false;
    case ImageFormat::RGBA_Vec4f:   return false;
    case ImageFormat::A_F32:        return false;

    default:                        return false;
    }
}

//------------------------------------------------------------------------

void Image::blit(
    const ImageFormat& dstFormat, U8* dstPtr, S64 dstStride,
    const ImageFormat& srcFormat, const U8* srcPtr, S64 srcStride,
    const Vec2i& size)
{
    FW_ASSERT(size.min() >= 0);
    if (size.min() == 0)
        return;

    // Same format?

    if (dstFormat == srcFormat)
    {
        int scanBytes = size.x * dstFormat.getBPP();
        for (int y = 0; y < size.y; y++)
            memcpy(dstPtr + dstStride * y, srcPtr + srcStride * y, scanBytes);
        return;
    }

    // To ABGR_8888?

    if (dstFormat.getID() == ImageFormat::ABGR_8888 && canBlitDirectly(srcFormat))
    {
        for (int y = 0; y < size.y; y++)
            blitToABGR((U32*)(dstPtr + dstStride * y), srcFormat, srcPtr + srcStride * y, size.x);
        return;
    }

    // From ABGR_8888?

    if (srcFormat.getID() == ImageFormat::ABGR_8888 && canBlitDirectly(dstFormat))
    {
        for (int y = 0; y < size.y; y++)
            blitFromABGR(dstFormat, dstPtr + dstStride * y, (const U32*)(srcPtr + srcStride * y), size.x);
        return;
    }

    // From integer-based format to another => convert thru ABGR_8888.

    if (canBlitDirectly(srcFormat) && canBlitDirectly(dstFormat) && canBlitThruABGR(srcFormat))
    {
        Array<U32> tmp(NULL, size.x);
        for (int y = 0; y < size.y; y++)
        {
            blitToABGR(tmp.getPtr(), srcFormat, srcPtr + srcStride * y, size.x);
            blitFromABGR(dstFormat, dstPtr + dstStride * y, tmp.getPtr(), size.x);
        }
        return;
    }

    // General case.

    S64 dstBPP = dstFormat.getBPP();
    S64 srcBPP = srcFormat.getBPP();
    Array<F32> dv(NULL, dstFormat.getNumChannels());
    Array<F32> sv(NULL, srcFormat.getNumChannels());
    Array<Vec2i> map;

    for (int i = 0; i < dstFormat.getNumChannels(); i++)
    {
        ImageFormat::ChannelType t = dstFormat.getChannel(i).Type;
        dv[i] = (t == ImageFormat::ChannelType_A) ? 1.0f : 0.0f;
        int si = srcFormat.findChannel(t);
        if (si != -1)
            map.add(Vec2i(i, si));
    }

    for (int y = 0; y < size.y; y++)
    {
        U8* dstPixel = dstPtr + dstStride * y;
        const U8* srcPixel = srcPtr + srcStride * y;

        for (int x = 0; x < size.x; x++)
        {
            getChannels(sv.getPtr(), srcPixel, srcFormat, 0, sv.getSize());
            for (int i = 0; i < map.getSize(); i++)
                dv[map[i].x] = sv[map[i].y];
            setChannels(dstPixel, dv.getPtr(), dstFormat, 0, dv.getSize());

            dstPixel += dstBPP;
            srcPixel += srcBPP;
        }
    }
}

//------------------------------------------------------------------------

void Image::blitToABGR(U32* dstPtr,  const ImageFormat& srcFormat, const U8* srcPtr, int width)
{
    FW_ASSERT(width > 0);
    FW_ASSERT(dstPtr && srcPtr);
    FW_ASSERT(canBlitDirectly(srcFormat));

    const U8*       s8  = srcPtr;
    const U16*      s16 = (const U16*)srcPtr;
    const Vec3f*    sv3 = (const Vec3f*)srcPtr;
    const Vec4f*    sv4 = (const Vec4f*)srcPtr;
    const F32*      sf  = (const F32*)srcPtr;

    switch (srcFormat.getID())
    {
    case ImageFormat::R8_G8_B8:     for (int x = width; x > 0; x--) { *dstPtr++ = s8[0] | (s8[1] << 8) | (s8[2] << 16) | 0xFF000000; s8 += 3; } break;
    case ImageFormat::R8_G8_B8_A8:  for (int x = width; x > 0; x--) { *dstPtr++ = s8[0] | (s8[1] << 8) | (s8[2] << 16) | (s8[3] << 24); s8 += 4; } break;
    case ImageFormat::A8:           for (int x = width; x > 0; x--) *dstPtr++ = *s8++ << 24; break;
    case ImageFormat::XBGR_8888:    memcpy(dstPtr, srcPtr, width * sizeof(U32)); break;
    case ImageFormat::ABGR_8888:    memcpy(dstPtr, srcPtr, width * sizeof(U32)); break;

    case ImageFormat::RGB_565:      for (int x = width; x > 0; x--) { U16 v = *s16++; *dstPtr++ = RGB_565_TO_ABGR_8888(v); } break;
    case ImageFormat::RGBA_5551:    for (int x = width; x > 0; x--) { U16 v = *s16++; *dstPtr++ = RGBA_5551_TO_ABGR_8888(v); } break;

    case ImageFormat::RGB_Vec3f:    for (int x = width; x > 0; x--) *dstPtr++ = Vec4f(*sv3++, 1.0f).toABGR(); break;
    case ImageFormat::RGBA_Vec4f:   for (int x = width; x > 0; x--) *dstPtr++ = (sv4++)->toABGR(); break;
    case ImageFormat::A_F32:        for (int x = width; x > 0; x--) *dstPtr++ = clamp((int)(*sf++ * 255.0f + 0.5f), 0x00, 0xFF) << 24; break;

    default:                        FW_ASSERT(false); break;
    }
}

//------------------------------------------------------------------------

void Image::blitFromABGR(const ImageFormat& dstFormat, U8* dstPtr, const U32* srcPtr, int width)
{
    FW_ASSERT(width > 0);
    FW_ASSERT(dstPtr && srcPtr);
    FW_ASSERT(canBlitDirectly(dstFormat));

    U8*         d8  = dstPtr;
    U16*        d16 = (U16*)dstPtr;
    Vec3f*      dv3 = (Vec3f*)dstPtr;
    Vec4f*      dv4 = (Vec4f*)dstPtr;
    F32*        df  = (F32*)dstPtr;

    switch (dstFormat.getID())
    {
    case ImageFormat::R8_G8_B8:     for (int x = width; x > 0; x--) { U32 v = *srcPtr++; *d8++ = (U8)v; *d8++ = (U8)(v >> 8); *d8++ = (U8)(v >> 16); } break;
    case ImageFormat::R8_G8_B8_A8:  for (int x = width; x > 0; x--) { U32 v = *srcPtr++; *d8++ = (U8)v; *d8++ = (U8)(v >> 8); *d8++ = (U8)(v >> 16); *d8++ = (U8)(v >> 24); } break;
    case ImageFormat::A8:           for (int x = width; x > 0; x--) *d8++ = (U8)(*srcPtr++ >> 24); break;
    case ImageFormat::XBGR_8888:    memcpy(dstPtr, srcPtr, width * sizeof(U32)); break;
    case ImageFormat::ABGR_8888:    memcpy(dstPtr, srcPtr, width * sizeof(U32)); break;

    case ImageFormat::RGB_565:      for (int x = width; x > 0; x--) { U32 v = *srcPtr++; *d16++ = ABGR_8888_TO_RGB_565(v); } break;
    case ImageFormat::RGBA_5551:    for (int x = width; x > 0; x--) { U32 v = *srcPtr++; *d16++ = ABGR_8888_TO_RGBA_5551(v); } break;

    case ImageFormat::RGB_Vec3f:    for (int x = width; x > 0; x--) *dv3++ = Vec4f::fromABGR(*srcPtr++).getXYZ(); break;
    case ImageFormat::RGBA_Vec4f:   for (int x = width; x > 0; x--) *dv4++ = Vec4f::fromABGR(*srcPtr++); break;
    case ImageFormat::A_F32:        for (int x = width; x > 0; x--) *df++ = (F32)(*srcPtr++ >> 24) / 255.0f; break;

    default:                        FW_ASSERT(false); break;
    }
}

//------------------------------------------------------------------------

void Image::getChannels(F32* values, const U8* pixelPtr, const ImageFormat& format, int first, int num)
{
    FW_ASSERT(num >= 0);
    FW_ASSERT((values && pixelPtr) || !num);
    FW_ASSERT(first >= 0 && first + num <= format.getNumChannels());

    for (int i = 0; i < num; i++)
    {
        const ImageFormat::Channel& c = format.getChannel(i + first);
        const U8* wordPtr = pixelPtr + c.wordOfs;
        U32 field;
        switch (c.wordSize)
        {
        case 1:     field = *wordPtr; break;
        case 2:     field = *(const U16*)wordPtr; break;
        case 4:     field = *(const U32*)wordPtr; break;
        default:    FW_ASSERT(false); return;
        }
        field >>= c.fieldOfs;

        U32 mask = (1 << c.fieldSize) - 1;
        switch (c.format)
        {
        case ImageFormat::ChannelFormat_Clamp:  values[i] = (F32)(field & mask) / (F32)mask; break;
        case ImageFormat::ChannelFormat_Int:    values[i] = (F32)(field & mask); break;
        case ImageFormat::ChannelFormat_Float:  FW_ASSERT(c.fieldSize == 32); values[i] = bitsToFloat(field); break;
        default:                                FW_ASSERT(false); return;
        }
    }
}

//------------------------------------------------------------------------

void Image::setChannels(U8* pixelPtr, const F32* values, const ImageFormat& format, int first, int num)
{
    FW_ASSERT(num >= 0);
    FW_ASSERT((pixelPtr && values) || !num);
    FW_ASSERT(first >= 0 && first + num <= format.getNumChannels());

    memset(pixelPtr, 0, format.getBPP());

    for (int i = 0; i < num; i++)
    {
        const ImageFormat::Channel& c = format.getChannel(i + first);
        U32 mask = (1 << c.fieldSize) - 1;
        U32 field;
        switch (c.format)
        {
        case ImageFormat::ChannelFormat_Clamp:  field = min((U32)max(values[i] * (F32)mask + 0.5f, 0.0f), mask); break;
        case ImageFormat::ChannelFormat_Int:    field = min((U32)max(values[i] + 0.5f, 0.0f), mask); break;
        case ImageFormat::ChannelFormat_Float:  FW_ASSERT(c.fieldSize == 32); field = floatToBits(values[i]); break;
        default:                                FW_ASSERT(false); return;
        }
        field <<= c.fieldOfs;

        U8* wordPtr = pixelPtr + c.wordOfs;
        switch (c.wordSize)
        {
        case 1:     *wordPtr |= (U8)field; break;
        case 2:     *(U16*)wordPtr |= (U16)field; break;
        case 4:     *(U32*)wordPtr |= field; break;
        default:    FW_ASSERT(false); return;
        }
    }
}

//------------------------------------------------------------------------
#endif