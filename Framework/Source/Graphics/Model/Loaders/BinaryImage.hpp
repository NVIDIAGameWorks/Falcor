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
// #include "gpu/Buffer.hpp"
// #include "io/FileFormats.hpp"
#include <vector>
using namespace Falcor;

namespace FW
{
//------------------------------------------------------------------------
    using S32 = int32_t;
    using U32 = uint32_t;
    using U8 = uint8_t;
#define FW_ASSERT assert
#define FW_ARRAY_SIZE arraysize

class ImageFormat
{
public:
    enum ID
    {
        R8_G8_B8 = 0,
        R8_G8_B8_A8,
        A8,
        XBGR_8888,
        ABGR_8888,

        RGB_565,
        RGBA_5551,

        RGB_Vec3f,
        RGBA_Vec4f,
        A_F32,

        BGRA_8888,
        BGR_888,
        RG_88,
        R8,

        // Compressed formats
        S3TC_DXT1,
        S3TC_DXT3,
        S3TC_DXT5,
        RGTC_R,
        RGTC_RG,

        ID_Generic,
        ID_Max
    };

    enum ChannelType // allows arbitrary values
    {
        ChannelType_R   = 0,
        ChannelType_G,
        ChannelType_B,
        ChannelType_A,
        ChannelType_Generic,

        ChannelType_Max
    };

    enum ChannelFormat
    {
        ChannelFormat_Clamp = 0,    // [0, 1]
        ChannelFormat_Int,          // [0, n[
        ChannelFormat_Float,        // any

        ChannelFormat_Max
    };

    struct Channel
    {
        ChannelType     Type;
        ChannelFormat   format;
        S32             wordOfs;    // bytes
        S32             wordSize;   // bytes
        S32             fieldOfs;   // bits
        S32             fieldSize;  // bits
    };

    struct StaticFormat
    {
        S32             bpp;
        S32             numChannels;
        Channel         channels[4];
//         GLenum          glInternalFormat;
//         GLenum          glFormat;
//         GLenum          glType;
//        bool            glLittleEndian;
    };

public:
                        ImageFormat     (void)                              { clear(); }
                        ImageFormat     (ID id)                             : m_id(id) { FW_ASSERT(id >= 0 && id < ID_Generic); }
                        ImageFormat     (const ImageFormat& other)          { set(other); }
                        ~ImageFormat    (void)                              {}

    ID                  getID           (void) const;
    const StaticFormat* getStaticFormat (void) const;
    int                 getBPP          (void) const;
    int                 getNumChannels  (void) const;
    const Channel&      getChannel      (int idx) const;
    int                 findChannel     (ChannelType Type) const;
    bool                hasChannel      (ChannelType Type) const            { return (findChannel(Type) != -1); }

    void                set             (const ImageFormat& other);
    void                clear           (void);
    void                addChannel      (const Channel& channel);

    ID                  getGLFormat     (void) const;

    ImageFormat&        operator=       (const ImageFormat& other)          { set(other); return *this; }
    bool                operator==      (const ImageFormat& other) const;
    bool                operator!=      (const ImageFormat& other) const    { return (!operator==(other)); }

private:
    static const StaticFormat s_staticFormats[];

    mutable ID          m_id;           // ID_Max if unknown
    S32                 m_genericBPP;   // only if m_id >= ID_Generic
    std::vector<Channel>      m_genericChannels; // only if m_id >= ID_Generic
};

//------------------------------------------------------------------------
#if 0
class Image
{
public:
                        Image           (const Vec2i& size, const ImageFormat& format = ImageFormat::ABGR_8888) { init(size, format); createBuffer(); }
                        Image           (const Vec2i& size, const ImageFormat& format, void* ptr, S64 stride);
                        Image           (const Vec2i& size, const ImageFormat& format, Buffer& buffer, S64 ofs, S64 stride);
                        Image           (const Image& other)                { init(other.getSize(), other.getFormat()); createBuffer(); set(other); }
                        ~Image          (void);

    bool                contains        (const Vec2i& pos, const Vec2i& size) const { return (pos.x >= 0 && pos.y >= 0 && pos.x + size.x <= m_size.x && pos.y + size.y <= m_size.y); }

    const Vec2i&        getSize         (void) const                        { return m_size; }
    const ImageFormat&  getFormat       (void) const                        { return m_format; }
    int                 getBPP          (void) const                        { return m_format.getBPP(); }
    S64                 getStride       (void) const                        { return m_stride; }

    Buffer&             getBuffer       (void) const                        { return *m_buffer; }
    S64                 getOffset       (const Vec2i& pos = 0) const        { FW_ASSERT(contains(pos, 0)); return m_offset + pos.x * getBPP() + pos.y * getStride(); }
    const U8*           getPtr          (const Vec2i& pos = 0) const        { return (const U8*)m_buffer->getPtr(getOffset(pos)); }
    U8*                 getMutablePtr   (const Vec2i& pos = 0)              { return (U8*)m_buffer->getMutablePtr(getOffset(pos)); }

    void                read            (const ImageFormat& format, void* ptr, S64 stride, const Vec2i& pos, const Vec2i& size) const   { FW_ASSERT(contains(pos, size)); blit(format, (U8*)ptr, stride, getFormat(), getPtr(pos), getStride(), size); }
    void                read            (const ImageFormat& format, void* ptr, S64 stride) const                                        { blit(format, (U8*)ptr, stride, getFormat(), getPtr(), getStride(), getSize()); }
    void                write           (const ImageFormat& format, const void* ptr, S64 stride, const Vec2i& pos, const Vec2i& size)   { FW_ASSERT(contains(pos, size)); blit(getFormat(), getMutablePtr(pos), getStride(), format, (const U8*)ptr, stride, size); }
    void                write           (const ImageFormat& format, const void* ptr, S64 stride)                                        { blit(getFormat(), getMutablePtr(), getStride(), format, (const U8*)ptr, stride, getSize()); }
    void                set             (const Vec2i& dstPos, const Image& src, const Vec2i& srcPos, const Vec2i& size)                 { FW_ASSERT(contains(dstPos, size) && src.contains(srcPos, size)); blit(getFormat(), getMutablePtr(dstPos), getStride(), src.getFormat(), src.getPtr(srcPos), src.getStride(), size); }
    void                set             (const Image& src)                                                                              { blit(getFormat(), getMutablePtr(), getStride(), src.getFormat(), src.getPtr(), src.getStride(), Vec2i(min(getSize().x, src.getSize().x), min(getSize().y, src.getSize().y))); }

    void                clear           (U32 abgr = 0)                      { if (m_size.min() != 0) setABGR(0, abgr); replicatePixel(); }
    void                clear           (const Vec4f& color)                { if (m_size.min() != 0) setVec4f(0, color); replicatePixel(); }

    U32                 getABGR         (const Vec2i& pos) const;
    void                setABGR         (const Vec2i& pos, U32 value);
    Vec4f               getVec4f        (const Vec2i& pos) const;
    void                setVec4f        (const Vec2i& pos, const Vec4f& value);
    Vec3f               getVec3f        (const Vec2i& pos) const                { return getVec4f(pos).getXYZ(); }
    void                setVec3f        (const Vec2i& pos, const Vec3f& value)  { setVec4f(pos, Vec4f(value, 1.0f)); }

    U32                 getABGR         (int x, int y) const                    { return getABGR(Vec2i(x, y)); }
    void                setABGR         (int x, int y, U32 value)               { setABGR(Vec2i(x, y), value); }
    Vec4f               getVec4f        (int x, int y) const                    { return getVec4f(Vec2i(x, y)); }
    void                setVec4f        (int x, int y, const Vec4f& value)      { setVec4f(Vec2i(x, y), value); }
    Vec3f               getVec3f        (int x, int y) const                    { return getVec3f(Vec2i(x, y)); }
    void                setVec3f        (int x, int y, const Vec3f& value)      { setVec3f(Vec2i(x, y), value); }

    void                getChannels     (F32* values, const Vec2i& pos, int first, int num) const   { getChannels(values, getPtr(pos), getFormat(), first, num); }
    void                getChannels     (F32* values, const Vec2i& pos) const                       { getChannels(values, getPtr(pos), getFormat(), 0, getFormat().getNumChannels()); }
    const Array<F32>&   getChannels     (const Vec2i& pos) const                                    { getChannels(m_channelTmp.getPtr(), getPtr(pos), getFormat(), 0, getFormat().getNumChannels()); return m_channelTmp; }
    F32                 getChannel      (const Vec2i& pos, int idx) const                           { F32 res; getChannels(&res, getPtr(pos), getFormat(), idx, 1); return res; }
    void                setChannels     (const Vec2i& pos, const F32* values, int first, int num)   { setChannels(getMutablePtr(pos), values, getFormat(), first, num); }
    void                setChannels     (const Vec2i& pos, const F32* values)                       { setChannels(getMutablePtr(pos), values, getFormat(), 0, getFormat().getNumChannels()); }
    void                setChannel      (const Vec2i& pos, int idx, F32 value)                      { setChannels(getMutablePtr(pos), &value, getFormat(), idx, 1); }

    void                flipX           (void);
    void                flipY           (void);

    GLuint              createGLTexture (ImageFormat::ID desiredFormat = ImageFormat::ID_Max, bool generateMipmaps = true) const;

    ImageFormat         chooseCudaFormat(CUDA_ARRAY_DESCRIPTOR* desc = NULL, ImageFormat::ID desiredFormat = ImageFormat::ID_Max) const;
    CUarray             createCudaArray (ImageFormat::ID desiredFormat = ImageFormat::ID_Max, ImageFormat* formatOut = NULL, CUDA_ARRAY_DESCRIPTOR* arrayDescOut = NULL) const;
    CUtexObject         createCudaTexObject(ImageFormat::ID desiredFormat = ImageFormat::ID_Max, ImageFormat* formatOut = NULL, CUDA_ARRAY_DESCRIPTOR* arrayDescOut = NULL, CUarray* arrayOut = NULL) const;

    Image*              downscale2x     (void) const; // Returns ImageFormat::ABGR_8888, or NULL if size <= 1x1.

    Image&              operator=       (const Image& other)                { if (&other != this) set(other); return *this; }

private:
    void                init            (const Vec2i& size, const ImageFormat& format);
    void                createBuffer    (void);
    void                replicatePixel  (void);

    static bool         canBlitDirectly (const ImageFormat& format);
    static bool         canBlitThruABGR (const ImageFormat& format);

    static void         blit            (const ImageFormat& dstFormat, U8* dstPtr, S64 dstStride,
                                         const ImageFormat& srcFormat, const U8* srcPtr, S64 srcStride,
                                         const Vec2i& size);

    static void         blitToABGR      (U32* dstPtr,  const ImageFormat& srcFormat, const U8* srcPtr, int width);
    static void         blitFromABGR    (const ImageFormat& dstFormat, U8* dstPtr, const U32* srcPtr, int width);

    static void         getChannels     (F32* values, const U8* pixelPtr, const ImageFormat& format, int first, int num);
    static void         setChannels     (U8* pixelPtr, const F32* values, const ImageFormat& format, int first, int num);

private:
    Vec2i               m_size;
    ImageFormat         m_format;
    S64                 m_stride;
    Buffer*             m_buffer;
    bool                m_ownBuffer;
    S64                 m_offset;

    mutable Array<F32>  m_channelTmp;
};
#endif
//------------------------------------------------------------------------
}
