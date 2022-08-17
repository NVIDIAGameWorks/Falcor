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
#include "CryptoUtils.h"

namespace Falcor
{
    SHA1::SHA1() :
        mIndex(0),
        mBits(0)
    {
        mState[0] = 0x67452301;
        mState[1] = 0xefcdab89;
        mState[2] = 0x98badcfe;
        mState[3] = 0x10325476;
        mState[4] = 0xc3d2e1f0;
    }

    void SHA1::update(uint8_t byte)
    {
        addByte(byte);
        mBits += 8;
    }

    void SHA1::update(const void* data, size_t len)
    {
        if (!data) return;

        const uint8_t *ptr = reinterpret_cast<const uint8_t*>(data);

        // Fill up buffer if not full.
        while (len > 0 && mIndex != 0)
        {
            update(*ptr++);
            len--;
        }

        // Process full blocks.
        while (len >= sizeof(mBuf))
        {
            processBlock(ptr);
            ptr += sizeof(mBuf);
            len -= sizeof(mBuf);
            mBits += sizeof(mBuf) * 8;
        }

        // Process remaining bytes.
        while (len > 0)
        {
            update(*ptr++);
            len--;
        }
    }

    SHA1::MD SHA1::finalize()
    {
        // Finalize with 0x80, some zero padding and the length in bits.
        addByte(0x80);
        while (mIndex % 64 != 56)
        {
            addByte(0);
        }
        for (int i = 7; i >= 0; --i)
        {
            addByte(mBits >> i * 8);
        }

        MD md;
        for (int i = 0; i < 5; i++)
        {
            for (int j = 3; j >= 0; j--)
            {
                md[i * 4 + j] = (mState[i] >> ((3 - j) * 8)) & 0xff;
            }
        }

        return md;
    }

    SHA1::MD SHA1::compute(const void* data, size_t len)
    {
        SHA1 sha1;
        sha1.update(data, len);
        return sha1.finalize();
    }

    void SHA1::addByte(uint8_t byte)
    {
        mBuf[mIndex++] = byte;

        if (mIndex >= sizeof(mBuf)){
            mIndex = 0;
            processBlock(mBuf);
        }
    }

    void SHA1::processBlock(const uint8_t* ptr)
    {
        auto rol32 = [](uint32_t x, uint32_t n) {
            return (x << n) | (x >> (32 - n));
        };

        auto makeWord = [](const uint8_t* p)
        {
            return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | (uint32_t)p[3];
        };

        const uint32_t c0 = 0x5a827999;
        const uint32_t c1 = 0x6ed9eba1;
        const uint32_t c2 = 0x8f1bbcdc;
        const uint32_t c3 = 0xca62c1d6;

        uint32_t a = mState[0];
        uint32_t b = mState[1];
        uint32_t c = mState[2];
        uint32_t d = mState[3];
        uint32_t e = mState[4];

        uint32_t w[16];

        for (size_t i = 0; i < 16; i++)
        {
            w[i] = makeWord(ptr + i * 4);
        }

#define SHA1_LOAD(i) w[i&15] = rol32(w[(i + 13) & 15] ^ w[(i + 8) & 15] ^ w[(i + 2) & 15] ^ w[i & 15], 1);
#define SHA1_ROUND_0(v,u,x,y,z,i)              z += ((u & (x ^ y)) ^ y) + w[i & 15] + c0 + rol32(v, 5); u = rol32(u, 30);
#define SHA1_ROUND_1(v,u,x,y,z,i) SHA1_LOAD(i) z += ((u & (x ^ y)) ^ y) + w[i & 15] + c0 + rol32(v, 5); u = rol32(u, 30);
#define SHA1_ROUND_2(v,u,x,y,z,i) SHA1_LOAD(i) z += (u ^ x ^ y) + w[i & 15] + c1 + rol32(v, 5); u = rol32(u, 30);
#define SHA1_ROUND_3(v,u,x,y,z,i) SHA1_LOAD(i) z += (((u | x) & y) | (u & x)) + w[i & 15] + c2 + rol32(v, 5); u = rol32(u, 30);
#define SHA1_ROUND_4(v,u,x,y,z,i) SHA1_LOAD(i) z += (u ^ x ^ y) + w[i & 15] + c3 + rol32(v, 5); u = rol32(u, 30);

        SHA1_ROUND_0(a, b, c, d, e,  0);
        SHA1_ROUND_0(e, a, b, c, d,  1);
        SHA1_ROUND_0(d, e, a, b, c,  2);
        SHA1_ROUND_0(c, d, e, a, b,  3);
        SHA1_ROUND_0(b, c, d, e, a,  4);
        SHA1_ROUND_0(a, b, c, d, e,  5);
        SHA1_ROUND_0(e, a, b, c, d,  6);
        SHA1_ROUND_0(d, e, a, b, c,  7);
        SHA1_ROUND_0(c, d, e, a, b,  8);
        SHA1_ROUND_0(b, c, d, e, a,  9);
        SHA1_ROUND_0(a, b, c, d, e, 10);
        SHA1_ROUND_0(e, a, b, c, d, 11);
        SHA1_ROUND_0(d, e, a, b, c, 12);
        SHA1_ROUND_0(c, d, e, a, b, 13);
        SHA1_ROUND_0(b, c, d, e, a, 14);
        SHA1_ROUND_0(a, b, c, d, e, 15);
        SHA1_ROUND_1(e, a, b, c, d, 16);
        SHA1_ROUND_1(d, e, a, b, c, 17);
        SHA1_ROUND_1(c, d, e, a, b, 18);
        SHA1_ROUND_1(b, c, d, e, a, 19);
        SHA1_ROUND_2(a, b, c, d, e, 20);
        SHA1_ROUND_2(e, a, b, c, d, 21);
        SHA1_ROUND_2(d, e, a, b, c, 22);
        SHA1_ROUND_2(c, d, e, a, b, 23);
        SHA1_ROUND_2(b, c, d, e, a, 24);
        SHA1_ROUND_2(a, b, c, d, e, 25);
        SHA1_ROUND_2(e, a, b, c, d, 26);
        SHA1_ROUND_2(d, e, a, b, c, 27);
        SHA1_ROUND_2(c, d, e, a, b, 28);
        SHA1_ROUND_2(b, c, d, e, a, 29);
        SHA1_ROUND_2(a, b, c, d, e, 30);
        SHA1_ROUND_2(e, a, b, c, d, 31);
        SHA1_ROUND_2(d, e, a, b, c, 32);
        SHA1_ROUND_2(c, d, e, a, b, 33);
        SHA1_ROUND_2(b, c, d, e, a, 34);
        SHA1_ROUND_2(a, b, c, d, e, 35);
        SHA1_ROUND_2(e, a, b, c, d, 36);
        SHA1_ROUND_2(d, e, a, b, c, 37);
        SHA1_ROUND_2(c, d, e, a, b, 38);
        SHA1_ROUND_2(b, c, d, e, a, 39);
        SHA1_ROUND_3(a, b, c, d, e, 40);
        SHA1_ROUND_3(e, a, b, c, d, 41);
        SHA1_ROUND_3(d, e, a, b, c, 42);
        SHA1_ROUND_3(c, d, e, a, b, 43);
        SHA1_ROUND_3(b, c, d, e, a, 44);
        SHA1_ROUND_3(a, b, c, d, e, 45);
        SHA1_ROUND_3(e, a, b, c, d, 46);
        SHA1_ROUND_3(d, e, a, b, c, 47);
        SHA1_ROUND_3(c, d, e, a, b, 48);
        SHA1_ROUND_3(b, c, d, e, a, 49);
        SHA1_ROUND_3(a, b, c, d, e, 50);
        SHA1_ROUND_3(e, a, b, c, d, 51);
        SHA1_ROUND_3(d, e, a, b, c, 52);
        SHA1_ROUND_3(c, d, e, a, b, 53);
        SHA1_ROUND_3(b, c, d, e, a, 54);
        SHA1_ROUND_3(a, b, c, d, e, 55);
        SHA1_ROUND_3(e, a, b, c, d, 56);
        SHA1_ROUND_3(d, e, a, b, c, 57);
        SHA1_ROUND_3(c, d, e, a, b, 58);
        SHA1_ROUND_3(b, c, d, e, a, 59);
        SHA1_ROUND_4(a, b, c, d, e, 60);
        SHA1_ROUND_4(e, a, b, c, d, 61);
        SHA1_ROUND_4(d, e, a, b, c, 62);
        SHA1_ROUND_4(c, d, e, a, b, 63);
        SHA1_ROUND_4(b, c, d, e, a, 64);
        SHA1_ROUND_4(a, b, c, d, e, 65);
        SHA1_ROUND_4(e, a, b, c, d, 66);
        SHA1_ROUND_4(d, e, a, b, c, 67);
        SHA1_ROUND_4(c, d, e, a, b, 68);
        SHA1_ROUND_4(b, c, d, e, a, 69);
        SHA1_ROUND_4(a, b, c, d, e, 70);
        SHA1_ROUND_4(e, a, b, c, d, 71);
        SHA1_ROUND_4(d, e, a, b, c, 72);
        SHA1_ROUND_4(c, d, e, a, b, 73);
        SHA1_ROUND_4(b, c, d, e, a, 74);
        SHA1_ROUND_4(a, b, c, d, e, 75);
        SHA1_ROUND_4(e, a, b, c, d, 76);
        SHA1_ROUND_4(d, e, a, b, c, 77);
        SHA1_ROUND_4(c, d, e, a, b, 78);
        SHA1_ROUND_4(b, c, d, e, a, 79);

#undef SHA1_LOAD
#undef SHA1_ROUND_0
#undef SHA1_ROUND_1
#undef SHA1_ROUND_2
#undef SHA1_ROUND_3
#undef SHA1_ROUND_4

        mState[0] += a;
        mState[1] += b;
        mState[2] += c;
        mState[3] += d;
        mState[4] += e;
    }
}
