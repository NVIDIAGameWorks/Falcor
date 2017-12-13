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
#include "Font.h"
#include "Utils/Platform/OS.h"
#include <fstream>
#include "Graphics/TextureHelper.h"
#include "API/Texture.h"

namespace Falcor
{
    std::string GetFontFilename(const std::string& FontName, float size)
    {
        std::string Filename = FontName + std::to_string(size);
        return Filename;
    }

    Font::UniquePtr Font::create()
    {
        UniquePtr pFont = UniquePtr(new Font());
        bool b = pFont->loadFromFile("DejaVu Sans Mono", 14);
        if(b == false)
        {
            logError("Failed to create font resource");
            pFont = nullptr;
        }

        return pFont;
    }

    static const uint32_t FontMagicNumber = 0xDEAD0001;

#pragma pack(1)
    struct FontFileHeader
    {
        uint32_t StructSize;
        uint32_t CharDataSize;
        uint32_t MagicNumber;
        uint32_t CharCount;
        float FontHeight;
        float TabWidth;
        float LetterSpacing;
    };

#pragma pack(1)
    struct FontCharData
    {
        char Char;
        float TopLeftX;
        float TopLeftY;
        float Width;
        float Height;
    };

    Font::~Font() = default;

    bool Font::loadFromFile(const std::string& FontName, float size)
    {
        std::string Filename = "Framework/Fonts/" + GetFontFilename(FontName, size);
        std::string TextureFilename;
        findFileInDataDirectories(Filename + ".dds", TextureFilename);
        std::string DataFilename;
        findFileInDataDirectories(Filename + ".bin", DataFilename);
        if((doesFileExist(TextureFilename) == false) || (doesFileExist(DataFilename) == false))
        {
            return false;
        }

        // Load the data
        std::ifstream Data(DataFilename, std::ios::binary);
        FontFileHeader Header;
        // Read the header
        Data.read((char*)&Header, sizeof(Header));
        bool bValid = (Header.StructSize == sizeof(Header));
        bValid = bValid && (Header.MagicNumber == FontMagicNumber);
        bValid = bValid && (Header.CharDataSize == sizeof(FontCharData));
        bValid = bValid && (Header.CharCount == mCharCount);

        if(bValid == false)
        {
            Data.close();
            return false;
        }

        mTabWidth = Header.TabWidth;
        mFontHeight = Header.FontHeight;

        mLetterSpacing = 0;
        // Load the char data
        for(auto i = 0; i < mCharCount; i++)
        {
            FontCharData CharData;
            Data.read((char*)&CharData, sizeof(FontCharData));
            if(CharData.Char != i + mFirstChar)
            {
                Data.close();
                return false;
            }

            mCharDesc[i].topLeft.x = CharData.TopLeftX;
            mCharDesc[i].topLeft.y = CharData.TopLeftY;
            mCharDesc[i].size.x = CharData.Width;
            mCharDesc[i].size.y = CharData.Height;
            mLetterSpacing = max(mLetterSpacing, CharData.Width);
        }

        // Load the texture
        mpTexture = createTextureFromFile(TextureFilename, false, false);
        return true;
    }
}