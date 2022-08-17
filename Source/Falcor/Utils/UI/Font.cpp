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
#include "Font.h"
#include "Core/Errors.h"
#include "Core/API/Texture.h"
#include <fstream>

namespace Falcor
{
    Font::Font()
    {
        if (!loadFromFile("DejaVu Sans Mono", 14))
        {
            throw RuntimeError("Failed to create font resource");
        }
    }

    Font::UniquePtr Font::create()
    {
        return UniquePtr(new Font());
    }

    static const uint32_t kFontMagicNumber = 0xDEAD0001;

#pragma pack(1)
    struct FontFileHeader
    {
        uint32_t structSize;
        uint32_t charDataSize;
        uint32_t magicNumber;
        uint32_t charCount;
        float fontHeight;
        float tabWidth;
        float letterSpacing;
    };

#pragma pack(1)
    struct FontCharData
    {
        char character;
        float topLeftX;
        float topLeftY;
        float width;
        float height;
    };

    Font::~Font() = default;

    bool Font::loadFromFile(const std::string& fontName, float size)
    {
        std::string baseName = "Framework/Fonts/" + fontName + std::to_string(size);
        std::filesystem::path texturePath;
        std::filesystem::path dataPath;
        if (!findFileInDataDirectories(baseName + ".dds", texturePath)) return false;
        if (!findFileInDataDirectories(baseName + ".bin", dataPath)) return false;

        // Load the data
        std::ifstream data(dataPath, std::ios::binary);
        FontFileHeader header;
        // Read the header
        data.read((char*)&header, sizeof(header));
        bool valid = (header.structSize == sizeof(header));
        valid = valid && (header.magicNumber == kFontMagicNumber);
        valid = valid && (header.charDataSize == sizeof(FontCharData));
        valid = valid && (header.charCount == mCharCount);

        if (!valid) return false;

        mTabWidth = header.tabWidth;
        mFontHeight = header.fontHeight;

        mLetterSpacing = 0;
        // Load the char data
        for(uint32_t i = 0; i < mCharCount; i++)
        {
            FontCharData charData;
            data.read((char*)&charData, sizeof(FontCharData));
            if(charData.character != i + mFirstChar)
            {
                data.close();
                return false;
            }

            mCharDesc[i].topLeft.x = charData.topLeftX;
            mCharDesc[i].topLeft.y = charData.topLeftY;
            mCharDesc[i].size.x = charData.width;
            mCharDesc[i].size.y = charData.height;
            mLetterSpacing = std::max(mLetterSpacing, charData.width);
        }

        // Load the texture
        mpTexture = Texture::createFromFile(texturePath, false, false);
        return mpTexture != nullptr;
    }
}
