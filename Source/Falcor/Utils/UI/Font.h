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
#pragma once
#include "Core/API/Texture.h"

namespace Falcor
{
    /** This class holds data and texture used to render text.
        It represents a mono-spaced font.
    */
    class Font
    {
    public:
        using UniquePtr = std::unique_ptr<Font>;
        using UniqueConstPtr = std::unique_ptr<const Font>;

        ~Font();

        /** Create a font object.
            \return New object, or throws an exception if creation failed.
        */
        static UniquePtr create();

        /** The structs contains information on the location of the character in the texture
        */
        struct CharTexCrdDesc
        {
            float2 topLeft; ///< Non-normalized origin of the character in the texture
            float2 size;    ///< Size in pixels of the character. This should be used to initialize the texture-coordinate when rendering.
        };

        /** Get the texture containing the characters
        */
        Texture::SharedPtr getTexture() const { return mpTexture; }

        /** Get the character descriptor
        */
        const CharTexCrdDesc& getCharDesc(char c) const
        {
            assert(c >= mFirstChar && c <= mLastChar);
            return mCharDesc[c - mFirstChar];
        }

        /** Get the height in pixels of the font
        */
        float getFontHeight() const { return mFontHeight; }

        /** Get the width in pixels of the tab character
        */
        float getTabWidth() const { return mTabWidth; }

        /** Get the spacing in pixels between 2 characters. This is measured as (start-of-char-2) - (start-of-char-1).
        */
        float getLettersSpacing() const { return mLetterSpacing; }

    private:
        Font();
        Font(const Font&) = delete;
        Font& operator=(const Font&) = delete;

        bool loadFromFile(const std::string& fontName, float size);

        static const char mFirstChar = '!';
        static const char mLastChar = '~';
        static const uint32_t mCharCount = mLastChar - mFirstChar + 1;
        static const uint32_t mTexWidth = 1024;

        Texture::SharedPtr mpTexture;
        CharTexCrdDesc mCharDesc[mCharCount];
        float mFontHeight;
        float mTabWidth;
        float mLetterSpacing;
    };
}
