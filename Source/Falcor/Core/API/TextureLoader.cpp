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
#include "Core/API/Texture.h"
#include "Utils/BinaryFileStream.h"
#include "Utils/StringUtils.h"
#include <cstring>
#include "Utils/Image/ImageIO.h"

static const bool kTopDown = true;

namespace Falcor
{
    Texture::SharedPtr Texture::createFromFile(const std::string& filename, bool generateMipLevels, bool loadAsSrgb, Texture::BindFlags bindFlags)
    {
        std::string fullpath;
        if (findFileInDataDirectories(filename, fullpath) == false)
        {
            logWarning("Error when loading image file. Can't find image file '" + filename + "'");
            return nullptr;
        }

        Texture::SharedPtr pTex;
        if (hasSuffix(filename, ".dds"))
        {
            pTex = ImageIO::loadTextureFromDDS(filename, loadAsSrgb);
        }
        else
        {
            Bitmap::UniqueConstPtr pBitmap = Bitmap::createFromFile(fullpath, kTopDown);
            if (pBitmap)
            {
                ResourceFormat texFormat = pBitmap->getFormat();
                if (loadAsSrgb)
                {
                    texFormat = linearToSrgbFormat(texFormat);
                }

                pTex = Texture::create2D(pBitmap->getWidth(), pBitmap->getHeight(), texFormat, 1, generateMipLevels ? Texture::kMaxPossible : 1, pBitmap->getData(), bindFlags);
            }
        }

        if (pTex != nullptr)
        {
            pTex->setSourceFilename(fullpath);
        }

        return pTex;
    }
}
