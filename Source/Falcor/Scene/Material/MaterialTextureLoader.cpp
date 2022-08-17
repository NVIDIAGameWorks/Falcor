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
#include "MaterialTextureLoader.h"
#include "Utils/Logger.h"

namespace Falcor
{
    MaterialTextureLoader::MaterialTextureLoader(const TextureManager::SharedPtr& pTextureManager, bool useSrgb)
        : mpTextureManager(pTextureManager)
        , mUseSrgb(useSrgb)
    {
    }

    MaterialTextureLoader::~MaterialTextureLoader()
    {
        assignTextures();
    }

    void MaterialTextureLoader::loadTexture(const Material::SharedPtr& pMaterial, Material::TextureSlot slot, const std::filesystem::path& path)
    {
        FALCOR_ASSERT(pMaterial);
        if (!pMaterial->hasTextureSlot(slot))
        {
            logWarning("MaterialTextureLoader::loadTexture() - Material '{}' does not have texture slot '{}'. Ignoring call.", pMaterial->getName(), to_string(slot));
            return;
        }

        bool srgb = mUseSrgb && pMaterial->getTextureSlotInfo(slot).srgb;

        // Request texture to be loaded.
        auto handle = mpTextureManager->loadTexture(path, true, srgb);

        // Store assignment to material for later.
        mTextureAssignments.emplace_back(TextureAssignment{ pMaterial, slot, handle });
    }

    void MaterialTextureLoader::assignTextures()
    {
        mpTextureManager->waitForAllTexturesLoading();

        // Assign textures to materials.
        for (const auto& assignment : mTextureAssignments)
        {
            auto pTexture = mpTextureManager->getTexture(assignment.handle);
            assignment.pMaterial->setTexture(assignment.textureSlot, pTexture);
        }
    }
}
