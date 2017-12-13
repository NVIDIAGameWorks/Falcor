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
#include "Graphics/Material/MaterialEditor.h"
#include "Utils/Gui.h"
#include "Utils/Platform/OS.h"
#include "Graphics/TextureHelper.h"
#include "API/Texture.h"
#include "Graphics/Scene/Scene.h"
#include "Graphics/Scene/SceneExporter.h"
#include <cstring>

namespace Falcor
{

    Gui::DropdownList MaterialEditor::kLayerTypeDropdown =
    {
        { MatLambert,    "Lambert" },
        { MatConductor,  "Conductor" },
        { MatDielectric, "Dielectric" },
        { MatEmissive,   "Emissive" },
        { MatUser,       "Custom" }
    };

    Gui::DropdownList MaterialEditor::kLayerBlendDropdown =
    {
        { BlendFresnel,  "Fresnel" },
        { BlendAdd,      "Additive" }/*,
        { BlendConstant, "Constant Factor" }*/
    };

    Gui::DropdownList MaterialEditor::kLayerNDFDropdown =
    {
        { NDFBeckmann, "Beckmann" },
        { NDFGGX,      "GGX" },
        { NDFUser,     "User Defined" }
    };

    Texture::SharedPtr loadTexture(bool useSrgb)
    {
        std::string filename;
        Texture::SharedPtr pTexture = nullptr;
        if(openFileDialog(nullptr, filename) == true)
        {
            pTexture = createTextureFromFile(filename, true, useSrgb);
            if(pTexture)
            {
                pTexture->setName(filename);
            }
        }
        return pTexture;
    }

    MaterialEditor::UniquePtr MaterialEditor::create(const Material::SharedPtr& pMaterial, std::function<void(void)> editorFinishedCB)
    {
        return UniquePtr(new MaterialEditor(pMaterial, editorFinishedCB));
    }

    void MaterialEditor::renderGui(Gui* pGui)
    {
        pGui->pushWindow("Material Editor", 400, 600, 440, 300);

        if (closeEditor(pGui))
        {
            return;
        }

        pGui->addSeparator();

        setName(pGui);
        setId(pGui);
        setDoubleSided(pGui);
        pGui->addSeparator();

        setNormalMap(pGui);
        setAlphaMap(pGui);
        setHeightMap(pGui);
        setAmbientOcclusionMap(pGui);

        setHeightModifiers(pGui);
        setAlphaThreshold(pGui);

        for (uint32_t i = 0; i < mpMaterial->getNumLayers(); i++)
        {
            std::string groupName("Layer " + std::to_string(i));

            if (pGui->beginGroup(groupName.c_str()))
            {
                setLayerTexture(pGui, i);
                setLayerType(pGui, i);
                setLayerNdf(pGui, i);
                setLayerBlend(pGui, i);

                const auto layer = mpMaterial->getLayer(i);

                switch (layer.type)
                {
                case Material::Layer::Type::Lambert:
                case Material::Layer::Type::Emissive:
                    setLayerAlbedo(pGui, i);
                    break;

                case Material::Layer::Type::Conductor:
                    setLayerAlbedo(pGui, i);
                    setLayerRoughness(pGui, i);
                    setConductorLayerParams(pGui, i);
                    break;

                case Material::Layer::Type::Dielectric:
                    setLayerAlbedo(pGui, i);
                    setLayerRoughness(pGui, i);
                    setDielectricLayerParams(pGui, i);
                    break;

                default:
                    break;
                }

                bool layerRemoved = removeLayer(pGui, i);

                pGui->endGroup();

                if (layerRemoved)
                {
                    break;
                }
            }
        }

        if (mpMaterial->getNumLayers() < MatMaxLayers)
        {
            pGui->addSeparator();
            addLayer(pGui);
        }

        pGui->popWindow();
    }

    bool MaterialEditor::closeEditor(Gui* pGui)
    {
        if (pGui->addButton("Close Editor"))
        {
            pGui->popWindow();
            if (mpEditorFinishedCB != nullptr)
            {
                mpEditorFinishedCB();
            }
            return true;
        }

        return false;
    }

    void MaterialEditor::setName(Gui* pGui)
    {
        char nameBuf[256];
        std::strcpy(nameBuf, mpMaterial->getName().c_str());

        if (pGui->addTextBox("Name", nameBuf, arraysize(nameBuf)))
        {
            mpMaterial->setName(nameBuf);
        }
    }

    void MaterialEditor::setId(Gui* pGui)
    {
        int32_t id = mpMaterial->getId();

        if (pGui->addIntVar("ID", id, 0))
        {
            mpMaterial->setID(id);
        }
    }

    void MaterialEditor::setDoubleSided(Gui* pGui)
    {
        bool doubleSided = mpMaterial->isDoubleSided();
        if (pGui->addCheckBox("Double Sided", doubleSided))
        {
            mpMaterial->setDoubleSided(doubleSided);
        }
    }

    void MaterialEditor::setHeightModifiers(Gui* pGui)
    {
        glm::vec2 heightMods = mpMaterial->getHeightModifiers();

        pGui->addFloatVar("Height Bias", heightMods[0], -FLT_MAX, FLT_MAX);
        pGui->addFloatVar("Height Scale", heightMods[1], 0.0f, FLT_MAX);

        mpMaterial->setHeightModifiers(heightMods);
    }

    void MaterialEditor::setAlphaThreshold(Gui* pGui)
    {
        float a = mpMaterial->getAlphaThreshold();

        if(pGui->addFloatVar("Alpha Threshold", a, 0.0f, 1.0f))
        {
            mpMaterial->setAlphaThreshold(a);
        }
    }

    void MaterialEditor::addLayer(Gui* pGui)
    {
        if (pGui->addButton("Add Layer"))
        {
            if (mpMaterial->getNumLayers() >= MatMaxLayers)
            {
                msgBox("Exceeded the number of supported layers. Can't add anymore");
                return;
            }

            mpMaterial->addLayer(Material::Layer());
        }
    }

    void MaterialEditor::setLayerType(Gui* pGui, uint32_t layerID)
    {
        uint32_t type = (uint32_t)mpMaterial->getLayer(layerID).type;

        std::string label("Type##" + std::to_string(layerID));
        if (pGui->addDropdown(label.c_str(), kLayerTypeDropdown, type))
        {
            mpMaterial->setLayerType(layerID, (Material::Layer::Type)type);
        }
    }

    void MaterialEditor::setLayerNdf(Gui* pGui, uint32_t layerID)
    {
        uint32_t ndf = (uint32_t)mpMaterial->getLayer(layerID).ndf;

        std::string label("NDF##" + std::to_string(layerID));
        if (pGui->addDropdown(label.c_str(), kLayerNDFDropdown, ndf))
        {
            mpMaterial->setLayerNdf(layerID, (Material::Layer::NDF)ndf);
        }
    }

    void MaterialEditor::setLayerBlend(Gui* pGui, uint32_t layerID)
    {
        uint32_t blend = (uint32_t)mpMaterial->getLayer(layerID).blend;

        std::string label("Blend##" + std::to_string(layerID));
        if (pGui->addDropdown(label.c_str(), kLayerBlendDropdown, blend))
        {
            mpMaterial->setLayerBlend(layerID, (Material::Layer::Blend)blend);
        }
    }

    void MaterialEditor::setLayerAlbedo(Gui* pGui, uint32_t layerID)
    {
        glm::vec4 albedo = mpMaterial->getLayer(layerID).albedo;

        std::string label("Albedo##" + std::to_string(layerID));
        if (pGui->addRgbaColor(label.c_str(), albedo))
        {
            mpMaterial->setLayerAlbedo(layerID, albedo);
        }
    }

    void MaterialEditor::setLayerRoughness(Gui* pGui, uint32_t layerID)
    {
        float roughness = mpMaterial->getLayer(layerID).roughness[0];

        std::string label("Roughness##" + std::to_string(layerID));
        if (pGui->addFloatVar(label.c_str(), roughness, 0.0f, 1.0f))
        {
            mpMaterial->setLayerRoughness(layerID, glm::vec4(roughness));
        }
    }

    void MaterialEditor::setLayerTexture(Gui* pGui, uint32_t layerID)
    {
        const auto pTexture = mpMaterial->getLayer(layerID).pTexture;

        auto pNewTexture = changeTexture(pGui, "Texture##" + std::to_string(layerID), pTexture, true);
        if (pNewTexture != pTexture)
        {
            mpMaterial->setLayerTexture(layerID, pNewTexture);
        }
    }

    void MaterialEditor::setConductorLayerParams(Gui* pGui, uint32_t layerID)
    {
        if (pGui->beginGroup("IoR"))
        {
            const auto layer = mpMaterial->getLayer(layerID);
            float r = layer.extraParam[0];
            float i = layer.extraParam[1];

            pGui->addFloatVar("Real", r, 0.0f, FLT_MAX);
            pGui->addFloatVar("Imaginary", i, 0.0f, FLT_MAX);

            mpMaterial->setLayerUserParam(layerID, glm::vec4(r, i, 0.0f, 0.0f));

            pGui->endGroup();
        }
    }

    void MaterialEditor::setDielectricLayerParams(Gui* pGui, uint32_t layerID)
    {
        const auto layer = mpMaterial->getLayer(layerID);
        float ior = layer.extraParam[0];

        if (pGui->addFloatVar("IoR", ior, 0.0f, FLT_MAX))
        {
            mpMaterial->setLayerUserParam(layerID, glm::vec4(ior, 0.0f, 0.0f, 0.0f));
        }
    }

    bool MaterialEditor::removeLayer(Gui* pGui, uint32_t layerID)
    {
        std::string label("Remove##" + std::to_string(layerID));
        if (pGui->addButton(label.c_str()))
        {
            mpMaterial->removeLayer(layerID);
            return true;
        }

        return false;
    }

    void MaterialEditor::setNormalMap(Gui* pGui)
    {
        const auto pTexture = mpMaterial->getNormalMap();

        auto pNewTexture = changeTexture(pGui, "Normal Map", pTexture, false);
        if (pNewTexture != pTexture)
        {
            mpMaterial->setNormalMap(pNewTexture);
        }
    }

    void MaterialEditor::setAlphaMap(Gui* pGui)
    {
        const auto pTexture = mpMaterial->getAlphaMap();

        auto pNewTexture = changeTexture(pGui, "Alpha Map", pTexture, false);
        if (pNewTexture != pTexture)
        {
            mpMaterial->setAlphaMap(pNewTexture);
        }
    }

    void MaterialEditor::setHeightMap(Gui* pGui)
    {
        const auto pTexture = mpMaterial->getHeightMap();

        auto pNewTexture = changeTexture(pGui, "Height Map", pTexture, false);
        if (pNewTexture != pTexture)
        {
            mpMaterial->setHeightMap(pNewTexture);
        }
    }

    void MaterialEditor::setAmbientOcclusionMap(Gui* pGui)
    {
        const auto pTexture = mpMaterial->getAmbientOcclusionMap();

        auto pNewTexture = changeTexture(pGui, "AO Map", pTexture, true);
        if (pNewTexture != pTexture)
        {
            mpMaterial->setAmbientOcclusionMap(pNewTexture);
        }
    }

    Texture::SharedPtr MaterialEditor::changeTexture(Gui* pGui, const std::string& label, const Texture::SharedPtr& pCurrentTexture, bool useSRGB)
    {
        std::string texPath(pCurrentTexture ? pCurrentTexture->getSourceFilename() : "");

        char texPathBuff[1024];
        std::strcpy(texPathBuff, texPath.c_str());

        pGui->addTextBox(label.c_str(), texPathBuff, arraysize(texPathBuff));

        std::string changeLabel("Change##" + label);
        if (pGui->addButton(changeLabel.c_str()))
        {
            return loadTexture(useSRGB);
        }

        std::string removeLabel("Remove##" + label);
        if (pGui->addButton(removeLabel.c_str(), true))
        {
            return nullptr;
        }

        return pCurrentTexture;
    }
}
