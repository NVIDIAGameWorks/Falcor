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
#include "BasicMaterial.h"
#include "Data/HostDeviceData.h"
#include <cstring>

namespace Falcor
{
    BasicMaterial::BasicMaterial()
    {
        std::memset(pTextures, 0x00, sizeof(pTextures)); 
    }

    Material::SharedPtr BasicMaterial::convertToMaterial()
    {
        Material::SharedPtr pMaterial = Material::create("");
        std::string name = "BasicMaterial" + std::to_string(pMaterial->getId());
        pMaterial->setName(name);

        size_t currentLayer = 0;

        /* Parse diffuse layer */
        if(luminance(diffuseColor) > 0.f || pTextures[MapType::DiffuseMap] != nullptr)
        {
            Material::Layer layer;
            layer.type = Material::Layer::Type::Lambert;
            layer.blend = Material::Layer::Blend::Additive;

            glm::vec3 albedo = diffuseColor;
            /* Special case: if albedo color is zero but texture is not, ignore the color */
            if(luminance(albedo) == 0.f)
            {
                albedo = glm::vec3(1.f);
            }

            layer.albedo = glm::vec4(albedo, opacity);
            layer.pTexture = pTextures[MapType::DiffuseMap];
            pMaterial->addLayer(layer);
        }

        /* Parse conductor layer */
        if(luminance(specularColor) > 0.f || pTextures[MapType::SpecularMap] != nullptr)
        {
            Material::Layer layer;
            layer.type = Material::Layer::Type::Conductor;
            layer.blend = Material::Layer::Blend::Additive;

            glm::vec3 specular = specularColor;
            /* Special case: if albedo color is zero but texture is not, ignore the color */
            if (luminance(specular) == 0.f)
            {
                specular = glm::vec3(1.f);
            }

            layer.albedo = glm::vec4(specular, 1.f);
            layer.pTexture = pTextures[MapType::SpecularMap];
            layer.roughness = glm::vec4(convertShininessToRoughness(shininess));
            
            /* Average chrome IoR and kappa */
            layer.extraParam = glm::vec4(3.f, 4.2f, 0.f, 0.f);
            pMaterial->addLayer(layer);
        }

        /* Parse dielectric layer */
        if((luminance(transparentColor) * (1.f - opacity) > 0.f) && currentLayer < MatMaxLayers)
        {
            Material::Layer layer;
            layer.type = Material::Layer::Type::Dielectric;
            layer.blend = Material::Layer::Blend::Fresnel;

            layer.albedo = glm::vec4(transparentColor * (1.f - opacity), 1.f);

            /* Always assume a smooth dielectric */
            layer.roughness = glm::vec4(0.f);
            layer.extraParam = glm::vec4(IoR, 0.f, 0.f, 0.f);

            pMaterial->addLayer(layer);
        }

        /* Parse emissive layer */
        if(luminance(emissiveColor) > 0.f || pTextures[MapType::EmissiveMap] != nullptr)
        {
            Material::Layer layer;
            layer.type = Material::Layer::Type::Emissive;
            layer.blend = Material::Layer::Blend::Additive;

            glm::vec3 albedo = emissiveColor;
            /* Special case: if albedo color is zero but texture is not, ignore the color */
            if(luminance(albedo) == 0.f)
            {
                albedo = glm::vec3(1.f);
            }

            layer.albedo = glm::vec4(albedo, 1.f);
            layer.pTexture = pTextures[MapType::EmissiveMap];
            pMaterial->addLayer(layer);
        }

        /* Initialize modifiers */
        pMaterial->setNormalMap(pTextures[MapType::NormalMap]);
        pMaterial->setAlphaMap(pTextures[MapType::AlphaMap]);
        pMaterial->setAlphaThreshold(0.5f);
        pMaterial->setAmbientOcclusionMap(pTextures[MapType::AmbientMap]);
        pMaterial->setHeightModifiers(glm::vec2(bumpScale, bumpOffset));
        pMaterial->setHeightMap(pTextures[MapType::HeightMap]);

        return pMaterial;
    }

    static inline uint32_t getLayersCountByType(const Material* pMaterial, Material::Layer::Type layerType)
    {
        uint32_t numLayers = 0;
        int32_t layerId = pMaterial->getNumLayers();
        while(--layerId >= 0)
        {
            Material::Layer layer = pMaterial->getLayer(layerId);
            if(layer.type == layerType)
            {
                numLayers++;
            }
        }

        return numLayers;
    }

    static bool getLayerFromType(const Material* pMaterial, Material::Layer::Type layerType, Material::Layer& layer)
    {
        int32_t layerId = pMaterial->getNumLayers();;
        while(--layerId >= 0)
        {
            layer = pMaterial->getLayer(layerId);
            if(layer.type == layerType)
            {
                return true;
            }
        }

        return false;
    }

    void BasicMaterial::initializeFromMaterial(const Material* pMaterial)
    {
        if(getLayersCountByType(pMaterial, Material::Layer::Type::Lambert) > 1 ||
            getLayersCountByType(pMaterial, Material::Layer::Type::Conductor) > 1 ||
            getLayersCountByType(pMaterial, Material::Layer::Type::Dielectric) > 1)
        {
            logWarning("BasicMaterial::initializeFromMaterial(): Material " + pMaterial->getName() + " was exported with loss of data");
        }

        Material::Layer layer;
        /* Parse diffuse layer */
        if(getLayerFromType(pMaterial, Material::Layer::Type::Lambert, layer))
        {
            pTextures[MapType::DiffuseMap] = layer.pTexture;
            diffuseColor = glm::vec3(layer.albedo);
        }

        /* Parse emissive layer */
        if(getLayerFromType(pMaterial, Material::Layer::Type::Emissive, layer))
        {
            pTextures[MapType::EmissiveMap] = layer.pTexture;
            emissiveColor = glm::vec3(layer.albedo);
        }

        /* Parse specular layer */
        if(getLayerFromType(pMaterial, Material::Layer::Type::Conductor, layer))
        {
            pTextures[MapType::SpecularMap] = layer.pTexture;
            specularColor = glm::vec3(layer.albedo);
            shininess = convertRoughnessToShininess(layer.roughness.x);
        }

        /* Parse transparent layer */
        if(getLayerFromType(pMaterial, Material::Layer::Type::Dielectric, layer))
        {
            transparentColor = glm::vec3(layer.albedo);
            IoR = layer.extraParam.x;
            if(layer.pTexture)
            {
                logWarning("Material::initializeFromMaterial: Material " + pMaterial->getName() + " has an unsupported transparency texture");
            }
        }

        /* Parse modifiers */
        pTextures[MapType::AlphaMap] = pMaterial->getAlphaMap();
        opacity = pMaterial->getAlphaThreshold();
        pTextures[MapType::AmbientMap] = pMaterial->getAmbientOcclusionMap();
        pTextures[MapType::NormalMap] = pMaterial->getNormalMap();
        pTextures[MapType::HeightMap] = pMaterial->getHeightMap();
        bumpScale = pMaterial->getHeightModifiers().x;
        bumpOffset = pMaterial->getHeightModifiers().y;
    }
}