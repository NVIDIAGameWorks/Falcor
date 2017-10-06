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
#include "Material.h"

namespace Falcor
{
    class Texture;
    class ConstantBuffer;

    /** Basic container for material property values. Primarily used as a helper for import/export operations.
        For regular use cases, see Material in Material.h.
    */
    struct BasicMaterial
    {
        /** Material map Type. All maps are 2D textures.
        */
        enum MapType
        {
            DiffuseMap,     ///< Diffuse reflectance
            SpecularMap,    ///< Specular reflectance
            EmissiveMap,    ///< Emissive intensity
            NormalMap,      ///< Normal map
            AlphaMap,       ///< Either opacity map or transparency map, depends on the model
            HeightMap,      ///< Height/displacement map
            AmbientMap,     ///< Ambient occlusion map

            Count
        };

        glm::vec3   diffuseColor = glm::vec3(0, 0, 0);      ///< Diffuse albedo of a Lambertian BRDF
        glm::vec3   specularColor = glm::vec3(0, 0, 0);     ///< Specular reflection color
        float       shininess = 2.f;                        ///< Specular power, i.e. an exponent of a Phong BRDF
        float       opacity = 1.f;                          ///< Opacity of the material
        glm::vec3   transparentColor = glm::vec3(0, 0, 0);  ///< Refraction color of a transparent dielectric material
        float       IoR = 1.f;                              ///< Index of refraction
        glm::vec3   emissiveColor = glm::vec3(0, 0, 0);     ///< Emissive color of the material
        float       bumpScale = 1.f;                        ///< Object-space scale of a bump/displacement map
        float       bumpOffset = 0.f;                       ///< Object-space zero-based offset of a bump/displacement map

        Texture::SharedPtr  pTextures[MapType::Count];      ///< Array of available types of textures

        BasicMaterial();

        /** Returns a regular Material object containing the same properties.
        */
        Material::SharedPtr convertToMaterial();

        /** Initializes values from a regular Material object.
        */
        void initializeFromMaterial(const Material* pMaterial);
    };
}