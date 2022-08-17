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
#pragma once
#include "Scene.h"
#include "Animation/Animation.h"
#include "Camera/Camera.h"
#include "Lights/EnvMap.h"
#include "Lights/Light.h"
#include "Volume/Grid.h"
#include "Volume/GridVolume.h"
#include "Material/BasicMaterial.h"
#include "Material/MaterialSystem.h"
#include "Material/MaterialTextureLoader.h"

#include "Core/Macros.h"
#include "Utils/CryptoUtils.h"

#include <filesystem>
#include <string>
#include <vector>

namespace Falcor
{
    /** Helper class for reading and writing scene cache files.
        The scene cache is used to heavily reduce load times of more complex assets.
        The cache stores a binary representation of `Scene::SceneData` which contains everything to re-create a `Scene`.
    */
    class FALCOR_API SceneCache
    {
    public:
        using Key = SHA1::MD;

        /** Check if there is a valid scene cache for a given cache key.
            \param[in] key Cache key.
            \return Returns true if a valid cache exists.
        */
        static bool hasValidCache(const Key& key);

        /** Write a scene cache.
            \param[in] sceneData Scene data.
            \param[in] key Cache key.
        */
        static void writeCache(const Scene::SceneData& sceneData, const Key& key);

        /** Read a scene cache.
            \param[in] key Cache key.
            \return Returns the loaded scene data.
        */
        static Scene::SceneData readCache(const Key& key);

    private:
        class OutputStream;
        class InputStream;

        static std::filesystem::path getCachePath(const Key& key);

        static void writeSceneData(OutputStream& stream, const Scene::SceneData& sceneData);
        static Scene::SceneData readSceneData(InputStream& stream);

        static void writeMetadata(OutputStream& stream, const Scene::Metadata& metadata);
        static Scene::Metadata readMetadata(InputStream& stream);

        static void writeCamera(OutputStream& stream, const Camera::SharedPtr& pCamera);
        static Camera::SharedPtr readCamera(InputStream& stream);

        static void writeLight(OutputStream& stream, const Light::SharedPtr& pLight);
        static Light::SharedPtr readLight(InputStream& stream);

        static void writeMaterials(OutputStream& stream, const MaterialSystem::SharedPtr& pMaterials);
        static void writeMaterial(OutputStream& stream, const Material::SharedPtr& pMaterial);
        static void writeBasicMaterial(OutputStream& stream, const BasicMaterial::SharedPtr& pMaterial);
        static void readMaterials(InputStream& stream, const MaterialSystem::SharedPtr& pMaterials, MaterialTextureLoader& materialTextureLoader);
        static Material::SharedPtr readMaterial(InputStream& stream, MaterialTextureLoader& materialTextureLoader);
        static void readBasicMaterial(InputStream& stream, MaterialTextureLoader& materialTextureLoader, const BasicMaterial::SharedPtr& pMaterial);

        static void writeSampler(OutputStream& stream, const Sampler::SharedPtr& pSampler);
        static Sampler::SharedPtr readSampler(InputStream& stream);

        static void writeGridVolume(OutputStream& stream, const GridVolume::SharedPtr& pVolume, const std::vector<Grid::SharedPtr>& grids);
        static GridVolume::SharedPtr readGridVolume(InputStream& stream, const std::vector<Grid::SharedPtr>& grids);

        static void writeGrid(OutputStream& stream, const Grid::SharedPtr& pGrid);
        static Grid::SharedPtr readGrid(InputStream& stream);

        static void writeEnvMap(OutputStream& stream, const EnvMap::SharedPtr& pEnvMap);
        static EnvMap::SharedPtr readEnvMap(InputStream& stream);

        static void writeTransform(OutputStream& stream, const Transform& transform);
        static Transform readTransform(InputStream& stream);

        static void writeAnimation(OutputStream& stream, const Animation::SharedPtr& pAnimation);
        static Animation::SharedPtr readAnimation(InputStream& stream);

        static void writeMarker(OutputStream& stream, const std::string& id);
        static void readMarker(InputStream& stream, const std::string& id);
    };
}
