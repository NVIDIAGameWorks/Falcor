/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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
#include "SceneCache.h"
#include "Material/MaterialTextureLoader.h"

#include <lz4_stream/lz4_stream.h>

namespace Falcor
{
    namespace
    {
        /** Specfies the current cache file version.
            This needs to be incremented every time the file format changes!
        */
        const uint32_t kVersion = 14;

        /** Scene cache directory (subdirectory in the application data directory).
        */
        const std::string kDirectory = "NVIDIA/Falcor/SceneCache";

        const size_t kBlockSize = 1 * 1024 * 1024;

        const char* kMagic = "FalcorS$";
        struct Header
        {
            uint8_t magic[8]{};
            uint32_t version{};

            bool isValid() const
            {
                return std::memcmp(magic, kMagic, sizeof(Header::magic)) == 0 && version == kVersion;
            }
        };
    }

    /** Wrapper around std::ostream to ease serialization of basic types.
    */
    class SceneCache::OutputStream
    {
    public:
        OutputStream(std::ostream& stream) : mStream(stream) {}

        void write(const void* data, size_t len)
        {
            mStream.write(reinterpret_cast<const char*>(data), len);
        }

        template<typename T>
        void write(const T& value)
        {
            write(&value, sizeof(T));
        }

        template<>
        void write(const std::string& value)
        {
            uint64_t len = value.size();
            write(len);
            write(value.data(), len);
        }

        template<typename T>
        void write(const std::vector<T>& vec)
        {
            uint64_t len = vec.size();
            write(len);
            if constexpr (std::is_trivial<T>::value && !std::is_same<T, bool>::value)
            {
                write(vec.data(), len * sizeof(T));
            }
            else
            {
                for (const auto& item : vec) write<T>(item);
            }
        }

        template<typename T>
        void write(const std::optional<T>& opt)
        {
            bool hasValue = opt.has_value();
            write(hasValue);
            if (hasValue) write(opt.value());
        }

    private:
        std::ostream& mStream;
    };

    /** Wrapper around std::istream to ease serialization of basic types.
    */
    class SceneCache::InputStream
    {
    public:
        InputStream(std::istream& stream) : mStream(stream) {}

        void read(void* data, size_t len)
        {
            mStream.read(reinterpret_cast<char*>(data), len);
        }

        template<typename T>
        void read(T& value)
        {
            read(&value, sizeof(T));
        }

        template<>
        void read(std::string& value)
        {
            uint64_t len = read<uint64_t>();
            value.resize(len);
            read(value.data(), len);
        }

        template<typename T>
        T read()
        {
            T value;
            read(value);
            return value;
        }

        template<typename T>
        void read(std::vector<T>& vec)
        {
            uint64_t len = read<uint64_t>();
            vec.resize(len);
            if constexpr (std::is_trivial<T>::value && !std::is_same<T, bool>::value)
            {
                read(vec.data(), len * sizeof(T));
            }
            else
            {
                for (auto& item : vec) read<T>(item);
            }
        }

        template<typename T>
        void read(std::optional<T>& opt)
        {
            bool hasValue = read<bool>();
            if (hasValue) opt = read<T>();
        }

    private:
        std::istream& mStream;
    };

    bool SceneCache::hasValidCache(const Key& key)
    {
        auto cachePath = getCachePath(key);
        if (!std::filesystem::exists(cachePath)) return false;

        // Open file.
        std::ifstream fs(cachePath.c_str(), std::ios_base::binary);
        if (fs.bad()) return false;

        // Verify header.
        Header header;
        fs.read(reinterpret_cast<char*>(&header), sizeof(header));
        return !fs.eof() && header.isValid();
    }

    void SceneCache::writeCache(const Scene::SceneData& sceneData, const Key& key)
    {
        auto cachePath = getCachePath(key);

        logInfo("Writing scene cache to " + cachePath.string());

        // Create directories if not existing.
        std::filesystem::create_directories(cachePath.parent_path());

        // Open file.
        std::ofstream fs(cachePath.c_str(), std::ios_base::binary);
        if (fs.bad()) throw std::runtime_error("Failed to create scene cache file '" + cachePath.string() + "'!");

        // Write header (uncompressed).
        Header header;
        std::memcpy(header.magic, kMagic, sizeof(Header::magic));
        header.version = kVersion;
        fs.write(reinterpret_cast<const char*>(&header), sizeof(header));

        // Write cache (compressed).
        lz4_stream::basic_ostream<kBlockSize> zs(fs);
        OutputStream stream(zs);
        writeSceneData(stream, sceneData);
        if (fs.bad()) throw std::runtime_error("Failed to write scene cache file to '" + cachePath.string() + "'!");
    }

    Scene::SceneData SceneCache::readCache(const Key& key)
    {
        auto cachePath = getCachePath(key);

        logInfo("Loading scene cache from " + cachePath.string());

        // Open file.
        std::ifstream fs(cachePath.c_str(), std::ios_base::binary);
        if (fs.bad()) throw std::runtime_error("Failed to open scene cache file '" + cachePath.string() + "'!");

        // Read header (uncompressed).
        Header header;
        fs.read(reinterpret_cast<char*>(&header), sizeof(header));
        if (!header.isValid()) throw std::runtime_error("Invalid header in scene cache file '" + cachePath.string() + "'!");

        // Read cache (compressed).
        lz4_stream::basic_istream<kBlockSize, kBlockSize> zs(fs);
        InputStream stream(zs);
        auto sceneData = readSceneData(stream);
        if (fs.bad()) throw std::runtime_error("Failed to read scene cache file from '" + cachePath.string() + "'!");
        return sceneData;
    }

    std::filesystem::path SceneCache::getCachePath(const Key& key)
    {
        std::stringstream ss;
        ss << std::hex << std::setfill('0') << std::setw(2);
        for (auto c : key) ss << (int)c;
        return std::filesystem::path(getAppDataDirectory()) / kDirectory / ss.str();
    }

    // SceneData

    void SceneCache::writeSceneData(OutputStream& stream, const Scene::SceneData& sceneData)
    {
        writeMarker(stream, "Filename");
        stream.write(sceneData.filename);

        writeMarker(stream, "RenderSettings");
        stream.write(sceneData.renderSettings);

        writeMarker(stream, "Cameras");
        stream.write((uint32_t)sceneData.cameras.size());
        for (const auto& pCamera : sceneData.cameras) writeCamera(stream, pCamera);
        stream.write(sceneData.selectedCamera);
        stream.write(sceneData.cameraSpeed);

        writeMarker(stream, "Lights");
        stream.write((uint32_t)sceneData.lights.size());
        for (const auto& pLight : sceneData.lights) writeLight(stream, pLight);

        writeMarker(stream, "Grids");
        stream.write((uint32_t)sceneData.grids.size());
        for (const auto& pGrid : sceneData.grids) writeGrid(stream, pGrid);

        writeMarker(stream, "Volumes");
        stream.write((uint32_t)sceneData.volumes.size());
        for (const auto& pVolume : sceneData.volumes) writeVolume(stream, pVolume, sceneData.grids);

        writeMarker(stream, "EnvMap");
        bool hasEnvMap = sceneData.pEnvMap != nullptr;
        stream.write(hasEnvMap);
        if (hasEnvMap) writeEnvMap(stream, sceneData.pEnvMap);

        writeMarker(stream, "Materials");
        stream.write((uint32_t)sceneData.materials.size());
        for (const auto& pMaterial : sceneData.materials) writeMaterial(stream, pMaterial);

        writeMarker(stream, "SceneGraph");
        stream.write((uint32_t)sceneData.sceneGraph.size());
        for (const auto& node : sceneData.sceneGraph)
        {
            stream.write(node.name);
            stream.write(node.parent);
            stream.write(node.transform);
            stream.write(node.meshBind);
            stream.write(node.localToBindSpace);
        }

        writeMarker(stream, "Animations");
        stream.write((uint32_t)sceneData.animations.size());
        for (const auto& pAnimation : sceneData.animations)
        {
            writeAnimation(stream, pAnimation);
        }

        writeMarker(stream, "Metadata");
        writeMetadata(stream, sceneData.metadata);

        writeMarker(stream, "Meshes");
        stream.write(sceneData.meshDesc);
        stream.write(sceneData.meshNames);
        stream.write(sceneData.meshBBs);
        stream.write(sceneData.meshInstanceData);
        stream.write(sceneData.displacedMeshInstanceCount);
        stream.write((uint32_t)sceneData.meshIdToInstanceIds.size());
        for (const auto& item : sceneData.meshIdToInstanceIds)
        {
            stream.write(item);
        }
        stream.write((uint32_t)sceneData.meshGroups.size());
        for (const auto& group : sceneData.meshGroups)
        {
            stream.write(group.meshList);
            stream.write(group.isStatic);
            stream.write(group.isDisplaced);
        }
        stream.write((uint32_t)sceneData.cachedMeshes.size());
        for (const auto& cachedMesh : sceneData.cachedMeshes)
        {
            stream.write(cachedMesh.meshId);
            stream.write(cachedMesh.timeSamples);
            stream.write((uint32_t)cachedMesh.vertexData.size());
            for (const auto& data : cachedMesh.vertexData) stream.write(data);
        }
        stream.write(sceneData.has16BitIndices);
        stream.write(sceneData.has32BitIndices);
        stream.write(sceneData.meshDrawCount);
        stream.write(sceneData.meshIndexData);
        stream.write(sceneData.meshStaticData);
        stream.write(sceneData.meshDynamicData);

        writeMarker(stream, "Curves");
        stream.write(sceneData.curveDesc);
        stream.write(sceneData.curveBBs);
        stream.write(sceneData.curveInstanceData);
        stream.write(sceneData.curveIndexData);
        stream.write(sceneData.curveStaticData);

        stream.write((uint32_t)sceneData.cachedCurves.size());
        for (const auto& cachedCurve : sceneData.cachedCurves)
        {
            stream.write(cachedCurve.timeSamples);
            stream.write(cachedCurve.indexData);
            stream.write((uint32_t)cachedCurve.vertexData.size());
            for (const auto& data : cachedCurve.vertexData) stream.write(data);
        }

        writeMarker(stream, "CustomPrimitives");
        stream.write(sceneData.customPrimitiveDesc);
        stream.write(sceneData.customPrimitiveAABBs);

        writeMarker(stream, "End");
    }

    Scene::SceneData SceneCache::readSceneData(InputStream& stream)
    {
        Scene::SceneData sceneData;

        readMarker(stream, "Filename");
        stream.read(sceneData.filename);

        readMarker(stream, "RenderSettings");
        stream.read(sceneData.renderSettings);

        readMarker(stream, "Cameras");
        sceneData.cameras.resize(stream.read<uint32_t>());
        for (auto& pCamera : sceneData.cameras) pCamera = readCamera(stream);
        stream.read(sceneData.selectedCamera);
        stream.read(sceneData.cameraSpeed);

        readMarker(stream, "Lights");
        sceneData.lights.resize(stream.read<uint32_t>());
        for (auto& pLight : sceneData.lights) pLight = readLight(stream);

        readMarker(stream, "Grids");
        sceneData.grids.resize(stream.read<uint32_t>());
        for (auto& pGrid : sceneData.grids) pGrid = readGrid(stream);

        readMarker(stream, "Volumes");
        sceneData.volumes.resize(stream.read<uint32_t>());
        for (auto& pVolume : sceneData.volumes) pVolume = readVolume(stream, sceneData.grids);

        readMarker(stream, "EnvMap");
        auto hasEnvMap = stream.read<bool>();
        if (hasEnvMap) sceneData.pEnvMap = readEnvMap(stream);

        // Material textures are loaded asynchronously to allow loading other data
        // in parallel while loading textures from files and uploading them to the GPU.
        // Due to the current implementation, we need to make sure no other GPU operations (transfers)
        // are executed while loading material textures. Due to this, we load volume grids and the envmap
        // before material textures, as they upload buffers to the GPU when created.
        // Make sure no other GPU operations are executed until calling pMaterialTextureLoader.reset()
        // further down which blocks until all textures are loaded.
        auto pMaterialTextureLoader = std::make_unique<MaterialTextureLoader>(true);

        readMarker(stream, "Materials");
        sceneData.materials.resize(stream.read<uint32_t>());
        for (auto& pMaterial : sceneData.materials) pMaterial = readMaterial(stream, *pMaterialTextureLoader);

        readMarker(stream, "SceneGraph");
        sceneData.sceneGraph.resize(stream.read<uint32_t>());
        for (auto &node : sceneData.sceneGraph)
        {
            stream.read(node.name);
            stream.read(node.parent);
            stream.read(node.transform);
            stream.read(node.meshBind);
            stream.read(node.localToBindSpace);
        }

        readMarker(stream, "Animations");
        sceneData.animations.resize(stream.read<uint32_t>());
        for (auto& pAnimation : sceneData.animations) pAnimation = readAnimation(stream);

        readMarker(stream, "Metadata");
        sceneData.metadata = readMetadata(stream);

        readMarker(stream, "Meshes");
        stream.read(sceneData.meshDesc);
        stream.read(sceneData.meshNames);
        stream.read(sceneData.meshBBs);
        stream.read(sceneData.meshInstanceData);
        stream.read(sceneData.displacedMeshInstanceCount);
        sceneData.meshIdToInstanceIds.resize(stream.read<uint32_t>());
        for (auto& item : sceneData.meshIdToInstanceIds)
        {
            stream.read(item);
        }
        sceneData.meshGroups.resize(stream.read<uint32_t>());
        for (auto& group : sceneData.meshGroups)
        {
            stream.read(group.meshList);
            stream.read(group.isStatic);
            stream.read(group.isDisplaced);
        }
        sceneData.cachedMeshes.resize(stream.read<uint32_t>());
        for (auto& cachedMesh : sceneData.cachedMeshes)
        {
            stream.read(cachedMesh.meshId);
            stream.read(cachedMesh.timeSamples);
            cachedMesh.vertexData.resize(stream.read<uint32_t>());
            for (auto& data : cachedMesh.vertexData) stream.read(data);
        }
        stream.read(sceneData.has16BitIndices);
        stream.read(sceneData.has32BitIndices);
        stream.read(sceneData.meshDrawCount);
        stream.read(sceneData.meshIndexData);
        stream.read(sceneData.meshStaticData);
        stream.read(sceneData.meshDynamicData);

        readMarker(stream, "Curves");
        stream.read(sceneData.curveDesc);
        stream.read(sceneData.curveBBs);
        stream.read(sceneData.curveInstanceData);
        stream.read(sceneData.curveIndexData);
        stream.read(sceneData.curveStaticData);

        sceneData.cachedCurves.resize(stream.read<uint32_t>());
        for (auto& cachedCurve : sceneData.cachedCurves)
        {
            stream.read(cachedCurve.timeSamples);
            stream.read(cachedCurve.indexData);
            cachedCurve.vertexData.resize(stream.read<uint32_t>());
            for (auto& data : cachedCurve.vertexData) stream.read(data);
        }

        readMarker(stream, "CustomPrimitives");
        stream.read(sceneData.customPrimitiveDesc);
        stream.read(sceneData.customPrimitiveAABBs);

        readMarker(stream, "End");

        pMaterialTextureLoader.reset();

        return sceneData;
    }

    // Metadata

    void SceneCache::writeMetadata(OutputStream& stream, const Scene::Metadata& metadata)
    {
        stream.write(metadata.fNumber);
        stream.write(metadata.filmISO);
        stream.write(metadata.shutterSpeed);
        stream.write(metadata.samplesPerPixel);
        stream.write(metadata.maxDiffuseBounces);
        stream.write(metadata.maxSpecularBounces);
        stream.write(metadata.maxTransmissionBounces);
        stream.write(metadata.maxVolumeBounces);
        std::optional<bool> dummy;
        stream.write(dummy);
    }

    Scene::Metadata SceneCache::readMetadata(InputStream& stream)
    {
        Scene::Metadata metadata;
        stream.read(metadata.fNumber);
        stream.read(metadata.filmISO);
        stream.read(metadata.shutterSpeed);
        stream.read(metadata.samplesPerPixel);
        stream.read(metadata.maxDiffuseBounces);
        stream.read(metadata.maxSpecularBounces);
        stream.read(metadata.maxTransmissionBounces);
        stream.read(metadata.maxVolumeBounces);
        std::optional<bool> dummy;
        stream.read(dummy);
        return metadata;
    }

    // Camera

    void SceneCache::writeCamera(OutputStream& stream, const Camera::SharedPtr& pCamera)
    {
        stream.write(pCamera->mHasAnimation);
        stream.write(pCamera->mIsAnimated);
        stream.write(pCamera->mNodeID);

        stream.write(pCamera->mName);
        stream.write(pCamera->mPreserveHeight);
        stream.write(pCamera->mData);
    }

    Camera::SharedPtr SceneCache::readCamera(InputStream& stream)
    {
        auto pCamera = Camera::create();

        stream.read(pCamera->mHasAnimation);
        stream.read(pCamera->mIsAnimated);
        stream.read(pCamera->mNodeID);

        stream.read(pCamera->mName);
        stream.read(pCamera->mPreserveHeight);
        stream.read(pCamera->mData);

        return pCamera;
    }

    // Light

    void SceneCache::writeLight(OutputStream& stream, const Light::SharedPtr& pLight)
    {
        LightType type = pLight->getType();
        stream.write(type);

        stream.write(pLight->mHasAnimation);
        stream.write(pLight->mIsAnimated);
        stream.write(pLight->mNodeID);

        stream.write(pLight->mName);
        stream.write(pLight->mActive);
        stream.write(pLight->mData);

        switch (type)
        {
        case LightType::Point:
        case LightType::Directional:
            break;
        case LightType::Distant:
            stream.write(std::static_pointer_cast<DistantLight>(pLight)->mAngle);
            break;
        case LightType::Rect:
        case LightType::Disc:
        case LightType::Sphere:
            stream.write(std::static_pointer_cast<AnalyticAreaLight>(pLight)->mScaling);
            stream.write(std::static_pointer_cast<AnalyticAreaLight>(pLight)->mTransformMatrix);
            break;
        }
    }

    Light::SharedPtr SceneCache::readLight(InputStream& stream)
    {
        Light::SharedPtr pLight;
        auto type = stream.read<LightType>();

        switch (type)
        {
        case LightType::Point:
            pLight = PointLight::create();
            break;
        case LightType::Directional:
            pLight = DirectionalLight::create();
            break;
        case LightType::Distant:
            pLight = DistantLight::create();
            break;
        case LightType::Rect:
            pLight = RectLight::create();
            break;
        case LightType::Disc:
            pLight = DiscLight::create();
            break;
        case LightType::Sphere:
            pLight = SphereLight::create();
            break;
        }

        stream.read(pLight->mHasAnimation);
        stream.read(pLight->mIsAnimated);
        stream.read(pLight->mNodeID);

        stream.read(pLight->mName);
        stream.read(pLight->mActive);
        stream.read(pLight->mData);

        switch (type)
        {
        case LightType::Point:
        case LightType::Directional:
            break;
        case LightType::Distant:
            stream.read(std::static_pointer_cast<DistantLight>(pLight)->mAngle);
            break;
        case LightType::Rect:
        case LightType::Disc:
        case LightType::Sphere:
            stream.read(std::static_pointer_cast<AnalyticAreaLight>(pLight)->mScaling);
            stream.read(std::static_pointer_cast<AnalyticAreaLight>(pLight)->mTransformMatrix);
            break;
        }

        return pLight;
    }

    // Material

    void SceneCache::writeMaterial(OutputStream& stream, const Material::SharedPtr& pMaterial)
    {
        auto writeTextureSlot = [&stream, &pMaterial] (Material::TextureSlot slot)
        {
            const auto& pTexture = pMaterial->getTexture(slot);
            bool hasTexture = pTexture != nullptr;
            stream.write(hasTexture);
            if (hasTexture)
            {
                stream.write(pTexture->getSourceFilename());
            }
        };

        stream.write(pMaterial->mName);
        stream.write(pMaterial->mData);
        for (uint32_t slot = 0; slot < (uint32_t)Material::TextureSlot::Count; ++slot)
        {
            writeTextureSlot(Material::TextureSlot(slot));
        }
        writeTransform(stream, pMaterial->mTextureTransform);
        stream.write(pMaterial->mDoubleSided);
        stream.write(pMaterial->mAlphaRange);
        stream.write(pMaterial->mIsTexturedBaseColorConstant);
        stream.write(pMaterial->mIsTexturedAlphaConstant);
    }

    Material::SharedPtr SceneCache::readMaterial(InputStream& stream, MaterialTextureLoader& materialTextureLoader)
    {
        Material::SharedPtr pMaterial = Material::create("");

        auto readTextureSlot = [&] (Material::TextureSlot slot)
        {
            auto hasTexture = stream.read<bool>();
            if (hasTexture)
            {
                auto filename = stream.read<std::string>();
                materialTextureLoader.loadTexture(pMaterial, slot, filename);
            }
        };

        stream.read(pMaterial->mName);
        stream.read(pMaterial->mData);
        for (uint32_t slot = 0; slot < (uint32_t)Material::TextureSlot::Count; ++slot)
        {
            readTextureSlot(Material::TextureSlot(slot));
        }
        pMaterial->mTextureTransform = readTransform(stream);
        stream.read(pMaterial->mDoubleSided);
        stream.read(pMaterial->mAlphaRange);
        stream.read(pMaterial->mIsTexturedBaseColorConstant);
        stream.read(pMaterial->mIsTexturedAlphaConstant);

        return pMaterial;
    }

    // Volume

    void SceneCache::writeVolume(OutputStream& stream, const Volume::SharedPtr& pVolume, const std::vector<Grid::SharedPtr>& grids)
    {
        stream.write(pVolume->mHasAnimation);
        stream.write(pVolume->mIsAnimated);
        stream.write(pVolume->mNodeID);

        stream.write(pVolume->mName);
        for (const auto& gridSequence : pVolume->mGrids)
        {
            stream.write((uint32_t)gridSequence.size());
            for (const auto& pGrid : gridSequence)
            {
                uint32_t id = pGrid ? (uint32_t)std::distance(grids.begin(), std::find(grids.begin(), grids.end(), pGrid)) : uint32_t(-1);
                stream.write(id);
            }
        }
        stream.write(pVolume->mGridFrame);
        stream.write(pVolume->mGridFrameCount);
        stream.write(pVolume->mBounds);
        stream.write(pVolume->mData);
    }

    Volume::SharedPtr SceneCache::readVolume(InputStream& stream, const std::vector<Grid::SharedPtr>& grids)
    {
        Volume::SharedPtr pVolume = Volume::create("");

        stream.read(pVolume->mHasAnimation);
        stream.read(pVolume->mIsAnimated);
        stream.read(pVolume->mNodeID);

        stream.read(pVolume->mName);
        for (auto& gridSequence : pVolume->mGrids)
        {
            gridSequence.resize(stream.read<uint32_t>());
            for (auto& pGrid : gridSequence)
            {
                auto id = stream.read<uint32_t>();
                pGrid = id == uint32_t(-1) ? nullptr : grids[id];
            }
        }
        stream.read(pVolume->mGridFrame);
        stream.read(pVolume->mGridFrameCount);
        stream.read(pVolume->mBounds);
        stream.read(pVolume->mData);

        return pVolume;
    }

    // Grid

    void SceneCache::writeGrid(OutputStream& stream, const Grid::SharedPtr& pGrid)
    {
        const nanovdb::HostBuffer& buffer = pGrid->mGridHandle.buffer();
        stream.write((uint64_t)buffer.size());
        stream.write(buffer.data(), buffer.size());
    }

    Grid::SharedPtr SceneCache::readGrid(InputStream& stream)
    {
        uint64_t size = stream.read<uint64_t>();
        auto buffer = nanovdb::HostBuffer::create(size);
        stream.read(buffer.data(), buffer.size());
        return Grid::SharedPtr(new Grid(nanovdb::GridHandle<nanovdb::HostBuffer>(std::move(buffer))));
    }

    // EnvMap

    void SceneCache::writeEnvMap(OutputStream& stream, const EnvMap::SharedPtr& pEnvMap)
    {
        auto filename = pEnvMap->getEnvMap()->getSourceFilename();
        stream.write(filename);
        stream.write(pEnvMap->mData);
        stream.write(pEnvMap->mRotation);
    }

    EnvMap::SharedPtr SceneCache::readEnvMap(InputStream& stream)
    {
        auto filename = stream.read<std::string>();
        auto pEnvMap = EnvMap::create(filename);
        stream.read(pEnvMap->mData);
        stream.read(pEnvMap->mRotation);
        return pEnvMap;
    }

    // Transform

    void SceneCache::writeTransform(OutputStream& stream, const Transform& transform)
    {
        stream.write(transform.mTranslation);
        stream.write(transform.mScaling);
        stream.write(transform.mRotation);
    }

    Transform SceneCache::readTransform(InputStream& stream)
    {
        Transform transform;
        stream.read(transform.mTranslation);
        stream.read(transform.mScaling);
        stream.read(transform.mRotation);
        return transform;
    }

    // Animation

    void SceneCache::writeAnimation(OutputStream& stream, const Animation::SharedPtr& pAnimation)
    {
        stream.write(pAnimation->mName);
        stream.write(pAnimation->mNodeID);
        stream.write(pAnimation->mDuration);
        stream.write(pAnimation->mPreInfinityBehavior);
        stream.write(pAnimation->mPostInfinityBehavior);
        stream.write(pAnimation->mInterpolationMode);
        stream.write(pAnimation->mEnableWarping);
        stream.write(pAnimation->mKeyframes);
    }

    Animation::SharedPtr SceneCache::readAnimation(InputStream& stream)
    {
        Animation::SharedPtr pAnimation = Animation::create("", 0, 0.0);
        stream.read(pAnimation->mName);
        stream.read(pAnimation->mNodeID);
        stream.read(pAnimation->mDuration);
        stream.read(pAnimation->mPreInfinityBehavior);
        stream.read(pAnimation->mPostInfinityBehavior);
        stream.read(pAnimation->mInterpolationMode);
        stream.read(pAnimation->mEnableWarping);
        stream.read(pAnimation->mKeyframes);
        return pAnimation;
    }

    // Marker

    void SceneCache::writeMarker(OutputStream& stream, const std::string& id)
    {
        stream.write(id);
    }

    void SceneCache::readMarker(InputStream& stream, const std::string& id)
    {
        auto str = stream.read<std::string>();
        if (id != str) throw std::exception("Found invalid marker in cache!");
    }
}
