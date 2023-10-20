/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
#include "SceneBuilderDump.h"
#include "Scene/SceneBuilder.h"
#include "Utils/Math/FNVHash.h"
#include <fmt/format.h>
#include <BS_thread_pool.hpp>

/// SceneBuilder printing is split off to its own file to avoid polluting the SceneBuilder.cpp with debug prints

namespace Falcor
{

namespace
{
std::string printMatrix(const float4x4& m)
{
    if (m == float4x4::identity())
        return "identity";
    return fmt::format("{}", m);
}

std::string printNodeID(NodeID id)
{
    if (!id.isValid())
        return "N/A";
    return fmt::format("{}", id);
}

std::string toString(const StaticVertexData& vertex)
{
    return fmt::format("pos: {}; nrm: {}; tan: {}; uvs: {}", vertex.position, vertex.normal, vertex.tangent, vertex.texCrd);
}

std::string toString(const PackedStaticVertexData& vertex)
{
    return toString(vertex.unpack());
}

std::string toString(const StaticCurveVertexData& vertex)
{
    return fmt::format("pos: {}; rad: {}; uvs: {}", vertex.position, vertex.radius, vertex.texCrd);
}

std::string toString(const DynamicCurveVertexData& vertex)
{
    return fmt::format("pos: {}", vertex.position);
}

template<typename T>
uint64_t hash64(const std::vector<T>& v)
{
    return fnvHashArray64(v.data(), v.size() * sizeof(T));
}
} // namespace

std::map<std::string, std::string> SceneBuilderDump::getDebugContent(const SceneBuilder& sceneBuilder)
{
    struct FullMeshDesc
    {
        MeshID id;
        const SceneBuilder::MeshSpec* meshSpec;
        const CachedMesh* cachedMesh{nullptr};
        const CachedCurve* cachedCurve{nullptr}; // when using polytube hair
        std::string deduplicatedName;
    };

    struct FullCurveDesc
    {
        CurveID id;
        const SceneBuilder::CurveSpec* curveSpec;
        const CachedCurve* cachedCurve{nullptr};
        std::string deduplicatedName;
    };

    std::vector<FullMeshDesc> sortedMeshes(sceneBuilder.mMeshes.size());
    std::vector<FullCurveDesc> sortedCurves(sceneBuilder.mCurves.size());

    for (size_t i = 0; i < sceneBuilder.mMeshes.size(); ++i)
    {
        sortedMeshes[i].id = MeshID{i};
        sortedMeshes[i].meshSpec = &sceneBuilder.mMeshes[i];
    }

    for (size_t i = 0; i < sceneBuilder.mCurves.size(); ++i)
    {
        sortedCurves[i].id = CurveID{i};
        sortedCurves[i].curveSpec = &sceneBuilder.mCurves[i];
    }

    for (size_t i = 0; i < sceneBuilder.mSceneData.cachedCurves.size(); ++i)
    {
        const CachedCurve& cached = sceneBuilder.mSceneData.cachedCurves[i];
        if (cached.tessellationMode == CurveTessellationMode::LinearSweptSphere)
        {
            sortedCurves[cached.geometryID.get()].cachedCurve = &sceneBuilder.mSceneData.cachedCurves[i];
        }
        else
        {
            sortedMeshes[cached.geometryID.get()].cachedCurve = &sceneBuilder.mSceneData.cachedCurves[i];
        }
    }

    for (size_t i = 0; i < sceneBuilder.mSceneData.cachedMeshes.size(); ++i)
    {
        const CachedMesh& cached = sceneBuilder.mSceneData.cachedMeshes[i];
        sortedMeshes[cached.meshID.get()].cachedCurve = &sceneBuilder.mSceneData.cachedCurves[i];
    }

    std::sort(
        sortedMeshes.begin(),
        sortedMeshes.end(),
        [](const FullMeshDesc& lhs, const FullMeshDesc& rhs) { return lhs.meshSpec->name < rhs.meshSpec->name; }
    );

    std::sort(
        sortedCurves.begin(),
        sortedCurves.end(),
        [](const FullCurveDesc& lhs, const FullCurveDesc& rhs) { return lhs.curveSpec->name < rhs.curveSpec->name; }
    );

    std::map<std::string, std::string> result;
    auto deduplicateName = [&](std::string name) -> std::string
    {
        if (auto it = result.find(name); it == result.end())
            return name;
        int counter = 0;
        while (true)
        {
            std::string temp = fmt::format("{}_{}", name, counter);
            if (auto it = result.find(temp); it == result.end())
                return temp;
            ++counter;
        }
    };

    auto sanityName = [&](std::string name) -> std::string
    {
        auto pos = name.find_first_of("Prototype");
        if (pos == std::string::npos)
            return deduplicateName(name);
        pos = name.find_first_of("/", pos);
        if (pos == std::string::npos)
            return deduplicateName(name);
        name.erase(name.begin(), name.begin() + pos);
        return deduplicateName(name);
    };

    for (auto& it : sortedMeshes)
    {
        it.deduplicatedName = sanityName(it.meshSpec->name);;
        result[it.deduplicatedName] = it.deduplicatedName;
    }
    for (auto& it : sortedCurves)
    {
        it.deduplicatedName = sanityName(it.curveSpec->name);;
        result[it.deduplicatedName] = it.deduplicatedName;
    }

    std::mutex resultMutex;
    auto genMesh = [&](int i)
    {
        const auto& meshDesc = sortedMeshes[i];
        const SceneBuilder::MeshSpec& mesh = *meshDesc.meshSpec;
        std::string name = meshDesc.deduplicatedName;

        std::string res;
        res += fmt::format("Mesh: {}\n", name);
        res += fmt::format("   material: {}\n", sceneBuilder.mSceneData.pMaterials->getMaterial(mesh.materialId)->getName());
        res += fmt::format("   skinned: {}\n", mesh.isSkinned() ? "YES" : "NO");
        res += fmt::format("   mesh.staticData: {}\n", hash64(mesh.staticData));
        // res += fmt::format("   mesh.staticData:\n");
        // for (auto& it : mesh.staticData)
        //     res += fmt::format("      {}\n", toString(it));
        res += fmt::format("   mesh.indexData: {}\n", hash64(mesh.indexData));
        if (mesh.isSkinned())
        {
            NodeID bindMatrixID = *mesh.instances.begin();
            if (bindMatrixID.isValid())
                res += fmt::format("   bindXform: {}\n", printMatrix(sceneBuilder.mSceneGraph[bindMatrixID.get()].meshBind));

            NodeID skeletonMatrixID = (mesh.skeletonNodeID == NodeID::Invalid()) ? *mesh.instances.begin() : mesh.skeletonNodeID;
            if (skeletonMatrixID.isValid())
                res += fmt::format("   skeletonXform: {}\n", printMatrix(sceneBuilder.mSceneGraph[skeletonMatrixID.get()].transform));

            for (size_t j = 0; j < mesh.skinningData.size(); ++j)
            {
                const SkinningVertexData& skinningData = mesh.skinningData[j];
                res += fmt::format("   Vertex#{}\n", j);
                for (int k = 0; k < 4; ++k)
                {
                    NodeID boneID{skinningData.boneID[k]};
                    float weight = skinningData.boneWeight[k];
                    if (weight == 0 || !boneID.isValid())
                        continue;
                    res += fmt::format("      Bone#{}\n", k);
                    res += fmt::format("         boneXform: {} x w{}\n", printMatrix(sceneBuilder.mSceneGraph[boneID.get()].transform), weight);
                }
            }
        }

        if (meshDesc.cachedMesh)
        {
            res += "   Cached meshes:";
            for (auto t : meshDesc.cachedCurve->timeSamples)
                res += fmt::format(" {}", t);
            res += "\n";

            for (size_t t = 0; t < meshDesc.cachedMesh->timeSamples.size(); ++t)
                res += fmt::format("         t{}: {}\n", t, hash64(meshDesc.cachedMesh->vertexData[t]));
        }

        if (meshDesc.cachedCurve)
        {
            const auto& cached = *meshDesc.cachedCurve;

            res += "   Cached curves:\n";
            for (auto t : cached.timeSamples)
                res += fmt::format(" {}", t);
            res += "\n";
            res += fmt::format("      Index data: {}\n", hash64(cached.indexData));

            for (size_t t = 0; t < cached.timeSamples.size(); ++t)
                res += fmt::format("         t{}: {}\n", t, hash64(cached.vertexData[t]));
        }
        std::lock_guard<std::mutex> lock(resultMutex);
        result[name] = std::move(res);
    };

    auto genCurve = [&](int i)
    {
        const auto& curveDesc = sortedCurves[i];
        const SceneBuilder::CurveSpec& curve = *curveDesc.curveSpec;
        std::string name = curveDesc.deduplicatedName;

        std::string res;
        res += fmt::format("Curve: {}\n", name);
        res += fmt::format("   material: {}\n", sceneBuilder.mSceneData.pMaterials->getMaterial(curve.materialId)->getName());
        res += fmt::format("   staticVertexCount: {}\n", curve.staticVertexCount);
        res += fmt::format("   indexCount: {}\n", curve.indexCount);
        res += fmt::format("   vertexCount: {}\n", curve.vertexCount);
        res += fmt::format("   degree: {}\n", curve.degree);
        res += fmt::format("   curve.staticData: {}\n", hash64(curve.staticData));
        res += fmt::format("   curve.indexData: {}\n", hash64(curve.indexData));

        if (curveDesc.cachedCurve)
        {
            const auto& cached = *curveDesc.cachedCurve;

            res += "   Cached curves:\n";
            for (auto t : cached.timeSamples)
                res += fmt::format(" {}", t);
            res += "\n";
            res += fmt::format("      Index data: {}\n", hash64(cached.indexData));

            for (size_t t = 0; t < cached.timeSamples.size(); ++t)
                res += fmt::format("         t{}: {}\n", t, hash64(cached.vertexData[t]));
        }

        std::lock_guard<std::mutex> lock(resultMutex);
        result[name] = std::move(res);
    };

    BS::thread_pool threadPool;

    for (size_t i = 0; i < sortedMeshes.size(); ++i)
        threadPool.push_task([&,i]{ genMesh(i); });
    for (size_t i = 0; i < sortedCurves.size(); ++i)
        threadPool.push_task([&,i]{ genCurve(i); });

    threadPool.wait_for_tasks();

    return result;
}

} // namespace Falcor
