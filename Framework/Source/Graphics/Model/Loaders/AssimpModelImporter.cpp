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
#include "AssimpModelImporter.h"
#include "Graphics/Model/Model.h"
#include "assimp/Importer.hpp"
#include "assimp/postprocess.h"
#include "assimp/scene.h"
#include "Graphics/Model/Animation.h"
#include "Graphics/Model/Mesh.h"
#include "Graphics/Model/AnimationController.h"
#include "glm/common.hpp"
#include "glm/geometric.hpp"
#include "API/Texture.h"
#include "API/Buffer.h"
#include "glm/matrix.hpp"
#include "Utils/OS.h"
#include "Graphics/TextureHelper.h"
#include "API/VertexLayout.h"
#include "Data/VertexAttrib.h"
#include "Utils/StringUtils.h"
#include "API/Device.h"

namespace Falcor
{
    static_assert(Mesh::kMaxBonesPerVertex == 4, "Fix the weights and IDs container below");
    using VertexWeightsVec = std::vector<float4>;
    struct uvec8_4
    {
        uint8_t u[4];
        uint8_t& operator[](uint32_t i) { return u[i]; }
    };

    using VertexIdsVec = std::vector<uvec8_4>;

    template<typename posType>
    void generateSubmeshTangentData(
        const std::vector<uint32_t>& indices,
        uint32_t vertexCount,
        const posType* vertexPosData,
        const glm::vec3* vertexNormalData,
        const glm::vec2* texCrdData,
        uint32_t texCrdCount,
        glm::vec3* bitangentData);


    void loadBones(const aiMesh* pAiMesh, VertexWeightsVec& weights, VertexIdsVec& ids, uint32_t vertexCount, const std::map<std::string, uint32_t>& boneNameToIdMap)
    {
        if (pAiMesh->mNumBones > 0xff)
        {
            logError("Too many bones");
        }

        weights.resize(vertexCount);
        ids.resize(vertexCount);

        for (uint32_t bone = 0; bone < pAiMesh->mNumBones; bone++)
        {
            const aiBone* pAiBone = pAiMesh->mBones[bone];
            uint32_t aiBoneID = boneNameToIdMap.at(std::string(pAiBone->mName.C_Str()));

            // The way Assimp works, the weights holds the IDs of the vertices it affects.
            // We loop over all the weights, initializing the vertices data along the way
            for (uint32_t weightID = 0; weightID < pAiBone->mNumWeights; weightID++)
            {
                // Get the vertex the current weight affects
                const aiVertexWeight& aiWeight = pAiBone->mWeights[weightID];

                // Get the address of the Bone ID and weight for the current vertex
                uvec8_4& vertexIds = ids[aiWeight.mVertexId];
                vec4& vertexWeights = weights[aiWeight.mVertexId];

                // Find the next unused slot in the bone array of the vertex, and initialize it with the current value
                bool emptySlotFound = false;
                for (uint32_t j = 0; j < Mesh::kMaxBonesPerVertex; j++)
                {
                    if (vertexWeights[j] == 0)
                    {
                        vertexIds[j] = (uint8_t)aiBoneID;
                        vertexWeights[j] = aiWeight.mWeight;
                        emptySlotFound = true;
                        break;
                    }
                }

                if (emptySlotFound == false)
                {
                    logError("Too many bones");
                }
            }
        }

        // Now we need to normalize the weights for each vertex, since in some models the sum is larger than 1
        for (uint32_t i = 0; i < vertexCount; i++)
        {
            vec4& w = weights[i];

            float f = 0;
            // Sum the weights
            for (int j = 0; j < Mesh::kMaxBonesPerVertex; j++)
            {
                f += w[j];
            }
            w /= f;
        }
    }

    std::vector<uint32_t> createIndexBufferData(const aiMesh* pAiMesh)
    {
        uint32_t indexCount = pAiMesh->mNumFaces * pAiMesh->mFaces[0].mNumIndices;
        std::vector<uint32_t> indices(indexCount);
        const uint32_t firstFacePrimSize = pAiMesh->mFaces[0].mNumIndices;

        for (uint32_t i = 0; i < pAiMesh->mNumFaces; i++)
        {
            uint32_t primSize = pAiMesh->mFaces[i].mNumIndices;
            assert(primSize == firstFacePrimSize); // Mesh contains mixed primitive types, can be solved using aiProcess_SortByPType
            for (uint32_t j = 0; j < firstFacePrimSize; j++)
            {
                indices[i * firstFacePrimSize + j] = (uint32_t)(pAiMesh->mFaces[i].mIndices[j]);
            }
        }

        return indices;
    }

    void genTangentSpace(const aiMesh* pAiMesh)
    {
        if (pAiMesh->mFaces[0].mNumIndices == 3)
        {
            aiMesh* pMesh = const_cast<aiMesh*>(pAiMesh);
            pMesh->mBitangents = new aiVector3D[pMesh->mNumVertices];

            const glm::vec3* pPos = (glm::vec3*)pMesh->mVertices;
            glm::vec3* pBi = (glm::vec3*)pMesh->mBitangents;
            glm::vec3* pNormals = (glm::vec3*)pMesh->mNormals;
            std::vector<uint32_t> indices = createIndexBufferData(pAiMesh);

            uint32_t texCrdCount = 0;
            std::vector<glm::vec2> texCrd;
            if (pMesh->mTextureCoords[0] != nullptr)
            {
                texCrdCount = 1;
                texCrd.resize(pMesh->mNumVertices);
                for (size_t i = 0; i < pMesh->mNumVertices; ++i)
                {
                    texCrd[i] = glm::vec2(pMesh->mTextureCoords[0][i].x, pMesh->mTextureCoords[0][i].y);
                }
            }

            generateSubmeshTangentData<glm::vec3>(indices, pAiMesh->mNumVertices, pPos, pNormals, (texCrdCount > 0 ? texCrd.data() : nullptr), texCrdCount, pBi);
        }
    }

    struct layoutsData
    {
        uint32_t pos;
        std::string name;
        ResourceFormat format;
    };

    static const layoutsData kLayoutData[VERTEX_LOCATION_COUNT] =
    {
        { VERTEX_POSITION_LOC,      VERTEX_POSITION_NAME,       ResourceFormat::RGB32Float },
        { VERTEX_NORMAL_LOC,        VERTEX_NORMAL_NAME,         ResourceFormat::RGB32Float },
        { VERTEX_BITANGENT_LOC,     VERTEX_BITANGENT_NAME,      ResourceFormat::RGB32Float },
        { VERTEX_TEXCOORD_LOC,      VERTEX_TEXCOORD_NAME,       ResourceFormat::RGB32Float }, //for some reason this is rgb
        { VERTEX_LIGHTMAP_UV_LOC,   VERTEX_LIGHTMAP_UV_NAME,    ResourceFormat::RGB32Float }, //for some reason this is rgb
        { VERTEX_BONE_WEIGHT_LOC,   VERTEX_BONE_WEIGHT_NAME,    ResourceFormat::RGBA32Float },
        { VERTEX_BONE_ID_LOC,       VERTEX_BONE_ID_NAME,        ResourceFormat::RGBA8Uint },
        { VERTEX_DIFFUSE_COLOR_LOC, VERTEX_DIFFUSE_COLOR_NAME,  ResourceFormat::RGBA32Float },
    };


    glm::mat4 aiMatToGLM(const aiMatrix4x4& aiMat)
    {
        glm::mat4 glmMat;
        glmMat[0][0] = aiMat.a1; glmMat[0][1] = aiMat.a2; glmMat[0][2] = aiMat.a3; glmMat[0][3] = aiMat.a4;
        glmMat[1][0] = aiMat.b1; glmMat[1][1] = aiMat.b2; glmMat[1][2] = aiMat.b3; glmMat[1][3] = aiMat.b4;
        glmMat[2][0] = aiMat.c1; glmMat[2][1] = aiMat.c2; glmMat[2][2] = aiMat.c3; glmMat[2][3] = aiMat.c4;
        glmMat[3][0] = aiMat.d1; glmMat[3][1] = aiMat.d2; glmMat[3][2] = aiMat.d3; glmMat[3][3] = aiMat.d4;

        return glm::transpose(glmMat);
    }

    BasicMaterial::MapType getFalcorTexTypeFromAi(aiTextureType type, const bool isObjFile)
    {
        switch (type)
        {
        case aiTextureType_DIFFUSE:
            return BasicMaterial::MapType::DiffuseMap;
        case aiTextureType_SPECULAR:
            return BasicMaterial::MapType::SpecularMap;
        case aiTextureType_EMISSIVE:
            return BasicMaterial::MapType::EmissiveMap;
        case aiTextureType_HEIGHT:
            // OBJ doesn't support normal maps, so they are usually placed in the height map slot. For consistency with other formats, we move them to the normal map slot.
            return isObjFile ? BasicMaterial::MapType::NormalMap : BasicMaterial::MapType::HeightMap;
        case aiTextureType_NORMALS:
            return BasicMaterial::MapType::NormalMap;
        case aiTextureType_OPACITY:
            return BasicMaterial::MapType::AlphaMap;
        case aiTextureType_AMBIENT:
            return BasicMaterial::MapType::AmbientMap;
        default:
            return BasicMaterial::MapType::Count;
        }
    }

    bool isSrgbRequired(aiTextureType aiType, bool isSrgbRequested)
    {
        if (isSrgbRequested == false)
        {
            return false;
        }
        switch (aiType)
        {
        case aiTextureType_DIFFUSE:
        case aiTextureType_SPECULAR:
        case aiTextureType_AMBIENT:
        case aiTextureType_EMISSIVE:
        case aiTextureType_LIGHTMAP:
        case aiTextureType_REFLECTION:
            return true;
        case aiTextureType_HEIGHT:
        case aiTextureType_NORMALS:
        case aiTextureType_SHININESS:
        case aiTextureType_OPACITY:
        case aiTextureType_DISPLACEMENT:
            return false;
        default:
            should_not_get_here();
            return false;
        }
    }

    void AssimpModelImporter::loadTextures(const aiMaterial* pAiMaterial, const std::string& folder, BasicMaterial* pMaterial, bool isObjFile, bool useSrgb)
    {
        for (int i = 0; i < AI_TEXTURE_TYPE_MAX; ++i)
        {
            aiTextureType aiType = (aiTextureType)i;

            uint32_t textureCount = pAiMaterial->GetTextureCount(aiType);

            if (textureCount >= 1)
            {
                if (textureCount != 1)
                {
                    logError("Can't create material with more then one texture per Type");
                    return;
                }

                // Get the texture name
                aiString path;
                pAiMaterial->GetTexture(aiType, 0, &path);
                std::string s(path.data);
                Texture::SharedPtr pTex = nullptr;

                if (s.empty())
                {
                    logWarning("Texture has empty file name, ignoring.");
                    continue;
                }

                // Check if the texture was already loaded
                const auto& a = mTextureCache.find(s);
                if (a != mTextureCache.end())
                {
                    pTex = a->second;
                }
                else
                {
                    // create a new texture
                    std::string fullpath = folder + '\\' + s;
                    pTex = createTextureFromFile(fullpath, true, isSrgbRequired(aiType, useSrgb));
                    if (pTex)
                    {
                        mTextureCache[s] = pTex;
                    }
                }

                assert(pTex != nullptr);
                BasicMaterial::MapType texSlot = getFalcorTexTypeFromAi(aiType, isObjFile);
                if (texSlot != BasicMaterial::MapType::Count)
                {
                    pMaterial->pTextures[texSlot] = pTex;
                }
                else
                {
                    logWarning("Texture '" + s + "' is not supported by the material system\n");
                }
            }
        }

        // Flush upload heap after every material so we don't accumulate a ton of memory usage when loading a model with a lot of textures
        gpDevice->flushAndSync();
    }

    Material::SharedPtr AssimpModelImporter::createMaterial(const aiMaterial* pAiMaterial, const std::string& folder, bool isObjFile, bool useSrgb)
    {
        aiString name;
        pAiMaterial->Get(AI_MATKEY_NAME, name);
        std::string nameStr = std::string(name.C_Str());
        std::transform(nameStr.begin(), nameStr.end(), nameStr.begin(), ::tolower);

        BasicMaterial basicMaterial;
        loadTextures(pAiMaterial, folder, &basicMaterial, isObjFile, useSrgb);

        // Opacity
        float opacity;
        if (pAiMaterial->Get(AI_MATKEY_OPACITY, opacity) == AI_SUCCESS)
        {
            basicMaterial.opacity = opacity;
        }

        // Bump scaling
        float bumpScaling;
        if (pAiMaterial->Get(AI_MATKEY_BUMPSCALING, bumpScaling) == AI_SUCCESS)
        {
            basicMaterial.bumpScale = bumpScaling;
        }

        // Shininess
        float shininess;
        if (pAiMaterial->Get(AI_MATKEY_SHININESS, shininess) == AI_SUCCESS)
        {
            basicMaterial.shininess = shininess;
        }

        // Refraction
        float refraction;
        if (pAiMaterial->Get(AI_MATKEY_REFRACTI, refraction) == AI_SUCCESS)
        {
            basicMaterial.IoR = refraction;
        }

        // Diffuse color
        aiColor3D color;
        if (pAiMaterial->Get(AI_MATKEY_COLOR_DIFFUSE, color) == AI_SUCCESS)
        {
            basicMaterial.diffuseColor = glm::vec3(color.r, color.g, color.b);
        }

        // Specular color
        if (pAiMaterial->Get(AI_MATKEY_COLOR_SPECULAR, color) == AI_SUCCESS)
        {
            basicMaterial.specularColor = glm::vec3(color.r, color.g, color.b);
        }

        // Emissive color
        if (pAiMaterial->Get(AI_MATKEY_COLOR_EMISSIVE, color) == AI_SUCCESS)
        {
            basicMaterial.emissiveColor = glm::vec3(color.r, color.g, color.b);
            if (isObjFile && luminance(basicMaterial.emissiveColor) > 0)
            {
                basicMaterial.pTextures[BasicMaterial::MapType::EmissiveMap] = basicMaterial.pTextures[BasicMaterial::MapType::DiffuseMap];
            }
        }

        // Transparent color
        if (pAiMaterial->Get(AI_MATKEY_COLOR_TRANSPARENT, color) == AI_SUCCESS)
        {
            basicMaterial.transparentColor = glm::vec3(color.r, color.g, color.b);
        }

        auto pMaterial = basicMaterial.convertToMaterial();

        // Double-Sided
        int isDoubleSided;
        if (pAiMaterial->Get(AI_MATKEY_TWOSIDED, isDoubleSided) == AI_SUCCESS)
        {
            pMaterial->setDoubleSided((isDoubleSided != 0));
        }

        return pMaterial;
    }

    bool verifyScene(const aiScene* pScene)
    {
        bool b = true;

        // No internal textures
        if (pScene->mTextures != 0)
        {
            logError("Model has internal textures");
            b = false;
        }
        return b;
    }

    AssimpModelImporter::AssimpModelImporter(Model& model, Model::LoadFlags flags) : mFlags(flags), mModel(model)
    {
    }

    bool AssimpModelImporter::createAllMaterials(const aiScene* pScene, const std::string& modelFolder, bool isObjFile, bool useSrgb)
    {
        for (uint32_t i = 0; i < pScene->mNumMaterials; i++)
        {
            const aiMaterial* pAiMaterial = pScene->mMaterials[i];
            auto pMaterial = createMaterial(pAiMaterial, modelFolder, isObjFile, useSrgb);
            if (pMaterial == nullptr)
            {
                logError("Can't allocate memory for material");
                return false;
            }
            auto pAdded = checkForExistingMaterial(pMaterial);
            if (pMaterial != pAdded)
            {
                // Material already exists
                pMaterial = pAdded;
            }
            mAiMaterialToFalcor[i] = pMaterial;
        }

        return true;
    }

    bool AssimpModelImporter::parseAiSceneNode(const aiNode* pCurrent, const aiScene* pScene, IdToMesh& aiToFalcorMesh)
    {
        if (pCurrent->mNumMeshes)
        {
            // Init the transformation
            aiMatrix4x4 transform = pCurrent->mTransformation;
            const aiNode* pParent = pCurrent->mParent;
            while (pParent)
            {
                transform *= pParent->mTransformation;
                pParent = pParent->mParent;
            }

            // Initialize the meshes
            for (uint32_t i = 0; i < pCurrent->mNumMeshes; i++)
            {
                uint32_t aiId = pCurrent->mMeshes[i];

                // New mesh
                if (aiToFalcorMesh.find(aiId) == aiToFalcorMesh.end())
                {
                    // Cache mesh
                    aiToFalcorMesh[aiId] = createMesh(pScene->mMeshes[aiId]);
                }

                mModel.addMeshInstance(aiToFalcorMesh[aiId], aiMatToGLM(transform));
            }
        }

        bool b = true;
        // visit the children
        for (uint32_t i = 0; i < pCurrent->mNumChildren; i++)
        {
            b |= parseAiSceneNode(pCurrent->mChildren[i], pScene, aiToFalcorMesh);
        }
        return b;
    }

    bool AssimpModelImporter::createDrawList(const aiScene* pScene)
    {
        createAnimationController(pScene);
        IdToMesh aiToFalcorMeshId;
        aiNode* pRoot = pScene->mRootNode;
        return parseAiSceneNode(pRoot, pScene, aiToFalcorMeshId);
    }

    bool AssimpModelImporter::initModel(const std::string& filename)
    {
        std::string fullpath;
        if (findFileInDataDirectories(filename, fullpath) == false)
        {
            logError(std::string("Can't find model file ") + filename, true);
            return nullptr;
        }

        uint32_t AssimpFlags = aiProcessPreset_TargetRealtime_MaxQuality |
            aiProcess_OptimizeGraph |
            aiProcess_FlipUVs |
            // aiProcess_FixInfacingNormals | // causes incorrect facing normals for crytek-sponza
            0;

        // aiProcessPreset_TargetRealtime_MaxQuality enabled some optimizations the user might not want
        if(is_set(mFlags, Model::LoadFlags::FindDegeneratePrimitives) == false)
        {
            AssimpFlags &= ~aiProcess_FindDegenerates;
        }

        // Avoid merging original meshes
        if(is_set(mFlags, Model::LoadFlags::DontMergeMeshes))
        {
            AssimpFlags &= ~aiProcess_OptimizeGraph;
        }

        // Never use Assimp's tangent gen code
        AssimpFlags &= ~(aiProcess_CalcTangentSpace);

        Assimp::Importer importer;
        const aiScene* pScene = importer.ReadFile(fullpath, AssimpFlags);

        if((pScene == nullptr) || (verifyScene(pScene) == false))
        {
            std::string str("Can't open model file '");
            str = str + std::string(filename) + "'\n" + importer.GetErrorString();
            logError(str, true);
            return false;
        }

        // Extract the folder name
        auto last = fullpath.find_last_of("/\\");
        std::string modelFolder = fullpath.substr(0, last);

        // Order of initialization matters, materials, bones and animations need to loaded before mesh initialization
        bool isObjFile = hasSuffix(filename, ".obj", false);
        bool useSrgbTextures = !is_set(mFlags, Model::LoadFlags::AssumeLinearSpaceTextures);
        if(createAllMaterials(pScene, modelFolder, isObjFile, useSrgbTextures) == false)
        {
            logError(std::string("Can't create materials for model ") + filename, true);
            return false;
        }

        if (createDrawList(pScene) == false)
        {
            logError(std::string("Can't create draw lists for model ") + filename, true);
            return false;
        }

        return true;
    }

    bool AssimpModelImporter::import(Model& model, const std::string& filename, Model::LoadFlags flags)
    {
        AssimpModelImporter loader(model, flags);
        return loader.initModel(filename);
    }

    uint32_t AssimpModelImporter::initBone(const aiNode* pCurNode, uint32_t parentID, uint32_t boneID)
    {
        assert(mBoneNameToIdMap.find(pCurNode->mName.C_Str()) != mBoneNameToIdMap.end());
        assert(pCurNode->mNumMeshes == 0);
        mBoneNameToIdMap[pCurNode->mName.C_Str()] = boneID;

        assert(boneID < mBones.size());
        Bone& bone = mBones[boneID];
        bone.name = pCurNode->mName.C_Str();
        bone.parentID = parentID;
        bone.boneID = boneID;
        bone.localTransform = aiMatToGLM(pCurNode->mTransformation);
        bone.originalLocalTransform = bone.localTransform;
        bone.globalTransform = bone.localTransform;

        if (parentID != AnimationController::kInvalidBoneID)
        {
            bone.globalTransform *= mBones[parentID].globalTransform;
        }

        boneID++;

        for (uint32_t i = 0; i < pCurNode->mNumChildren; i++)
        {
            // Check that the child is actually used
            if (mBoneNameToIdMap.find(pCurNode->mChildren[i]->mName.C_Str()) != mBoneNameToIdMap.end())
            {
                boneID = initBone(pCurNode->mChildren[i], bone.boneID, boneID);
            }
        }
        return boneID;
    }

    void AssimpModelImporter::initializeBonesOffsetMatrices(const aiScene* pScene)
    {
        for (uint32_t meshID = 0; meshID < pScene->mNumMeshes; meshID++)
        {
            const aiMesh* pAiMesh = pScene->mMeshes[meshID];

            for (uint32_t boneID = 0; boneID < pAiMesh->mNumBones; boneID++)
            {
                const aiBone* pAiBone = pAiMesh->mBones[boneID];
                auto boneIt = mBoneNameToIdMap.find(pAiBone->mName.C_Str());
                assert(boneIt != mBoneNameToIdMap.end());
                Bone& bone = mBones[boneIt->second];
                bone.offset = aiMatToGLM(pAiBone->mOffsetMatrix);
            }
        }
    }

    void AssimpModelImporter::initializeBones(const aiScene* pScene)
    {
        // Go over all the meshes, and find the bones that are being used
        for (uint32_t i = 0; i < pScene->mNumMeshes; i++)
        {
            const aiMesh* pMesh = pScene->mMeshes[i];
            if (pMesh->HasBones())
            {
                for (uint32_t j = 0; j < pMesh->mNumBones; j++)
                {
                    mBoneNameToIdMap[pMesh->mBones[j]->mName.C_Str()] = AnimationController::kInvalidBoneID;
                }
            }
        }

        if (mBoneNameToIdMap.size() != 0)
        {
            // For every bone used, all its ancestors are bones too. Mark them
            auto it = mBoneNameToIdMap.begin();
            while (it != mBoneNameToIdMap.end())
            {
                aiNode* pCurNode = pScene->mRootNode->FindNode(it->first.c_str());
                while (pCurNode)
                {
                    mBoneNameToIdMap[pCurNode->mName.C_Str()] = AnimationController::kInvalidBoneID;
                    pCurNode = pCurNode->mParent;
                }
                it++;
            }

            // Now create the hierarchy
            mBones.resize(mBoneNameToIdMap.size());
            uint32_t bonesCount = initBone(pScene->mRootNode, AnimationController::kInvalidBoneID, 0);
            assert(mBoneNameToIdMap.size() == bonesCount);

            initializeBonesOffsetMatrices(pScene);
        }
    }

    void AssimpModelImporter::createAnimationController(const aiScene* pScene)
    {
        initializeBones(pScene);

        if (pScene->HasAnimations())
        {
            auto pAnimCtrl = AnimationController::create(mBones);

            for (uint32_t i = 0; i < pScene->mNumAnimations; i++)
            {
                Animation::UniquePtr pAnimation = createAnimation(pScene->mAnimations[i]);
                pAnimCtrl->addAnimation(std::move(pAnimation));
            }

            mModel.setAnimationController(std::move(pAnimCtrl));
        }
    }


    Animation::UniquePtr AssimpModelImporter::createAnimation(const aiAnimation* pAiAnim)
    {
        assert(pAiAnim->mNumMeshChannels == 0);
        float duration = float(pAiAnim->mDuration);
        float ticksPerSecond = pAiAnim->mTicksPerSecond ? float(pAiAnim->mTicksPerSecond) : 25;

        std::vector<Animation::AnimationSet> animationSets;
        animationSets.resize(pAiAnim->mNumChannels);

        for (uint32_t i = 0; i < pAiAnim->mNumChannels; i++)
        {
            const aiNodeAnim* pAiNode = pAiAnim->mChannels[i];
            // If the bone is not used, skip it
            const auto& idIt = mBoneNameToIdMap.find(pAiNode->mNodeName.C_Str());
            if (idIt == mBoneNameToIdMap.end()) continue;;

            animationSets[i].boneID = idIt->second;

            for (uint32_t j = 0; j < pAiNode->mNumPositionKeys; j++)
            {
                const aiVectorKey& key = pAiNode->mPositionKeys[j];
                Animation::AnimationKey<glm::vec3> position;
                position.value = glm::vec3(key.mValue.x, key.mValue.y, key.mValue.z);
                position.time = float(key.mTime);
                animationSets[i].translation.keys.push_back(position);
            }

            for (uint32_t j = 0; j < pAiNode->mNumScalingKeys; j++)
            {
                const aiVectorKey& key = pAiNode->mScalingKeys[j];
                Animation::AnimationKey<glm::vec3> scale;
                scale.value = glm::vec3(key.mValue.x, key.mValue.y, key.mValue.z);
                scale.time = float(key.mTime);
                animationSets[i].scaling.keys.push_back(scale);
            }

            for (uint32_t j = 0; j < pAiNode->mNumRotationKeys; j++)
            {
                const aiQuatKey& key = pAiNode->mRotationKeys[j];
                Animation::AnimationKey<glm::quat> rotation;
                rotation.value = glm::quat(key.mValue.w, key.mValue.x, key.mValue.y, key.mValue.z);
                rotation.time = float(key.mTime);
                animationSets[i].rotation.keys.push_back(rotation);
            }
        }

        return Animation::create(std::string(pAiAnim->mName.C_Str()), animationSets, duration, ticksPerSecond);
    }

    BoundingBox createMeshBbox(const aiMesh* pAiMesh)
    {
        glm::vec3 boxMin, boxMax;
        for (uint32_t vertexID = 0; vertexID < pAiMesh->mNumVertices; vertexID++)
        {
            glm::vec3 xyz(pAiMesh->mVertices[vertexID].x, pAiMesh->mVertices[vertexID].y, pAiMesh->mVertices[vertexID].z);
            boxMin = glm::min(boxMin, xyz);
            boxMax = glm::max(boxMax, xyz);
        }

        return BoundingBox::fromMinMax(boxMin, boxMax);
    }

    Mesh::SharedPtr AssimpModelImporter::createMesh(const aiMesh* pAiMesh)
    {
        uint32_t vertexCount = pAiMesh->mNumVertices;
        uint32_t indexCount = pAiMesh->mNumFaces * pAiMesh->mFaces[0].mNumIndices;
        auto pIB = createIndexBuffer(pAiMesh);
        BoundingBox boundingBox = createMeshBbox(pAiMesh);

        const bool generateTangentSpace = (pAiMesh->HasTangentsAndBitangents() == false) && (is_set(mFlags, Model::LoadFlags::DontGenerateTangentSpace) == false);
        if (generateTangentSpace)
        {
            genTangentSpace(pAiMesh);
        }

        VertexLayout::SharedPtr pLayout = createVertexLayout(pAiMesh);
        if (pLayout == nullptr)
        {
            assert(0);
            return nullptr;
        }

        std::vector<Buffer::SharedPtr> pVBs(pLayout->getBufferCount());

        // Initialize the bones data
        VertexWeightsVec weights;
        VertexIdsVec ids;
        if (pAiMesh->HasBones())
        {
            loadBones(pAiMesh, weights, ids, vertexCount, mBoneNameToIdMap);
        }

        // Create corresponding vertex buffers
        for (uint32_t i = 0; i < pLayout->getBufferCount(); i++)
        {
            const VertexBufferLayout* pVbLayout = pLayout->getBufferLayout(i).get();
            pVBs[i] = createVertexBuffer(pAiMesh, pVbLayout, (uint8_t*)ids.data(), weights.data());
        }

        Vao::Topology topology;
        switch (pAiMesh->mFaces[0].mNumIndices)
        {
        case 1:
            topology = Vao::Topology::PointList;
            break;
        case 2:
            topology = Vao::Topology::LineList;
            break;
        case 3:
            topology = Vao::Topology::TriangleList;
            break;
        default:
            logError(std::string("Error when creating mesh. Unknown topology with " + std::to_string(pAiMesh->mFaces[0].mNumIndices) + " indices."));
            assert(0);
        }

        auto pMaterial = mAiMaterialToFalcor[pAiMesh->mMaterialIndex];
        assert(pMaterial);

        Mesh::SharedPtr pMesh = Mesh::create(pVBs, vertexCount, pIB, indexCount, pLayout, topology, pMaterial, boundingBox, pAiMesh->HasBones());

        if (generateTangentSpace)
        {
            aiMesh* pM = const_cast<aiMesh*>(pAiMesh);
            safe_delete_array(pM->mBitangents);
        }

        return pMesh;
    }

    Buffer::SharedPtr AssimpModelImporter::createIndexBuffer(const aiMesh* pAiMesh)
    {
        std::vector<uint32_t> indices = createIndexBufferData(pAiMesh);
        const uint32_t size = (uint32_t)(sizeof(uint32_t) * indices.size());
        Buffer::BindFlags bindFlags = Buffer::BindFlags::Index;
        if (is_set(mFlags, Model::LoadFlags::BuffersAsShaderResource))
        {
            bindFlags |= Buffer::BindFlags::ShaderResource;
        }
        return Buffer::create(size, bindFlags, Buffer::CpuAccess::None, indices.data());;
    }


    bool isElementUsed(const aiMesh* pAiMesh, uint32_t location)
    {
        switch (location)
        {
        case VERTEX_POSITION_LOC:
            return true;
        case VERTEX_NORMAL_LOC:
            return pAiMesh->HasNormals();
        case VERTEX_BITANGENT_LOC:
            return (pAiMesh->mBitangents != nullptr); // ASSIMP doesn't have a function that checks only for bitangents
        case VERTEX_BONE_WEIGHT_LOC:
        case VERTEX_BONE_ID_LOC:
            return pAiMesh->HasBones();
        case VERTEX_DIFFUSE_COLOR_LOC:
            return pAiMesh->HasVertexColors(0);
        case VERTEX_TEXCOORD_LOC:
            return pAiMesh->HasTextureCoords(0);
        case VERTEX_LIGHTMAP_UV_LOC:
            return pAiMesh->HasTextureCoords(1);
        default:
            should_not_get_here();
            return false;
        }
    }

    VertexLayout::SharedPtr AssimpModelImporter::createVertexLayout(const aiMesh* pAiMesh)
    {
        static const uint32_t kMaxSupportedUVs = 2;
        // Must have position!!!
        if (pAiMesh->HasPositions() == false)
        {
            logError("AssimpModelImporter: Loaded mesh with no positions!");
            return nullptr;
        }

        if (pAiMesh->GetNumUVChannels() > kMaxSupportedUVs)
        {
            logWarning("AssimpModelImporter: Loaded mesh with more then " + std::to_string(kMaxSupportedUVs) + " UV channels. Ignoring extra UVs");
        }

        for (uint32_t i = 0; i < min(pAiMesh->GetNumUVChannels(), kMaxSupportedUVs); i++)
        {
            if (pAiMesh->HasTextureCoords(i) == false)
            {
                logError("AssimpModelImporter: Unsupported texture coordinate set used in model.");
                return nullptr;
            }
        }

        VertexLayout::SharedPtr pLayout = VertexLayout::create();

        uint32_t bufferCount = 0;
        for (uint32_t location = 0; location < VERTEX_LOCATION_COUNT; ++location)
        {
            if (isElementUsed(pAiMesh, location))
            {
                VertexBufferLayout::SharedPtr pVbLayout = VertexBufferLayout::create();
                pVbLayout->addElement(kLayoutData[location].name, 0, kLayoutData[location].format, 1, location);
                pLayout->addBufferLayout(bufferCount, pVbLayout);
                bufferCount++;
            }
        }

        return pLayout;
    }

    Buffer::SharedPtr AssimpModelImporter::createVertexBuffer(const aiMesh* pAiMesh, const VertexBufferLayout* pLayout, const uint8_t* pBoneIds, const vec4* pBoneWeights)
    {
        const uint32_t vertexStride = pLayout->getStride();
        std::vector<uint8_t> initData(vertexStride * pAiMesh->mNumVertices, 0);

        for (uint32_t vertexID = 0; vertexID < pAiMesh->mNumVertices; vertexID++)
        {
            uint8_t* pVertex = &initData[vertexStride * vertexID];

            for (uint32_t elementID = 0; elementID < pLayout->getElementCount(); elementID++)
            {
                uint32_t offset = pLayout->getElementOffset(elementID);
                uint32_t location = pLayout->getElementShaderLocation(elementID);
                uint8_t* pDst = pVertex + offset;

                uint8_t* pSrc = nullptr;
                uint32_t size = 0;
                switch (location)
                {
                case VERTEX_POSITION_LOC:
                    pSrc = (uint8_t*)(&pAiMesh->mVertices[vertexID]);
                    size = sizeof(pAiMesh->mVertices[0]);
                    break;
                case VERTEX_NORMAL_LOC:
                    pSrc = (uint8_t*)(&pAiMesh->mNormals[vertexID]);
                    size = sizeof(pAiMesh->mNormals[0]);
                    break;
                case VERTEX_BITANGENT_LOC:
                    pSrc = (uint8_t*)(&pAiMesh->mBitangents[vertexID]);
                    size = sizeof(pAiMesh->mBitangents[0]);
                    break;
                case VERTEX_DIFFUSE_COLOR_LOC:
                    pSrc = (uint8_t*)(&pAiMesh->mColors[0][vertexID]);
                    size = sizeof(pAiMesh->mColors[0][0]);
                    break;
                case VERTEX_TEXCOORD_LOC:
                    pSrc = (uint8_t*)(&pAiMesh->mTextureCoords[0][vertexID]);
                    size = sizeof(pAiMesh->mTextureCoords[0][vertexID]);
                    break;
                case VERTEX_LIGHTMAP_UV_LOC:
                    pSrc = (uint8_t*)(&pAiMesh->mTextureCoords[1][vertexID]);
                    size = sizeof(pAiMesh->mTextureCoords[1][vertexID]);
                    break;
                case VERTEX_BONE_WEIGHT_LOC:
                    pSrc = (uint8_t*)(&pBoneWeights[vertexID]);
                    size = sizeof(pBoneWeights[vertexID]);
                    break;
                case VERTEX_BONE_ID_LOC:
                    pSrc = (uint8_t*)(&pBoneIds[vertexID * 4]);
                    size = sizeof(uint8_t) * 4;
                    break;
                default:
                    should_not_get_here();
                    continue;
                }

                memcpy(pDst, pSrc, size);
            }
        }
        Buffer::BindFlags bindFlags = Buffer::BindFlags::Vertex;
        if (is_set(mFlags, Model::LoadFlags::BuffersAsShaderResource))
        {
            bindFlags |= Buffer::BindFlags::ShaderResource;
        }

        return Buffer::create(vertexStride * pAiMesh->mNumVertices, bindFlags, Buffer::CpuAccess::None, initData.data());;
    }
}