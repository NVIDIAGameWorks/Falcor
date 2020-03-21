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
 **************************************************************************/
#include "stdafx.h"
#include "assimp/Importer.hpp"
#include "assimp/postprocess.h"
#include "assimp/scene.h"
#include "assimp/pbrmaterial.h"
#include "AssimpImporter.h"
#include "Utils/StringUtils.h"
#include "Core/API/Device.h"
#include "Scene/SceneBuilder.h"

namespace Falcor
{
    namespace
    {
        using BoneMeshMap = std::map<std::string, std::vector<uint32_t>>;
        using MeshInstanceList = std::vector<std::vector<const aiNode*>>;

        /** Converts specular power to roughness. Note there is no "the conversion".
            Reference: http://simonstechblog.blogspot.com/2011/12/microfacet-brdf.html
            \param specPower specular power of an obsolete Phong BSDF
        */
        float convertSpecPowerToRoughness(float specPower)
        {
            return clamp(sqrt(2.0f / (specPower + 2.0f)), 0.f, 1.f);
        }

        enum class ImportMode {
            Default,
            OBJ,
            GLTF2,
        };

        mat4 aiCast(const aiMatrix4x4& aiMat)
        {
            mat4 glmMat;
            glmMat[0][0] = aiMat.a1; glmMat[0][1] = aiMat.a2; glmMat[0][2] = aiMat.a3; glmMat[0][3] = aiMat.a4;
            glmMat[1][0] = aiMat.b1; glmMat[1][1] = aiMat.b2; glmMat[1][2] = aiMat.b3; glmMat[1][3] = aiMat.b4;
            glmMat[2][0] = aiMat.c1; glmMat[2][1] = aiMat.c2; glmMat[2][2] = aiMat.c3; glmMat[2][3] = aiMat.c4;
            glmMat[3][0] = aiMat.d1; glmMat[3][1] = aiMat.d2; glmMat[3][2] = aiMat.d3; glmMat[3][3] = aiMat.d4;

            return transpose(glmMat);
        }

        vec3 aiCast(const aiColor3D& ai)
        {
            return vec3(ai.r, ai.g, ai.b);
        }

        vec3 aiCast(const aiVector3D& val)
        {
            return vec3(val.x, val.y, val.z);
        }

        quat aiCast(const aiQuaternion& q)
        {
            return quat(q.w, q.x, q.y, q.z);
        }

        /** Texture types used in Falcor materials.
        */
        enum class TextureType
        {
            BaseColor,
            Specular,
            Emissive,
            Normal,
            Occlusion,
        };

        /** Mapping from ASSIMP to Falcor texture type.
        */
        struct TextureMapping
        {
            aiTextureType aiType;
            unsigned int aiIndex;
            TextureType targetType;
        };

        /** Mapping tables for different import modes.
        */
        static const std::vector<TextureMapping> kTextureMappings[3] =
        {
            // Default mappings,
            {
                { aiTextureType_DIFFUSE, 0, TextureType::BaseColor },
                { aiTextureType_SPECULAR, 0, TextureType::Specular },
                { aiTextureType_EMISSIVE, 0, TextureType::Emissive },
                { aiTextureType_NORMALS, 0, TextureType::Normal },
                { aiTextureType_AMBIENT, 0, TextureType::Occlusion },
            },
            // OBJ mappings,
            {
                { aiTextureType_DIFFUSE, 0, TextureType::BaseColor },
                { aiTextureType_SPECULAR, 0, TextureType::Specular },
                { aiTextureType_EMISSIVE, 0, TextureType::Emissive },
                { aiTextureType_AMBIENT, 0, TextureType::Occlusion },
                // OBJ does not offer a normal map, thus we use the bump map instead.
                { aiTextureType_HEIGHT, 0, TextureType::Normal },
                { aiTextureType_DISPLACEMENT, 0, TextureType::Normal },
            },
            // GLTF2 mapings,
            {
                { aiTextureType_DIFFUSE, 0, TextureType::BaseColor },
                { aiTextureType_EMISSIVE, 0, TextureType::Emissive },
                { aiTextureType_NORMALS, 0, TextureType::Normal },
                { aiTextureType_AMBIENT, 0, TextureType::Occlusion },
                // GLTF2 exposes metallic roughness texture.
                { AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_METALLICROUGHNESS_TEXTURE, TextureType::Specular },
            }
        };

        void setTexture(TextureType type, Material* pMaterial, Texture::SharedPtr pTexture)
        {
            switch (type)
            {
            case TextureType::BaseColor:
                pMaterial->setBaseColorTexture(pTexture);
                break;
            case TextureType::Specular:
                pMaterial->setSpecularTexture(pTexture);
                break;
            case TextureType::Emissive:
                pMaterial->setEmissiveTexture(pTexture);
                break;
            case TextureType::Normal:
                pMaterial->setNormalMap(pTexture);
                break;
            case TextureType::Occlusion:
                pMaterial->setOcclusionMap(pTexture);
                break;
            default:
                should_not_get_here();
            }
        }

        bool isSrgbRequired(TextureType type, uint32_t shadingModel)
        {
            switch (type)
            {
            case TextureType::Specular:
                assert(shadingModel == ShadingModelMetalRough || shadingModel == ShadingModelSpecGloss);
                return (shadingModel == ShadingModelSpecGloss);
            case TextureType::BaseColor:
            case TextureType::Emissive:
            case TextureType::Occlusion:
                return true;
            case TextureType::Normal:
                return false;
            default:
                should_not_get_here();
                return false;
            }
        }

        class ImporterData
        {
        public:
            ImporterData(const aiScene* pAiScene, SceneBuilder& sceneBuilder, const SceneBuilder::InstanceMatrices& modelInstances_) : pScene(pAiScene), modelInstances(modelInstances_), builder(sceneBuilder) {}
            const aiScene* pScene;

            SceneBuilder& builder;
            std::map<uint32_t, Material::SharedPtr> materialMap;
            std::map<uint32_t, size_t> meshMap;
            std::map<const std::string, Texture::SharedPtr> textureCache;
            const SceneBuilder::InstanceMatrices& modelInstances;
            std::map<std::string, mat4> localToBindPoseMatrices;

            size_t getFalcorNode(const aiNode* pNode) const
            {
                return mAiToFalcorNode.at(pNode);
            }

            size_t getFalcorNode(const std::string& aiNodeName, uint32_t index) const
            {
                try
                {
                    return getFalcorNode(mAiNodes.at(aiNodeName)[index]);
                }
                catch (const std::exception&)
                {
                    return SceneBuilder::kInvalidNode;
                }
            }

            uint32_t getNodeInstanceCount(const std::string& nodeName) const
            {
                return (uint32_t)mAiNodes.at(nodeName).size();
            }

            void addAiNode(const aiNode* pNode, size_t falcorID)
            {
                assert(mAiToFalcorNode.find(pNode) == mAiToFalcorNode.end());
                mAiToFalcorNode[pNode] = falcorID;

                if (mAiNodes.find(pNode->mName.C_Str()) == mAiNodes.end())
                {
                    mAiNodes[pNode->mName.C_Str()] = {};
                }
                mAiNodes[pNode->mName.C_Str()].push_back(pNode);
            }

        private:
            std::map<const aiNode*, size_t> mAiToFalcorNode;
            std::map<const std::string, std::vector<const aiNode*>> mAiNodes;

        };

        using KeyframeList = std::list<Animation::Keyframe>;

        struct AnimationChannelData
        {
            uint32_t posIndex = 0;
            uint32_t rotIndex = 0;
            uint32_t scaleIndex = 0;
        };

        template<typename AiType, typename FalcorType>
        bool parseAnimationChannel(const AiType* pKeys, uint32_t count, double time, uint32_t& currentIndex, FalcorType& falcor)
        {
            if (currentIndex >= count) return true;

            if (pKeys[currentIndex].mTime == time)
            {
                falcor = aiCast(pKeys[currentIndex].mValue);
                currentIndex++;
            }

            return currentIndex >= count;
        }

        void resetNegativeKeyframeTimes(aiNodeAnim* pAiNode)
        {
            auto resetTime = [](auto keys, uint32_t count)
            {
                if (count > 1) assert(keys[1].mTime >= 0);
                if (keys[0].mTime < 0) keys[0].mTime = 0;
            };
            resetTime(pAiNode->mPositionKeys, pAiNode->mNumPositionKeys);
            resetTime(pAiNode->mRotationKeys, pAiNode->mNumRotationKeys);
            resetTime(pAiNode->mScalingKeys, pAiNode->mNumScalingKeys);
        }

        Animation::SharedPtr createAnimation(ImporterData& data, const aiAnimation* pAiAnim)
        {
            assert(pAiAnim->mNumMeshChannels == 0);
            double duration = pAiAnim->mDuration;
            double ticksPerSecond = pAiAnim->mTicksPerSecond ? pAiAnim->mTicksPerSecond : 25;
            double durationInSeconds = duration / ticksPerSecond;

            Animation::SharedPtr pAnimation = Animation::create(pAiAnim->mName.C_Str(), durationInSeconds);

            for (uint32_t i = 0; i < pAiAnim->mNumChannels; i++)
            {
                aiNodeAnim* pAiNode = pAiAnim->mChannels[i];
                resetNegativeKeyframeTimes(pAiNode);

                std::vector<size_t> channels;
                for (uint32_t i = 0; i < data.getNodeInstanceCount(pAiNode->mNodeName.C_Str()); i++)
                {
                    channels.push_back(pAnimation->addChannel(data.getFalcorNode(pAiNode->mNodeName.C_Str(), i)));
                }

                uint32_t pos = 0, rot = 0, scale = 0;
                Animation::Keyframe keyframe;
                bool done = false;

                auto nextKeyTime = [&]()
                {
                    double time = -std::numeric_limits<double>::max();
                    if (pos < pAiNode->mNumPositionKeys) time = max(time, pAiNode->mPositionKeys[pos].mTime);
                    if (rot < pAiNode->mNumRotationKeys) time = max(time, pAiNode->mRotationKeys[rot].mTime);
                    if (scale < pAiNode->mNumScalingKeys) time = max(time, pAiNode->mScalingKeys[scale].mTime);
                    assert(time != -std::numeric_limits<double>::max());
                    return time;
                };

                while (!done)
                {
                    double time = nextKeyTime();
                    assert(time == 0 || (time / ticksPerSecond) > keyframe.time);
                    keyframe.time = time / ticksPerSecond;

                    // Note the order of the logical-and, we don't want to short-circuit the function calls
                    done = parseAnimationChannel(pAiNode->mPositionKeys, pAiNode->mNumPositionKeys, time, pos, keyframe.translation);
                    done = parseAnimationChannel(pAiNode->mRotationKeys, pAiNode->mNumRotationKeys, time, rot, keyframe.rotation) && done;
                    done = parseAnimationChannel(pAiNode->mScalingKeys, pAiNode->mNumScalingKeys, time, scale, keyframe.scaling) && done;
                    for(auto c : channels) pAnimation->addKeyframe(c, keyframe);
                }
            }

            return pAnimation;
        }

        bool createCamera(ImporterData& data, ImportMode importMode)
        {
            if (data.pScene->mNumCameras == 0) return true;
            if (data.pScene->mNumCameras > 1)
            {
                logWarning("Model file contains more then a single camera. Falcor only supports one camera per scene, we will use the first camera");
            }

            if (data.builder.hasCamera())
            {
                logWarning("Found cameras in model file, but the scene already contains a camera. Ignoring the new camera");
                return true;
            }

            const aiCamera* pAiCamera = data.pScene->mCameras[0];
            pAiCamera = pAiCamera;
            Camera::SharedPtr pCamera = Camera::create();
            pCamera->setPosition(aiCast(pAiCamera->mPosition));
            pCamera->setUpVector(aiCast(pAiCamera->mUp));
            pCamera->setTarget(aiCast(pAiCamera->mLookAt) + aiCast(pAiCamera->mPosition));
            // Some importers don't provide the aspect ratio, use default for that case.
            float aspectRatio = pAiCamera->mAspect != 0.f ? pAiCamera->mAspect : pCamera->getAspectRatio();
            // Load focal length only when using GLTF2, use fixed 35mm for backwards compatibility with FBX files.
            float focalLength = importMode == ImportMode::GLTF2 ? fovYToFocalLength(pAiCamera->mHorizontalFOV * 2 / aspectRatio, pCamera->getFrameHeight()) : 35.f;
            pCamera->setFocalLength(focalLength);
            pCamera->setAspectRatio(aspectRatio);
            pCamera->setDepthRange(pAiCamera->mClipPlaneNear, pAiCamera->mClipPlaneFar);

            size_t nodeID = data.getFalcorNode(pAiCamera->mName.C_Str(), 0);

            if(nodeID != SceneBuilder::kInvalidNode)
            {
                SceneBuilder::Node n;
                n.name = "Camera.BaseMatrix";
                n.parent = nodeID;
                n.transform = pCamera->getViewMatrix();
                // GLTF2 has the view direction reversed.
                if (importMode == ImportMode::GLTF2) n.transform[2] = -n.transform[2];
                nodeID = data.builder.addNode(n);
            }

            // Find if the camera is affected by a node
            data.builder.setCamera(pCamera, nodeID);
            return true;
        }

        bool addLightCommon(Light::ConstSharedPtrRef pLight, const mat4& baseMatrix, ImporterData& data, const aiLight* pAiLight)
        {
            pLight->setName(pAiLight->mName.C_Str());
            assert(pAiLight->mColorDiffuse == pAiLight->mColorSpecular);
            pLight->setIntensity(aiCast(pAiLight->mColorSpecular));

            // Find if the light is affected by a node
            size_t nodeID = data.getFalcorNode(pAiLight->mName.C_Str(), 0);
            if (nodeID != SceneBuilder::kInvalidNode)
            {
                SceneBuilder::Node n;
                n.name = pLight->getName() + ".BaseMatrix";
                n.parent = nodeID;
                n.transform = baseMatrix;
                nodeID = data.builder.addNode(n);
            }
            data.builder.addLight(pLight, nodeID);

            return true;
        }

        bool createDirLight(ImporterData& data, const aiLight* pAiLight)
        {
            DirectionalLight::SharedPtr pLight = DirectionalLight::create();
            vec3 direction = normalize(aiCast(pAiLight->mDirection));
            pLight->setWorldDirection(direction);
            mat4 base;
            base[2] = vec4(direction, 0);
            return addLightCommon(pLight, base, data, pAiLight);
        }

        bool createPointLight(ImporterData& data, const aiLight* pAiLight)
        {
            PointLight::SharedPtr pLight = PointLight::create();
            vec3 position = aiCast(pAiLight->mPosition);
            vec3 lookAt = normalize(aiCast(pAiLight->mDirection));
            vec3 up = normalize(aiCast(pAiLight->mUp));
            pLight->setWorldPosition(position);
            pLight->setWorldDirection(lookAt);
            pLight->setOpeningAngle(pAiLight->mAngleOuterCone);
            pLight->setPenumbraAngle(pAiLight->mAngleOuterCone - pAiLight->mAngleInnerCone);

            vec3 right = cross(up, lookAt);
            mat4 base;
            base[0] = vec4(right, 0);
            base[1] = vec4(up, 0);
            base[2] = vec4(lookAt, 0);
            base[3] = vec4(position, 1);

            return addLightCommon(pLight, base, data, pAiLight);
        }

        bool createLights(ImporterData& data)
        {
            for (uint32_t i = 0; i < data.pScene->mNumLights; i++)
            {
                const aiLight* pAiLight = data.pScene->mLights[i];
                switch (pAiLight->mType)
                {
                case aiLightSource_DIRECTIONAL:
                    if (!createDirLight(data, pAiLight)) return false;
                    break;
                case aiLightSource_POINT:
                case aiLightSource_SPOT:
                    if (!createPointLight(data, pAiLight)) return false;
                    break;
                default:
                    logWarning("Unsupported ASSIMP light type " + to_string(pAiLight->mType));
                    continue;
                }
            }

            return true;
        }

        bool createAnimations(ImporterData& data)
        {
            for (uint32_t i = 0; i < data.pScene->mNumAnimations; i++)
            {
                Animation::SharedPtr pAnimation = createAnimation(data, data.pScene->mAnimations[i]);
                data.builder.addAnimation(0, pAnimation);
            }
            return true;
        }

        std::vector<vec2> createTexCrdList(const aiVector3D* pAiTexCrd, uint32_t count)
        {
            std::vector<vec2> v2(count);
            for (uint32_t i = 0; i < count; i++)
            {
                assert(pAiTexCrd[i].z == 0);
                v2[i] = vec2(pAiTexCrd[i].x, pAiTexCrd[i].y);
            }
            return v2;
        }

        std::vector<uint32_t> createIndexList(const aiMesh* pAiMesh)
        {
            const uint32_t perFaceIndexCount = pAiMesh->mFaces[0].mNumIndices;
            const uint32_t indexCount = pAiMesh->mNumFaces * perFaceIndexCount;

            std::vector<uint32_t> indices(indexCount);

            for (uint32_t i = 0; i < pAiMesh->mNumFaces; i++)
            {
                assert(pAiMesh->mFaces[i].mNumIndices == perFaceIndexCount); // Mesh contains mixed primitive types, can be solved using aiProcess_SortByPType
                for (uint32_t j = 0; j < perFaceIndexCount; j++)
                {
                    indices[i * perFaceIndexCount + j] = (uint32_t)(pAiMesh->mFaces[i].mIndices[j]);
                }
            }
            return indices;
        }

        void loadBones(const aiMesh* pAiMesh, const ImporterData& data, std::vector<vec4>& weights, std::vector<uvec4>& ids)
        {
            const uint32_t vertexCount = pAiMesh->mNumVertices;
            weights.resize(vertexCount);
            ids.resize(vertexCount);

            for (uint32_t bone = 0; bone < pAiMesh->mNumBones; bone++)
            {
                const aiBone* pAiBone = pAiMesh->mBones[bone];
                assert(data.getNodeInstanceCount(pAiBone->mName.C_Str()) == 1);
                size_t aiBoneID = data.getFalcorNode(pAiBone->mName.C_Str(), 0);

                // The way Assimp works, the weights holds the IDs of the vertices it affects.
                // We loop over all the weights, initializing the vertices data along the way
                for (uint32_t weightID = 0; weightID < pAiBone->mNumWeights; weightID++)
                {
                    // Get the vertex the current weight affects
                    const aiVertexWeight& aiWeight = pAiBone->mWeights[weightID];

                    // Get the address of the Bone ID and weight for the current vertex
                    uvec4& vertexIds = ids[aiWeight.mVertexId];
                    vec4& vertexWeights = weights[aiWeight.mVertexId];

                    // Find the next unused slot in the bone array of the vertex, and initialize it with the current value
                    bool emptySlotFound = false;
                    for (uint32_t j = 0; j < Scene::kMaxBonesPerVertex; j++)
                    {
                        if (vertexWeights[j] == 0)
                        {
                            vertexIds[j] = (uint32_t)aiBoneID;
                            vertexWeights[j] = aiWeight.mWeight;
                            emptySlotFound = true;
                            break;
                        }
                    }

                    if (emptySlotFound == false) logError("One of the vertices has too many bones attached to it. If you'll continue, this bone will be ignored and the animation might not look correct");
                }
            }

            // Now we need to normalize the weights for each vertex, since in some models the sum is larger than 1
            for (uint32_t i = 0; i < vertexCount; i++)
            {
                vec4& w = weights[i];
                float f = 0;
                for (int j = 0; j < Scene::kMaxBonesPerVertex; j++) f += w[j];
                w /= f;
            }
        }

        bool createMeshes(ImporterData& data)
        {
            const aiScene* pScene = data.pScene;
            for (uint32_t i = 0; i < pScene->mNumMeshes; i++)
            {
                SceneBuilder::Mesh mesh;
                const aiMesh* pAiMesh = pScene->mMeshes[i];

                // Indices
                const auto indexList = createIndexList(pAiMesh);
                mesh.indexCount = (uint32_t)indexList.size();
                mesh.pIndices = indexList.data();

                // Vertices
                assert(pAiMesh->mVertices);
                mesh.vertexCount = pAiMesh->mNumVertices;
                static_assert(sizeof(pAiMesh->mVertices[0]) == sizeof(mesh.pPositions[0]));
                static_assert(sizeof(pAiMesh->mNormals[0]) == sizeof(mesh.pNormals[0]));
                static_assert(sizeof(pAiMesh->mBitangents[0]) == sizeof(mesh.pBitangents[0]));
                mesh.pPositions = (vec3*)pAiMesh->mVertices;
                mesh.pNormals = (vec3*)pAiMesh->mNormals;
                mesh.pBitangents = (vec3*)pAiMesh->mBitangents;
                const auto& texCrd = pAiMesh->HasTextureCoords(0) ? createTexCrdList(pAiMesh->mTextureCoords[0], pAiMesh->mNumVertices) : std::vector<vec2>();
                mesh.pTexCrd = texCrd.size() ? texCrd.data() : nullptr;

                std::vector<uvec4> boneIds;
                std::vector<vec4> boneWeights;

                if (pAiMesh->HasBones())
                {
                    loadBones(pAiMesh, data, boneWeights, boneIds);
                    mesh.pBoneIDs = boneIds.data();
                    mesh.pBoneWeights = boneWeights.data();
                }

                switch (pAiMesh->mFaces[0].mNumIndices)
                {
                case 1: mesh.topology = Vao::Topology::PointList; break;
                case 2: mesh.topology = Vao::Topology::LineList; break;
                case 3: mesh.topology = Vao::Topology::TriangleList; break;
                default:
                    logError("Error when creating mesh. Unknown topology with " + std::to_string(pAiMesh->mFaces[0].mNumIndices) + " indices.");
                    should_not_get_here();
                }

                mesh.pMaterial = data.materialMap.at(pAiMesh->mMaterialIndex);
                assert(mesh.pMaterial);
                size_t meshID = data.builder.addMesh(mesh);
                if (meshID == SceneBuilder::kInvalidNode) return false;
                data.meshMap[i] = meshID;
            }

            return true;
        }

        bool isBone(ImporterData& data, const std::string& name)
        {
            return data.localToBindPoseMatrices.find(name) != data.localToBindPoseMatrices.end();
        }

        std::string getNodeType(ImporterData& data, const aiNode* pNode)
        {
            if (pNode->mNumMeshes > 0) return "mesh instance";
            if (isBone(data, pNode->mName.C_Str())) return "bone";
            else return "local transform";
        }

        void dumpSceneGraphHierarchy(ImporterData& data, const std::string& filename, aiNode* pRoot)
        {
            std::ofstream dotfile;
            dotfile.open(filename.c_str());

            std::function<void(const aiNode* pNode)> dumpNode = [&dotfile, &dumpNode, &data](const aiNode* pNode)
            {
                for (uint32_t i = 0; i < pNode->mNumChildren; i++)
                {
                    const aiNode* pChild = pNode->mChildren[i];
                    std::string parent = pNode->mName.C_Str();
                    std::string parentType = getNodeType(data, pNode);
                    std::string parentID = to_string(data.getFalcorNode(pNode));
                    std::string me = pChild->mName.C_Str();
                    std::string myType = getNodeType(data, pChild);
                    std::string myID = to_string(data.getFalcorNode(pChild));
                    std::replace(parent.begin(), parent.end(), '.', '_');
                    std::replace(me.begin(), me.end(), '.', '_');
                    std::replace(parent.begin(), parent.end(), '$', '_');
                    std::replace(me.begin(), me.end(), '$', '_');

                    dotfile << parentID << " " << parent << " (" << parentType << ") " << " -> " << myID << " " << me << " (" << myType << ") " << std::endl;

                    dumpNode(pChild);
                }
            };

            // Header
            dotfile << "digraph SceneGraph {" << std::endl;
            dumpNode(pRoot);
            // Close the file
            dotfile << "}" << std::endl; // closing graph scope
            dotfile.close();
        }

        mat4 getLocalToBindPoseMatrix(ImporterData& data, const std::string& name)
        {
            return isBone(data, name) ? data.localToBindPoseMatrices[name] : identity<mat4>();
        }

        bool parseNode(ImporterData& data, const aiNode* pCurrent, bool hasBoneAncestor)
        {
            SceneBuilder::Node n;
            n.name = pCurrent->mName.C_Str();
            bool currentIsBone = isBone(data, n.name);
            assert(currentIsBone == false || pCurrent->mNumMeshes == 0);

            n.parent = pCurrent->mParent ? data.getFalcorNode(pCurrent->mParent) : SceneBuilder::kInvalidNode;
            n.transform = aiCast(pCurrent->mTransformation);
            n.localToBindPose = getLocalToBindPoseMatrix(data, n.name);

            data.addAiNode(pCurrent, data.builder.addNode(n));

            bool b = true;
            // visit the children
            for (uint32_t i = 0; i < pCurrent->mNumChildren; i++)
            {
                b |= parseNode(data, pCurrent->mChildren[i], currentIsBone || hasBoneAncestor);
            }
            return b;
        }

        void createBoneList(ImporterData& data)
        {
            const aiScene* pScene = data.pScene;
            auto& boneMatrices = data.localToBindPoseMatrices;

            for (uint32_t meshID = 0; meshID < pScene->mNumMeshes; meshID++)
            {
                const aiMesh* pMesh = pScene->mMeshes[meshID];
                if (pMesh->HasBones() == false) continue;;
                for (uint32_t boneID = 0; boneID < pMesh->mNumBones; boneID++)
                {
                    boneMatrices[pMesh->mBones[boneID]->mName.C_Str()] = aiCast(pMesh->mBones[boneID]->mOffsetMatrix);
                }
            }
        }

        bool createSceneGraph(ImporterData& data)
        {
            createBoneList(data);
            aiNode* pRoot = data.pScene->mRootNode;
            assert(isBone(data, pRoot->mName.C_Str()) == false);
            bool success = parseNode(data, pRoot, false);
            //dumpSceneGraphHierarchy(data, "graph.dotfile", pRoot); // used for debugging
            return success;
        }

        bool addMeshes(ImporterData& data, aiNode* pNode)
        {
            size_t nodeID = data.getFalcorNode(pNode);
            for (size_t mesh = 0; mesh < pNode->mNumMeshes; mesh++)
            {
                size_t meshID = data.meshMap.at(pNode->mMeshes[mesh]);

                if (data.modelInstances.size())
                {
                    for(size_t instance = 0 ; instance < data.modelInstances.size() ; instance++)
                    {
                        size_t instanceNode = nodeID;
                        if(data.modelInstances[instance] != mat4())
                        {
                            // Add nodes
                            SceneBuilder::Node n;
                            n.name = "Node" + to_string(nodeID) + ".instance" + to_string(instance);
                            n.parent = nodeID;
                            n.transform = data.modelInstances[instance];
                            instanceNode = data.builder.addNode(n);
                        }
                        data.builder.addMeshInstance(instanceNode, meshID);
                    }
                }
                else data.builder.addMeshInstance(nodeID, meshID);
            }

            bool b = true;
            // visit the children
            for (uint32_t i = 0; i < pNode->mNumChildren; i++)
            {
                b |= addMeshes(data, pNode->mChildren[i]);
            }
            return b;
        }

        void loadTextures(ImporterData& data, const aiMaterial* pAiMaterial, const std::string& folder, Material* pMaterial, ImportMode importMode, bool useSrgb)
        {
            const auto& textureMappings = kTextureMappings[int(importMode)];

            for (const auto& source : textureMappings)
            {
                // Skip if texture of requested type is not available
                if (pAiMaterial->GetTextureCount(source.aiType) < source.aiIndex + 1) continue;

                // Get the texture name
                aiString aiPath;
                pAiMaterial->GetTexture(source.aiType, source.aiIndex, &aiPath);
                std::string path(aiPath.data);
                if (path.empty())
                {
                    logWarning("Texture has empty file name, ignoring.");
                    continue;
                }

                // Check if the texture was already loaded
                Texture::SharedPtr pTex;
                const auto& cacheItem = data.textureCache.find(path);
                if (cacheItem != data.textureCache.end())
                {
                    pTex = cacheItem->second;
                }
                else
                {
                    // create a new texture
                    std::string fullpath = folder + '/' + path;
                    fullpath = replaceSubstring(fullpath, "\\", "/");
                    pTex = Texture::createFromFile(fullpath, true, useSrgb && isSrgbRequired(source.targetType, pMaterial->getShadingModel()));
                    if (pTex)
                    {
                        data.textureCache[path] = pTex;
                    }
                }

                assert(pTex != nullptr);
                setTexture(source.targetType, pMaterial, pTex);
            }

            // Flush upload heap after every material so we don't accumulate a ton of memory usage when loading a model with a lot of textures
            gpDevice->flushAndSync();
        }

        Material::SharedPtr createMaterial(ImporterData& data, const aiMaterial* pAiMaterial, const std::string& folder, ImportMode importMode, bool useSrgb)
        {
            aiString name;
            pAiMaterial->Get(AI_MATKEY_NAME, name);

            // Parse the name
            std::string nameStr = std::string(name.C_Str());
            if (nameStr.empty())
            {
                logWarning("Material with no name found -> renaming to `unnamed`");
                nameStr = "unnamed";
            }
            Material::SharedPtr pMaterial = Material::create(nameStr);

            // Determine shading model.
            // MetalRough is the default for everything except OBJ. Check that both flags aren't set simultaneously.
            SceneBuilder::Flags builderFlags = data.builder.getFlags();
            assert(!(is_set(builderFlags, SceneBuilder::Flags::UseSpecGlossMaterials) && is_set(builderFlags, SceneBuilder::Flags::UseMetalRoughMaterials)));
            if (is_set(builderFlags, SceneBuilder::Flags::UseSpecGlossMaterials) || (importMode == ImportMode::OBJ && !is_set(builderFlags, SceneBuilder::Flags::UseMetalRoughMaterials)))
            {
                pMaterial->setShadingModel(ShadingModelSpecGloss);
            }

            // Load textures. Note that loading is affected by the current shading model.
            loadTextures(data, pAiMaterial, folder, pMaterial.get(), importMode, useSrgb);

            // Opacity
            float opacity = 1.f;
            if (pAiMaterial->Get(AI_MATKEY_OPACITY, opacity) == AI_SUCCESS)
            {
                vec4 diffuse = pMaterial->getBaseColor();
                diffuse.a = opacity;
                pMaterial->setBaseColor(diffuse);
            }

            // Bump scaling
            float bumpScaling;
            if (pAiMaterial->Get(AI_MATKEY_BUMPSCALING, bumpScaling) == AI_SUCCESS)
            {
                // TODO this should probably be a multiplier to the normal map
            }

            // Shininess
            float shininess;
            if (pAiMaterial->Get(AI_MATKEY_SHININESS, shininess) == AI_SUCCESS)
            {
                // Convert OBJ/MTL Phong exponent to glossiness.
                if (importMode == ImportMode::OBJ)
                {
                    float roughness = convertSpecPowerToRoughness(shininess);
                    shininess = 1.f - roughness;
                }
                vec4 spec = pMaterial->getSpecularParams();
                spec.a = shininess;
                pMaterial->setSpecularParams(spec);
            }

            // Refraction
            float refraction;
            if (pAiMaterial->Get(AI_MATKEY_REFRACTI, refraction) == AI_SUCCESS) pMaterial->setIndexOfRefraction(refraction);

            // Diffuse color
            aiColor3D color;
            if (pAiMaterial->Get(AI_MATKEY_COLOR_DIFFUSE, color) == AI_SUCCESS)
            {
                vec4 diffuse = vec4(color.r, color.g, color.b, pMaterial->getBaseColor().a);
                pMaterial->setBaseColor(diffuse);
            }

            // Specular color
            if (pAiMaterial->Get(AI_MATKEY_COLOR_SPECULAR, color) == AI_SUCCESS)
            {
                vec4 specular = vec4(color.r, color.g, color.b, pMaterial->getSpecularParams().a);
                pMaterial->setSpecularParams(specular);
            }

            // Emissive color
            if (pAiMaterial->Get(AI_MATKEY_COLOR_EMISSIVE, color) == AI_SUCCESS)
            {
                vec3 emissive = vec3(color.r, color.g, color.b);
                pMaterial->setEmissiveColor(emissive);
            }

            // Double-Sided
            int isDoubleSided;
            if (pAiMaterial->Get(AI_MATKEY_TWOSIDED, isDoubleSided) == AI_SUCCESS)
            {
                pMaterial->setDoubleSided((isDoubleSided != 0));
            }

            // Handle GLTF2 PBR materials
            if (importMode == ImportMode::GLTF2)
            {
                if (pAiMaterial->Get(AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_BASE_COLOR_FACTOR, color) == AI_SUCCESS)
                {
                    vec4 baseColor = vec4(color.r, color.g, color.b, pMaterial->getBaseColor().a);
                    pMaterial->setBaseColor(baseColor);
                }

                vec4 specularParams = pMaterial->getSpecularParams();

                float metallic;
                if (pAiMaterial->Get(AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_METALLIC_FACTOR, metallic) == AI_SUCCESS)
                {
                    specularParams.b = metallic;
                }

                float roughness;
                if (pAiMaterial->Get(AI_MATKEY_GLTF_PBRMETALLICROUGHNESS_ROUGHNESS_FACTOR, roughness) == AI_SUCCESS)
                {
                    specularParams.g = roughness;
                }

                pMaterial->setSpecularParams(specularParams);
            }

            // Parse the information contained in the name
            // Tokens following a '.' are interpreted as special flags
            auto nameVec = splitString(nameStr, ".");
            if (nameVec.size() > 1)
            {
                for (size_t i = 1; i < nameVec.size(); i++)
                {
                    std::string str = nameVec[i];
                    std::transform(str.begin(), str.end(), str.begin(), ::tolower);
                    if (str == "doublesided") pMaterial->setDoubleSided(true);
                    else logWarning("Unknown material property found in the material's name - `" + nameVec[i] + "`");
                }
            }

            // Use scalar opacity value for controlling specular transmission
            // TODO: Remove this workaround when we have a better way to define materials.
            if (opacity < 1.f)
            {
                pMaterial->setSpecularTransmission(1.f - opacity);
                pMaterial->setDoubleSided(true);
            }

            return pMaterial;
        }

        bool createAllMaterials(ImporterData& data, const std::string& modelFolder, ImportMode importMode)
        {
            bool useSrgb = !is_set(data.builder.getFlags(), SceneBuilder::Flags::AssumeLinearSpaceTextures);

            for (uint32_t i = 0; i < data.pScene->mNumMaterials; i++)
            {
                const aiMaterial* pAiMaterial = data.pScene->mMaterials[i];
                auto pMaterial = createMaterial(data, pAiMaterial, modelFolder, importMode, useSrgb);
                if (pMaterial == nullptr)
                {
                    logError("Can't allocate memory for material");
                    return false;
                }
                data.materialMap[i] = pMaterial;
            }

            return true;
        }

        BoneMeshMap createBoneMap(const aiScene* pScene)
        {
            BoneMeshMap boneMap;

            for (uint32_t meshID = 0; meshID < pScene->mNumMeshes; meshID++)
            {
                const aiMesh* pMesh = pScene->mMeshes[meshID];
                for (uint32_t boneID = 0; boneID < pMesh->mNumBones; boneID++)
                {
                    try
                    {
                        boneMap.at(pMesh->mBones[boneID]->mName.C_Str()).push_back(meshID);
                    }
                    catch (const std::out_of_range&)
                    {
                        std::vector<uint32_t> meshIDs;
                        meshIDs.push_back(meshID);
                        boneMap[pMesh->mBones[boneID]->mName.C_Str()] = meshIDs;
                    }
                }
            }

            return boneMap;
        }

        MeshInstanceList countMeshInstances(const aiScene* pScene)
        {
            MeshInstanceList meshInstances(pScene->mNumMeshes);

            std::function<void(const aiNode*)> countNodeMeshs = [&](const aiNode* pNode)
            {
                for (uint32_t i = 0; i < pNode->mNumMeshes; i++)
                {
                    meshInstances[pNode->mMeshes[i]].push_back(pNode);
                }

                for (uint32_t i = 0; i < pNode->mNumChildren; i++)
                {
                    countNodeMeshs(pNode->mChildren[i]);
                }
            };
            countNodeMeshs(pScene->mRootNode);

            return meshInstances;
        }

        bool validateBones(const aiScene* pScene)
        {
            // Make sure that each bone is only affecting a single mesh.
            // Our skinning system depends on that, because we apply the inverse world transformation to blended vertices. We do that because apparently ASSIMP's bone matrices are pre-multiplied with the final world transform
            // which results in the world-space blended-vertices, but we'd like them to be in local-space
            BoneMeshMap boneMap = createBoneMap(pScene);
            MeshInstanceList meshInstances = countMeshInstances(pScene);

            for (auto& b : boneMap)
            {
                for (uint32_t i = 0; i < b.second.size(); i++)
                {
                    if (meshInstances[b.second[i]].size() != 1)
                    {
                        logError(b.first + " references a mesh with multiple instances");
                        return false;
                    }

                    if (i > 0 && meshInstances[b.second[i]][0]->mTransformation != meshInstances[b.second[i - 1]][0]->mTransformation)
                    {
                        logError(b.first + " is contained within mesh instances with different world transform matrices");
                        return false;
                    }
                }
            }

            return true;
        }

        void verifyScene(const aiScene* pScene)
        {
            bool b = true;

            // No internal textures
            if (pScene->mTextures != 0)
            {
                b = false;
                logWarning("Model has internal textures which Falcor doesn't support");
            }

            b = validateBones(pScene);
            assert(b);
        }
    }

    bool AssimpImporter::import(const std::string& filename, SceneBuilder& builder, const InstanceMatrices& meshInstances)
    {
        std::string fullpath;
        if (findFileInDataDirectories(filename, fullpath) == false)
        {
            logError("Can't find model file " + filename);
            return false;
        }

        SceneBuilder::Flags builderFlags = builder.getFlags();
        uint32_t assimpFlags = aiProcessPreset_TargetRealtime_MaxQuality |
            aiProcess_FlipUVs |
            0;

        assimpFlags &= ~(aiProcess_CalcTangentSpace); // Never use Assimp's tangent gen code
        assimpFlags &= ~(aiProcess_FindDegenerates); // Avoid converting degenerated triangles to lines
        assimpFlags &= ~(aiProcess_OptimizeGraph); // Never use as it doesn't handle transforms with negative determinants
        assimpFlags &= ~(aiProcess_RemoveRedundantMaterials); // Avoid merging materials
        if (is_set(builderFlags, SceneBuilder::Flags::DontMergeMeshes)) assimpFlags &= ~aiProcess_OptimizeMeshes; // Avoid merging original meshes

        Assimp::Importer importer;
        const aiScene* pScene = importer.ReadFile(fullpath, assimpFlags);

        if (pScene == nullptr)
        {
            std::string str("Can't open model file '");
            str = str + std::string(filename) + "'\n" + importer.GetErrorString();
            logError(str);
            return false;
        }

        verifyScene(pScene);

        // Extract the folder name
        auto last = fullpath.find_last_of("/\\");
        std::string modelFolder = fullpath.substr(0, last);

        ImporterData data(pScene, builder, meshInstances);

        // Enable special treatment for obj and gltf files
        ImportMode importMode = ImportMode::Default;
        if (hasSuffix(filename, ".obj", false)) importMode = ImportMode::OBJ;
        if (hasSuffix(filename, ".gltf", false) || hasSuffix(filename, ".glb", false)) importMode = ImportMode::GLTF2;

        if (createAllMaterials(data, modelFolder, importMode) == false)
        {
            logError("Can't create materials for model " + filename);
            return false;
        }

        if (createSceneGraph(data) == false)
        {
            logError("Can't create draw lists for model " + filename);
            return false;
        }

        if (createMeshes(data) == false)
        {
            logError("Can't create meshes for model " + filename);
            return false;
        }

        if (addMeshes(data, data.pScene->mRootNode) == false)
        {
            logError("Can't add meshes for model " + filename);
            return false;
        }

        if (createAnimations(data) == false)
        {
            logError("Can't create animations for model " + filename);
            return false;
        }

        if (createCamera(data, importMode) == false)
        {
            logError("Can't create a camera for model " + filename);
            return false;
        }

        if (createLights(data) == false)
        {
            logError("Can't create a lights for model " + filename);
            return false;
        }
        return true;
    }

    bool AssimpImporter::import(const std::string& filename, SceneBuilder& builder)
    {
        InstanceMatrices meshInstances(1);
        return import(filename, builder, meshInstances);
    }
}
