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
#include "Light.h"
#include "Utils/Gui.h"
#include "API/Device.h"
#include "API/ConstantBuffer.h"
#include "API/Buffer.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include "Data/VertexAttrib.h"
#include "Graphics/Model/Model.h"
#include "Graphics/TextureHelper.h"
#include "Effects/Shadows/CSM.h"
#include "Graphics/Scene/Scene.h"
#include "API/Device.h"
#include <sstream>
namespace Falcor
{
    bool checkOffset(const std::string& structName, size_t cbOffset, size_t cppOffset, const char* field)
    {
        if (cbOffset != cppOffset)
        {
            logError("Light::" + std::string(structName) + ":: " + std::string(field) + " CB offset mismatch. CB offset is " + std::to_string(cbOffset) + ", C++ data offset is " + std::to_string(cppOffset));
            return false;
        }
        return true;
    }

    void Light::setIntoProgramVars(ProgramVars* pVars, ConstantBuffer* pCb, const std::string& varName)
    {
        size_t offset = pCb->getVariableOffset(varName);

#if _LOG_ENABLED
#define check_offset(_a) {static bool b = true; if(b) {assert(checkOffset("LightData", pCb->getVariableOffset(varName + "." + #_a) - offset, offsetof(LightData, _a), #_a));} b = false;}
        check_offset(dirW);
        check_offset(intensity);
        check_offset(penumbraAngle);
#undef check_offset
#endif

        setIntoProgramVars(pVars, pCb, offset);
    }

    void Light::setIntoProgramVars(ProgramVars* pVars, ConstantBuffer* pCb, size_t offset)
    {
        static_assert(kDataSize % sizeof(float) * 4 == 0, "LightData size should be a multiple of 16");
        assert(offset + kDataSize <= pCb->getSize());

        // Set everything except for the material
        pCb->setBlob(getRawData(), offset, kDataSize);
    }

    void Light::setIntoParameterBlock(ParameterBlock* pBlock, size_t offset, const std::string & varName)
    {
        auto pCb = pBlock->getConstantBuffer(pBlock->getReflection()->getName());

        static_assert(kDataSize % sizeof(float) * 4 == 0, "LightData size should be a multiple of 16");
        assert(offset + kDataSize <= pCb->getSize());

        // Set everything except for the material
        pCb->setBlob(getRawData(), offset, kDataSize);
    }

    glm::vec3 Light::getColorForUI()
    {
        if ((mUiLightIntensityColor * mUiLightIntensityScale) != getIntensityData())
        {
            float mag = max(getIntensityData().x, max(getIntensityData().y, getIntensityData().z));
            if (mag <= 1.f)
            {
                mUiLightIntensityColor = getIntensityData();
                mUiLightIntensityScale = 1.0f;
            }
            else
            {
                mUiLightIntensityColor = getIntensityData() / mag;
                mUiLightIntensityScale = mag;
            }
        }

        return mUiLightIntensityColor;
    }
    
    void Light::setColorFromUI(const glm::vec3& uiColor)
    {
        mUiLightIntensityColor = uiColor;
        getIntensityData() = (mUiLightIntensityColor * mUiLightIntensityScale);
        makeDirty();
    }

    float Light::getIntensityForUI()
    {
        if ((mUiLightIntensityColor * mUiLightIntensityScale) != getIntensityData())
        {
            float mag = max(getIntensityData().x, max(getIntensityData().y, getIntensityData().z));
            if (mag <= 1.f)
            {
                mUiLightIntensityColor = getIntensityData();
                mUiLightIntensityScale = 1.0f;
            }
            else
            {
                mUiLightIntensityColor = getIntensityData() / mag;
                mUiLightIntensityScale = mag;
            }
        }

        return mUiLightIntensityScale;
    }

    void Light::setIntensityFromUI(float intensity)
    {
        mUiLightIntensityScale = intensity;
        getIntensityData() = (mUiLightIntensityColor * mUiLightIntensityScale);
        makeDirty();
    }

    void Light::renderUI(Gui* pGui, const char* group)
    {
        if(!group || pGui->beginGroup(group))
        {
            glm::vec3 color = getColorForUI();
            if (pGui->addRgbColor("Color", color))
            {
                setColorFromUI(color);
            }
            float intensity = getIntensityForUI();
            if (pGui->addFloatVar("Intensity", intensity))
            {
                setIntensityFromUI(intensity);
            }

            if (group)
            {
                pGui->endGroup();
            }
        }
    }

    DirectionalLight::DirectionalLight() : mDistance(-1.0f)
    {
        mData.type = LightDirectional;
    }

    DirectionalLight::SharedPtr DirectionalLight::create()
    {
        DirectionalLight* pLight = new DirectionalLight();
        return SharedPtr(pLight);
    }

    DirectionalLight::~DirectionalLight() = default;

    void DirectionalLight::renderUI(Gui* pGui, const char* group)
    {
        if(!group || pGui->beginGroup(group))
        {
            if (pGui->addDirectionWidget("Direction", mData.dirW))
            {
                setWorldDirection(mData.dirW);
            }
            InfinitesimalLight::renderUI(pGui);
            if (group)
            {
                pGui->endGroup();
            }
        }
    }

    void DirectionalLight::setWorldDirection(const glm::vec3& dir)
    {
        mData.dirW = normalize(dir);
        mData.posW = mCenter - mData.dirW * mDistance; // Move light's position sufficiently far away
        makeDirty();
    }

    void DirectionalLight::setWorldParams(const glm::vec3& center, float radius)
    {
        mDistance = radius;
        mCenter = center;
        mData.posW = mCenter - mData.dirW * mDistance; // Move light's position sufficiently far away
        makeDirty();
    }

    float DirectionalLight::getPower() const
    {
        const float surfaceArea = (float)M_PI * mDistance * mDistance;
        return luminance(mData.intensity) * surfaceArea;
    }

    void DirectionalLight::move(const glm::vec3& position, const glm::vec3& target, const glm::vec3& up)
    {
        logError("DirectionalLight::move() is not used and thus not implemented for now.");
    }

    PointLight::SharedPtr PointLight::create()
    {
        PointLight* pLight = new PointLight;
        return SharedPtr(pLight);
    }

    PointLight::PointLight()
    {
        mData.type = LightPoint;
    }

    PointLight::~PointLight() = default;

    float PointLight::getPower() const
    {
        return luminance(mData.intensity) * 4.f * (float)M_PI;
    }

    void PointLight::renderUI(Gui* pGui, const char* group)
    {
        if(!group || pGui->beginGroup(group))
        {
            pGui->addFloat3Var("World Position", mData.posW, -FLT_MAX, FLT_MAX);
            pGui->addDirectionWidget("Direction", mData.dirW);

            if (pGui->addFloatVar("Opening Angle", mData.openingAngle, 0.f, (float)M_PI))
            {
                setOpeningAngle(mData.openingAngle);
            }
            if (pGui->addFloatVar("Penumbra Width", mData.penumbraAngle, 0.f, (float)M_PI))
            {
                setPenumbraAngle(mData.penumbraAngle);
            }
            InfinitesimalLight::renderUI(pGui);

            if (group)
            {
                pGui->endGroup();
            }
        }
    }

    void PointLight::setOpeningAngle(float openingAngle)
    {
        openingAngle = glm::clamp(openingAngle, 0.f, (float)M_PI);
        mData.openingAngle = openingAngle;
        /* Prepare an auxiliary cosine of the opening angle to quickly check whether we're within the cone of a spot light */
        mData.cosOpeningAngle = cos(openingAngle);
        makeDirty();
    }

    void PointLight::move(const glm::vec3& position, const glm::vec3& target, const glm::vec3& up)
    {
        mData.posW = position;
        mData.dirW = target - position;
    }

    AreaLight::SharedPtr AreaLight::create()
    {
        AreaLight* pLight = new AreaLight;
        return SharedPtr(pLight);
    }

    AreaLight::AreaLight()
    {
    }

    AreaLight::~AreaLight() = default;

    float AreaLight::getPower() const
    {
        return luminance(mAreaLightData.intensity) * (float)M_PI * mAreaLightData.surfaceArea;
    }

    void AreaLight::setIntoProgramVars(ProgramVars* pVars, ConstantBuffer* pCb, const std::string& varName)
    {
        // Set data except for material and mesh buffers
        size_t offset = pCb->getVariableOffset(varName);
        static_assert(kDataSize % sizeof(float) * 4 == 0, "AreaLightData size should be a multiple of 16");
        assert(offset + kAreaLightDataSize <= pCb->getSize());
        pCb->setBlob(&mAreaLightData, offset, kAreaLightDataSize);

#if _LOG_ENABLED
#define check_offset(_a) {static bool b = true; if(b) {assert(checkOffset("AreaLightData", pCb->getVariableOffset(varName + "." + #_a) - offset, offsetof(AreaLightData, _a), #_a));} b = false;}
        check_offset(dirW);
        check_offset(intensity);
        check_offset(tangent);
        check_offset(bitangent);
        check_offset(aabbMin);
        check_offset(aabbMax);
#undef check_offset
#endif

        // Set buffers and material
        const ParameterBlock::SharedPtr& pBlock = pVars->getDefaultBlock();
        pBlock->setRawBuffer(varName + ".resources.indexBuffer", mpIndexBuffer);
        pBlock->setRawBuffer(varName + ".resources.vertexBuffer", mpVertexBuffer);
        pBlock->setRawBuffer(varName + ".resources.texCoordBuffer", mpTexCoordBuffer);
        pBlock->setRawBuffer(varName + ".resources.meshCDFBuffer", mpMeshCDFBuffer);

        std::string matVarName = varName + ".resources.material";
        mpMeshInstance->getObject()->getMaterial()->setIntoProgramVars(pVars, pCb, matVarName.c_str());
    }

    void AreaLight::setIntoProgramVars(ProgramVars* pVars, ConstantBuffer* pCb, size_t offset)
    {
        logWarning("AreaLight::setIntoProgramVars() - Area light data contains resources that cannot be bound by offset. Ignoring call.");
    }

    void AreaLight::setIntoParameterBlock(ParameterBlock * pBlock, size_t offset, const std::string & varName)
    {
        auto pCb = pBlock->getConstantBuffer(pBlock->getReflection()->getName()).get();

        // Set data except for material and mesh buffers
        //size_t offset = pCb->getVariableOffset(varName);
        static_assert(kDataSize % sizeof(float) * 4 == 0, "AreaLightData size should be a multiple of 16");
        assert(offset + kAreaLightDataSize <= pCb->getSize());
        pCb->setBlob(&mAreaLightData, offset, kAreaLightDataSize);

#if _LOG_ENABLED
#define check_offset(_a) {static bool b = true; if(b) {assert(checkOffset("AreaLightData", pCb->getVariableOffset(varName + "." + #_a) - offset, offsetof(AreaLightData, _a), #_a));} b = false;}
        check_offset(dirW);
        check_offset(intensity);
        check_offset(tangent);
        check_offset(bitangent);
        check_offset(aabbMin);
        check_offset(aabbMax);
#undef check_offset
#endif

        // Set buffers and material
        pBlock->setRawBuffer(varName + ".resources.indexBuffer", mpIndexBuffer);
        pBlock->setRawBuffer(varName + ".resources.vertexBuffer", mpVertexBuffer);
        pBlock->setRawBuffer(varName + ".resources.texCoordBuffer", mpTexCoordBuffer);
        pBlock->setRawBuffer(varName + ".resources.meshCDFBuffer", mpMeshCDFBuffer);

        std::string matVarName = varName + ".resources.material.";
        mpMeshInstance->getObject()->getMaterial()->setIntoParameterBlock(pBlock, matVarName.c_str());
    }

    void AreaLight::renderUI(Gui* pGui, const char* group)
    {
        if(!group || pGui->beginGroup(group))
        {
            if (mpMeshInstance)
            {
                vec3 posW = mpMeshInstance->getTransformMatrix()[3];
                if (pGui->addFloat3Var("World Position", posW, -FLT_MAX, FLT_MAX))
                {
                    mpMeshInstance->setTranslation(posW, true);
                }
            }

            float intensity = mAreaLightData.intensity.r;
            if (pGui->addFloatVar("Intensity", intensity, 0.0f))
            {
                mAreaLightData.intensity = vec3(intensity);
            }

            if (group)
            {
                pGui->endGroup();
            }
        }
    }

    void AreaLight::setMeshData(const Model::MeshInstance::SharedPtr& pMeshInstance)
    {
        if (pMeshInstance && pMeshInstance != mpMeshInstance)
        {
            const auto& pMesh = pMeshInstance->getObject();
            assert(pMesh != nullptr);

            mpMeshInstance = pMeshInstance;

            // Fetch the mesh instance transformation
            mAreaLightData.transMat = mpMeshInstance->getTransformMatrix();

            const auto& vao = pMesh->getVao();
            setIndexBuffer(vao->getIndexBuffer());
            mAreaLightData.numIndices = uint32_t(mpIndexBuffer->getSize() / sizeof(glm::ivec3));

            int32_t posIdx = vao->getElementIndexByLocation(VERTEX_POSITION_LOC).vbIndex;
            assert(posIdx != Vao::ElementDesc::kInvalidIndex);
            setPositionsBuffer(vao->getVertexBuffer(posIdx));

            const int32_t uvIdx = vao->getElementIndexByLocation(VERTEX_TEXCOORD_LOC).vbIndex;
            bool hasUv = uvIdx != Vao::ElementDesc::kInvalidIndex;
            if (hasUv)
            {
                setTexCoordBuffer(vao->getVertexBuffer(VERTEX_TEXCOORD_LOC));
            }

            // Compute surface area of the mesh and generate probability
            // densities for importance sampling a triangle mesh
            computeSurfaceArea();

            // Check if this mesh has a material
            const Material::SharedPtr& pMaterial = pMesh->getMaterial();
            if (pMaterial)
            {
                mAreaLightData.intensity = pMaterial->getEmissiveColor();
            }
            makeDirty();
        }
    }

    void AreaLight::computeSurfaceArea()
    {
        if (mpMeshInstance && mpVertexBuffer && mpIndexBuffer)
        {
            const auto& pMesh = mpMeshInstance->getObject();
            assert(pMesh != nullptr);

            if (mpMeshInstance->getObject()->getPrimitiveCount() != 2 || mpMeshInstance->getObject()->getVertexCount() != 4)
            {
                logWarning("Only support sampling of rectangular light sources made of 2 triangles.");
                return;
            }

            // Read data from the buffers
            const glm::ivec3* pIndices = (const glm::ivec3*)mpIndexBuffer->map(Buffer::MapType::Read);
            const glm::vec3* pVertices = (const glm::vec3*)mpVertexBuffer->map(Buffer::MapType::Read);

            // Calculate surface area of the mesh
            mAreaLightData.surfaceArea = 0.f;
            mMeshCDF.push_back(0.f);
            for (uint32_t i = 0; i < pMesh->getPrimitiveCount(); ++i)
            {
                glm::ivec3 pId = pIndices[i];
                const vec3 p0(pVertices[pId.x]), p1(pVertices[pId.y]), p2(pVertices[pId.z]);

                mAreaLightData.surfaceArea += 0.5f * glm::length(glm::cross(p1 - p0, p2 - p0));

                // Add an entry using surface area measure as the discrete probability
                mMeshCDF.push_back(mMeshCDF[mMeshCDF.size() - 1] + mAreaLightData.surfaceArea);
            }

            // Normalize the probability densities
            if (mAreaLightData.surfaceArea > 0.f)
            {
                float invSurfaceArea = 1.f / mAreaLightData.surfaceArea;
                for (uint32_t i = 1; i < mMeshCDF.size(); ++i)
                {
                    mMeshCDF[i] *= invSurfaceArea;
                }

                mMeshCDF[mMeshCDF.size() - 1] = 1.f;
            }

            // Calculate basis tangent vectors and their lengths
            ivec3 pId = pIndices[0];
            const vec3 p0(pVertices[pId.x]), p1(pVertices[pId.y]), p2(pVertices[pId.z]);

            mAreaLightData.tangent = p0 - p1;
            mAreaLightData.bitangent = p2 - p1;

            // Create a CDF buffer
            mpMeshCDFBuffer.reset();
            mpMeshCDFBuffer = Buffer::create(sizeof(mMeshCDF[0])*mMeshCDF.size(), Buffer::BindFlags::Vertex, Buffer::CpuAccess::None, mMeshCDF.data());

            // Set the world position and world direction of this light
            if (mpIndexBuffer->getSize() != 0 && mpVertexBuffer->getSize() != 0)
            {
                glm::vec3 boxMin = pVertices[0];
                glm::vec3 boxMax = pVertices[0];
                for (uint32_t id = 1; id < mpMeshInstance->getObject()->getVertexCount(); ++id)
                {
                    boxMin = glm::min(boxMin, pVertices[id]);
                    boxMax = glm::max(boxMax, pVertices[id]);
                }

                mAreaLightData.posW = BoundingBox::fromMinMax(boxMin, boxMax).center;

                // This holds only for planar light sources
                const glm::vec3& p0 = pVertices[pIndices[0].x];
                const glm::vec3& p1 = pVertices[pIndices[0].y];
                const glm::vec3& p2 = pVertices[pIndices[0].z];

                // Take the normal of the first triangle as a light normal
                mAreaLightData.dirW = normalize(cross(p1 - p0, p2 - p0));

                // Save the axis-aligned bounding box
                mAreaLightData.aabbMin = boxMin;
                mAreaLightData.aabbMax = boxMax;
            }

            mpIndexBuffer->unmap();
            mpVertexBuffer->unmap();
        }
    }

    void AreaLight::move(const glm::vec3& position, const glm::vec3& target, const glm::vec3& up)
    {
        // Override target and up
        vec3 stillTarget = position + vec3(0, 0, 1);
        vec3 stillUp = vec3(0, 1, 0);
        mpMeshInstance->move(position, stillTarget, stillUp);
    }

    AreaLight::SharedPtr createAreaLight(const Model::MeshInstance::SharedPtr& pMeshInstance)
    {
        // Create an area light
        AreaLight::SharedPtr pAreaLight = AreaLight::create();
        if (pAreaLight)
        {
            // Set the geometry mesh
            pAreaLight->setMeshData(pMeshInstance);
        }

        return pAreaLight;
    }

    std::vector<AreaLight::SharedPtr> createAreaLightsForModel(const Model* pModel)
    {
        assert(pModel);
        std::vector<AreaLight::SharedPtr> areaLights;

        // Get meshes for this model
        for (uint32_t meshId = 0; meshId < pModel->getMeshCount(); ++meshId)
        {
            const Mesh::SharedPtr& pMesh = pModel->getMesh(meshId);

            // Obtain mesh instances for this mesh
            for (uint32_t instanceId = 0; instanceId < pModel->getMeshInstanceCount(meshId); ++instanceId)
            {
                // Check if this mesh has an emissive material
                const Material::SharedPtr& pMaterial = pMesh->getMaterial();
                if (pMaterial)
                {
                    if(EXTRACT_EMISSIVE_TYPE(pMaterial->getFlags()) != ChannelTypeUnused)
                    {
                        areaLights.push_back(createAreaLight(pModel->getMeshInstance(meshId, instanceId)));
                    }
                }
            }
        }
        return areaLights;
    }


    // LightEnv

    LightEnv::SharedPtr LightEnv::create()
    {
        return LightEnv::SharedPtr(new LightEnv());
    }

    LightEnv::LightEnv()
    {
        shaderTypeName = "AmbientLight";
        lightCollectionChanged = true;
    }

    void LightEnv::merge(LightEnv const* lightEnv)
    {
        mpLights.insert(mpLights.end(), lightEnv->mpLights.begin(), lightEnv->mpLights.end());
        mLightVersionIDs.resize(mpLights.size(), -1);
    }

    uint32_t LightEnv::addLight(const Light::SharedPtr& pLight)
    {
        mpLights.push_back(pLight);
        mLightVersionIDs.push_back(-1);
        lightCollectionChanged = true;
        return (uint32_t)mpLights.size() - 1;
    }

    void LightEnv::deleteLight(uint32_t lightID)
    {
        lightCollectionChanged = true;
        mpLights.erase(mpLights.begin() + lightID);
        mLightVersionIDs.erase(mLightVersionIDs.begin() + lightID);
    }

    void LightEnv::deleteAreaLights()
    {
        lightCollectionChanged = true;
        // Clean up the list before adding
        std::vector<Light::SharedPtr>::iterator it = mpLights.begin();
        std::vector<VersionID>::iterator vi = mLightVersionIDs.begin();
        for (; it != mpLights.end();)
        {
            if ((*it)->getType() == LightArea)
            {
                it = mpLights.erase(it);
                vi = mLightVersionIDs.erase(vi);
            }
            else
            {
                ++it;
                ++vi;
            }
        }
    }

    void LightEnv::deleteLightProbes()
    {
        lightCollectionChanged = true;
        // Clean up the list before adding
        std::vector<Light::SharedPtr>::iterator it = mpLights.begin();
        std::vector<VersionID>::iterator vi = mLightVersionIDs.begin();

        for (; it != mpLights.end();)
        {
            auto t = (*it)->getType();
            if (t >= LightProbeTypeStart && t < LightProbeTypeEnd)
            {
                it = mpLights.erase(it);
                vi = mLightVersionIDs.erase(vi);
            }
            else
            {
                ++it;
                ++vi;
            }
        }
    }

    // SLANG-INTEGRATION: forward declare
    ReflectionType::SharedPtr reflectType(slang::TypeLayoutReflection* pSlangType);

    ParameterBlockReflection::SharedConstPtr LightEnv::spBlockReflection;

    // Field offsets inside the parameter block.
    // TODO: these will need to be more dynamic once we are building a dynamic type...
    size_t LightEnv::sLightCountOffset = ConstantBuffer::kInvalidOffset;
    size_t LightEnv::sLightArrayOffset = ConstantBuffer::kInvalidOffset;
    size_t LightEnv::sAmbientLightOffset = ConstantBuffer::kInvalidOffset;

    ParameterBlock::SharedConstPtr LightEnv::getParameterBlock() const
    {
        auto env = const_cast<LightEnv*>(this);
        auto versionID = getVersionID();
        if (versionID > mParamBlockVersionID)
        {
            mParamBlockVersionID = versionID;
            // SLANG-INTEGRATION
            if (spBlockReflection == nullptr || lightCollectionChanged)
            {
                lightCollectionChanged = false;
                env->lightTypes.clear();
                for (auto light : mpLights)
                {
                    auto lightType = light->getTypeId();
                    auto findRs = env->lightTypes.find(lightType);
                    if (findRs == env->lightTypes.end())
                    {
                        LightTypeInfo newInfo;
                        newInfo.typeName = light->getShaderTypeName();
                        env->lightTypes[lightType] = newInfo;
                        findRs = env->lightTypes.find(lightType);
                    }
                    findRs->second.lights.push_back(light);
                }
                std::stringstream varNameSb, strStream;
                varNameSb << "lights.";
                for (auto & lt : env->lightTypes)
                {
                    strStream << "LightPair<LightArray<" << lt.second.typeName << ", " << lt.second.lights.size()
                        << ">, ";
                    lt.second.variableName = varNameSb.str() + "light1";
                    varNameSb << "light2.";
                }
                strStream << "AmbientLight";
                for (size_t i = 0u; i < env->lightTypes.size(); i++)
                    strStream << "> ";
                
                env->lightCollectionTypeName = strStream.str();
                env->shaderTypeName = "LightEnv<" + env->lightCollectionTypeName + " >";

                GraphicsProgram::SharedPtr pProgram = GraphicsProgram::createFromFile("", "Framework/Shaders/MaterialBlock.slang");
                ProgramReflection::SharedConstPtr pReflection = pProgram->getActiveVersion()->getReflector();
                auto slangReq = pProgram->getActiveVersion()->slangRequest;
                auto reflection = spGetReflection(slangReq);

                auto materialType = spReflection_FindTypeByName(reflection, env->shaderTypeName.c_str());
                auto layout = spReflection_GetTypeLayout(reflection, materialType, SLANG_LAYOUT_RULES_DEFAULT);
                auto blockType = reflectType((slang::TypeLayoutReflection*)layout);
                auto blockReflection = ParameterBlockReflection::create("");
                blockReflection->setElementType(blockType);
                blockReflection->finalize();
                spBlockReflection = blockReflection;
                assert(spBlockReflection);
                for (auto & lt : env->lightTypes)
                {
                    lt.second.cbOffset = blockType->findMember(lt.second.variableName)->getOffset();
                }
                const auto& pAmbientOffset = blockType->findMember(varNameSb.str() + "ambientLight");
                sAmbientLightOffset = pAmbientOffset ? pAmbientOffset->getOffset() : ConstantBuffer::kInvalidOffset;
            }
            mpParamBlock = ParameterBlock::create(spBlockReflection, true);
            mpParamBlock->setTypeName(shaderTypeName);
            mpParamBlock->genericTypeParamName = "TLightCollection";
            mpParamBlock->genericTypeArgumentName = lightCollectionTypeName;
            // Note: the following logic used to be in the `SceneRenderer`,
            // and so some stuff doesn't translate directly (e.g., we don't
            // currently have a representation of ambient lights in the
            // light environment, but we could/should).
            //
            glm::vec3 ambientIntensity = glm::vec3(0.0f);

            ConstantBuffer* pCB = mpParamBlock->getConstantBuffer(mpParamBlock->getReflection()->getName()).get();
            for (auto & lt : env->lightTypes)
            {
                uint32_t i = 0;
                for (auto l : lt.second.lights)
                {
                    l->setIntoParameterBlock(mpParamBlock.get(), i * Light::getShaderStructSize() + lt.second.cbOffset,
                        lt.second.variableName + ".lights[" + std::to_string(i) + "]");
                    i++;
                }
            }
            // set global params
            mpParamBlock->setSampler("params.probeSampler", lightProbeSampler);
            // Set lights
            if (sAmbientLightOffset != ConstantBuffer::kInvalidOffset)
            {
                pCB->setVariable(sAmbientLightOffset, ambientIntensity);
            }
        }
        else
        {
            // update shadow map parameters
            for (auto & lt : env->lightTypes)
            {
                int i = 0;
                for (auto l : lt.second.lights)
                {
                    if ((l->getTypeId() & LightType_ShadowBit))
                    {
                        auto infLight = dynamic_cast<InfinitesimalLight*>(l.get());
                        infLight->updateShadowParameters(mpParamBlock.get(), 
                            lt.second.variableName + ".lights[" + std::to_string(i) + "]");
                    }
                    i++;
                }
            }
        }
        return mpParamBlock;
    }

    void LightEnv::setIntoProgramVars(ProgramVars * vars)
    {
        auto block = vars->getDefaultBlock();
        //        block->setTexture("g_ltc_mat", texLtcMat);
        //        block->setTexture("g_ltc_mag", texLtcMag);
        //        block->setSampler("g_light_texSampler", linearSampler);
    }

    void LightEnv::bindSampler(Sampler::SharedPtr sampler)
    {
        lightProbeSampler = sampler;
        mVersionID++;
    }

    VersionID LightEnv::getVersionID() const
    {
        // check if any light has been modified
        bool dirty = lightCollectionChanged;

        auto ll = mpLights.begin(), le = mpLights.end();
        auto vv = mLightVersionIDs.begin();
        for (; ll != le; ++ll, ++vv)
        {
            auto oldVersionID = *vv;
            auto newVersionID = (*ll)->getVersionID();

            if (newVersionID > oldVersionID)
            {
                dirty = true;
                *vv = newVersionID;
            }
        }

        if (dirty)
        {
            mVersionID++;
        }
        return mVersionID;
    }
    InfinitesimalLight::~InfinitesimalLight()
    {
    }
    void InfinitesimalLight::enableShadowMap(Scene* pScene, int width, int height, int numCascades)
    {
        isShadowed = true;
        mCsm = CascadedShadowMaps::create(width, height, this, pScene, numCascades);

    }
    void InfinitesimalLight::disableShadowMap()
    {
        isShadowed = false;
        mCsm = nullptr;
    }
    void InfinitesimalLight::setIntoParameterBlock(ParameterBlock * pBlock, size_t offset, const std::string & varName)
    {
        Light::setIntoParameterBlock(pBlock, offset, varName);
        if (mCsm)
        {
            mCsm->setDataIntoParameterBlock(pBlock, varName + ".shadow.csm");
        }
    }
    void InfinitesimalLight::updateShadowParameters(ParameterBlock * pBlock, const std::string & varName)
    {
        if (mCsm)
        {
            mCsm->setDataIntoParameterBlock(pBlock, varName + ".shadow.csm");
        }
    }
    void InfinitesimalLight::renderUI(Gui * pGui, const char * group)
    {
        if (!group || pGui->beginGroup(group))
        {
            Light::renderUI(pGui, nullptr);
            if (mCsm)
            {
                mCsm->renderUi(pGui, "shadow");
            }
            if (group)
            {
                pGui->endGroup();
            }
        }
    }
}
