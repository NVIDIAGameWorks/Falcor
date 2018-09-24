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
#include "rapidjson/stringbuffer.h"
#include "rapidjson/prettywriter.h"

#include "Framework.h"
#include "SceneExporter.h"
#include <fstream>
#include "Utils/Platform/OS.h"
#include "Graphics/Scene/Editor/SceneEditor.h"

#define SCENE_EXPORTER
#include "SceneExportImportCommon.h"


namespace Falcor
{
    // Must be defined even though it's a const uint because value is passed as reference to functions
    const uint32_t SceneExporter::kVersion;

    bool SceneExporter::saveScene(const std::string& filename, const Scene::SharedPtr& pScene, uint32_t exportOptions)
    {
        SceneExporter exporter(filename, pScene);
        return exporter.save(exportOptions);
    }

    template<typename T>
    void addLiteral(rapidjson::Value& jval, rapidjson::Document::AllocatorType& jallocator, const std::string& key, const T& value)
    {
        rapidjson::Value jkey;
        jkey.SetString(key.c_str(), (uint32_t)key.size(), jallocator);
        jval.AddMember(jkey, value, jallocator);
    }

    void addJsonValue(rapidjson::Value& jval, rapidjson::Document::AllocatorType& jallocator, const std::string& key, rapidjson::Value& value)
    {
        rapidjson::Value jkey;
        jkey.SetString(key.c_str(), (uint32_t)key.size(), jallocator);
        jval.AddMember(jkey, value, jallocator);
    }

    void addString(rapidjson::Value& jval, rapidjson::Document::AllocatorType& jallocator, const std::string& key, const std::string& value)
    {
        rapidjson::Value jstring, jkey;
        jstring.SetString(value.c_str(), (uint32_t)value.size(), jallocator);
        jkey.SetString(key.c_str(), (uint32_t)key.size(), jallocator);

        jval.AddMember(jkey, jstring, jallocator);
    }

    void addBool(rapidjson::Value& jval, rapidjson::Document::AllocatorType& jallocator, const std::string& key, bool isValue)
    {
        rapidjson::Value jbool, jkey;
        jbool.SetBool(isValue);
        jkey.SetString(key.c_str(), (uint32_t)key.size(), jallocator);

        jval.AddMember(jkey, jbool, jallocator);
    }

    template<typename T>
    void addVector(rapidjson::Value& jval, rapidjson::Document::AllocatorType& jallocator, const std::string& key, const T& value)
    {
        rapidjson::Value jkey;
        jkey.SetString(key.c_str(), (uint32_t)key.size(), jallocator);
        rapidjson::Value jvec(rapidjson::kArrayType);
        for (int32_t i = 0; i < value.length(); i++)
        {
            jvec.PushBack(value[i], jallocator);
        }

        jval.AddMember(jkey, jvec, jallocator);
    }

    bool SceneExporter::save(uint32_t exportOptions)
    {
        mExportOptions = exportOptions;

        // create the file
        mJDoc.SetObject();

        // Write the version
        rapidjson::Value& JVal = mJDoc;
        auto& allocator = mJDoc.GetAllocator();
        addLiteral(JVal, allocator, SceneKeys::kVersion, kVersion);

        // Write everything else
        bool exportPaths = (exportOptions & ExportPaths) != 0;
        if (exportOptions & ExportGlobalSettings)    writeGlobalSettings(exportPaths);
        if (exportOptions & ExportModels)            writeModels();
        if (exportOptions & ExportLights)            writeLights();
        if (exportOptions & ExportCameras)           writeCameras();
        if (exportOptions & ExportUserDefined)       writeUserDefinedSection();
        if (exportOptions & ExportPaths)             writePaths();

        // Get the output string
        rapidjson::StringBuffer buffer;
        rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
        writer.SetIndent(' ', 4);
        mJDoc.Accept(writer);
        std::string str(buffer.GetString(), buffer.GetSize());

        // Output the file
        std::ofstream outputStream(mFilename.c_str());
        if (outputStream.fail())
        {
            logError("Can't open output scene file " + mFilename + ".\nExporting failed.");
            return false;
        }
        outputStream << str;
        outputStream.close();

        return true;
    }

    void SceneExporter::writeGlobalSettings(bool writeActivePath)
    {
        rapidjson::Value& jval = mJDoc;
        auto& Allocator = mJDoc.GetAllocator();

        addLiteral(jval, Allocator, SceneKeys::kCameraSpeed, mpScene->getCameraSpeed());
        addLiteral(jval, Allocator, SceneKeys::kLightingScale, mpScene->getLightingScale());

        if (mpScene->getCameraCount() > 0)
        {
            addString(jval, Allocator, SceneKeys::kActiveCamera, mpScene->getActiveCamera()->getName());
        }
    }

    void createModelValue(const Scene::SharedPtr& pScene, uint32_t modelID, rapidjson::Document::AllocatorType& allocator, rapidjson::Value& jmodel)
    {
        jmodel.SetObject();

        const Model* pModel = pScene->getModel(modelID).get();

        // Export model properties
        addString(jmodel, allocator, SceneKeys::kFilename, stripDataDirectories(pModel->getFilename()));
        addString(jmodel, allocator, SceneKeys::kName, pModel->getName());

        if (pScene->getModel(modelID)->hasAnimations())
        {
            addLiteral(jmodel, allocator, SceneKeys::kActiveAnimation, pModel->getActiveAnimation());
        }

        // Export model material properties
        rapidjson::Value materialValue;
        materialValue.SetObject();
        switch (pModel->getMesh(0)->getMaterial()->getShadingModel())
        {
        case ShadingModelMetalRough:
            addString(materialValue, allocator, SceneKeys::kShadingModel, SceneKeys::kShadingMetalRough);
            break;
        case ShadingModelSpecGloss:
            addString(materialValue, allocator, SceneKeys::kShadingModel, SceneKeys::kShadingSpecGloss);
            break;
        default:
            logWarning("SceneExporter: Unknown shading model found on model " + pModel->getName() + ", ignoring value");
        }
        addJsonValue(jmodel, allocator, SceneKeys::kMaterial, materialValue);

        // Export model instances
        rapidjson::Value jsonInstanceArray;
        jsonInstanceArray.SetArray();
        for (uint32_t i = 0; i < pScene->getModelInstanceCount(modelID); i++)
        {
            rapidjson::Value jsonInstance;
            jsonInstance.SetObject();
            auto& pInstance = pScene->getModelInstance(modelID, i);

            addString(jsonInstance, allocator, SceneKeys::kName, pInstance->getName());
            addVector(jsonInstance, allocator, SceneKeys::kTranslationVec, pInstance->getTranslation());
            addVector(jsonInstance, allocator, SceneKeys::kScalingVec, pInstance->getScaling());

            // Translate rotation to degrees
            glm::vec3 rotation = glm::degrees(pInstance->getRotation());
            addVector(jsonInstance, allocator, SceneKeys::kRotationVec, rotation);

            jsonInstanceArray.PushBack(jsonInstance, allocator);
        }

        addJsonValue(jmodel, allocator, SceneKeys::kModelInstances, jsonInstanceArray);
    }

    void SceneExporter::writeModels()
    {
        if (mpScene->getModelCount() == 0)
        {
            return;
        }

        rapidjson::Value jsonModelArray;
        jsonModelArray.SetArray();

        for (uint32_t i = 0; i < mpScene->getModelCount(); i++)
        {
            rapidjson::Value jsonModel;
            createModelValue(mpScene, i, mJDoc.GetAllocator(), jsonModel);
            jsonModelArray.PushBack(jsonModel, mJDoc.GetAllocator());
        }
        addJsonValue(mJDoc, mJDoc.GetAllocator(), SceneKeys::kModels, jsonModelArray);
    }

    void createPointLightValue(const PointLight* pLight, rapidjson::Document::AllocatorType& allocator, rapidjson::Value& jsonLight)
    {
        addString(jsonLight, allocator, SceneKeys::kName, pLight->getName());
        addString(jsonLight, allocator, SceneKeys::kType, SceneKeys::kPointLight);
        addVector(jsonLight, allocator, SceneKeys::kLightIntensity, pLight->getIntensity());
        addVector(jsonLight, allocator, SceneKeys::kLightPos, pLight->getWorldPosition());
        addVector(jsonLight, allocator, SceneKeys::kLightDirection, pLight->getWorldDirection());
        addLiteral(jsonLight, allocator, SceneKeys::kLightOpeningAngle, glm::degrees(pLight->getOpeningAngle()));
        addLiteral(jsonLight, allocator, SceneKeys::kLightPenumbraAngle, glm::degrees(pLight->getPenumbraAngle()));
    }

    void createDirectionalLightValue(const DirectionalLight* pLight, rapidjson::Document::AllocatorType& allocator, rapidjson::Value& jsonLight)
    {
        addString(jsonLight, allocator, SceneKeys::kName, pLight->getName());
        addString(jsonLight, allocator, SceneKeys::kType, SceneKeys::kDirLight);
        addVector(jsonLight, allocator, SceneKeys::kLightIntensity, pLight->getIntensity());
        addVector(jsonLight, allocator, SceneKeys::kLightDirection, pLight->getWorldDirection());
    }

    void createLightValue(const Scene::SharedPtr& pScene, uint32_t lightID, rapidjson::Document::AllocatorType& allocator, rapidjson::Value& jsonLight)
    {
        jsonLight.SetObject();
        const auto pLight = pScene->getLight(lightID);

        switch (pLight->getType())
        {
        case LightPoint:
            createPointLightValue((PointLight*)pLight.get(), allocator, jsonLight);
            break;
        case LightDirectional:
            createDirectionalLightValue((DirectionalLight*)pLight.get(), allocator, jsonLight);
            break;
        default:
            should_not_get_here();
            break;
        }
    }

    void SceneExporter::writeLights()
    {
        if (mpScene->getLightCount() == 0)
        {
            return;
        }

        rapidjson::Value jsonLightsArray(rapidjson::kArrayType);

        uint32_t numLightsSaved = 0;
        for (uint32_t i = 0; i < mpScene->getLightCount(); i++)
        {
            if (mpScene->getLights()[i]->getType() != LightPoint &&
                mpScene->getLights()[i]->getType() != LightDirectional)
            {
                continue;
            }
            rapidjson::Value jsonLight;
            createLightValue(mpScene, i, mJDoc.GetAllocator(), jsonLight);
            jsonLightsArray.PushBack(jsonLight, mJDoc.GetAllocator());
            numLightsSaved++;
        }
        if (numLightsSaved > 0)
        {
            addJsonValue(mJDoc, mJDoc.GetAllocator(), SceneKeys::kLights, jsonLightsArray);
        }
    }

    void createCameraValue(const Scene::SharedConstPtr& pScene, uint32_t cameraID, rapidjson::Document::AllocatorType& allocator, rapidjson::Value& jsonCamera)
    {
        jsonCamera.SetObject();
        const auto pCamera = pScene->getCamera(cameraID);
        addString(jsonCamera, allocator, SceneKeys::kName, pCamera->getName());
        addVector(jsonCamera, allocator, SceneKeys::kCamPosition, pCamera->getPosition());
        addVector(jsonCamera, allocator, SceneKeys::kCamTarget, pCamera->getTarget());
        addVector(jsonCamera, allocator, SceneKeys::kCamUp, pCamera->getUpVector());
        addLiteral(jsonCamera, allocator, SceneKeys::kCamFocalLength, pCamera->getFocalLength());
        glm::vec2 depthRange;
        depthRange[0] = pCamera->getNearPlane();
        depthRange[1] = pCamera->getFarPlane();
        addVector(jsonCamera, allocator, SceneKeys::kCamDepthRange, depthRange);
        addLiteral(jsonCamera, allocator, SceneKeys::kCamAspectRatio, pCamera->getAspectRatio());
    }

    void SceneExporter::writeCameras()
    {
        if (mpScene->getCameraCount() == 0)
        {
            return;
        }

        rapidjson::Value jsonCameraArray(rapidjson::kArrayType);
        for (uint32_t i = 0; i < mpScene->getCameraCount(); i++)
        {
            rapidjson::Value jsonCamera;
            createCameraValue(mpScene, i, mJDoc.GetAllocator(), jsonCamera);
            jsonCameraArray.PushBack(jsonCamera, mJDoc.GetAllocator());
        }
        addJsonValue(mJDoc, mJDoc.GetAllocator(), SceneKeys::kCameras, jsonCameraArray);
    }

    void SceneExporter::writePaths()
    {
        if (mpScene->getPathCount() == 0)
        {
            return;
        }

        auto& allocator = mJDoc.GetAllocator();

        // Loop over the paths
        rapidjson::Value jsonPathsArray(rapidjson::kArrayType);
        for (uint32_t pathID = 0; pathID < mpScene->getPathCount(); pathID++)
        {
            const auto pPath = mpScene->getPath(pathID);
            rapidjson::Value jsonPath;
            jsonPath.SetObject();
            addString(jsonPath, allocator, SceneKeys::kName, pPath->getName());
            addBool(jsonPath, allocator, SceneKeys::kPathLoop, pPath->isRepeatOn());

            // Add the keyframes
            rapidjson::Value jsonFramesArray(rapidjson::kArrayType);
            for (uint32_t frameID = 0; frameID < pPath->getKeyFrameCount(); frameID++)
            {
                const auto& frame = pPath->getKeyFrame(frameID);
                rapidjson::Value jsonFrame(rapidjson::kObjectType);
                addLiteral(jsonFrame, allocator, SceneKeys::kFrameTime, frame.time);
                addVector(jsonFrame, allocator, SceneKeys::kCamPosition, frame.position);
                addVector(jsonFrame, allocator, SceneKeys::kCamTarget, frame.target);
                addVector(jsonFrame, allocator, SceneKeys::kCamUp, frame.up);

                jsonFramesArray.PushBack(jsonFrame, allocator);
            }

            addJsonValue(jsonPath, allocator, SceneKeys::kPathFrames, jsonFramesArray);

            // Add attached objects
            rapidjson::Value jsonObjectsArray(rapidjson::kArrayType);
            for (uint32_t i = 0; i < pPath->getAttachedObjectCount(); i++)
            {
                rapidjson::Value jsonObject(rapidjson::kObjectType);

                const auto& pMovable = pPath->getAttachedObject(i);

                const auto& pModelInstance = std::dynamic_pointer_cast<Scene::ModelInstance>(pMovable);
                const auto& pCamera = std::dynamic_pointer_cast<Camera>(pMovable);
                const auto& pLight = std::dynamic_pointer_cast<Light>(pMovable);

                if (pModelInstance != nullptr)
                {
                    addString(jsonObject, allocator, SceneKeys::kType, SceneKeys::kModelInstance);
                    addString(jsonObject, allocator, SceneKeys::kName, pModelInstance->getName());
                }
                else if (pCamera != nullptr)
                {
                    addString(jsonObject, allocator, SceneKeys::kType, SceneKeys::kCamera);
                    addString(jsonObject, allocator, SceneKeys::kName, pCamera->getName());
                }
                else if (pLight != nullptr)
                {
                    addString(jsonObject, allocator, SceneKeys::kType, SceneKeys::kLight);
                    addString(jsonObject, allocator, SceneKeys::kName, pLight->getName());
                }

                jsonObjectsArray.PushBack(jsonObject, allocator);
            }

            addJsonValue(jsonPath, allocator, SceneKeys::kAttachedObjects, jsonObjectsArray);

            // Finish path
            jsonPathsArray.PushBack(jsonPath, allocator);
        }

        addJsonValue(mJDoc, allocator, SceneKeys::kPaths, jsonPathsArray);
    }

    void SceneExporter::writeUserDefinedSection()
    {
        if (mpScene->getUserVariableCount() == 0)
        {
            return;
        }

        rapidjson::Value jsonUserValues(rapidjson::kObjectType);
        auto& allocator = mJDoc.GetAllocator();

        // TODO -- use these. unused scenekeys to avoid linux warning
        (void)SceneKeys::kEnvMap;
        (void)SceneKeys::kAreaLightRect;
        (void)SceneKeys::kAreaLightSphere;
        (void)SceneKeys::kAreaLightDisc;

        for (uint32_t varID = 0; varID < mpScene->getUserVariableCount(); varID++)
        {
            std::string name;
            const auto& var = mpScene->getUserVariable(varID, name);

            switch (var.type)
            {
            case Scene::UserVariable::Type::Int:
                addLiteral(jsonUserValues, allocator, name, var.i32);
                break;
            case Scene::UserVariable::Type::Uint:
                addLiteral(jsonUserValues, allocator, name, var.u32);
                break;
            case Scene::UserVariable::Type::Int64:
                addLiteral(jsonUserValues, allocator, name, var.i64);
                break;
            case Scene::UserVariable::Type::Uint64:
                addLiteral(jsonUserValues, allocator, name, var.u64);
                break;
            case Scene::UserVariable::Type::Double:
                addLiteral(jsonUserValues, allocator, name, var.d64);
                break;
            case Scene::UserVariable::Type::String:
                addString(jsonUserValues, allocator, name, var.str);
                break;
            case Scene::UserVariable::Type::Vec2:
                addVector(jsonUserValues, allocator, name, var.vec2);
                break;
            case Scene::UserVariable::Type::Vec3:
                addVector(jsonUserValues, allocator, name, var.vec3);
                break;
            case Scene::UserVariable::Type::Vec4:
                addVector(jsonUserValues, allocator, name, var.vec4);
                break;
            case Scene::UserVariable::Type::Bool:
                addBool(jsonUserValues, allocator, name, var.b);
                break;
            default:
                should_not_get_here();
                return;
            }
        }

        addJsonValue(mJDoc, allocator, SceneKeys::kUserDefined, jsonUserValues);
    }
}
