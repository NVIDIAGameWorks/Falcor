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
#include "SceneImporter.h"
#include "rapidjson/error/en.h"
#include "Scene.h"
#include "Utils/Platform/OS.h"
#include <sstream>
#include <fstream>
#include <algorithm>
#include "Graphics/TextureHelper.h"
#include "API/Device.h"
#include "Data/HostDeviceSharedMacros.h"

#define SCENE_IMPORTER
#include "SceneExportImportCommon.h"

namespace Falcor
{
    bool SceneImporter::error(const std::string& msg)
    {
        std::string err = "Error when parsing scene file \"" + mFilename + "\".\n" + msg;
#if _LOG_ENABLED
        logError(err);
#else
        msgBox(err);
#endif
        return false;
    }

    template<uint32_t VecSize>
    bool SceneImporter::getFloatVec(const rapidjson::Value& jsonVal, const std::string& desc, float vec[VecSize])
    {
        if (jsonVal.IsArray() == false)
        {
            error("Trying to load a vector for " + desc + ", but JValue is not an array");
            return false;
        }

        if (jsonVal.Size() != VecSize)
        {
            return error("Trying to load a vector for " + desc + ", but vector size mismatches. Required size is " + std::to_string(VecSize) + ", array size is " + std::to_string(jsonVal.Size()));
        }

        for (uint32_t i = 0; i < jsonVal.Size(); i++)
        {
            if (jsonVal[i].IsNumber() == false)
            {
                return error("Trying to load a vector for " + desc + ", but one the elements is not a number.");
            }

            vec[i] = (float)(jsonVal[i].GetDouble());
        }
        return true;
    }

    bool SceneImporter::getFloatVecAnySize(const rapidjson::Value& jsonVal, const std::string& desc, std::vector<float>& vec)
    {
        if (jsonVal.IsArray() == false)
        {
            return error("Trying to load a vector for " + desc + ", but JValue is not an array");
        }

        vec.resize(jsonVal.Size());
        for (uint32_t i = 0; i < jsonVal.Size(); i++)
        {
            if (jsonVal[i].IsNumber() == false)
            {
                return error("Trying to load a vector for " + desc + ", but one the elements is not a number.");
            }

            vec[i] = (float)(jsonVal[i].GetDouble());
        }
        return true;
    }

    bool SceneImporter::loadScene(Scene& scene, const std::string& filename, Model::LoadFlags modelLoadFlags, Scene::LoadFlags sceneLoadFlags)
    {
        SceneImporter importer(scene);
        return importer.load(filename, modelLoadFlags, sceneLoadFlags);
    }

    bool SceneImporter::createModelInstances(const rapidjson::Value& jsonVal, const Model::SharedPtr& pModel)
    {
        if (jsonVal.IsArray() == false)
        {
            return error("Model instances should be an array of objects");
        }

        for (uint32_t i = 0; i < jsonVal.Size(); i++)
        {
            const auto& instance = jsonVal[i];
            glm::vec3 scaling(1, 1, 1);
            glm::vec3 translation(0, 0, 0);
            glm::vec3 rotation(0, 0, 0);
            std::string name = "Instance " + std::to_string(i);

            for (auto m = instance.MemberBegin(); m < instance.MemberEnd(); m++)
            {
                std::string key(m->name.GetString());
                if (key == SceneKeys::kName)
                {
                    if (m->value.IsString() == false)
                    {
                        return error("Model instance name should be a string value.");
                    }
                    name = std::string(m->value.GetString());
                }
                else if (key == SceneKeys::kTranslationVec)
                {
                    if (getFloatVec<3>(m->value, "Model instance translation vector", &translation[0]) == false)
                    {
                        return false;
                    }
                }
                else if (key == SceneKeys::kScalingVec)
                {
                    if (getFloatVec<3>(m->value, "Model instance scale vector", &scaling[0]) == false)
                    {
                        return false;
                    }
                }
                else if (key == SceneKeys::kRotationVec)
                {
                    if (getFloatVec<3>(m->value, "Model instance rotation vector", &rotation[0]) == false)
                    {
                        return false;
                    }

                    rotation = glm::radians(rotation);
                }
                else
                {
                    return error("Unknown key \"" + key + "\" when parsing model instance");
                }
            }

            if (isNameDuplicate(name, mInstanceMap, "model instances"))
            {
                return false;
            }
            else
            {
                auto pInstance = Scene::ModelInstance::create(pModel, translation, rotation, scaling, name);
                mInstanceMap[pInstance->getName()] = pInstance;
                mScene.addModelInstance(pInstance);
            }
        }

        return true;
    }

    bool SceneImporter::createModel(const rapidjson::Value& jsonModel)
    {
        // Model must have at least a filename
        if (jsonModel.HasMember(SceneKeys::kFilename) == false)
        {
            return error("Model must have a filename");
        }

        // Get Model name
        const auto& modelFile = jsonModel[SceneKeys::kFilename];
        if (modelFile.IsString() == false)
        {
            return error("Model filename must be a string");
        }

        std::string file = mDirectory + '/' + modelFile.GetString();
        if (doesFileExist(file) == false)
        {
            file = modelFile.GetString();
        }

        // Parse additional properties that affect loading
        Model::LoadFlags modelFlags = mModelLoadFlags;
        if (jsonModel.HasMember(SceneKeys::kMaterial))
        {
            const auto& materialSettings = jsonModel[SceneKeys::kMaterial];
            if (materialSettings.IsObject() == false)
            {
                return error("Material properties for \"" + file + "\" must be a JSON object");
            }

            for (auto m = materialSettings.MemberBegin(); m != materialSettings.MemberEnd(); m++)
            {
                if (m->name == SceneKeys::kShadingModel)
                {
                    if (m->value == SceneKeys::kShadingSpecGloss)
                    {
                        modelFlags |= Model::LoadFlags::UseSpecGlossMaterials;
                    }
                }
            }
        }

        // Load the model
        auto pModel = Model::createFromFile(file.c_str(), modelFlags);
        if (pModel == nullptr)
        {
            return error("Could not load model: " + file);
        }

        bool instanceAdded = false;

        // Loop over the other members
        for (auto jval = jsonModel.MemberBegin(); jval != jsonModel.MemberEnd(); jval++)
        {
            std::string keyName(jval->name.GetString());
            if (keyName == SceneKeys::kFilename)
            {
                // Already handled
            }
            else if (keyName == SceneKeys::kName)
            {
                if (jval->value.IsString() == false)
                {
                    return error("Model name should be a string value.");
                }
                pModel->setName(std::string(jval->value.GetString()));
            }
            else if (keyName == SceneKeys::kModelInstances)
            {
                if (createModelInstances(jval->value, pModel) == false)
                {
                    return false;
                }

                instanceAdded = true;
            }
            else if (keyName == SceneKeys::kActiveAnimation)
            {
                if (jval->value.IsUint() == false)
                {
                    return error("Model active animation should be an unsigned integer");
                }
                uint32_t activeAnimation = jval->value.GetUint();
                if (activeAnimation >= pModel->getAnimationsCount())
                {
                    std::string msg = "Warning when parsing scene file \"" + mFilename + "\".\nModel " + pModel->getName() + " was specified with active animation " + std::to_string(activeAnimation);
                    msg += ", but model only has " + std::to_string(pModel->getAnimationsCount()) + " animations. Ignoring field";
                    logWarning(msg);
                }
                else
                {
                    pModel->setActiveAnimation(activeAnimation);
                }
            }
            else if (keyName == SceneKeys::kMaterial)
            {
                // Existing parameters already handled
            }
            else
            {
                return error("Invalid key found in models array. Key == " + keyName + ".");
            }
        }

        // If no instances for the model were loaded from the scene file
        if (instanceAdded == false)
        {
            mScene.addModelInstance(pModel, "Instance 0");
        }

        return true;
    }

    bool SceneImporter::parseModels(const rapidjson::Value& jsonVal)
    {
        if (jsonVal.IsArray() == false)
        {
            return error("models section should be an array of objects.");
        }

        // Loop over the array
        for (uint32_t i = 0; i < jsonVal.Size(); i++)
        {
            if (createModel(jsonVal[i]) == false)
            {
                return false;
            }
        }
        return true;
    }

    bool SceneImporter::createDirLight(const rapidjson::Value& jsonLight)
    {
        auto pDirLight = DirectionalLight::create();

        for (auto it = jsonLight.MemberBegin(); it != jsonLight.MemberEnd(); it++)
        {
            std::string key(it->name.GetString());
            const auto& value = it->value;
            if (key == SceneKeys::kName)
            {
                if (value.IsString() == false)
                {
                    return error("Point light name should be a string");
                }
                std::string name = value.GetString();
                if (name.find(' ') != std::string::npos)
                {
                    return error("Point light name can't have spaces");
                }
                pDirLight->setName(name);
            }
            else if (key == SceneKeys::kType)
            {
                // Don't care
            }
            else if (key == SceneKeys::kLightIntensity)
            {
                glm::vec3 intensity;
                if (getFloatVec<3>(value, "Directional light intensity", &intensity[0]) == false)
                {
                    return false;
                }
                pDirLight->setIntensity(intensity);
            }
            else if (key == SceneKeys::kLightDirection)
            {
                glm::vec3 direction;
                if (getFloatVec<3>(value, "Directional light intensity", &direction[0]) == false)
                {
                    return false;
                }
                pDirLight->setWorldDirection(direction);
            }
            else
            {
                return error("Invalid key found in directional light object. Key == " + key + ".");
            }
        }
        mScene.addLight(pDirLight);
        return true;
    }

    bool SceneImporter::createPointLight(const rapidjson::Value& jsonLight)
    {
        auto pPointLight = PointLight::create();

        for (auto it = jsonLight.MemberBegin(); it != jsonLight.MemberEnd(); it++)
        {
            std::string key(it->name.GetString());
            const auto& value = it->value;
            if (key == SceneKeys::kName)
            {
                if (value.IsString() == false)
                {
                    return error("Dir light name should be a string");
                }
                std::string name = value.GetString();
                if (name.find(' ') != std::string::npos)
                {
                    return error("Dir light name can't have spaces");
                }
                pPointLight->setName(name);
            }
            else if (key == SceneKeys::kType)
            {
                // Don't care
            }
            else if (key == SceneKeys::kLightOpeningAngle)
            {
                if (value.IsNumber() == false)
                {
                    return error("Camera's FOV should be a number");
                }
                float angle = (float)value.GetDouble();
                // Convert to radiance
                angle = glm::radians(angle);
                pPointLight->setOpeningAngle(angle);
            }
            else if (key == SceneKeys::kLightPenumbraAngle)
            {
                if (value.IsNumber() == false)
                {
                    return error("Camera's FOV should be a number");
                }
                float angle = (float)value.GetDouble();
                // Convert to radiance
                angle = glm::radians(angle);
                pPointLight->setPenumbraAngle(angle);
            }
            else if (key == SceneKeys::kLightIntensity)
            {
                glm::vec3 intensity;
                if (getFloatVec<3>(value, "Point light intensity", &intensity[0]) == false)
                {
                    return false;
                }
                pPointLight->setIntensity(intensity);
            }
            else if (key == SceneKeys::kLightPos)
            {
                glm::vec3 position;
                if (getFloatVec<3>(value, "Point light position", &position[0]) == false)
                {
                    return false;
                }
                pPointLight->setWorldPosition(position);
            }
            else if (key == SceneKeys::kLightDirection)
            {
                glm::vec3 dir;
                if (getFloatVec<3>(value, "Point light direction", &dir[0]) == false)
                {
                    return false;
                }
                pPointLight->setWorldDirection(dir);
            }
            else
            {
                return error("Invalid key found in point light object. Key == " + key + ".");
            }
        }

        if (isNameDuplicate(pPointLight->getName(), mLightMap, "lights"))
        {
            return false;
        }
        else
        {
            mLightMap[pPointLight->getName()] = pPointLight;
            mScene.addLight(pPointLight);
        }

        return true;
    }

    // Support for analytic area lights
    bool SceneImporter::createAnalyticAreaLight(const rapidjson::Value& jsonLight)
    {
        auto pAreaLight = AnalyticAreaLight::create();

        glm::vec3 scaling(1, 1, 1);
        glm::vec3 translation(0, 0, 0);
        glm::vec3 rotation(0, 0, 0);

        for (auto it = jsonLight.MemberBegin(); it != jsonLight.MemberEnd(); it++)
        {
            std::string key(it->name.GetString());
            const auto& value = it->value;
            if (key == SceneKeys::kName)
            {
                if (value.IsString() == false)
                {
                    return error("Area light name should be a string");
                }
                std::string name = value.GetString();
                if (name.find(' ') != std::string::npos)
                {
                    return error("Area light name can't have spaces");
                }
                pAreaLight->setName(name);
            }
            else if (key == SceneKeys::kType)
            {
                if (value.IsString() == false)
                {
                    return error("Area light type should be a string");
                }

                std::string type = value.GetString();
                if (type == SceneKeys::kAreaLightRect)          pAreaLight->setType(LightAreaRect);
                else if (type == SceneKeys::kAreaLightSphere)   pAreaLight->setType(LightAreaSphere);
                else if (type == SceneKeys::kAreaLightDisc)     pAreaLight->setType(LightAreaDisc);
                else return error("Invalid area light type");
            }
            else if (key == SceneKeys::kLightIntensity)
            {
                glm::vec3 intensity;
                if (getFloatVec<3>(value, "Area light intensity", &intensity[0]) == false)
                {
                    return false;
                }
                pAreaLight->setIntensity(intensity);
            }
            else if (key == SceneKeys::kTranslationVec)
            {
                if (getFloatVec<3>(value, "Area light translation vector", &translation[0]) == false)
                {
                    return false;
                }
            }
            else if (key == SceneKeys::kScalingVec)
            {
                if (getFloatVec<3>(value, "Area light scale vector", &scaling[0]) == false)
                {
                    return false;
                }
            }
            else if (key == SceneKeys::kRotationVec)
            {
                if (getFloatVec<3>(value, "Area light rotation vector", &rotation[0]) == false)
                {
                    return false;
                }

                rotation = glm::radians(rotation);
            }
            else
            {
                return error("Invalid key found in area light object. Key == " + key + ".");
            }
        }

        // Set transform matrix for the light source
        pAreaLight->setScaling(scaling);
        glm::mat4 translationMtx = glm::translate(glm::mat4(), translation);
        glm::mat4 rotationMtx = glm::yawPitchRoll(rotation[0], rotation[1], rotation[2]);
        //glm::mat4 scalingMtx = glm::scale(glm::mat4(), scaling);
        glm::mat4 composite = translationMtx * rotationMtx;
        pAreaLight->setTransformMatrix(composite);

        if (isNameDuplicate(pAreaLight->getName(), mLightMap, "lights"))
        {
            return false;
        }
        else
        {
            mLightMap[pAreaLight->getName()] = pAreaLight;
            mScene.addLight(pAreaLight);
        }
        return true;
    }

    bool SceneImporter::parseLights(const rapidjson::Value& jsonVal)
    {
        if (jsonVal.IsArray() == false)
        {
            return error("lights section should be an array of objects.");
        }

        // Go over all the objects
        for (uint32_t i = 0; i < jsonVal.Size(); i++)
        {
            const auto& light = jsonVal[i];
            const auto& type = light.FindMember(SceneKeys::kType);
            if (type == light.MemberEnd())
            {
                return error("Light source must have a type.");
            }

            if (type->value.IsString() == false)
            {
                return error("Light source Type must be a string.");
            }

            std::string lightType(type->value.GetString());
            bool b;
            if (lightType == SceneKeys::kDirLight)
            {
                b = createDirLight(light);
            }
            else if (lightType == SceneKeys::kPointLight)
            {
                b = createPointLight(light);
            }
            else if (lightType == SceneKeys::kAreaLightRect || lightType == SceneKeys::kAreaLightSphere || lightType == SceneKeys::kAreaLightDisc)
            {
                b = createAnalyticAreaLight(light);
            }
            else
            {
                return error("Unrecognized light Type \"" + lightType + "\"");
            }

            if (b == false)
            {
                return false;
            }
        }

        return true;
    }

    bool SceneImporter::parseLightProbes(const rapidjson::Value& jsonVal)
    {
        if (jsonVal.IsArray() == false)
        {
            return error("Light probes should be an array of objects.");
        }

        for (uint32_t i = 0; i < jsonVal.Size(); i++)
        {
            const auto& lightProbe = jsonVal[i];

            if (lightProbe.HasMember(SceneKeys::kFilename) == false)
            {
                return error("An image file must be specified for a light probe.");
            }

            // Check if path is relative, if not, assume full path
            std::string imagePath = lightProbe[SceneKeys::kFilename].GetString();
            std::string actualPath = mDirectory + '/' + imagePath;
            if (doesFileExist(actualPath) == false)
            {
                actualPath = imagePath;
            }

            vec3 position;
            glm::vec3 intensity(1.0f);
            float radius = -1;
            uint32_t diffuseSamples = LightProbe::kDefaultDiffSamples;
            uint32_t specSamples = LightProbe::kDefaultSpecSamples;

            for (auto m = lightProbe.MemberBegin(); m < lightProbe.MemberEnd(); m++)
            {
                std::string key = m->name.GetString();
                const auto& value = m->value;
                if (key == SceneKeys::kLightIntensity)
                {
                    if (getFloatVec<3>(value, "Light probe intensity", &intensity[0]) == false)
                    {
                        return false;
                    }
                }
                else if (key == SceneKeys::kLightPos)
                {
                    if (getFloatVec<3>(value, "Light probe world position", &position[0]) == false)
                    {
                        return false;
                    }
                }
                else if (key == SceneKeys::kLightProbeRadius)
                {
                    if (value.IsUint() == false)
                    {
                        error("Light Probe radius must be a float.");
                        return false;
                    }
                    radius = float(value.GetDouble());
                }
                else if (key == SceneKeys::kLightProbeDiffSamples)
                {
                    if (value.IsUint() == false)
                    {
                        error("Light Probe diffuse sample count must be a uint.");
                        return false;
                    }
                    diffuseSamples = value.GetUint();
                }
                else if (key == SceneKeys::kLightProbeSpecSamples)
                {
                    if (value.IsUint() == false)
                    {
                        error("Light Probe specular sample count must be a uint.");
                        return false;
                    }
                    specSamples = value.GetUint();
                }
            }

            LightProbe::SharedPtr pLightProbe = LightProbe::create(gpDevice->getRenderContext().get(), actualPath, true, ResourceFormat::RGBA16Float, diffuseSamples, specSamples);
            pLightProbe->setPosW(position);
            pLightProbe->setIntensity(intensity);
            mScene.addLightProbe(pLightProbe);
        }

        return true;
    }

    bool SceneImporter::createPathFrames(ObjectPath* pPath, const rapidjson::Value& jsonFramesArray)
    {
        // an array of key frames
        if (jsonFramesArray.IsArray() == false)
        {
            return error("Camera path frames should be an array of key-frame objects");
        }

        for (uint32_t i = 0; i < jsonFramesArray.Size(); i++)
        {
            float time = 0;
            glm::vec3 pos, target, up;
            for (auto it = jsonFramesArray[i].MemberBegin(); it < jsonFramesArray[i].MemberEnd(); it++)
            {
                std::string key(it->name.GetString());
                auto& value = it->value;
                bool b = true;
                if (key == SceneKeys::kFrameTime)
                {
                    if (value.IsNumber() == false)
                    {
                        error("Camera path time should be a number");
                        b = false;
                    }

                    time = (float)value.GetDouble();
                }
                else if (key == SceneKeys::kCamPosition)
                {
                    b = getFloatVec<3>(value, "Camera path position", &pos[0]);
                }
                else if (key == SceneKeys::kCamTarget)
                {
                    b = getFloatVec<3>(value, "Camera path target", &target[0]);
                }
                else if (key == SceneKeys::kCamUp)
                {
                    b = getFloatVec<3>(value, "Camera path up vector", &up[0]);
                }

                if (b == false)
                {
                    return false;
                }
            }
            pPath->addKeyFrame(time, pos, target, up);
        }
        return true;
    }

    ObjectPath::SharedPtr SceneImporter::createPath(const rapidjson::Value& jsonPath)
    {
        auto pPath = ObjectPath::create();

        for (auto it = jsonPath.MemberBegin(); it != jsonPath.MemberEnd(); it++)
        {
            const std::string key(it->name.GetString());
            const auto& value = it->value;

            if (key == SceneKeys::kName)
            {
                if (value.IsString() == false)
                {
                    error("Path name should be a string");
                    return nullptr;
                }

                std::string pathName(value.GetString());
                pPath->setName(pathName);
            }
            else if (key == SceneKeys::kPathLoop)
            {
                if (value.IsBool() == false)
                {
                    error("Path loop should be a boolean value");
                    return nullptr;
                }

                bool b = value.GetBool();
                pPath->setAnimationRepeat(b);
            }
            else if (key == SceneKeys::kPathFrames)
            {
                if (createPathFrames(pPath.get(), value) == false)
                {
                    return nullptr;
                }
            }
            else if (key == SceneKeys::kAttachedObjects)
            {
                if (value.IsArray() == false)
                {
                    error("Path object list should be an array");
                    return nullptr;
                }

                for (uint32_t i = 0; i < value.Size(); i++)
                {
                    std::string type = value[i].FindMember(SceneKeys::kType)->value.GetString();
                    std::string name = value[i].FindMember(SceneKeys::kName)->value.GetString();

                    pPath->attachObject(getMovableObject(type, name));
                }
            }
            else
            {
                error("Unknown token \"" + key + "\" when parsing camera path");
                return nullptr;
            }
        }
        return pPath;
    }

    bool SceneImporter::parsePaths(const rapidjson::Value& jsonVal)
    {
        if (jsonVal.IsArray() == false)
        {
            return error("Paths should be an array");
        }

        for (uint32_t PathID = 0; PathID < jsonVal.Size(); PathID++)
        {
            auto pPath = createPath(jsonVal[PathID]);
            if (pPath)
            {
                mScene.addPath(pPath);
            }
            else
            {
                return false;
            }
        }
        return true;
    }

    bool SceneImporter::parseActivePath(const rapidjson::Value& jsonVal)
    {
        if (mScene.getVersion() != 1)
        {
            return true;
        }

        // Paths should already be initialized at this stage
        if (jsonVal.IsString() == false)
        {
            return error("Active path should be a string.");
        }

        std::string activePath = jsonVal.GetString();

        // Find the path
        for (uint32_t i = 0; i < mScene.getPathCount(); i++)
        {
            if (activePath == mScene.getPath(i)->getName())
            {
                mScene.getPath(i)->attachObject(mScene.getActiveCamera());
                return true;
            }
        }

        return error("Active path \"" + activePath + "\" not found.");
    }

    bool SceneImporter::createCamera(const rapidjson::Value& jsonCamera)
    {
        auto pCamera = Camera::create();
        std::string activePath;

        // Go over all the keys
        for (auto it = jsonCamera.MemberBegin(); it != jsonCamera.MemberEnd(); it++)
        {
            std::string key(it->name.GetString());
            const auto& value = it->value;
            if (key == SceneKeys::kName)
            {
                // Name
                if (value.IsString() == false)
                {
                    return error("Camera name should be a string value");
                }
                pCamera->setName(value.GetString());
            }
            else if (key == SceneKeys::kCamPosition)
            {
                glm::vec3 pos;
                if (getFloatVec<3>(value, "Camera's position", &pos[0]) == false)
                {
                    return false;
                }
                pCamera->setPosition(pos);
            }
            else if (key == SceneKeys::kCamTarget)
            {
                glm::vec3 target;
                if (getFloatVec<3>(value, "Camera's target", &target[0]) == false)
                {
                    return false;
                }
                pCamera->setTarget(target);
            }
            else if (key == SceneKeys::kCamUp)
            {
                glm::vec3 up;
                if (getFloatVec<3>(value, "Camera's up vector", &up[0]) == false)
                {
                    return false;
                }
                pCamera->setUpVector(up);
            }
            else if (key == SceneKeys::kCamFovY) // Version 1
            {
                if (mScene.getVersion() > 1)
                {
                    return error("Camera FOV is only valid in scene version 1. Ignoring value.");
                }

                if (value.IsNumber() == false)
                {
                    return error("Camera's FOV should be a number");
                }

                // Convert to radians
                float fovY = glm::radians((float)value.GetDouble());
                pCamera->setFocalLength(fovYToFocalLength(fovY, Camera::kDefaultFrameHeight));
            }
            else if (key == SceneKeys::kCamFocalLength) // Version 2
            {
                if (mScene.getVersion() != 2)
                {
                    return error("Camera focal length is only valid in scene version 2. Ignoring value.");
                }

                if (value.IsNumber() == false)
                {
                    return error("Camera's focal length should be a number");
                }

                pCamera->setFocalLength((float)value.GetDouble());
            }
            else if (key == SceneKeys::kCamDepthRange)
            {
                float depthRange[2];
                if (getFloatVec<2>(value, "Camera's depth-range", depthRange) == false)
                {
                    return false;
                }
                pCamera->setDepthRange(depthRange[0], depthRange[1]);
            }
            else if (key == SceneKeys::kCamAspectRatio)
            {
                if (value.IsNumber() == false)
                {
                    return error("Camera's aspect ratio should be a number");
                }
                pCamera->setAspectRatio((float)value.GetDouble());
            }
            else
            {
                return error("Invalid key found in cameras array. Key == " + key + ".");
            }
        }

        if (isNameDuplicate(pCamera->getName(), mCameraMap, "cameras"))
        {
            return false;
        }
        else
        {
            mCameraMap[pCamera->getName()] = pCamera;
            mScene.addCamera(pCamera);
        }

        return true;
    }

    bool SceneImporter::parseCameras(const rapidjson::Value& jsonVal)
    {
        if (jsonVal.IsArray() == false)
        {
            return error("cameras section should be an array of objects.");
        }

        // Go over all the objects
        for (uint32_t i = 0; i < jsonVal.Size(); i++)
        {
            if (createCamera(jsonVal[i]) == false)
            {
                return false;
            }
        }

        // Set the active camera to camera 0.
        mScene.setActiveCamera(0);

        return true;
    }

    bool SceneImporter::load(const std::string& filename, Model::LoadFlags modelLoadFlags, Scene::LoadFlags sceneLoadFlags)
    {
        std::string fullpath;
        mFilename = filename;
        mModelLoadFlags = modelLoadFlags;
        mSceneLoadFlags = sceneLoadFlags;

        if (is_set(mSceneLoadFlags, Scene::LoadFlags::GenerateAreaLights))
        {
            mModelLoadFlags |= Model::LoadFlags::BuffersAsShaderResource;
        }

        if (findFileInDataDirectories(filename, fullpath))
        {
            // Load the file
            std::string jsonData = readFile(fullpath);
            rapidjson::StringStream JStream(jsonData.c_str());

            // Get the file directory
            auto last = fullpath.find_last_of("/\\");
            mDirectory = fullpath.substr(0, last);

            // create the DOM
            mJDoc.ParseStream(JStream);

            if (mJDoc.HasParseError())
            {
                size_t line;
                line = std::count(jsonData.begin(), jsonData.begin() + mJDoc.GetErrorOffset(), '\n');
                return error(std::string("JSON Parse error in line ") + std::to_string(line) + ". " + rapidjson::GetParseError_En(mJDoc.GetParseError()));
            }

            if (topLevelLoop() == false)
            {
                return false;
            }

            if (is_set(mSceneLoadFlags, Scene::LoadFlags::GenerateAreaLights))
            {
                mScene.createAreaLights();
            }

            return true;
        }
        else
        {
            return error("File not found.");
        }
    }

    bool SceneImporter::parseAmbientIntensity(const rapidjson::Value& jsonVal)
    {
        logWarning("SceneImporter: Global ambient term is no longer supported. Ignoring value.");
        return true;
    }

    bool SceneImporter::parseLightingScale(const rapidjson::Value& jsonVal)
    {
        if (jsonVal.IsNumber() == false)
        {
            return error("Lighting scale should be a number.");
        }

        float f = (float)(jsonVal.GetDouble());
        mScene.setLightingScale(f);
        return true;
    }

    bool SceneImporter::parseCameraSpeed(const rapidjson::Value& jsonVal)
    {
        if (jsonVal.IsNumber() == false)
        {
            return error("Camera speed should be a number.");
        }

        float f = (float)(jsonVal.GetDouble());
        mScene.setCameraSpeed(f);
        return true;
    }

    bool SceneImporter::parseActiveCamera(const rapidjson::Value& jsonVal)
    {
        // Cameras should already be initialized at this stage
        if (jsonVal.IsString() == false)
        {
            return error("Active camera should be a string.");
        }

        std::string activeCamera = jsonVal.GetString();

        // Find the camera
        for (uint32_t i = 0; i < mScene.getCameraCount(); i++)
        {
            if (activeCamera == mScene.getCamera(i)->getName())
            {
                mScene.setActiveCamera(i);
                return true;
            }
        }

        return error("Active camera \"" + activeCamera + "\" not found.");
    }

    bool SceneImporter::parseVersion(const rapidjson::Value& jsonVal)
    {
        if (jsonVal.IsUint() == false)
        {
            return error("value should be an unsigned integer number");
        }
        mScene.setVersion(jsonVal.GetUint());
        return true;
    }

    bool SceneImporter::parseEnvMap(const rapidjson::Value& jsonVal)
    {
        if (mScene.getEnvironmentMap())
        {
            return error("Scene can't have more then one environment map");
        }
        
        if (jsonVal.IsString() == false)
        {
            return error(std::string(SceneKeys::kEnvMap) + " should be a string");
        }

        std::string filename = mDirectory + '/' + jsonVal.GetString();
        if (doesFileExist(filename) == false)
        {
            if (findFileInDataDirectories(jsonVal.GetString(), filename) == false)
            {
                return error("Can't find environment map file " + std::string(jsonVal.GetString()));
            }
        }

        auto pTex = createTextureFromFile(filename, false, true);
        mScene.setEnvironmentMap(pTex);
        return true;
    }

    bool SceneImporter::parseUserDefinedSection(const rapidjson::Value& jsonVal)
    {
        if (jsonVal.IsObject() == false)
        {
            return error("User defined section should be a JSON object.");
        }

        for (auto it = jsonVal.MemberBegin(); it != jsonVal.MemberEnd(); it++)
        {
            bool b;
            Scene::UserVariable userVar;
            std::string name(it->name.GetString());
            const auto& value = it->value;
            // Check if this is a vector
            if (value.IsArray())
            {
                for (uint32_t i = 0; i < value.Size(); i++)
                {
                    if (value[i].IsNumber() == false)
                    {
                        return error("User defined section contains an array, but some of the elements are not numbers.");
                    }
                }

                switch (value.Size())
                {
                case 2:
                    userVar.type = Scene::UserVariable::Type::Vec2;
                    b = getFloatVec<2>(value, "custom-field vec2", &userVar.vec2[0]);
                    break;
                case 3:
                    userVar.type = Scene::UserVariable::Type::Vec3;
                    b = getFloatVec<3>(value, "custom-field vec3", &userVar.vec3[0]);
                    break;
                case 4:
                    userVar.type = Scene::UserVariable::Type::Vec4;
                    b = getFloatVec<4>(value, "custom-field vec4", &userVar.vec4[0]);
                    break;
                default:
                    userVar.type = Scene::UserVariable::Type::Vector;
                    b = getFloatVecAnySize(value, "vector of floats", userVar.vector);
                    break;
                }
                if (b == false)
                {
                    return false;
                }
            }
            else
            {
                // Not an array. Must be a literal
                // The way rapidjson works, a uint is also an int, and a 32-bit number is also a 64-bit number, so the order in which we check the Type matters
                if (value.IsUint())
                {
                    userVar.type = Scene::UserVariable::Type::Uint;
                    userVar.u32 = value.GetUint();
                }
                else if (value.IsInt())
                {
                    userVar.type = Scene::UserVariable::Type::Int;
                    userVar.i32 = value.GetInt();
                }
                else if (value.IsUint64())
                {
                    userVar.type = Scene::UserVariable::Type::Uint64;
                    userVar.u64 = value.GetUint64();
                }
                else if (value.IsInt64())
                {
                    userVar.type = Scene::UserVariable::Type::Int64;
                    userVar.i64 = value.GetInt64();
                }
                else if (value.IsDouble())
                {
                    userVar.type = Scene::UserVariable::Type::Double;
                    userVar.d64 = value.GetDouble();
                }
                else if (value.IsString())
                {
                    userVar.type = Scene::UserVariable::Type::String;
                    userVar.str = value.GetString();
                }
                else if (value.IsBool())
                {
                    userVar.type = Scene::UserVariable::Type::Bool;
                    userVar.b = value.GetBool();
                }
                else
                {
                    return error("Error when parsing custom-field \"" + name + "\". Field Type invalid. Must be a literal number, string boolean or an array of 2/3/4 numbers.");
                }
            }
            mScene.addUserVariable(name, userVar);
        }
        return true;
    }

    bool SceneImporter::loadIncludeFile(const std::string& include)
    {
        // Find the file
        std::string fullpath = mDirectory + '/' + include;
        if (doesFileExist(fullpath) == false)
        {
            // Look in the data directories
            if (findFileInDataDirectories(include, fullpath) == false)
            {
                return error("Can't find include file " + include);
            }
        }

        Scene::SharedPtr pScene = Scene::create();
        SceneImporter::loadScene(*pScene, fullpath, mModelLoadFlags, mSceneLoadFlags);
        if (pScene == nullptr)
        {
            return false;
        }
        mScene.merge(pScene.get());

        return true;
    }

    bool SceneImporter::parseIncludes(const rapidjson::Value& jsonVal)
    {
        if (jsonVal.IsArray() == false)
        {
            return error("Include section should be an array of strings");
        }

        for (uint32_t i = 0; i < jsonVal.Size(); i++)
        {
            if (jsonVal[i].IsString() == false)
            {
                return error("Include element should be a string");
            }

            const std::string include = jsonVal[i].GetString();
            if (loadIncludeFile(include) == false)
            {
                return false;
            }
        }
        return true;
    }

    bool SceneImporter::isNameDuplicate(const std::string& name, const ObjectMap& objectMap, const std::string& objectType) const
    {
        if (objectMap.find(name) != objectMap.end())
        {
            const std::string msg = "Multiple " + objectType + " found the same name: " + name + ".\nObjects may not attach to paths correctly.\n\nContinue anyway?";

            // If user pressed ok, return false to ignore duplicate
            return msgBox(msg, MsgBoxType::OkCancel) == MsgBoxButton::Ok ? false : true;
        }

        return false;
    }

    IMovableObject::SharedPtr SceneImporter::getMovableObject(const std::string& type, const std::string& name) const
    {
        if (type == SceneKeys::kModelInstance)
        {
            return mInstanceMap.find(name)->second;
        }
        else if (type == SceneKeys::kCamera)
        {
            return mCameraMap.find(name)->second;
        }
        else if (type == SceneKeys::kLight)
        {
            return mLightMap.find(name)->second;
        }

        should_not_get_here();
        return nullptr;
    }

    const SceneImporter::FuncValue SceneImporter::kFunctionTable[] =
    {
        // The order matters here.
        {SceneKeys::kVersion, &SceneImporter::parseVersion},
        {SceneKeys::kEnvMap, &SceneImporter::parseEnvMap},
        {SceneKeys::kAmbientIntensity, &SceneImporter::parseAmbientIntensity},
        {SceneKeys::kLightingScale, &SceneImporter::parseLightingScale},
        {SceneKeys::kCameraSpeed, &SceneImporter::parseCameraSpeed},

        {SceneKeys::kModels, &SceneImporter::parseModels},
        {SceneKeys::kLights, &SceneImporter::parseLights},
        {SceneKeys::kLightProbes, &SceneImporter::parseLightProbes},
        {SceneKeys::kCameras, &SceneImporter::parseCameras},
        {SceneKeys::kActiveCamera, &SceneImporter::parseActiveCamera},  // Should come after ParseCameras
        {SceneKeys::kUserDefined, &SceneImporter::parseUserDefinedSection},

        {SceneKeys::kPaths, &SceneImporter::parsePaths},
        {SceneKeys::kActivePath, &SceneImporter::parseActivePath}, // Should come after ParsePaths
        {SceneKeys::kInclude, &SceneImporter::parseIncludes}
    };

    bool SceneImporter::validateSceneFile()
    {
        // Make sure the top-level is valid
        for (auto it = mJDoc.MemberBegin(); it != mJDoc.MemberEnd(); it++)
        {
            bool found = false;
            const std::string name(it->name.GetString());

            for (uint32_t i = 0; i < arraysize(kFunctionTable); i++)
            {
                // Check that we support this value
                if (kFunctionTable[i].token == name)
                {
                    found = true;
                    break;
                }
            }

            if (found == false)
            {
                return error("Invalid key found in top-level object. Key == " + std::string(it->name.GetString()) + ".");
            }
        }
        return true;
    }

    bool SceneImporter::topLevelLoop()
    {
        if (validateSceneFile() == false)
        {
            return false;
        }

        for (uint32_t i = 0; i < arraysize(kFunctionTable); i++)
        {
            const auto& jsonMember = mJDoc.FindMember(kFunctionTable[i].token.c_str());
            if (jsonMember != mJDoc.MemberEnd())
            {
                auto a = kFunctionTable[i].func;
                if ((this->*a)(jsonMember->value) == false)
                {
                    return false;
                }
            }
        }

        return true;
    }
}