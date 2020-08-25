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
#include "SceneImporter.h"
#include "rapidjson/document.h"
#include "rapidjson/error/en.h"
#include "Core/API/Device.h"
#include "glm/gtx/euler_angles.hpp"
#include "glm/gtx/transform.hpp"
#include <filesystem>

namespace Falcor
{
    namespace SceneKeys
    {

        static const char* kInclude = "include";

        // Not supported in exporter yet
        static const char* kLightProbes = "light_probes";
        static const char* kLightProbeRadius = "radius";
        static const char* kLightProbeDiffSamples = "diff_samples";
        static const char* kLightProbeSpecSamples = "spec_samples";

        // Keys for values in older scene versions that are not exported anymore
        static const char* kCamFovY = "fovY";
        static const char* kActivePath = "active_path";
        static const char* kAmbientIntensity = "ambient_intensity";
        static const char* kVersion = "version";
        static const char* kSceneUnit = "scene_unit";
        static const char* kCameraSpeed = "camera_speed";
        static const char* kActiveCamera = "active_camera";
        static const char* kLightingScale = "lighting_scale";

        static const char* kName = "name";
        static const char* kEnvMap = "env_map";

        static const char* kModels = "models";
        static const char* kFilename = "file";
        static const char* kModelInstances = "instances";
        static const char* kTranslationVec = "translation";
        static const char* kRotationVec = "rotation";
        static const char* kScalingVec = "scaling";
        static const char* kActiveAnimation = "active_animation";

        static const char* kCameras = "cameras";
        static const char* kCamPosition = "pos";
        static const char* kCamTarget = "target";
        static const char* kCamUp = "up";
        static const char* kCamFocalLength = "focal_length";
        static const char* kCamDepthRange = "depth_range";
        static const char* kCamAspectRatio = "aspect_ratio";
        static const char* kCamFocalDistance = "focal_distance";
        static const char* kCamApertureRadius = "aperture_radius";
        static const char* kCamSpeed = "speed";

        static const char* kLights = "lights";
        static const char* kType = "type";
        static const char* kDirLight = "dir_light";
        static const char* kDistantLight = "distant_light";
        static const char* kPointLight = "point_light";
        static const char* kAreaLightRect = "area_light_rect";
        static const char* kAreaLightSphere = "area_light_sphere";
        static const char* kAreaLightDisc = "area_light_disc";
        static const char* kLightIntensity = "intensity";
        static const char* kLightOpeningAngle = "opening_angle";
        static const char* kLightPenumbraAngle = "penumbra_angle";
        static const char* kLightPos = "pos";
        static const char* kLightDirection = "direction";

        static const char* kPaths = "paths";
        static const char* kAttachedObjects = "attached_objects";
        static const char* kModelInstance = "model_instance";
        static const char* kLight = "light";
        static const char* kCamera = "camera";

        static const char* kMaterial = "material";
        static const char* kShadingModel = "shading_model";
        static const char* kShadingSpecGloss = "spec_gloss";
        static const char* kShadingMetalRough = "metal_rough";

        static const char* kUserDefined = "user_defined";
    };

    class SceneImporterImpl
    {
    public:
        SceneImporterImpl(SceneBuilder& builder) : mBuilder(builder) {}
        bool load(const std::string& filename);

    private:
        bool parseVersion(const rapidjson::Value& jsonVal);
        bool parseSceneUnit(const rapidjson::Value& jsonVal);
        bool parseModels(const rapidjson::Value& jsonVal);
        bool parseLights(const rapidjson::Value& jsonVal);
        bool parseLightProbes(const rapidjson::Value& jsonVal);
        bool parseCameras(const rapidjson::Value& jsonVal);
        bool parseCamera(const rapidjson::Value& jsonVal);
        bool parseAmbientIntensity(const rapidjson::Value& jsonVal);
        bool parseActiveCamera(const rapidjson::Value& jsonVal);
        bool parseCameraSpeed(const rapidjson::Value& jsonVal);
        bool parseLightingScale(const rapidjson::Value& jsonVal);
        bool parsePaths(const rapidjson::Value& jsonVal);
        bool parseUserDefinedSection(const rapidjson::Value& jsonVal);
        bool parseActivePath(const rapidjson::Value& jsonVal);
        bool parseIncludes(const rapidjson::Value& jsonVal);
        bool parseEnvMap(const rapidjson::Value& jsonVal);

        bool topLevelLoop();

        bool loadIncludeFile(const std::string& Include);

        std::vector<glm::mat4> parseModelInstances(const rapidjson::Value& jsonVal);
        bool createModel(const rapidjson::Value& jsonModel);
        bool createPointLight(const rapidjson::Value& jsonLight);
        bool createDirLight(const rapidjson::Value& jsonLight);
        bool createDistantLight(const rapidjson::Value& jsonLight);
        bool createAnalyticAreaLight(const rapidjson::Value& jsonLight);

        bool error(const std::string& msg);

        template<uint32_t VecSize>
        bool getFloatVec(const rapidjson::Value& jsonVal, const std::string& desc, float vec[VecSize]);
        bool getFloatVecAnySize(const rapidjson::Value& jsonVal, const std::string& desc, std::vector<float>& vec);
        rapidjson::Document mJDoc;
        SceneBuilder& mBuilder;
        std::string mFilename;
        std::string mDirectory;

        using LightMap = std::map<std::string, Light::SharedPtr>;

        struct FuncValue
        {
            const std::string token;
            decltype(&SceneImporterImpl::parseModels) func;
        };

        static const FuncValue kFunctionTable[];
        bool validateSceneFile();
    };

    bool SceneImporterImpl::error(const std::string& msg)
    {
        logError("Error when parsing scene file '" + mFilename + "'.\n" + msg);
        return false;
    }

    template<uint32_t VecSize>
    bool SceneImporterImpl::getFloatVec(const rapidjson::Value& jsonVal, const std::string& desc, float vec[VecSize])
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

    bool SceneImporterImpl::getFloatVecAnySize(const rapidjson::Value& jsonVal, const std::string& desc, std::vector<float>& vec)
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

    std::vector<glm::mat4> SceneImporterImpl::parseModelInstances(const rapidjson::Value& jsonVal)
    {
        struct ModelInstance
        {
            float3 scaling = float3(1, 1, 1);
            float3 position = float3(0, 0, 0);
            float3 rotation = float3(0, 0, 0);
        };

        std::vector<ModelInstance> instances;

        if (jsonVal.IsArray() == false)
        {
            logError("Model instances should be an array of objects");
            return {};
        }

        for (uint32_t i = 0; i < jsonVal.Size(); i++)
        {
            const auto& jsonInstance = jsonVal[i];
            ModelInstance instance;

            for (auto m = jsonInstance.MemberBegin(); m < jsonInstance.MemberEnd(); m++)
            {
                std::string key(m->name.GetString());
                if (key == SceneKeys::kName) continue;
                else if (key == SceneKeys::kTranslationVec) getFloatVec<3>(m->value, "Model instance translation vector", &instance.position[0]);
                else if (key == SceneKeys::kScalingVec)     getFloatVec<3>(m->value, "Model instance scale vector", &instance.scaling[0]);
                else if (key == SceneKeys::kRotationVec)
                {
                    if (getFloatVec<3>(m->value, "Model instance rotation vector", &instance.rotation[0]))
                    {
                        instance.rotation = glm::radians(instance.rotation);
                    }
                }
                else logError("Unknown key '" + key + "' when parsing model instance");
            }

            instances.push_back(instance);
        }

        std::vector<glm::mat4> matrices(instances.size());
        for (size_t i = 0; i < matrices.size(); i++)
        {
            glm::mat4 T;
            T[3] = float4(instances[i].position, 1);
            glm::mat4 S = glm::scale(instances[i].scaling);
            glm::mat4 R = glm::yawPitchRoll(instances[i].rotation[0], instances[i].rotation[1], instances[i].rotation[2]);
            matrices[i] = T * R * S;
        }

        return matrices;
    }

    bool SceneImporterImpl::createModel(const rapidjson::Value& jsonModel)
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
        SceneBuilder::Flags buildFlags = mBuilder.getFlags();
        if (jsonModel.HasMember(SceneKeys::kMaterial))
        {
            const auto& materialSettings = jsonModel[SceneKeys::kMaterial];
            if (materialSettings.IsObject() == false)
            {
                return error("Material properties for '" + file + "' must be a JSON object");
            }

            for (auto m = materialSettings.MemberBegin(); m != materialSettings.MemberEnd(); m++)
            {
                if (m->name == SceneKeys::kShadingModel)
                {
                    logWarning("Model material key '" + std::string(SceneKeys::kShadingModel) + "' is not supported. Use the scene build flags.");
                }
            }
        }

        std::vector<glm::mat4> instances;

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
            }
            else if (keyName == SceneKeys::kModelInstances)
            {
                instances = parseModelInstances(jval->value);
            }
            else if (keyName == SceneKeys::kActiveAnimation)
            {
                logWarning("Model key '" + std::string(SceneKeys::kActiveAnimation) + "' is not supported.");
// #SCENEV2
//                 if (jval->value.IsUint() == false)
//                 {
//                     return error("Model active animation should be an unsigned integer");
//                 }
//                 uint32_t activeAnimation = jval->value.GetUint();
//                 if (activeAnimation >= pModel->getAnimationsCount())
//                 {
//                     std::string msg = "Warning when parsing scene file '" + mFilename + "'.\nModel " + pModel->getName() + " was specified with active animation " + std::to_string(activeAnimation);
//                     msg += ", but model only has " + std::to_string(pModel->getAnimationsCount()) + " animations. Ignoring field";
//                     logWarning(msg);
//                 }
//                 else
//                 {
//                     pModel->setActiveAnimation(activeAnimation);
//                 }
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

        assert(std::filesystem::path(file).extension() != ".fscene"); // #SCENE this will cause an endless recursion. We may want to fix it
        mBuilder.import(file.c_str(), instances);

        return true;
    }

    bool SceneImporterImpl::parseModels(const rapidjson::Value& jsonVal)
    {
        if (jsonVal.IsArray() == false)
        {
            return error("Models section should be an array of objects.");
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

    bool SceneImporterImpl::createDirLight(const rapidjson::Value& jsonLight)
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
                    return error("Directional light name should be a string");
                }
                std::string name = value.GetString();
                if (name.find(' ') != std::string::npos)
                {
                    return error("Directional light name can't have spaces");
                }
                pDirLight->setName(name);
            }
            else if (key == SceneKeys::kType)
            {
                // Don't care
            }
            else if (key == SceneKeys::kLightIntensity)
            {
                float3 intensity;
                if (getFloatVec<3>(value, "Directional light intensity", &intensity[0]) == false)
                {
                    return false;
                }
                pDirLight->setIntensity(intensity);
            }
            else if (key == SceneKeys::kLightDirection)
            {
                float3 direction;
                if (getFloatVec<3>(value, "Directional light direction", &direction[0]) == false)
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

        mBuilder.addLight(pDirLight);
        return true;
    }

    bool SceneImporterImpl::createDistantLight(const rapidjson::Value& jsonLight)
    {
        auto pDistLight = DistantLight::create();

        for (auto it = jsonLight.MemberBegin(); it != jsonLight.MemberEnd(); it++)
        {
            std::string key(it->name.GetString());
            const auto& value = it->value;
            if (key == SceneKeys::kName)
            {
                if (value.IsString() == false)
                {
                    return error("Distant light name should be a string");
                }
                std::string name = value.GetString();
                if (name.find(' ') != std::string::npos)
                {
                    return error("Distant light name can't have spaces");
                }
                pDistLight->setName(name);
            }
            else if (key == SceneKeys::kType)
            {
                // Don't care
            }
            else if (key == SceneKeys::kLightIntensity)
            {
                float3 intensity;
                if (getFloatVec<3>(value, "Distant light intensity", &intensity[0]) == false)
                {
                    return false;
                }
                pDistLight->setIntensity(intensity);
            }
            else if (key == SceneKeys::kLightDirection)
            {
                float3 direction;
                if (getFloatVec<3>(value, "Distant light direction", &direction[0]) == false)
                {
                    return false;
                }
                pDistLight->setWorldDirection(direction);
            }
            else
            {
                return error("Invalid key found in distant light object. Key == " + key + ".");
            }
        }

        mBuilder.addLight(pDistLight);
        return true;
    }

    bool SceneImporterImpl::createPointLight(const rapidjson::Value& jsonLight)
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
                    return error("Point light name should be a string");
                }
                std::string name = value.GetString();
                if (name.find(' ') != std::string::npos)
                {
                    return error("Point light name can't have spaces");
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
                    return error("Point light's FOV should be a number");
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
                    return error("Point light's FOV should be a number");
                }
                float angle = (float)value.GetDouble();
                // Convert to radiance
                angle = glm::radians(angle);
                pPointLight->setPenumbraAngle(angle);
            }
            else if (key == SceneKeys::kLightIntensity)
            {
                float3 intensity;
                if (getFloatVec<3>(value, "Point light intensity", &intensity[0]) == false)
                {
                    return false;
                }
                pPointLight->setIntensity(intensity);
            }
            else if (key == SceneKeys::kLightPos)
            {
                float3 position;
                if (getFloatVec<3>(value, "Point light position", &position[0]) == false)
                {
                    return false;
                }
                pPointLight->setWorldPosition(position);
            }
            else if (key == SceneKeys::kLightDirection)
            {
                float3 dir;
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

        mBuilder.addLight(pPointLight);

        return true;
    }

    // Support for analytic area lights
    bool SceneImporterImpl::createAnalyticAreaLight(const rapidjson::Value& jsonLight)
    {
        // Get the type of area light.
        auto typeKey = jsonLight.FindMember(SceneKeys::kType);
        if (typeKey == jsonLight.MemberEnd() || !typeKey->value.IsString()) error("Area light missing/invalid '" + std::string(SceneKeys::kType) + "' key");

        LightType type;
        if (typeKey->value.GetString() == SceneKeys::kAreaLightRect) type = LightType::Rect;
        else if (typeKey->value.GetString() == SceneKeys::kAreaLightSphere) type = LightType::Sphere;
        else if (typeKey->value.GetString() == SceneKeys::kAreaLightDisc) type = LightType::Disc;
        else return error("Invalid area light type");

        // Create the light.
        auto pAreaLight = AnalyticAreaLight::create(type);

        float3 scaling(1, 1, 1);
        float3 translation(0, 0, 0);
        float3 rotation(0, 0, 0);

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
                // Already parsed
            }
            else if (key == SceneKeys::kLightIntensity)
            {
                float3 intensity;
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

        mBuilder.addLight(pAreaLight);
        return true;
    }

    bool SceneImporterImpl::parseLights(const rapidjson::Value& jsonVal)
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
            else if (lightType == SceneKeys::kDistantLight)
            {
                b = createDistantLight(light);
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
                return error("Unrecognized light Type '" + lightType + "'");
            }

            if (b == false)
            {
                return false;
            }
        }

        return true;
    }

    bool SceneImporterImpl::parseLightProbes(const rapidjson::Value& jsonVal)
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

            float3 position;
            float3 intensity(1.0f);
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

            LightProbe::SharedPtr pLightProbe = LightProbe::create(gpDevice->getRenderContext(), actualPath, true, ResourceFormat::RGBA16Float, diffuseSamples, specSamples);
            pLightProbe->setPosW(position);
            pLightProbe->setIntensity(intensity);
            mBuilder.setLightProbe(pLightProbe);
        }

        return true;
    }

    bool SceneImporterImpl::parsePaths(const rapidjson::Value& jsonVal)
    {
        if (jsonVal.IsArray() == false)
        {
            return error("Paths should be an array");
        }

        logWarning("fscene paths are deprecated, please use Maya or other DCC tools to create a path directly in the model file");
        return true;
    }

    bool SceneImporterImpl::parseActivePath(const rapidjson::Value& jsonVal)
    {
        logWarning("fscene paths are deprecated, please use Maya or other DCC tools to create a path directly in the model file");
        return true;
    }

    bool SceneImporterImpl::parseCamera(const rapidjson::Value& jsonCamera)
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
            else if (key == SceneKeys::kCamSpeed)
            {
                if (value.IsNumber() == false)
                {
                    return error("Camera speed should be a number.");
                }

                float f = (float)(value.GetDouble());
                mBuilder.setCameraSpeed(f);
            }
            else if (key == SceneKeys::kCamPosition)
            {
                float3 pos;
                if (getFloatVec<3>(value, "Camera's position", &pos[0]) == false)
                {
                    return false;
                }
                pCamera->setPosition(pos);
            }
            else if (key == SceneKeys::kCamTarget)
            {
                float3 target;
                if (getFloatVec<3>(value, "Camera's target", &target[0]) == false)
                {
                    return false;
                }
                pCamera->setTarget(target);
            }
            else if (key == SceneKeys::kCamUp)
            {
                float3 up;
                if (getFloatVec<3>(value, "Camera's up vector", &up[0]) == false)
                {
                    return false;
                }
                pCamera->setUpVector(up);
            }
            else if (key == SceneKeys::kCamFocalLength) // Version 2
            {
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
            else if (key == SceneKeys::kCamFocalDistance)
            {
                if (value.IsNumber() == false)
                {
                    return error("Camera's focal distance should be a number");
                }
                pCamera->setFocalDistance((float)value.GetDouble());
            }
            else if (key == SceneKeys::kCamApertureRadius)
            {
                if (value.IsNumber() == false)
                {
                    return error("Camera's aperture radius should be a number");
                }
                pCamera->setApertureRadius((float)value.GetDouble());
            }
            else
            {
                return error("Invalid key found in cameras array. Key == " + key + ".");
            }
        }

        mBuilder.addCamera(pCamera);
        return true;
    }

    bool SceneImporterImpl::parseCameras(const rapidjson::Value& jsonVal)
    {
        if (jsonVal.IsArray() == false)
        {
            return error("Cameras section should be an array. If you want to use a single camera you can rename the section to 'camera'");
        }

        bool success = true;
        for (uint i = 0; i < jsonVal.Size(); i++)
        {
            success = parseCamera(jsonVal[0]) && success;
        }
        return success;
    }

    bool SceneImporterImpl::load(const std::string& filename)
    {
        std::string fullpath;
        mFilename = filename;

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

            return true;
        }
        else
        {
            return error("File not found.");
        }
    }

    bool SceneImporterImpl::parseAmbientIntensity(const rapidjson::Value& jsonVal)
    {
        logWarning("SceneImporterImpl: Global ambient term is no longer supported. Ignoring value.");
        return true;
    }

    bool SceneImporterImpl::parseLightingScale(const rapidjson::Value& jsonVal)
    {
        if (jsonVal.IsNumber() == false)
        {
            return error("Lighting scale should be a number.");
        }

        logWarning("Lighting scale is no longer supported");
        return true;
    }

    bool SceneImporterImpl::parseCameraSpeed(const rapidjson::Value& jsonVal)
    {
        if (jsonVal.IsNumber() == false)
        {
            return error("Camera speed should be a number.");
        }

        float f = (float)(jsonVal.GetDouble());
        mBuilder.setCameraSpeed(f);
        return true;
    }

    bool SceneImporterImpl::parseActiveCamera(const rapidjson::Value& jsonVal)
    {
        if (jsonVal.IsString() == false)
        {
            return error("Selected camera should be a name.");
        }

        std::string s = (std::string)(jsonVal.GetString());
        mBuilder.setCamera(s);
        return true;
    }

    bool SceneImporterImpl::parseVersion(const rapidjson::Value& jsonVal)
    {
        // Ignore this
        return true;
    }

    bool SceneImporterImpl::parseSceneUnit(const rapidjson::Value& jsonVal)
    {
        if (jsonVal.IsNumber() == false)
        {
            return error("Scene unit should be a number.");
        }

        logWarning("Scene unit is no longer supported");
        return true;
    }

    bool SceneImporterImpl::parseEnvMap(const rapidjson::Value& jsonVal)
    {
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

        mBuilder.setEnvMap(EnvMap::create(filename));
        return true;
    }

    bool SceneImporterImpl::parseUserDefinedSection(const rapidjson::Value& jsonVal)
    {
        if (jsonVal.IsObject() == false)
        {
            return error("User defined section should be a JSON object.");
        }

        logWarning("User defined section is no longer supported");
        return true;
    }

    bool SceneImporterImpl::loadIncludeFile(const std::string& include)
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
        return load(fullpath);
    }

    bool SceneImporterImpl::parseIncludes(const rapidjson::Value& jsonVal)
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

    const SceneImporterImpl::FuncValue SceneImporterImpl::kFunctionTable[] =
    {
        // The order matters here.
        {SceneKeys::kVersion, &SceneImporterImpl::parseVersion},
        {SceneKeys::kSceneUnit, &SceneImporterImpl::parseSceneUnit},
        {SceneKeys::kEnvMap, &SceneImporterImpl::parseEnvMap},
        {SceneKeys::kAmbientIntensity, &SceneImporterImpl::parseAmbientIntensity},
        {SceneKeys::kLightingScale, &SceneImporterImpl::parseLightingScale},
        {SceneKeys::kCameraSpeed, &SceneImporterImpl::parseCameraSpeed},

        {SceneKeys::kModels, &SceneImporterImpl::parseModels},
        {SceneKeys::kLights, &SceneImporterImpl::parseLights},
        {SceneKeys::kLightProbes, &SceneImporterImpl::parseLightProbes},
        {SceneKeys::kCameras, &SceneImporterImpl::parseCameras},
        {SceneKeys::kCamera, &SceneImporterImpl::parseCamera},
        {SceneKeys::kActiveCamera, &SceneImporterImpl::parseActiveCamera},  // Should come after ParseCameras
        {SceneKeys::kUserDefined, &SceneImporterImpl::parseUserDefinedSection},

        {SceneKeys::kPaths, &SceneImporterImpl::parsePaths},
        {SceneKeys::kActivePath, &SceneImporterImpl::parseActivePath}, // Should come after ParsePaths
        {SceneKeys::kInclude, &SceneImporterImpl::parseIncludes}
    };

    bool SceneImporterImpl::validateSceneFile()
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

    bool SceneImporterImpl::topLevelLoop()
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

    bool SceneImporter::import(const std::string& filename, SceneBuilder& builder, const SceneBuilder::InstanceMatrices& instances, const Dictionary& dict)
    {
        logWarning("fscene files are no longer supported in Falcor 4.0. Some properties may not be loaded.");
        if (!instances.empty()) logWarning("Scene importer does not support instancing.");

        SceneImporterImpl importer(builder);
        return importer.load(filename);
    }

    REGISTER_IMPORTER(
        SceneImporter,
        Importer::ExtensionList({
            "fscene"
        })
    )
}
