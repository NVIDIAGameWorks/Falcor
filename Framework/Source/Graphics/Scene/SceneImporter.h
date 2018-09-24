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
#include <string>
#include "rapidjson/document.h"
#include "Graphics/Material/Material.h"
#include "glm/vec2.hpp"
#include "glm/vec3.hpp"
#include "glm/vec4.hpp"
#include "Scene.h"

namespace Falcor
{
    class SceneImporter
    {
    public:
        static bool loadScene(Scene& scene, const std::string& filename, Model::LoadFlags modelLoadFlags, Scene::LoadFlags sceneLoadFlags);

    private:

        SceneImporter(Scene& scene) : mScene(scene) {}
        bool load(const std::string& filename, Model::LoadFlags modelLoadFlags, Scene::LoadFlags sceneLoadFlags);

        bool parseVersion(const rapidjson::Value& jsonVal);
        bool parseModels(const rapidjson::Value& jsonVal);
        bool parseLights(const rapidjson::Value& jsonVal);
        bool parseLightProbes(const rapidjson::Value& jsonVal);
        bool parseCameras(const rapidjson::Value& jsonVal);
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

        bool createModel(const rapidjson::Value& jsonModel);
        bool createModelInstances(const rapidjson::Value& jsonVal, const Model::SharedPtr& pModel);
        bool createPointLight(const rapidjson::Value& jsonLight);
        bool createDirLight(const rapidjson::Value& jsonLight);
        bool createAnalyticAreaLight(const rapidjson::Value& jsonLight);
        ObjectPath::SharedPtr createPath(const rapidjson::Value& jsonPath);
        bool createPathFrames(ObjectPath* pPath, const rapidjson::Value& jsonFramesArray);
        bool createCamera(const rapidjson::Value& jsonCamera);

        bool error(const std::string& msg);

        template<uint32_t VecSize>
        bool getFloatVec(const rapidjson::Value& jsonVal, const std::string& desc, float vec[VecSize]);
        bool getFloatVecAnySize(const rapidjson::Value& jsonVal, const std::string& desc, std::vector<float>& vec);
        rapidjson::Document mJDoc;
        Scene& mScene;
        std::string mFilename;
        std::string mDirectory;
        Model::LoadFlags mModelLoadFlags;
        Scene::LoadFlags mSceneLoadFlags;

        using ObjectMap = std::map<std::string, IMovableObject::SharedPtr>;
        bool isNameDuplicate(const std::string& name, const ObjectMap& objectMap, const std::string& objectType) const;
        IMovableObject::SharedPtr getMovableObject(const std::string& type, const std::string& name) const;

        ObjectMap mInstanceMap;
        ObjectMap mCameraMap;
        ObjectMap mLightMap;

        struct FuncValue
        {
            const std::string token;
            decltype(&SceneImporter::parseModels) func;
        };

        static const FuncValue kFunctionTable[];
        bool validateSceneFile();
    };
}