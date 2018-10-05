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
#include <vector>
#include <map>
#include "Graphics/Model/Model.h"
#include "Graphics/Light.h"
#include "Graphics/LightProbe.h"
#include "Graphics/Camera/Camera.h"
#include "Graphics/Camera/CameraController.h"
#include "Graphics/Paths/ObjectPath.h"
#include "Graphics/Model/ObjectInstance.h"
#include "Graphics/Model/SkinningCache.h"

namespace Falcor
{
    class Scene : public std::enable_shared_from_this<Scene>
    {
    public:
        using SharedPtr = std::shared_ptr<Scene>;
        using SharedConstPtr = std::shared_ptr<const Scene>;
        static const char* kFileFormatString;

        struct UserVariable
        {
            enum class Type
            {
                Unknown,    // Indicates an invalid/uninitialized variable
                Int,
                Uint,
                Int64,
                Uint64,
                Double,
                String,
                Vec2,
                Vec3,
                Vec4,
                Bool,
                Vector,
            };

            Type type = Type::Unknown;
            union
            {
                int32_t  i32;
                uint32_t u32;
                int64_t  i64;
                uint64_t u64;
                double   d64;
                bool     b;
            };
            std::string str;
            glm::vec2 vec2;
            glm::vec3 vec3;
            glm::vec4 vec4;
            std::vector<float> vector;

            UserVariable() { }
            UserVariable(const uint32_t&     v) : u32(v), type(Type::Uint) { }
            UserVariable(const int32_t&     v) : i32(v), type(Type::Int) { }
            UserVariable(const float&       v) : d64((double)v), type(Type::Double) { }
            UserVariable(const glm::vec2&   v) : vec2(v), type(Type::Vec2) { }
            UserVariable(const glm::vec3&   v) : vec3(v), type(Type::Vec3) { }
            UserVariable(const std::string& s) : str(s), type(Type::String) { }
        };

        using ModelInstance = ObjectInstance<Model>;
        using ModelInstanceList = std::vector<ModelInstance::SharedPtr>;

        /**
            Enum to generate light source(s)
        */
        enum class LoadFlags
        {
            None = 0x0,
            GenerateAreaLights = 0x1,    ///< Create area light(s) for meshes that have emissive material
        };

        static Scene::SharedPtr loadFromFile(const std::string& filename, Model::LoadFlags modelLoadFlags = Model::LoadFlags::None, Scene::LoadFlags sceneLoadFlags = LoadFlags::None);
        static Scene::SharedPtr create();

        virtual ~Scene();

        // Models
        uint32_t getModelCount() const { return (uint32_t)mModels.size(); }
        const Model::SharedPtr& getModel(uint32_t modelID) const { return mModels[modelID][0]->getObject(); };
        void deleteModel(uint32_t modelID);
        void deleteAllModels();

        // Model Instances
        virtual void addModelInstance(const ModelInstance::SharedPtr& pInstance);
        void addModelInstance(const Model::SharedPtr& pModel, const std::string& instanceName, const glm::vec3& translation = glm::vec3(), const glm::vec3& yawPitchRoll = glm::vec3(), const glm::vec3& scaling = glm::vec3(1));
        // Adds a model instance and shares ownership of it
        uint32_t getModelInstanceCount(uint32_t modelID) const;
        const ModelInstance::SharedPtr& getModelInstance(uint32_t modelID, uint32_t instanceID) const { return mModels[modelID][instanceID]; };
        void deleteModelInstance(uint32_t modelID, uint32_t instanceID);

        // Light Sources
        uint32_t addLight(const Light::SharedPtr& pLight);
        void deleteLight(uint32_t lightID);
        uint32_t getLightCount() const { return (uint32_t)mpLights.size(); }
        const Light::SharedPtr& getLight(uint32_t index) const { return mpLights[index]; }
        const std::vector<Light::SharedPtr>& getLights() const { return mpLights; }

        // Light Probes
        uint32_t addLightProbe(const LightProbe::SharedPtr& pLightProbe);
        void deleteLightProbe(uint32_t lightID);
        uint32_t getLightProbeCount() const { return (uint32_t)mpLightProbes.size(); }
        const LightProbe::SharedPtr& getLightProbe(uint32_t index) const { return mpLightProbes[index]; }
        const std::vector<LightProbe::SharedPtr>& getLightProbes() const { return mpLightProbes; }

        // Area Lights
        uint32_t addAreaLight(const AreaLight::SharedPtr& pAreaLight);
        void deleteAreaLight(uint32_t lightID);
        uint32_t getAreaLightCount() const { return (uint32_t)mpAreaLights.size(); }
        const AreaLight::SharedPtr& getAreaLight(uint32_t index) const { return mpAreaLights[index]; }
        const std::vector<AreaLight::SharedPtr>& getAreaLights() const { return mpAreaLights; }

        float getLightingScale() const { return mLightingScale; }
        void setLightingScale(float lightingScale) { mLightingScale = lightingScale; }

        // Object Paths
        uint32_t addPath(const ObjectPath::SharedPtr& pPath);
        void deletePath(uint32_t pathID);

        const ObjectPath::SharedPtr& getPath(uint32_t pathID) const { return mpPaths[pathID]; }
        uint32_t getPathCount() const { return (uint32_t)mpPaths.size(); }

        // Camera
        uint32_t addCamera(const Camera::SharedPtr& pCamera);
        void deleteCamera(uint32_t cameraID);

        uint32_t getCameraCount() const { return (uint32_t)mCameras.size(); }
        const Camera::SharedPtr getCamera(uint32_t index) const { return index < getCameraCount() ? mCameras[index] : nullptr; }
        const Camera::SharedPtr getActiveCamera() const { return getCamera(mActiveCameraID); }
        uint32_t getActiveCameraIndex() const { return mActiveCameraID; }

        void setActiveCamera(uint32_t camID);
        float getCameraSpeed() const { return mCameraSpeed; }
        void setCameraSpeed(float speed) { mCameraSpeed = speed; }

        // Camera update
        virtual bool update(double currentTime, CameraController* cameraController = nullptr);

        // User variables
        uint32_t getVersion() const { return mVersion; }
        void setVersion(uint32_t version) { mVersion = version; }
        void addUserVariable(const std::string& name, const UserVariable& var) { mUserVars[name] = var; }

        // If the name is not found, returns an invalid var (Type == Unknown)
        const UserVariable& getUserVariable(const std::string& name) const;
        const UserVariable& getUserVariable(uint32_t varID, std::string& varName) const;
        uint32_t getUserVariableCount() const { return (uint32_t)mUserVars.size(); }

        const uint32_t getId() const { return mId; }

        static const uint32_t kNoPath = (uint32_t)-1;

        void merge(const Scene* pFrom);

        /** Set scene unit.
            \param[in] sceneUnit Size of scene unit in meters.
        */
        void setSceneUnit(float sceneUnit) { mSceneUnit = sceneUnit; }

        /** Get scene unit.
            \return Size of scene unit in meters.
        */
        float getSceneUnit() const { return mSceneUnit; }

        /**
            Return scene extents in scene units
        */
        const vec3& getCenter() { updateExtents(); return mCenter; }
        const float getRadius() { updateExtents(); return mRadius; }

        /**
            This routine creates area light(s) in the scene. All meshes that
            have emissive material are treated as area lights.
        */
        void createAreaLights();

        /** Bind a sampler to all the materials in the scene
        */
        void bindSampler(Sampler::SharedPtr pSampler);

        /** Attach skinning cache to all models in scene.
        */
        void attachSkinningCacheToModels(SkinningCache::SharedPtr pSkinningCache);

        /** Set an environment-map texture
        */
        void setEnvironmentMap(const Texture::SharedPtr& pMap) { mpEnvMap = pMap; }

        /** Get the env-map texture
        */
        const Texture::SharedPtr& getEnvironmentMap() const { return mpEnvMap; }

        /** Get the filename
        */
        const std::string& getFilename() const { return mFilename; }

        /** Set a new aspect ratio for all the cameras in the scene
        */
        void setCamerasAspectRatio(float ratio);
    protected:

        Scene();
        /**
            Update changed scene extents (radius and center).
        */
        void updateExtents();

        static uint32_t sSceneCounter;

        uint32_t mId;

        std::vector<ModelInstanceList> mModels;
        std::vector<Light::SharedPtr> mpLights;
        std::vector<Camera::SharedPtr> mCameras;
        std::vector<ObjectPath::SharedPtr> mpPaths;
        std::vector<LightProbe::SharedPtr> mpLightProbes;
        std::vector<AreaLight::SharedPtr> mpAreaLights;
        Texture::SharedPtr mpEnvMap;

        uint32_t mActiveCameraID = 0;
        float mCameraSpeed = 1;
        float mLightingScale = 1.0f;
        uint32_t mVersion = 1;
        float mSceneUnit = 1.0f;            ///< Scene unit in meters (default 1 unit = 1 m)

        float mRadius = -1.f;
        vec3 mCenter = vec3(0, 0, 0);

        bool mExtentsDirty = true;

        std::string mFilename;

        using string_uservar_map = std::map<const std::string, UserVariable>;
        string_uservar_map mUserVars;
        static const UserVariable kInvalidVar;
    };

    enum_class_operators(Scene::LoadFlags);

#define flag_str(a) case Scene::LoadFlags::a: return #a
    inline std::string to_string(Scene::LoadFlags f)
    {
        switch (f)
        {
            flag_str(None);
            flag_str(GenerateAreaLights);
        default:
            should_not_get_here();
            return "";
        }
    }
#undef flag_str
}