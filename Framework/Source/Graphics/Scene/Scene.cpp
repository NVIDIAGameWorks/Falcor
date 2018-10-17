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
#include "Scene.h"
#include "SceneImporter.h"
#include "glm/gtx/euler_angles.hpp"
#include "glm/gtc/matrix_transform.hpp"

namespace Falcor
{
    uint32_t Scene::sSceneCounter = 0;

    const Scene::UserVariable Scene::kInvalidVar;

    const char* Scene::kFileFormatString = "Scene files\0*.fscene\0\0";

    Scene::SharedPtr Scene::loadFromFile(const std::string& filename, Model::LoadFlags modelLoadFlags, Scene::LoadFlags sceneLoadFlags)
    {
        Scene::SharedPtr pScene = create(filename);
        if (SceneImporter::loadScene(*pScene, filename, modelLoadFlags, sceneLoadFlags) == false)
        {
            pScene = nullptr;
        }
        return pScene;
    }

    Scene::SharedPtr Scene::create(const std::string& filename)
    {
        return SharedPtr(new Scene(filename));
    }

    Scene::Scene(const std::string& filename) : mId(sSceneCounter++), mFilename(filename)
    {
        // Reset all global id counters recursively
        Model::resetGlobalIdCounter();
    }

    Scene::~Scene() = default;

    void Scene::updateExtents()
    {
        if (mExtentsDirty)
        {
            mExtentsDirty = false;

            BoundingBox sceneAABB;
            bool first = true;
            for (uint32_t i = 0; i < getModelCount(); ++i)
            {
                for (uint32_t j = 0; j < getModelInstanceCount(i); ++j)
                {
                    const auto& pInst = getModelInstance(i, j);
                    if (first)
                    {
                        sceneAABB = pInst->getBoundingBox();
                        first = false;
                    }
                    else
                    {
                        sceneAABB = BoundingBox::fromUnion(sceneAABB, pInst->getBoundingBox());
                    }
                }
            }

            mCenter = sceneAABB.center;
            mRadius = length(sceneAABB.extent) * 0.5f;

            // Update light extents
            for (auto& pLight : mpLights)
            {
                if (pLight->getType() == LightDirectional)
                {
                    auto pDirLight = std::dynamic_pointer_cast<DirectionalLight>(pLight);
                    pDirLight->setWorldParams(getCenter(), getRadius());
                }
            }
        }
    }

    bool Scene::update(double currentTime, CameraController* cameraController)
    {
        bool changed = false;
        for (auto& path : mpPaths)
        {
            if (path->animate(currentTime))
            {
                changed = true;
            }
        }

        for (uint32_t i = 0; i < mModels.size(); i++)
        {
            if (mModels[i][0]->getObject()->animate(currentTime))
            {
                changed = true;
            }
        }

        mExtentsDirty = mExtentsDirty || changed;

        if (getCameraCount() > 0)
        {
            getActiveCamera()->beginFrame();
        }

        // Ignore the elapsed time we got from the user. This will allow camera movement in cases where the time is frozen
        if (cameraController)
        {
            cameraController->attachCamera(getActiveCamera());
            cameraController->setCameraSpeed(getCameraSpeed());
            changed |= cameraController->update();
        }

        return changed;
    }

    void Scene::deleteModel(uint32_t modelID)
    {
        // Delete entire vector of instances
        mModels.erase(mModels.begin() + modelID);
        mExtentsDirty = true;
    }

    void Scene::deleteAllModels()
    {
        mModels.clear();
        mExtentsDirty = true;
    }

    uint32_t Scene::getModelInstanceCount(uint32_t modelID) const
    {
        return (uint32_t)(mModels[modelID].size());
    }

    void Scene::addModelInstance(const Model::SharedPtr& pModel, const std::string& instanceName, const glm::vec3& translation, const glm::vec3& yawPitchRoll, const glm::vec3& scaling)
    {
        ModelInstance::SharedPtr pInstance = ModelInstance::create(pModel, translation, yawPitchRoll, scaling, instanceName);
        addModelInstance(pInstance);
        mExtentsDirty = true;
    }

    void Scene::addModelInstance(const ModelInstance::SharedPtr& pInstance)
    {
        // Checking for existing instance list for model
        for (uint32_t modelID = 0; modelID < (uint32_t)mModels.size(); modelID++)
        {
            // If found, add to that list
            if (getModel(modelID) == pInstance->getObject())
            {
                mModels[modelID].push_back(pInstance);
                return;
            }
        }

        // If not found, add a new list
        mModels.emplace_back();
        mModels.back().push_back(pInstance);
        mExtentsDirty = true;
    }

    void Scene::deleteModelInstance(uint32_t modelID, uint32_t instanceID)
    {
        // Delete instance
        auto& instances = mModels[modelID];

        //  Check if there is only one instance left.
        if (instances.size() == 1)
        {
            //  Delete the entire model, since it will also erase the corresponding instance.
            assert(instanceID == 0);
            deleteModel(modelID);
        }
        else
        {
            //  Erase the instance.
            instances.erase(instances.begin() + instanceID);
        }

        //  Extents will be dirty in either case.
        mExtentsDirty = true;
    }

    const Scene::UserVariable& Scene::getUserVariable(const std::string& name) const
    {
        const auto& a = mUserVars.find(name);
        if (a == mUserVars.end())
        {
            logWarning("Can't find user variable " + name + " in scene.");
            return kInvalidVar;
        }
        else
        {
            return a->second;
        }
    }

    const Scene::UserVariable& Scene::getUserVariable(uint32_t varID, std::string& varName) const
    {
        for (const auto& a : mUserVars)
        {
            if (varID == 0)
            {
                varName = a.first;
                return a.second;
            }
            --varID;
        }

        should_not_get_here();
        varName = "";
        return mUserVars.begin()->second;
    }

    uint32_t Scene::addLight(const Light::SharedPtr& pLight)
    {
        if (pLight->getType() == LightArea)
        {
            logWarning("Use Scene::addAreaLight() for area lights.");
            return uint32(-1);
        }

        mpLights.push_back(pLight);
        mExtentsDirty = true;
        return (uint32_t)mpLights.size() - 1;
    }

    void Scene::deleteLight(uint32_t lightID)
    {
        mpLights.erase(mpLights.begin() + lightID);
        mExtentsDirty = true;
    }

    uint32_t Scene::addLightProbe(const LightProbe::SharedPtr& pLightProbe)
    {
        mpLightProbes.push_back(pLightProbe);
        return (uint32_t)mpLightProbes.size() - 1;
    }

    void Scene::deleteLightProbe(uint32_t lightID)
    {
        mpLightProbes.erase(mpLightProbes.begin() + lightID);
    }

    uint32_t Scene::addAreaLight(const AreaLight::SharedPtr& pAreaLight)
    {
        mpAreaLights.push_back(pAreaLight);
        return (uint32_t)mpAreaLights.size() - 1;
    }

    void Scene::deleteAreaLight(uint32_t lightID)
    {
        mpAreaLights.erase(mpAreaLights.begin() + lightID);
    }

    uint32_t Scene::addPath(const ObjectPath::SharedPtr& pPath)
    {
        mpPaths.push_back(pPath);
        return (uint32_t)mpPaths.size() - 1;
    }

    void Scene::deletePath(uint32_t pathID)
    {
        mpPaths.erase(mpPaths.begin() + pathID);
    }

    uint32_t Scene::addCamera(const Camera::SharedPtr& pCamera)
    {
        mCameras.push_back(pCamera);
        return (uint32_t)mCameras.size() - 1;
    }

    void Scene::deleteCamera(uint32_t cameraID)
    {
        mCameras.erase(mCameras.begin() + cameraID);

        if (cameraID == mActiveCameraID)
        {
            mActiveCameraID = 0;
        }
    }

    void Scene::setActiveCamera(uint32_t camID)
    {
        mActiveCameraID = camID;
    }

    void Scene::merge(const Scene* pFrom)
    {
#define merge(name_) name_.insert(name_.end(), pFrom->name_.begin(), pFrom->name_.end());

        merge(mModels);
        merge(mpLights);
        merge(mpPaths);
        merge(mCameras);
#undef merge
        mUserVars.insert(pFrom->mUserVars.begin(), pFrom->mUserVars.end());
        mExtentsDirty = true;
    }

    void Scene::createAreaLights()
    {
        // Clean up area light(s) before adding
        mpAreaLights.clear();

        // Go through all models in the scene
        for (uint32_t modelId = 0; modelId < getModelCount(); ++modelId)
        {
            const Model::SharedPtr& pModel = getModel(modelId);
            if (pModel)
            {
                // TODO: Create area lights per model instance
                std::vector<AreaLight::SharedPtr> areaLights = createAreaLightsForModel(pModel.get());
                mpAreaLights.insert(mpAreaLights.end(), areaLights.begin(), areaLights.end());
            }
        }
    }

    void Scene::bindSampler(Sampler::SharedPtr pSampler)
    {
        for (auto& model : mModels)
        {
            model[0]->getObject()->bindSamplerToMaterials(pSampler);
        }

        for (auto& probe : mpLightProbes)
        {
            probe->setSampler(pSampler);
        }
    }

    void Scene::attachSkinningCacheToModels(SkinningCache::SharedPtr pSkinningCache)
    {
        for (auto& model : mModels)
        {
            model[0]->getObject()->attachSkinningCache(pSkinningCache);
        }
    }

    void Scene::setCamerasAspectRatio(float ratio)
    {
        for (auto& c : mCameras) c->setAspectRatio(ratio);
    }
}
