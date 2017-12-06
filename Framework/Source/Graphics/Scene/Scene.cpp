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
        Scene::SharedPtr pScene = create();
        if (SceneImporter::loadScene(*pScene, filename, modelLoadFlags, sceneLoadFlags) == false)
        {
            pScene = nullptr;
        }
        return pScene;
    }

    Scene::SharedPtr Scene::create()
    {
        return SharedPtr(new Scene());
    }

    Scene::Scene()
        : mId(sSceneCounter++)
    {
        // Reset all global id counters recursively
        Model::resetGlobalIdCounter();

        mpMaterialHistory = MaterialHistory::create();
    }

    Scene::~Scene() = default;

    void Scene::updateExtents()
    {
        if (mExtentsDirty)
        {
            mExtentsDirty = false;

            mRadius = 0.f;
            float k = 0.f;
            mCenter = vec3(0, 0, 0);
            for (uint32_t i = 0; i < getModelCount(); ++i)
            {
                const auto& model = getModel(i);
                const float r = model->getRadius();
                const vec3 c = model->getCenter();
                for (uint32_t j = 0; j < getModelInstanceCount(i); ++j)
                {
                    const auto& inst = getModelInstance(i, j);
                    const vec3 instC = vec3(vec4(c, 1.f) * inst->getTransformMatrix());
                    const vec3 scaling = inst->getScaling();
                    const float instR = r * max(scaling.x, max(scaling.y, scaling.z));

                    if (k == 0.f)
                    {
                        mCenter = instC;
                        mRadius = instR;
                    }
                    else
                    {
                        vec3 dir = instC - mCenter;
                        if (length(dir) > 1e-6f)
                            dir = normalize(dir);
                        vec3 a = mCenter - dir * mRadius;
                        vec3 b = instC + dir * instR;

                        mCenter = (a + b) * 0.5f;
                        mRadius = length(a - b);
                    }
                    k++;
                }
            }

            // Update light extents
            for (auto& light : mpLights)
            {
                if (light->getType() == LightDirectional)
                {
                    auto pDirLight = std::dynamic_pointer_cast<DirectionalLight>(light);
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
            mModels[i][0]->getObject()->animate(currentTime);
        }

        mExtentsDirty = mExtentsDirty || changed;

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
        if (mpMaterialHistory != nullptr)
        {
            mpMaterialHistory->onModelRemoved(getModel(modelID).get());
        }

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
        mpLights.push_back(pLight);
        mExtentsDirty = true;
        return (uint32_t)mpLights.size() - 1;
    }

    void Scene::deleteLight(uint32_t lightID)
    {
        mpLights.erase(mpLights.begin() + lightID);
        mExtentsDirty = true;
    }

    void Scene::deleteMaterial(uint32_t materialID)
    {
        if (mpMaterialHistory != nullptr)
        {
            mpMaterialHistory->onMaterialRemoved(getMaterial(materialID).get());
        }

        mpMaterials.erase(mpMaterials.begin() + materialID);
    }

    void Scene::deleteMaterialHistory()
    {
        mpMaterialHistory = nullptr;
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
        merge(mpMaterials);
        merge(mCameras);
#undef merge
        mUserVars.insert(pFrom->mUserVars.begin(), pFrom->mUserVars.end());
        mExtentsDirty = true;
    }

    void Scene::createAreaLights()
    {
        // Clean up area light(s) before adding
        deleteAreaLights();

        // Go through all models in the scene
        for (uint32_t modelId = 0; modelId < getModelCount(); ++modelId)
        {
            const Model::SharedPtr& pModel = getModel(modelId);
            if (pModel)
            {
                // Retrieve model instances for this model
                for (uint32_t modelInstanceId = 0; modelInstanceId < getModelInstanceCount(modelId); ++modelInstanceId)
                {
                    // #TODO This should probably create per model instance
                    AreaLight::createAreaLightsForModel(pModel, mpLights);
                }
            }
        }
    }

    void Scene::deleteAreaLights()
    {
        // Clean up the list before adding
        std::vector<Light::SharedPtr>::iterator it = mpLights.begin();

        for (; it != mpLights.end();)
        {
            if ((*it)->getType() == LightArea)
            {
                it = mpLights.erase(it);
            }
            else
            {
                ++it;
            }
        }
    }

    void Scene::bindSamplerToMaterials(Sampler::SharedPtr pSampler)
    {
        for (auto& pMat : mpMaterials)
        {
            pMat->setSampler(pSampler);
        }
    }

    void Scene::bindSamplerToModels(Sampler::SharedPtr pSampler)
    {
        for (auto& model : mModels)
        {
            model[0]->getObject()->bindSamplerToMaterials(pSampler);
        }
    }
}
