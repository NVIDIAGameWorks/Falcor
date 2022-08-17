/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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

// This code is based on pbrt:
// pbrt is Copyright(c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// The pbrt source code is licensed under the Apache License, Version 2.0.
// SPDX: Apache-2.0

#include "Builder.h"
#include "Helpers.h"
#include "Core/Assert.h"
#include "Utils/Logger.h"

namespace Falcor
{
    namespace pbrt
    {
        std::string to_string(const MaterialRef& materialRef)
        {
            if (const uint32_t* pIndex = std::get_if<uint32_t>(&materialRef))
            {
                return fmt::format("<index:{}>", *pIndex);
            }
            else if (const std::string* pName = std::get_if<std::string>(&materialRef))
            {
                return fmt::format("<name:{}>", *pName);
            }
            else
            {
                return "<null>";
            }
        }

        BasicScene::BasicScene(const std::filesystem::path& searchPath)
            : mSearchPath(searchPath)
        {
        }

        void BasicScene::setOptions(SceneEntity filter, SceneEntity film, CameraSceneEntity camera,
                                    SceneEntity sampler, SceneEntity integrator, SceneEntity accelerator)
        {
            mFilter = filter;
            mFilm = film;
            mCamera = camera;
            mSampler = sampler;
            mIntegrator = integrator;
            mAccelerator = accelerator;
        }

        void BasicScene::addNamedMaterial(std::string name, MaterialSceneEntity material)
        {
            mNamedMaterials.emplace(name, material);
        }

        uint32_t BasicScene::addMaterial(MaterialSceneEntity material)
        {
            mMaterials.push_back(material);
            return (uint32_t)(mMaterials.size() - 1);
        }

        void BasicScene::addMedium(MediumSceneEntity medium)
        {
            mMedia.push_back(medium);
        }

        void BasicScene::addFloatTexture(std::string name, TextureSceneEntity texture)
        {
            mFloatTextures.emplace(name, texture);
        }

        void BasicScene::addSpectrumTexture(std::string name, TextureSceneEntity texture)
        {
            mSpectrumTextures.emplace(name, texture);
        }

        void BasicScene::addLight(LightSceneEntity light)
        {
            mLights.push_back(light);
        }

        uint32_t BasicScene::addAreaLight(SceneEntity light)
        {
            mAreaLights.push_back(light);
            return (uint32_t)(mAreaLights.size() - 1);
        }

        void BasicScene::addShapes(std::vector<ShapeSceneEntity>& shapes)
        {
            std::move(shapes.begin(), shapes.end(), std::back_inserter(mShapes));
        }

        void BasicScene::addInstanceDefinition(InstanceDefinitionSceneEntity instanceDefinition)
        {
            mInstanceDefinitions.emplace(instanceDefinition.name, instanceDefinition);
        }

        void BasicScene::addInstances(std::vector<InstanceSceneEntity>& instances)
        {
            std::move(instances.begin(), instances.end(), std::back_inserter(mInstances));
        }

        const MaterialSceneEntity& BasicScene::getMaterial(const MaterialRef& materialRef) const
        {
            if (const uint32_t* pIndex = std::get_if<uint32_t>(&materialRef))
            {
                FALCOR_ASSERT(*pIndex >= 0 && *pIndex <= mMaterials.size());
                return mMaterials[*pIndex];
            }
            else if (const std::string* pName = std::get_if<std::string>(&materialRef))
            {
                auto it = mNamedMaterials.find(*pName);
                FALCOR_ASSERT(it != mNamedMaterials.end());
                return it->second;
            }
            else
            {
                throw RuntimeError("Expected valid material reference (index or name).");
            }
        }

        const SceneEntity& BasicScene::getAreaLight(int lightIndex)
        {
            FALCOR_ASSERT(lightIndex >= 0 && lightIndex < mAreaLights.size());
            return mAreaLights[lightIndex];
        }

        std::filesystem::path BasicScene::resolvePath(const std::filesystem::path& path) const
        {
            if (path.is_absolute()) return path;
            return mSearchPath / path;
        }

        std::string BasicScene::toString() const
        {
            std::string str;

            auto printEntities = [&](std::string_view name, auto entities)
            {
                str += fmt::format("{}=[\n", name);
                for (const auto& entity : entities) str += fmt::format("{}\n", entity.toString());
                str += "]\n";
            };

            auto printNamedEntities = [&](std::string_view name, auto entities)
            {
                str += fmt::format("{}=[\n", name);
                for (const auto& [name, entity] : entities) str += fmt::format("{}={}\n", name, entity.toString());
                str += "]\n";
            };

            str += fmt::format("filter={}\n", mFilter.toString());
            str += fmt::format("film={}\n", mFilm.toString());
            str += fmt::format("camera={}\n", mCamera.toString());
            str += fmt::format("sampler={}\n", mSampler.toString());
            str += fmt::format("integrator={}\n", mIntegrator.toString());
            str += fmt::format("accelerator={}\n", mAccelerator.toString());
            printNamedEntities("namedMaterials", mNamedMaterials);
            printEntities("materials", mMaterials);
            printEntities("media", mMedia);
            printNamedEntities("floatTextures", mFloatTextures);
            printNamedEntities("spectrumTextures", mSpectrumTextures);
            printEntities("lights", mLights);
            printEntities("shapes", mShapes);
            printEntities("areaLights", mAreaLights);
            printNamedEntities("instanceDefinitions", mInstanceDefinitions);
            printEntities("instances", mInstances);
            return str;
        }


// API State Macros
#define VERIFY_OPTIONS(func)                                                                        \
    if (mCurrentBlock == BlockState::WorldBlock)                                                    \
    {                                                                                               \
        throwError(loc, "Options cannot be set inside world block. '{}' is not allowed.", func);    \
    } else /* swallow trailing semicolon */

#define VERIFY_WORLD(func)                                                                          \
    if (mCurrentBlock == BlockState::OptionsBlock)                                                  \
    {                                                                                               \
        throwError(loc, "Scene description must be inside world block. '{}' is not allowed.", func);\
    } else /* swallow trailing semicolon */


        BasicSceneBuilder::BasicSceneBuilder(BasicScene& scene)
            : mScene(scene)
        {
        }

        void BasicSceneBuilder::onReverseOrientation(FileLoc loc)
        {
            VERIFY_WORLD("ReverseOrientation");
            mGraphicsState.reverseOrientation = !mGraphicsState.reverseOrientation;
        }

        void BasicSceneBuilder::onColorSpace(const std::string& name, FileLoc loc)
        {
            logWarning(loc, "Color spaces are currently not supported and ignored. Always using RGB Rec.709 color space.");
        }

        void BasicSceneBuilder::onIdentity(FileLoc loc)
        {
            mGraphicsState.forActiveTransforms([](auto t) { return rmcv::identity<rmcv::mat4>(); });
        }

        void BasicSceneBuilder::onTranslate(Float dx, Float dy, Float dz, FileLoc loc)
        {
            mGraphicsState.forActiveTransforms([=](auto t) { return t * rmcv::translate(float3(dx, dy, dz)); });
        }

        void BasicSceneBuilder::onCoordinateSystem(const std::string& name, FileLoc loc)
        {
            mNamedCoordinateSystems[name] = mGraphicsState.ctm;
        }

        void BasicSceneBuilder::onCoordSysTransform(const std::string& name, FileLoc loc)
        {
            if (mNamedCoordinateSystems.find(name) != mNamedCoordinateSystems.end())
            {
                mGraphicsState.ctm = mNamedCoordinateSystems[name];
            }
            else
            {
                logWarning(loc, "Couldn't find named coordinate system '{}'.", name);
            }
        }

        void BasicSceneBuilder::onCamera(const std::string& name, ParsedParameterVector params, FileLoc loc)
        {
            VERIFY_OPTIONS("Camera");

            TransformSet cameraFromWorld = mGraphicsState.ctm;
            TransformSet worldFromCamera = inverse(cameraFromWorld);
            mNamedCoordinateSystems["camera"] = worldFromCamera;

            ParameterDictionary dict(std::move(params), mGraphicsState.pColorSpace);
            mCamera = CameraSceneEntity(name, std::move(dict), loc, worldFromCamera[0], mGraphicsState.currentOutsideMedium);
        }

        void BasicSceneBuilder::onAttributeBegin(FileLoc loc)
        {
            VERIFY_WORLD("AttributeBegin");

            mStack.push_back({ StackEntry::Type::Attribute, loc, mGraphicsState });
        }

        void BasicSceneBuilder::onAttributeEnd(FileLoc loc)
        {
            VERIFY_WORLD("AttributeEnd");

            // Issue warning on unmatched AttributeEnd.
            if (mStack.empty())
            {
                logWarning(loc, "Unmatched AttributeEnd encountered. Ignoring it.");
                return;
            }

            if (mStack.back().type == StackEntry::Type::Object)
            {
                throwError(loc, "Mismatched nesting: open ObjectBegin from {} at AttributeEnd.", mStack.back().loc.toString());
            }
            else
            {
                FALCOR_ASSERT(mStack.back().type == StackEntry::Type::Attribute);
            }

            mGraphicsState = std::move(mStack.back().graphicsState);
            mStack.pop_back();
        }

        void BasicSceneBuilder::onAttribute(const std::string& target, ParsedParameterVector attrib, FileLoc loc)
        {
            ParsedParameterVector* currentAttributes = nullptr;
            if (target == "shape") currentAttributes = &mGraphicsState.shapeAttributes;
            else if (target == "light") currentAttributes = &mGraphicsState.lightAttributes;
            else if (target == "material") currentAttributes = &mGraphicsState.materialAttributes;
            else if (target == "medium") currentAttributes = &mGraphicsState.mediumAttributes;
            else if (target == "texture") currentAttributes = &mGraphicsState.textureAttributes;
            else throwError(loc, "Unknown attribute target '{}'. Must be 'shape', 'light', 'material', 'medium' or 'texture'.", target);

            // Note that we hold on to the current color space and associate it with the parameters...
            for (auto& p : attrib)
            {
                p.mayBeUnused = true;
                p.colorSpace = mGraphicsState.pColorSpace;
                currentAttributes->push_back(p);
            }
        }

        void BasicSceneBuilder::onSampler(const std::string& name, ParsedParameterVector params, FileLoc loc)
        {
            VERIFY_OPTIONS("Sampler");
            ParameterDictionary dict(std::move(params), mGraphicsState.pColorSpace);
            mSampler = SceneEntity(name, std::move(dict), loc);
        }

        void BasicSceneBuilder::onWorldBegin(FileLoc loc)
        {
            VERIFY_OPTIONS("WorldBegin");
            mCurrentBlock = BlockState::WorldBlock;

            // Reset graphics state.
            for (uint32_t i = 0; i < kMaxTransforms; ++i)
            {
                mGraphicsState.ctm[i] = rmcv::identity<rmcv::mat4>();
            }
            mGraphicsState.activeTransformBits = kAllTransformsBits;
            mNamedCoordinateSystems["world"] = mGraphicsState.ctm;

            // Add pre-WorldBegin entities to scene.
            mScene.setOptions(mFilter, mFilm, mCamera, mSampler, mIntegrator, mAccelerator);
        }

        void BasicSceneBuilder::onMakeNamedMedium(const std::string& name, ParsedParameterVector params, FileLoc loc)
        {
            // Issue error if medium _name_ is multiply defined.
            if (mMediumNames.find(name) != mMediumNames.end())
            {
                throwError(loc, "Redefining named medium '{}'.", name);
            }
            mMediumNames.insert(name);

            // Create _ParameterDictionary_ for medium and call _AddMedium()_
            ParameterDictionary dict(std::move(params), mGraphicsState.mediumAttributes, mGraphicsState.pColorSpace);
            mScene.addMedium(MediumSceneEntity(name, std::move(dict), loc, getTransform()));
        }

        void BasicSceneBuilder::onLightSource(const std::string& name, ParsedParameterVector params, FileLoc loc)
        {
            VERIFY_WORLD("LightSource");
            ParameterDictionary dict(std::move(params), mGraphicsState.lightAttributes, mGraphicsState.pColorSpace);
            mScene.addLight(LightSceneEntity(name, std::move(dict), loc, getTransform(), mGraphicsState.currentOutsideMedium));
        }

        void BasicSceneBuilder::onShape(const std::string& name, ParsedParameterVector params, FileLoc loc)
        {
            VERIFY_WORLD("Shape");
            ParameterDictionary dict(std::move(params), mGraphicsState.shapeAttributes, mGraphicsState.pColorSpace);

            int areaLightIndex = -1;
            if (!mGraphicsState.areaLightName.empty())
            {
                auto areaLight = SceneEntity(mGraphicsState.areaLightName, mGraphicsState.areaLightParams, mGraphicsState.areaLightLoc);
                areaLightIndex = mScene.addAreaLight(std::move(areaLight));
                if (mpActiveInstanceDefinition)
                {
                    logWarning(loc, "Area lights not supported with object instancing.");
                }
            }

            ShapeSceneEntity shape(name, std::move(dict), loc, getTransform(), mGraphicsState.reverseOrientation,
                mGraphicsState.currentMaterial, areaLightIndex, mGraphicsState.currentInsideMedium, mGraphicsState.currentOutsideMedium);

            if (mpActiveInstanceDefinition)
            {
                mpActiveInstanceDefinition->entity.shapes.push_back(std::move(shape));
            }
            else
            {
                mShapes.push_back(std::move(shape));
            }
        }

        void BasicSceneBuilder::onObjectBegin(const std::string& name, FileLoc loc)
        {
            VERIFY_WORLD("ObjectBegin");

            mStack.push_back({ StackEntry::Type::Object, loc, mGraphicsState });

            if (mpActiveInstanceDefinition)
            {
                throwError(loc, "ObjectBegin called inside of instance definition.");
            }

            if (mInstanceNames.find(name) != mInstanceNames.end())
            {
                throwError(loc, "{}: trying to redefine an object instance.", name);
            }

            mInstanceNames.insert(name);
            mpActiveInstanceDefinition = std::make_unique<ActiveInstanceDefinition>(name, loc);
        }

        void BasicSceneBuilder::onObjectEnd(FileLoc loc)
        {
            VERIFY_WORLD("ObjectEnd");

            if (!mpActiveInstanceDefinition)
            {
                throwError(loc, "ObjectEnd called outside of instance definition.");
            }

            if (mStack.back().type == StackEntry::Type::Attribute)
            {
                throwError(loc, "Mismatched nesting: open AttributeBegin from {} at ObjectEnd.", mStack.back().loc.toString());
            }
            else
            {
                FALCOR_ASSERT(mStack.back().type == StackEntry::Type::Object);
            }

            mGraphicsState = std::move(mStack.back().graphicsState);
            mStack.pop_back();

            mScene.addInstanceDefinition(std::move(mpActiveInstanceDefinition->entity));

            mpActiveInstanceDefinition = nullptr;
        }

        void BasicSceneBuilder::onObjectInstance(const std::string& name, FileLoc loc)
        {
            VERIFY_WORLD("ObjectInstance");

            if (mpActiveInstanceDefinition)
            {
                throwError(loc, "ObjectInstance can't be called inside instance definition");
            }

            InstanceSceneEntity instance(name, loc, getTransform());
            mInstances.push_back(std::move(instance));
        }

        void BasicSceneBuilder::onEndOfFiles()
        {
            if (mCurrentBlock != BlockState::WorldBlock)
            {
                throwError("End of files before 'WorldBegin'.");
            }

            // Ensure there are no pushed graphics states.
            if (!mStack.empty())
            {
                throwError("Missing end to AttributeBegin.");
            }

            mScene.addShapes(mShapes);
            mScene.addInstances(mInstances);
        }

        void BasicSceneBuilder::onOption(const std::string& name, const std::string& value, FileLoc loc)
        {
            // Options:
            // disablepixeljitter, disabletexturefiltering, disablewavelengthjitter, displacementedgescale
            // msereferenceimage, msereferenceout, rendercoordsys, seed, forcediffuse, pixelstats, wavefront
            logWarning(loc, "Option '{}' is currently not supported and ignored.", name);
        }

        void BasicSceneBuilder::onTransform(Float tr[16], FileLoc loc)
        {
            // The PBRT file-format has matrices for row-vectors (v.M), so needs transpose
            rmcv::mat4 m = rmcv::transpose(rmcv::mat4({
                tr[0], tr[1], tr[2], tr[3],
                tr[4], tr[5], tr[6], tr[7],
                tr[8], tr[9], tr[10], tr[11],
                tr[12], tr[13], tr[14], tr[15]
                }));

            mGraphicsState.forActiveTransforms([=](auto t) { return m; });
        }

        void BasicSceneBuilder::onConcatTransform(Float tr[16], FileLoc loc)
        {
            // The PBRT file-format has matrices for row-vectors (v.M), so needs transpose
            rmcv::mat4 m = rmcv::transpose(rmcv::mat4({
                tr[0], tr[1], tr[2], tr[3],
                tr[4], tr[5], tr[6], tr[7],
                tr[8], tr[9], tr[10], tr[11],
                tr[12], tr[13], tr[14], tr[15]
                }));

            mGraphicsState.forActiveTransforms([=](auto t) { return t * m; });
        }

        void BasicSceneBuilder::onRotate(Float angle, Float dx, Float dy, Float dz, FileLoc loc)
        {
            mGraphicsState.forActiveTransforms([=](auto t) { return t * rmcv::rotate(glm::radians(angle), float3(dx, dy, dz)); });
        }

        void BasicSceneBuilder::onScale(Float sx, Float sy, Float sz, FileLoc loc)
        {
            mGraphicsState.forActiveTransforms([=](auto t) { return t * rmcv::scale(float3(sx, sy, sz)); });
        }

        void BasicSceneBuilder::onLookAt(Float ex, Float ey, Float ez, Float lx, Float ly, Float lz, Float ux, Float uy, Float uz, FileLoc loc)
        {
            auto lookAt = rmcv::lookAtLH(float3(ex, ey, ez), float3(lx, ly, lz), float3(ux, uy, uz));
            mGraphicsState.forActiveTransforms([=](auto t) { return t * lookAt; });
        }

        void BasicSceneBuilder::onActiveTransformAll(FileLoc loc)
        {
            mGraphicsState.activeTransformBits = kAllTransformsBits;
        }

        void BasicSceneBuilder::onActiveTransformEndTime(FileLoc loc)
        {
            mGraphicsState.activeTransformBits = kEndTransformBits;
        }

        void BasicSceneBuilder::onActiveTransformStartTime(FileLoc loc)
        {
            mGraphicsState.activeTransformBits = kStartTransformBits;
        }

        void BasicSceneBuilder::onTransformTimes(Float start, Float end, FileLoc loc)
        {
            VERIFY_OPTIONS("TransformTimes");
            mGraphicsState.transformStartTime = start;
            mGraphicsState.transformEndTime = end;
            logWarning(loc, "Animated transforms are currently not supported and ignored.");
        }

        void BasicSceneBuilder::onPixelFilter(const std::string& name, ParsedParameterVector params, FileLoc loc)
        {
            VERIFY_OPTIONS("PixelFilter");
            ParameterDictionary dict(std::move(params), mGraphicsState.pColorSpace);
            mFilter = SceneEntity(name, std::move(dict), loc);
        }

        void BasicSceneBuilder::onFilm(const std::string& type, ParsedParameterVector params, FileLoc loc)
        {
            VERIFY_OPTIONS("Film");
            ParameterDictionary dict(std::move(params), mGraphicsState.pColorSpace);
            mFilm = SceneEntity(type, std::move(dict), loc);
        }

        void BasicSceneBuilder::onAccelerator(const std::string& name, ParsedParameterVector params, FileLoc loc)
        {
            VERIFY_OPTIONS("Accelerator");
            ParameterDictionary dict(std::move(params), mGraphicsState.pColorSpace);
            mAccelerator = SceneEntity(name, std::move(dict), loc);
        }

        void BasicSceneBuilder::onIntegrator(const std::string& name, ParsedParameterVector params, FileLoc loc)
        {
            VERIFY_OPTIONS("Integrator");
            ParameterDictionary dict(std::move(params), mGraphicsState.pColorSpace);
            mIntegrator = SceneEntity(name, std::move(dict), loc);
        }

        void BasicSceneBuilder::onMediumInterface(const std::string& insideName, const std::string& outsideName, FileLoc loc)
        {
            mGraphicsState.currentInsideMedium = insideName;
            mGraphicsState.currentOutsideMedium = outsideName;
        }

        void BasicSceneBuilder::onTexture(const std::string& name, const std::string& type, const std::string& texname, ParsedParameterVector params, FileLoc loc)
        {
            VERIFY_WORLD("Texture");
            ParameterDictionary dict(std::move(params), mGraphicsState.textureAttributes, mGraphicsState.pColorSpace);

            if (type != "float" && type != "spectrum")
            {
                throwError(loc, "'{}' texture type unknown. Must be 'float' or 'spectrum'.", type);
            }

            auto& textureNames = (type == "float") ? mFloatTextureNames : mSpectrumTextureNames;
            if (textureNames.find(name) != textureNames.end())
            {
                throwError(loc, "Redefining texture '{}'.", name);
            }
            textureNames.insert(name);

            if (type == "float")
            {
                mScene.addFloatTexture(name, TextureSceneEntity(texname, std::move(dict), loc, getTransform()));
            }
            else
            {
                mScene.addSpectrumTexture(name, TextureSceneEntity(texname, std::move(dict), loc, getTransform()));
            }
        }

        void BasicSceneBuilder::onMaterial(const std::string& name, ParsedParameterVector params, FileLoc loc)
        {
            VERIFY_WORLD("Material");
            ParameterDictionary dict(std::move(params), mGraphicsState.materialAttributes, mGraphicsState.pColorSpace);

            mGraphicsState.currentMaterial = mScene.addMaterial(MaterialSceneEntity(fmt::format("Unnamed{}", mUnamedMaterialIndex++), name, std::move(dict), loc));
        }

        void BasicSceneBuilder::onMakeNamedMaterial(const std::string& name, ParsedParameterVector params, FileLoc loc)
        {
            VERIFY_WORLD("MakeNamedMaterial");
            ParameterDictionary dict(std::move(params), mGraphicsState.materialAttributes, mGraphicsState.pColorSpace);

            if (mNamedMaterialNames.find(name) != mNamedMaterialNames.end())
            {
                throwError(loc, "Redefining named material '{}'.", name);
            }
            mNamedMaterialNames.insert(name);

            auto type = dict.getString("type", "");
            if (type.empty())
            {
                throwError(loc, "'type' parameter not provided for named material.");
            }

            mScene.addNamedMaterial(name, MaterialSceneEntity(name, type, std::move(dict), loc));
        }

        void BasicSceneBuilder::onNamedMaterial(const std::string& name, FileLoc loc)
        {
            VERIFY_WORLD("NamedMaterial");
            mGraphicsState.currentMaterial = name;
        }

        void BasicSceneBuilder::onAreaLightSource(const std::string& name, ParsedParameterVector params, FileLoc loc)
        {
            VERIFY_WORLD("AreaLightSource");
            mGraphicsState.areaLightName = name;
            mGraphicsState.areaLightParams = ParameterDictionary(std::move(params), mGraphicsState.lightAttributes, mGraphicsState.pColorSpace);
            mGraphicsState.areaLightLoc = loc;
        }
    }
}
