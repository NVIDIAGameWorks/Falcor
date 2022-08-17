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

#pragma once
#include "Types.h"
#include "Parser.h"
#include "Core/Assert.h"
#include "Utils/Math/Matrix.h"

#include <glm/gtx/string_cast.hpp>

#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <variant>
#include <vector>

namespace Falcor
{
    namespace pbrt
    {
        using MaterialRef = std::variant<std::monostate, uint32_t, std::string>;

        std::string to_string(const MaterialRef& materialRef);

        struct SceneEntity
        {
            SceneEntity() = default;
            SceneEntity(const std::string& name, ParameterDictionary params, FileLoc loc)
                : name(name)
                , params(params)
                , loc(loc)
            {}

            std::string toString() const
            {
                return fmt::format("SceneEntity(name='{}', params={})", name, params.toString());
            }

            std::string name;
            FileLoc loc;
            ParameterDictionary params;
        };

        struct MaterialSceneEntity : public SceneEntity
        {
            MaterialSceneEntity() = default;
            MaterialSceneEntity(const std::string& name, const std::string& type, ParameterDictionary params, FileLoc loc)
                : SceneEntity(name, params, loc)
                , type(type)
            {}

            std::string toString() const
            {
                return fmt::format("SceneMaterialEntity(name='{}', type='{}', params={})", name, type, params.toString());
            }

            std::string type;
        };

        struct TransformedSceneEntity : public SceneEntity
        {
            TransformedSceneEntity() = default;
            TransformedSceneEntity(const std::string& name, ParameterDictionary params, FileLoc loc, const rmcv::mat4& transform)
                : SceneEntity(name, params, loc)
                , transform(transform)
            {}

            std::string toString() const
            {
                return fmt::format("TransformedSceneEntity(name='{}', params={}, transform={})", name, params.toString(), rmcv::to_string(transform));
            }

            rmcv::mat4 transform;
        };

        struct CameraSceneEntity : public TransformedSceneEntity
        {
            CameraSceneEntity() = default;
            CameraSceneEntity(const std::string& name, ParameterDictionary params, FileLoc loc, const rmcv::mat4& transform, const std::string& medium)
                : TransformedSceneEntity(name, params, loc, transform)
                , medium(medium)
            {}

            std::string toString() const
            {
                return fmt::format("CameraSceneEntity(name='{}', params={}, transform={}, medium='{}')", name, params.toString(), rmcv::to_string(transform), medium);
            }

            std::string medium;
        };

        struct LightSceneEntity : public TransformedSceneEntity
        {
            LightSceneEntity() = default;
            LightSceneEntity(const std::string& name, ParameterDictionary params, FileLoc loc, const rmcv::mat4& transform, const std::string& medium)
                : TransformedSceneEntity(name, params, loc, transform)
                , medium(medium)
            {}

            std::string toString() const
            {
                return fmt::format("LightSceneEntity(name='{}', params={}, transform={}, medium='{}')", name, params.toString(), rmcv::to_string(transform), medium);
            }

            std::string medium;
        };

        struct MediumSceneEntity : public TransformedSceneEntity
        {
            MediumSceneEntity() = default;
            MediumSceneEntity(const std::string& name, ParameterDictionary params, FileLoc loc, const rmcv::mat4& transform)
                : TransformedSceneEntity(name, params, loc, transform)
            {}

            std::string toString() const
            {
                return fmt::format("MediumSceneEntity(name='{}', params={}, transform={})", name, params.toString(), rmcv::to_string(transform));
            }
        };

        struct TextureSceneEntity : public TransformedSceneEntity
        {
            TextureSceneEntity() = default;
            TextureSceneEntity(const std::string& name, ParameterDictionary params, FileLoc loc, const rmcv::mat4& transform)
                : TransformedSceneEntity(name, params, loc, transform)
            {}

            std::string toString() const
            {
                return fmt::format("TextureSceneEntity(name='{}', params={}, transform={})", name, params.toString(), rmcv::to_string(transform));
            }
        };

        struct ShapeSceneEntity : public TransformedSceneEntity
        {
            ShapeSceneEntity() = default;
            ShapeSceneEntity(const std::string& name, ParameterDictionary params, FileLoc loc, const rmcv::mat4& transform,
                             bool reverseOrientation, MaterialRef materialRef, int lightIndex,
                             const std::string& insideMedium, const std::string& outsideMedium)
                : TransformedSceneEntity(name, params, loc, transform),
                reverseOrientation(reverseOrientation),
                materialRef(materialRef),
                lightIndex(lightIndex),
                insideMedium(insideMedium),
                outsideMedium(outsideMedium)
            {}

            std::string toString() const
            {
                return fmt::format("ShapeSceneEntity(name='{}', params={}, transform={}, reverseOrientation={}, "
                    "materialRef={}. lightIndex={}, insideMedium='{}', outsideMedium='{}')",
                    name, params.toString(), rmcv::to_string(transform), reverseOrientation,
                    to_string(materialRef), lightIndex, insideMedium, outsideMedium);
            }

            bool reverseOrientation = false;
            MaterialRef materialRef;
            int lightIndex = -1;
            std::string insideMedium, outsideMedium;
        };

        struct InstanceDefinitionSceneEntity
        {
            InstanceDefinitionSceneEntity() = default;
            InstanceDefinitionSceneEntity(const std::string& name, FileLoc loc)
                : name(name)
                , loc(loc)
            {}

            std::string toString() const
            {
                return fmt::format("InstanceDefinitionSceneEntity(name='{}', shapes='{}')", name, shapes.size());
            }

            std::string name;
            FileLoc loc;
            std::vector<ShapeSceneEntity> shapes;
        };

        struct InstanceSceneEntity
        {
            InstanceSceneEntity() = default;
            InstanceSceneEntity(const std::string& name, FileLoc loc, const rmcv::mat4& transform)
                : name(name)
                , loc(loc)
                , transform(transform)
            {}

            std::string toString() const
            {
                return fmt::format("InstanceSceneEntity(name='{}', transform='{}')", name, rmcv::to_string(transform));
            }

            std::string name;
            FileLoc loc;
            rmcv::mat4 transform;
        };

        class BasicScene
        {
        public:
            BasicScene(const std::filesystem::path& searchPath);

            void setOptions(SceneEntity filter, SceneEntity film, CameraSceneEntity camera,
                            SceneEntity sampler, SceneEntity integrator, SceneEntity accelerator);

            void addNamedMaterial(std::string name, MaterialSceneEntity material);
            uint32_t addMaterial(MaterialSceneEntity material);
            void addMedium(MediumSceneEntity medium);
            void addFloatTexture(std::string name, TextureSceneEntity texture);
            void addSpectrumTexture(std::string name, TextureSceneEntity texture);
            void addLight(LightSceneEntity light);
            uint32_t addAreaLight(SceneEntity light);
            void addShapes(std::vector<ShapeSceneEntity>& shapes);
            void addInstanceDefinition(InstanceDefinitionSceneEntity instanceDefinition);
            void addInstances(std::vector<InstanceSceneEntity>& instances);

            const CameraSceneEntity& getCamera() const { return mCamera; }

            const std::map<std::string, MaterialSceneEntity>& getNamedMaterials() const { return mNamedMaterials; }
            const std::vector<MaterialSceneEntity>& getMaterials() const { return mMaterials; }
            const std::vector<MediumSceneEntity>& getMedia() const { return mMedia; }
            const std::map<std::string, TextureSceneEntity>& getFloatTextures() const { return mFloatTextures; }
            const std::map<std::string, TextureSceneEntity>& getSpectrumTextures() const { return mSpectrumTextures; }
            const std::vector<LightSceneEntity>& getLights() const { return mLights; }
            const std::vector<ShapeSceneEntity>& getShapes() const { return mShapes; }
            const std::map<std::string, InstanceDefinitionSceneEntity>& getInstanceDefinitions() const { return mInstanceDefinitions; }
            const std::vector<InstanceSceneEntity>& getInstances() const { return mInstances; }

            /** Get a named or unnamed material.
            */
            const MaterialSceneEntity& getMaterial(const MaterialRef& materialRef) const;

            const SceneEntity& getAreaLight(int lightIndex);

            std::filesystem::path resolvePath(const std::filesystem::path& path) const;

            std::string toString() const;

        private:
            std::filesystem::path mSearchPath;

            SceneEntity mFilter;
            SceneEntity mFilm;
            CameraSceneEntity mCamera;
            SceneEntity mSampler;
            SceneEntity mIntegrator;
            SceneEntity mAccelerator;

            std::map<std::string, MaterialSceneEntity> mNamedMaterials;
            std::vector<MaterialSceneEntity> mMaterials;
            std::vector<MediumSceneEntity> mMedia;
            std::map<std::string, TextureSceneEntity> mFloatTextures;
            std::map<std::string, TextureSceneEntity> mSpectrumTextures;
            std::vector<LightSceneEntity> mLights;
            std::vector<ShapeSceneEntity> mShapes;
            std::vector<SceneEntity> mAreaLights;

            std::map<std::string, InstanceDefinitionSceneEntity> mInstanceDefinitions;
            std::vector<InstanceSceneEntity> mInstances;
        };

        constexpr uint32_t kMaxTransforms = 2;

        using Transform = rmcv::mat4;

        struct TransformSet
        {
            TransformSet()
            {
                for (uint32_t i = 0; i < kMaxTransforms; ++i) t[i] = rmcv::identity<rmcv::mat4>();
            }

            Transform& operator[](uint32_t i)
            {
                FALCOR_ASSERT(i < kMaxTransforms);
                return t[i];
            }

            const Transform& operator[](uint32_t i) const
            {
                FALCOR_ASSERT(i < kMaxTransforms);
                return t[i];
            }

            friend TransformSet inverse(const TransformSet& ts)
            {
                TransformSet tInv;
                for (uint32_t i = 0; i < kMaxTransforms; ++i)
                {
                    tInv.t[i] = rmcv::inverse(ts.t[i]);
                }
                return tInv;
            }

            bool isAnimated() const
            {
                for (uint32_t i = 0; i < kMaxTransforms - 1; ++i)
                {
                    if (t[i] != t[i + 1]) return true;
                }
                return false;
            }

        private:
            Transform t[kMaxTransforms];
        };

        class BasicSceneBuilder : public ParserTarget
        {
        public:
            BasicSceneBuilder(BasicScene& scene);

            void onOption(const std::string& name, const std::string& value, FileLoc loc) override;
            void onIdentity(FileLoc loc) override;
            void onTranslate(Float dx, Float dy, Float dz, FileLoc loc) override;
            void onRotate(Float angle, Float ax, Float ay, Float az, FileLoc loc) override;
            void onScale(Float sx, Float sy, Float sz, FileLoc loc) override;
            void onLookAt(Float ex, Float ey, Float ez, Float lx, Float ly, Float lz, Float ux, Float uy, Float uz, FileLoc loc) override;
            void onConcatTransform(Float transform[16], FileLoc loc) override;
            void onTransform(Float transform[16], FileLoc loc) override;
            void onCoordinateSystem(const std::string&, FileLoc loc) override;
            void onCoordSysTransform(const std::string&, FileLoc loc) override;
            void onActiveTransformAll(FileLoc loc) override;
            void onActiveTransformEndTime(FileLoc loc) override;
            void onActiveTransformStartTime(FileLoc loc) override;
            void onTransformTimes(Float start, Float end, FileLoc loc) override;
            void onColorSpace(const std::string& n, FileLoc loc) override;
            void onPixelFilter(const std::string& name, ParsedParameterVector params, FileLoc loc) override;
            void onFilm(const std::string& type, ParsedParameterVector params, FileLoc loc) override;
            void onSampler(const std::string& name, ParsedParameterVector params, FileLoc loc) override;
            void onAccelerator(const std::string& name, ParsedParameterVector params, FileLoc loc) override;
            void onIntegrator(const std::string& name, ParsedParameterVector params, FileLoc loc) override;
            void onCamera(const std::string&, ParsedParameterVector params, FileLoc loc) override;
            void onMakeNamedMedium(const std::string& name, ParsedParameterVector params, FileLoc loc) override;
            void onMediumInterface(const std::string& insideName, const std::string& outsideName, FileLoc loc) override;
            void onWorldBegin(FileLoc loc) override;
            void onAttributeBegin(FileLoc loc) override;
            void onAttributeEnd(FileLoc loc) override;
            void onAttribute(const std::string& target, ParsedParameterVector params, FileLoc loc) override;
            void onTexture(const std::string& name, const std::string& type, const std::string& texname, ParsedParameterVector params, FileLoc loc) override;
            void onMaterial(const std::string& name, ParsedParameterVector params, FileLoc loc) override;
            void onMakeNamedMaterial(const std::string& name, ParsedParameterVector params, FileLoc loc) override;
            void onNamedMaterial(const std::string& name, FileLoc loc) override;
            void onLightSource(const std::string& name, ParsedParameterVector params, FileLoc loc) override;
            void onAreaLightSource(const std::string& name, ParsedParameterVector params, FileLoc loc) override;
            void onShape(const std::string& name, ParsedParameterVector params, FileLoc loc) override;
            void onReverseOrientation(FileLoc loc) override;
            void onObjectBegin(const std::string& name, FileLoc loc) override;
            void onObjectEnd(FileLoc loc) override;
            void onObjectInstance(const std::string& name, FileLoc loc) override;

            void onEndOfFiles() override;

        private:
            rmcv::mat4 getTransform() const { return mGraphicsState.ctm[0]; }

            static constexpr int kStartTransformBits = 1 << 0;
            static constexpr int kEndTransformBits = 1 << 1;
            static constexpr int kAllTransformsBits = (1 << kMaxTransforms) - 1;

            struct GraphicsState
            {
                template <typename F>
                void forActiveTransforms(F func)
                {
                    for (int i = 0; i < kMaxTransforms; ++i)
                    {
                        if (activeTransformBits & (1 << i)) ctm[i] = func(ctm[i]);
                    }
                }

                std::string currentInsideMedium, currentOutsideMedium;

                MaterialRef currentMaterial;

                std::string areaLightName;
                ParameterDictionary areaLightParams;
                FileLoc areaLightLoc;

                ParsedParameterVector shapeAttributes;
                ParsedParameterVector lightAttributes;
                ParsedParameterVector materialAttributes;
                ParsedParameterVector mediumAttributes;
                ParsedParameterVector textureAttributes;
                bool reverseOrientation = false;
                const RGBColorSpace* pColorSpace = nullptr;
                TransformSet ctm;
                uint32_t activeTransformBits = kAllTransformsBits;
                Float transformStartTime = 0, transformEndTime = 1;
            };

            BasicScene& mScene;

            enum class BlockState { OptionsBlock, WorldBlock };
            BlockState mCurrentBlock = BlockState::OptionsBlock;

            GraphicsState mGraphicsState;
            std::map<std::string, TransformSet> mNamedCoordinateSystems;

            struct StackEntry
            {
                enum class Type { Attribute, Object };
                Type type;
                FileLoc loc;
                GraphicsState graphicsState;
            };
            std::vector<StackEntry> mStack;

            struct ActiveInstanceDefinition
            {
                ActiveInstanceDefinition(std::string name, FileLoc loc) : entity(name, loc) {}
                InstanceDefinitionSceneEntity entity;
            };
            std::unique_ptr<ActiveInstanceDefinition> mpActiveInstanceDefinition;

            uint32_t mUnamedMaterialIndex = 0;
            std::set<std::string> mNamedMaterialNames;
            std::set<std::string> mMediumNames;
            std::set<std::string> mFloatTextureNames;
            std::set<std::string> mSpectrumTextureNames;
            std::set<std::string> mInstanceNames;

            SceneEntity mFilter;
            SceneEntity mFilm;
            CameraSceneEntity mCamera;
            SceneEntity mSampler;
            SceneEntity mIntegrator;
            SceneEntity mAccelerator;

            std::vector<ShapeSceneEntity> mShapes;
            std::vector<InstanceSceneEntity> mInstances;
        };
    }
}
