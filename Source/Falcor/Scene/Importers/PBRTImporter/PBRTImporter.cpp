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

/** pbrt-v4 scene importer.

    This implements a scene importer for pbrt-v4. As Falcor only supports
    a small subset of the features available in pbrt-v4, the importer has
    to take some approximations when importing scenes or simply ignore
    certain objects/attributes in the scene.

    The following describes how the importer works on a high level
    (see PBRTImporter::import() for more details):
    - A scene file is parsed using pbrt::parseFile() or pbrt::parseString().
    - The parser dispatches commands via the pbrt::ParserTarget interface.
    - The pbrt::BasicSceneBuilder (implementing pbrt::ParserTarget) builds
      a pbrt::BasicScene representing the parsed scene.
    - The buildScene() function in this file takes a pbrt::BasicScene
      and generates the Falcor scene using Falcor::SceneBuilder.

    The parser code and pbrt::BasicScene are derived directly from pbrt-v4
    code. The code was simplified in a few areas but should more or less
    reflect what is done in pbrt-v4.

    The code to convert to a Falcor scene is mostly contained in this file.
    This is where a lot of approximations take place, e.g. for material
    conversion. The current code is trying to emit warnings for all
    unhandled object types and parameters. Also, each handler is annotated
    with the parameter set that should be handled. All of this information
    was collected from reading pbrt-v4 code, as there is no specification
    available for the scene format. This means that while it reflects the
    current state of pbrt-v4 (as of March 2022), things may change in the
    future.
*/


#include "stdafx.h"
#include "PBRTImporter.h"
#include "Parser.h"
#include "Builder.h"
#include "Helpers.h"
#include "LoopSubdivide.h"
#include "EnvMapConverter.h"

#include <glm/gtx/transform.hpp>
#include <glm/gtx/euler_angles.hpp>

namespace Falcor
{
    namespace pbrt
    {
        const glm::mat4 kYtoZ = {
            1.f, 0.f, 0.f, 0.f,
            0.f, 0.f, 1.f, 0.f,
            0.f, 1.f, 0.f, 0.f,
            0.f, 0.f, 0.f, 1.f,
        };

        const glm::mat4 kInvertZ = {
            1.f, 0.f, 0.f, 0.f,
            0.f, 1.f, 0.f, 0.f,
            0.f, 0.f, -1.f, 0.f,
            0.f, 0.f, 0.f, 1.f,
        };

        /** Holds the results from creating a camera.
        */
        struct Camera
        {
            Falcor::Camera::SharedPtr pCamera;
            glm::mat4 transform;
        };

        /** Holds the results from creating a light.
        */
        struct Light
        {
            Falcor::Light::SharedPtr pLight;
            Falcor::EnvMap::SharedPtr pEnvMap;
        };

        /** Represents a float texture.
            These can be unassigned (std::monostate), a constant float or a texture.
            Note: pbrt-v4 supports many additional texture types that we currently don't support and represent here.
        */
        struct FloatTexture
        {
            std::variant<std::monostate, float, Falcor::Texture::SharedPtr> texture;
            glm::mat4 transform = glm::identity<glm::mat4>();

            bool isConstant() const { return std::holds_alternative<float>(texture); }
            float getConstant() const { FALCOR_ASSERT(isConstant()); return std::get<float>(texture); }
        };

        /** Represents a spectrum texture.
            These can be unassigned (std::monostate), a constant spectrum or a texture.
            Note: pbrt-v4 supports many additional texture types that we currently don't support and represent here.
        */
        struct SpectrumTexture
        {
            SpectrumType spectrumType = SpectrumType::Albedo;
            std::variant<std::monostate, Spectrum, Falcor::Texture::SharedPtr> texture;
            glm::mat4 transform = glm::identity<glm::mat4>();

            bool isConstant() const { return std::holds_alternative<Spectrum>(texture); }
            Spectrum getConstant() const { FALCOR_ASSERT(isConstant()); return std::get<Spectrum>(texture); }
        };

        struct Medium
        {
        };

        /** Holds the results from creating a shape.
        */
        struct Shape
        {
            Falcor::TriangleMesh::SharedPtr pTriangleMesh;
            glm::mat4 transform;
            Falcor::Material::SharedPtr pMaterial;
        };

        struct InstanceDefinition
        {
            std::vector<std::pair<uint32_t, glm::mat4>> meshes; // List of meshID + transform
            std::vector<std::pair<uint32_t, glm::mat4>> curves; // List of curveID + transfrom
        };

        struct BuilderContext
        {
            BasicScene& scene;
            SceneBuilder& builder;

            std::map<std::string, FloatTexture> floatTextures;
            std::map<std::string, SpectrumTexture> spectrumTextures;

            std::map<std::string, Medium> media;

            std::map<std::string, Falcor::Material::SharedPtr> namedMaterials;
            std::vector<Falcor::Material::SharedPtr> materials;

            Falcor::Material::SharedPtr pDefaultMaterial;

            std::map<std::string, InstanceDefinition> instanceDefinitions;

            size_t curveCount = 0;

            Falcor::Material::SharedPtr getMaterial(const MaterialRef& materialRef)
            {
                Falcor::Material::SharedPtr pMaterial;

                if (const uint32_t* pIndex = std::get_if<uint32_t>(&materialRef))
                {
                    FALCOR_ASSERT(*pIndex >= 0 && *pIndex < materials.size());
                    pMaterial = materials[*pIndex];
                }
                else if (const std::string* pName = std::get_if<std::string>(&materialRef))
                {
                    auto it = namedMaterials.find(*pName);
                    FALCOR_ASSERT(it != namedMaterials.end());
                    pMaterial = it->second;
                }

                if (!pMaterial)
                {
                    if (!pDefaultMaterial)
                    {
                        pDefaultMaterial = Falcor::StandardMaterial::create("Default");
                        pDefaultMaterial->setDoubleSided(true);
                    }
                    return pDefaultMaterial;

                }

                return pMaterial;
            }

            Resolver resolver = [this](const std::filesystem::path& path)
            {
                return scene.resolvePath(path);
            };
        };

        inline void warnUnsupportedType(const FileLoc& loc, std::string_view category, std::string_view name)
        {
            logWarning(loc, "{} type '{}' is currently not supported and ignored.", category, name);
        }

        inline void warnUnsupportedParameters(const ParameterDictionary& params, std::vector<std::string> names)
        {
            for (const auto& name : names)
            {
                if (params.hasParameter(name))
                {
                    logWarning(params.getParameterLoc(name), "Parameter '{}' is currently not supported and ignored.", name);
                }
            }
        }

        float3 spectrumToRGB(const Spectrum& spectrum, SpectrumType spectrumType)
        {
            // TODO: Handle spectrum type.
            if (auto pRGB = std::get_if<float3>(&spectrum))
            {
                return *pRGB;
            }
            else if (auto pPiecewiseLinearSpectrum = std::get_if<PiecewiseLinearSpectrum>(&spectrum))
            {
                return spectrumToRGB(*pPiecewiseLinearSpectrum);
            }
            else if (auto pBlackbodySpectrum = std::get_if<BlackbodySpectrum>(&spectrum))
            {
                return spectrumToRGB(*pBlackbodySpectrum);
            }
            else
            {
                throw RuntimeError("Unhandled spectrum variant.");
            }
        }

        float3 getSpectrumAsRGB(BuilderContext& ctx, const ParameterDictionary& params, const std::string& name, float3 def, SpectrumType spectrumType)
        {
            auto spectrum = params.getSpectrum(name, Spectrum(def), ctx.resolver);
            return spectrumToRGB(spectrum, spectrumType);
        }

        std::optional<SpectrumTexture> getSpectrumTextureOrNull(BuilderContext& ctx, const ParameterDictionary& params, const std::string& name, SpectrumType spectrumType)
        {
            if (params.hasTexture(name))
            {
                auto texture = params.getTexture(name);
                auto loc = params.getParameterLoc(name);
                if (texture.empty()) throwError(loc, "No texture name provided for parameter '{}'.", name);

                auto it = ctx.spectrumTextures.find(texture);
                if (it == ctx.spectrumTextures.end()) throwError(loc, "Cannot find spectrum texture named '{}' for parameter '{}'.", texture, name);

                SpectrumTexture spectrumTexture = it->second;
                spectrumTexture.spectrumType = spectrumType;
                return spectrumTexture;
            }
            else if (params.hasSpectrum(name))
            {
                SpectrumTexture spectrumTexture;
                spectrumTexture.texture = params.getSpectrum(name, Spectrum(float3(0.f)), ctx.resolver);
                spectrumTexture.spectrumType = spectrumType;
                return spectrumTexture;
            }

            return {};
        }

        SpectrumTexture getSpectrumTexture(BuilderContext& ctx, const ParameterDictionary& params, const std::string& name, float3 def, SpectrumType spectrumType)
        {
            if (auto spectrumTexture = getSpectrumTextureOrNull(ctx, params, name, spectrumType))
            {
                return *spectrumTexture;
            }

            SpectrumTexture spectrumTexture;
            spectrumTexture.texture = Spectrum(def);
            spectrumTexture.spectrumType = spectrumType;
            return spectrumTexture;
        }

        void assignSpectrumTexture(const SpectrumTexture& spectrumTexture, std::function<void(float3)> constantSetter, std::function<void(Falcor::Texture::SharedPtr)> textureSetter)
        {
            if (const auto *pSpectrum = std::get_if<Spectrum>(&spectrumTexture.texture))
            {
                constantSetter(spectrumToRGB(*pSpectrum, spectrumTexture.spectrumType));
            }
            else if (const auto *pTexture = std::get_if<Falcor::Texture::SharedPtr>(&spectrumTexture.texture))
            {
                textureSetter(*pTexture);
            }
        }

        std::optional<FloatTexture> getFloatTextureOrNull(BuilderContext& ctx, const ParameterDictionary& params, const std::string& name)
        {
            if (params.hasTexture(name))
            {
                auto texture = params.getTexture(name);
                auto loc = params.getParameterLoc(name);
                if (texture.empty()) throwError(loc, "No texture name provided for parameter '{}'.", name);

                auto it = ctx.floatTextures.find(texture);
                if (it == ctx.floatTextures.end()) throwError(loc, "Cannot find float texture named '{}' for parameter '{}'.", texture, name);

                return it->second;
            }
            else if (params.hasFloat(name))
            {
                FloatTexture floatTexture;
                floatTexture.texture = params.getFloat(name, 0.f);
                return floatTexture;
            }

            return {};
        }

        FloatTexture getFloatTexture(BuilderContext& ctx, const ParameterDictionary& params, const std::string& name, float def)
        {
            if (auto floatTexture = getFloatTextureOrNull(ctx, params, name))
            {
                return *floatTexture;
            }

            FloatTexture floatTexture;
            floatTexture.texture = def;
            return floatTexture;
        }

        float getScalarRoughness(BuilderContext& ctx, const SceneEntity& entity,
            const std::string& roughnessName = "roughness",
            const std::string& uroughnessName = "uroughness",
            const std::string& vroughnessName = "vroughness")
        {
            const auto& params = entity.params;

            auto uroughness = getFloatTextureOrNull(ctx, params, uroughnessName);
            auto vroughness = getFloatTextureOrNull(ctx, params, vroughnessName);
            if (!uroughness) uroughness = getFloatTexture(ctx, params, roughnessName, 0.f);
            if (!vroughness) vroughness = getFloatTexture(ctx, params, roughnessName, 0.f);
            auto remaproughness = params.getBool("remaproughness", true);

            if (!uroughness->isConstant() || !vroughness->isConstant())
            {
                float fallback = 0.f;
                logWarning(entity.loc, "Non-constant roughness is currently not supported. Using constant roughness of {} instead.", fallback);
                return fallback;
            }

            if (uroughness->getConstant() != vroughness->getConstant())
            {
                logWarning(entity.loc, "Anisotropic roughness is currently not supported. Using average of u and v instead.");
            }

            float roughness = 0.5f * (uroughness->getConstant() + vroughness->getConstant());

            // "remaproughness" determines if roughness represents a "linear" roughness value and should be converted to the NDF "alpha" value.
            // PBRT always uses the Trowbridge-Reitz / GGX NDF.
            if (!remaproughness) roughness = std::sqrt(roughness);

            return roughness;
        }

        float getScalarEta(BuilderContext& ctx, const SceneEntity& entity,
            const std::string& etaName = "eta")
        {
            const auto& params = entity.params;

            float eta = params.getFloat(etaName, 1.5f);
            if (params.hasSpectrum(etaName))
            {
                // This is a very crude approximation to get a scalar index of refraction value from a spectrum.
                auto rgb = getSpectrumAsRGB(ctx, params, etaName, float3(1.5f), SpectrumType::Unbounded);
                eta = (rgb.r + rgb.g + rgb.b) / 3.f;
            }

            return eta;
        }

        float3 fresnelDieletricConductor(float3 eta, float3 k, float cosTheta)
        {
            float cosTheta2 = cosTheta * cosTheta;
            float sinTheta2 = 1.f - cosTheta2;
            float3 eta2 = eta * eta;
            float3 k2 = k * k;

            float3 t0 = eta2 - k2 - sinTheta2;
            float3 a2plusb2 = sqrt(t0 * t0 + 4.f * eta2 * k2);
            float3 t1 = a2plusb2 + cosTheta2;
            float3 a = sqrt(0.5f * (a2plusb2 + t0));
            float3 t2 = 2.f * a * cosTheta;
            float3 Rs = (t1 - t2) / (t1 + t2);

            float3 t3 = cosTheta2 * a2plusb2 + sinTheta2 * sinTheta2;
            float3 t4 = t2 * sinTheta2;
            float3 Rp = Rs * (t3 - t4) / (t3 + t4);

            return 0.5f * (Rp + Rs);
        }

        float3 getSpecularAlbedo(BuilderContext& ctx, const SceneEntity& entity,
            const std::string& reflectanceName = "reflectance",
            const std::string& etaName = "eta",
            const std::string& kName = "k")
        {
            const auto& params = entity.params;

            auto eta = getSpectrumTextureOrNull(ctx, params, etaName, SpectrumType::Unbounded);
            auto k = getSpectrumTextureOrNull(ctx, params, kName, SpectrumType::Unbounded);
            auto reflectance = getSpectrumTextureOrNull(ctx, params, reflectanceName, SpectrumType::Albedo);

            if (reflectance && (eta || k))
            {
                throwError(entity.loc, "Both '{}' and '{}' and '{}' can't be provided.", reflectanceName, etaName, kName);
            }

            if (reflectance)
            {
                if (!reflectance->isConstant())
                {
                    float3 fallback = float3(0.5f);
                    logWarning(entity.loc, "Non-constant '{}' is not currently supported. Using constant {},{},{} instead.", reflectanceName, fallback.x, fallback.y, fallback.z);
                    return fallback;
                }
                return spectrumToRGB(reflectance->getConstant(), SpectrumType::Albedo);
            }

            float3 etaRgb = spectrumToRGB(Spectrum(*Spectra::getNamedSpectrum("metal-Cu-eta")), SpectrumType::Unbounded);
            float3 kRgb = spectrumToRGB(Spectrum(*Spectra::getNamedSpectrum("metal-Cu-k")), SpectrumType::Unbounded);

            if (eta)
            {
                if (eta->isConstant())
                {
                    etaRgb = spectrumToRGB(eta->getConstant(), SpectrumType::Unbounded);
                }
                else
                {
                    logWarning(entity.loc, "Non-constant '{}' is not currently supported. Using constant {},{},{} instead.", etaName, etaRgb.r, etaRgb.g, etaRgb.b);
                }
            }

            if (k)
            {
                if (k->isConstant())
                {
                    kRgb = spectrumToRGB(k->getConstant(), SpectrumType::Unbounded);
                }
                else
                {
                    logWarning(entity.loc, "Non-constant '{}' is not currently supported. Using constant {},{},{} instead.", kName, kRgb.r, kRgb.g, kRgb.b);
                }
            }

            // Approximate by reflectance at incident angle.
            return fresnelDieletricConductor(etaRgb, kRgb, 1.f);
        }

        Camera createCamera(BuilderContext& ctx, const CameraSceneEntity& entity)
        {
            auto warnUnsupported = [&]() { warnUnsupportedType(entity.loc, "Camera", entity.name); };

            const auto& type = entity.name;
            const auto& params = entity.params;

            Camera camera;

            if (type == "perspective")
            {
                // Parameters:
                // Float lensradius, Float focaldistance, Float fov, Float frameaspectratio, Float[4] screenwindow
                warnUnsupportedParameters(params, { "frameaspectratio", "screenwindow" });

                auto lensradius = params.getFloat("lensradius", 0.f);
                auto focaldistance = params.getFloat("focaldistance", 1e6f);
                auto fov = params.getFloat("fov", 90.f);

                auto pCamera = Falcor::Camera::create("Camera");
                pCamera->setApertureRadius(lensradius);
                pCamera->setFocalDistance(focaldistance);
                float focalLength = fovYToFocalLength(glm::radians(fov), 24.f);
                pCamera->setFocalLength(focalLength);

                camera.pCamera = pCamera;
                camera.transform = entity.transform;
            }
            else if (type == "orthographic")
            {
                // Parameters:
                // Float lensradius, Float focaldistance, Float frameaspectratio, Float[4] screenwindow
                warnUnsupported();
            }
            else if (type == "realistic")
            {
                // Parameters:
                // String lensfile, Float aperturediameter, Float focusdistance, String aperture
                warnUnsupported();
            }
            else if (type == "spherical")
            {
                // Parameters:
                // Float lensradius, Float focaldistance, Float frameaspectratio, Float[4] screenwindow, String mapping
                warnUnsupported();
            }
            else
            {
                throwError(entity.loc, "Unknown camera type '{}'.", type);
            }

            return camera;
        }

        Light createLight(BuilderContext& ctx, const LightSceneEntity& entity)
        {
            auto warnUnsupported = [&]() { warnUnsupportedType(entity.loc, "Light", entity.name); };

            const auto& type = entity.name;
            const auto& params = entity.params;

            Light light;

            if (type == "point")
            {
                // Parameters:
                // Spectrum I, Float scale, Float power, Point3 from
                warnUnsupported();
            }
            else if (type == "spot")
            {
                // Parameters:
                // Spectrum I, Float scale, Float power, Float coneangle, Float conedeltaangle, Point3 from, Point3 to
                warnUnsupported();
            }
            else if (type == "goniometric")
            {
                // Parameters:
                // Spectrum I, Float scale, String filename, Float power
                warnUnsupported();
            }
            else if (type == "projection")
            {
                // Parameters:
                // Float scale, Float power, Float fov, String filename
                warnUnsupported();
            }
            else if (type == "distant")
            {
                // Parameters:
                // Spectrum L, Float scale, Point3 from, Point3 to, Float illuminance
                auto L = params.getSpectrum("L", Spectrum(*Spectra::getNamedSpectrum("stdillum-D65")), ctx.resolver);
                auto scale = params.getFloat("scale", 1.f);
                auto from = params.getPoint3("from", float3(0.f, 0.f, 0.f));
                auto to = params.getPoint3("to", float3(0.f, 0.f, 1.f));
                auto illuminance = params.getFloat("illuminance", -1.f);

                // TODO: Missing spectrum normalization to 1 nit

                float3 intensity = spectrumToRGB(L, SpectrumType::Illuminant);
                intensity *= scale;
                if (illuminance > 0.f) intensity *= illuminance;

                float3 direction = normalize(glm::mat3(entity.transform) * (to - from));

                auto pDirectionalLight = Falcor::DirectionalLight::create("DirectionalLight");
                pDirectionalLight->setIntensity(intensity);
                pDirectionalLight->setWorldDirection(direction);
                light.pLight = pDirectionalLight;
            }
            else if (type == "infinite")
            {
                // Parameters:
                // Spectrum[] L, Float scale, Point3[] portal, String filename, Float illuminance
                warnUnsupportedParameters(params, { "portal", "illuminance" });

                auto L = params.getSpectrumArray("L", ctx.resolver);
                auto scale = params.getFloat("scale", 1.f);
                auto filename = params.getString("filename", "");

                if (!L.empty() && !filename.empty())
                {
                    throwError(entity.loc, "Can't specify both emission 'L' and 'filename' for infinite light.");
                }

                if (!L.empty())
                {
                    // Falcor doesn't have constant infinite emitter.
                    // We create a one pixel env map for now.
                    float4 data { spectrumToRGB(L[0], SpectrumType::Illuminant), 0.f };
                    auto pTexture = Texture::create2D(1, 1, ResourceFormat::RGBA32Float, 1, Texture::kMaxPossible, &data);
                    auto pEnvMap = EnvMap::create(pTexture);

                    light.pEnvMap = pEnvMap;

                }
                else if (!filename.empty())
                {
                    auto path = ctx.resolver(filename);
                    auto pOctTexture = Falcor::Texture::createFromFile(path, false, false);
                    // TODO: Use equal-area octahedral parametrization when env map supports it.
                    logWarning(entity.loc, "Environment map is converted from equal-area octahedral to lat-long parametrization. Exact results cannot be expected.");
                    EnvMapConverter envMapConverter;
                    auto pLatLongTexture = envMapConverter.convertEqualAreaOctToLatLong(gpDevice->getRenderContext(), pOctTexture);
                    auto pEnvMap = Falcor::EnvMap::create(pLatLongTexture);
                    pEnvMap->setIntensity(scale);

                    float3 rotation;
                    glm::extractEulerAngleXYZ(entity.transform, rotation.x, rotation.y, rotation.z);
                    pEnvMap->setRotation(glm::degrees(rotation));

                    light.pEnvMap = pEnvMap;
                }
            }
            else
            {
                throwError(entity.loc, "Unknown light type '{}'.", type);
            }

            return light;
        }

        FloatTexture createFloatTexture(BuilderContext& ctx, const TextureSceneEntity& entity)
        {
            auto warnUnsupported = [&]() { warnUnsupportedType(entity.loc, "Float texture", entity.name); };

            const auto& type = entity.name;
            const auto& params = entity.params;

            FloatTexture floatTexture;

            floatTexture.transform = entity.transform;
            if (floatTexture.transform != glm::identity<glm::mat4>())
            {
                logWarning(entity.loc, "Texture transforms are currently not supported and ignored.");
            }

            if (type == "constant")
            {
                // Parameters:
                // Float value
                floatTexture.texture = params.getFloat("value", 1.f);
            }
            else if (type == "scale")
            {
                // Parameters:
                // FloatTexture tex, FloatTexture scale
                warnUnsupported();
            }
            else if (type == "mix")
            {
                // Parameters:
                // FloatTexture tex1, FloatTexture tex2, FloatTexture amount
                warnUnsupported();
            }
            else if (type == "directionmix")
            {
                // Parameters:
                // FloatTexture tex1, FloatTexture tex2, Vector3 dir
                warnUnsupported();
            }
            else if (type == "bilerp")
            {
                // Parameters:
                // Float v00, Float v01, Float v10, Float v11
                warnUnsupported();
            }
            else if (type == "imagemap")
            {
                // Parameters:
                // Float maxanisotropy, String filter, String wrap
                // Float scale, Bool invert
                // String filename, String encoding
                warnUnsupportedParameters(params, { "maxanisotropy", "wrap", "scale", "invert" });

                auto path = ctx.resolver(params.getString("filename", ""));

                auto filter = params.getString("filter", "bilinear");
                // "ewa", "point" filter is not currently supported.
                if (filter != "bilinear" && filter != "trilinear")
                {
                    logWarning(entity.loc, "Filter '{}' is currently not supported, using 'bilinear' instead.");
                    filter = "bilinear";
                }
                bool generateMips = filter == "trilinear";

                std::string defaultEncoding = hasExtension(path, "png") ? "sRGB" : "linear";
                auto encoding = params.getString("encoding", defaultEncoding);
                // "gamma x" encoding is not currently supported.
                if (encoding != "linear" && encoding != "sRGB")
                {
                    logWarning(entity.loc, "Encoding '{}' is currently not supported, using '{}' instead.", encoding, defaultEncoding);
                    encoding = defaultEncoding;
                }
                bool sRGB = encoding == "sRGB";

                floatTexture.texture = Falcor::Texture::createFromFile(path, generateMips, sRGB);
            }
            else if (type == "checkerboard")
            {
                // Parameters:
                // FloatTexture tex1, FloatTexture tex2, Int dimension
                warnUnsupported();
            }
            else if (type == "dots")
            {
                // Parameters:
                // FloatTexture inside, FloatTexture outside
                warnUnsupported();
            }
            else if (type == "fbm")
            {
                // Parameters:
                // Int octaves, Float roughness
                warnUnsupported();
            }
            else if (type == "wrinkled")
            {
                // Parameters:
                // Int octaves, Float roughness
                warnUnsupported();
            }
            else if (type == "windy")
            {
                warnUnsupported();
            }
            else if (type == "ptex")
            {
                // Parameters:
                // String filename, String encoding, Float scale
                warnUnsupported();
            }
            else
            {
                throwError(entity.loc, "Unknown float texture type '{}'.", type);
            }

            return floatTexture;
        }

        SpectrumTexture createSpectrumTexture(BuilderContext& ctx, const TextureSceneEntity& entity)
        {
            auto warnUnsupported = [&]() { warnUnsupportedType(entity.loc, "Spectrum texture", entity.name); };

            const auto& type = entity.name;
            const auto& params = entity.params;

            SpectrumTexture spectrumTexture;

            spectrumTexture.spectrumType = SpectrumType::Albedo;
            spectrumTexture.transform = entity.transform;
            if (spectrumTexture.transform != glm::identity<glm::mat4>())
            {
                logWarning(entity.loc, "Texture transforms are currently not supported and ignored.");
            }

            if (type == "constant")
            {
                // Parameters:
                // Spectrum value
                spectrumTexture.texture = params.getSpectrum("value", Spectrum(1.f), ctx.resolver);
            }
            else if (type == "scale")
            {
                // Parameters:
                // SpectrumTexture tex, FloatTexture scale
                warnUnsupported();
            }
            else if (type == "mix")
            {
                // Parameters:
                // SpectrumTexture tex1, SpectrumTexture tex2, FloatTexture amount
                warnUnsupported();
            }
            else if (type == "directionmix")
            {
                // Parameters:
                // SpectrumTexture tex1, SpectrumTexture tex2, Vector3 dir
                warnUnsupported();
            }
            else if (type == "bilerp")
            {
                // Parameters:
                // Spectrum v00, Spectrum v01, Spectrum v10, Spectrum v11
                warnUnsupported();
            }
            else if (type == "imagemap")
            {
                // Parameters:
                // Float maxanisotropy, String filter, String wrap
                // Float scale, Bool invert
                // String filename, String encoding
                warnUnsupportedParameters(params, { "maxanisotropy", "wrap", "scale", "invert" });

                auto path = ctx.resolver(params.getString("filename", ""));

                auto filter = params.getString("filter", "bilinear");
                // "ewa", "point" filter is not currently supported.
                if (filter != "bilinear" && filter != "trilinear")
                {
                    logWarning(entity.loc, "Filter '{}' is currently not supported, using 'bilinear' instead.");
                    filter = "bilinear";
                }
                bool generateMips = filter == "trilinear";

                std::string defaultEncoding = hasExtension(path, "png") ? "sRGB" : "linear";
                auto encoding = params.getString("encoding", defaultEncoding);
                // "gamma x" encoding is not currently supported.
                if (encoding != "linear" && encoding != "sRGB")
                {
                    logWarning(entity.loc, "Encoding '{}' is currently not supported, using '{}' instead.", encoding, defaultEncoding);
                    encoding = defaultEncoding;
                }
                bool sRGB = encoding == "sRGB";

                spectrumTexture.texture = Falcor::Texture::createFromFile(path, generateMips, sRGB);
            }
            else if (type == "checkerboard")
            {
                // Parameters:
                // SpectrumTexture tex1, SpectrumTexture tex2, Int dimension
                warnUnsupported();
            }
            else if (type == "dots")
            {
                // Parameters:
                // SpectrumTexture inside, SpectrumTexture outside
                warnUnsupported();
            }
            else if (type == "marble")
            {
                // Parameters:
                // Int octaves, Float roughness, Float scale, Float variation
                warnUnsupported();
            }
            else if (type == "ptex")
            {
                // Parameters:
                // String filename, String encoding, Float scale
                warnUnsupported();
            }
            else
            {
                throwError(entity.loc, "Unknown float texture type '{}'.", type);
            }

            return spectrumTexture;
        }

        Falcor::Material::SharedPtr createMaterial(BuilderContext& ctx, const MaterialSceneEntity& entity)
        {
            const auto& type = entity.type;
            const auto& params = entity.params;

            auto warnUnsupported = [&]() { warnUnsupportedType(entity.loc, "Material", type); };

            Falcor::Material::SharedPtr pMaterial;

            if (type == "" || type == "none")
            {
                logWarning(entity.loc, "Material type '{}' is deprecated, use 'interface' instead.", type);
            }
            else if (type == "interface")
            {
                // Nothing to do.
            }
            else if (type == "diffuse")
            {
                // Parameters:
                // SpectrumTexture reflectance
                // FloatTexture displacement
                warnUnsupportedParameters(params, { "displacement" });

                auto reflectance = getSpectrumTexture(ctx, params, "reflectance", float3(0.5f), SpectrumType::Albedo);

                auto pStandardMaterial = StandardMaterial::create(entity.name);
                pStandardMaterial->setMetallic(0.f);
                pStandardMaterial->setRoughness(1.f);
                assignSpectrumTexture(reflectance,
                    [&](float3 rgb) { pStandardMaterial->setBaseColor(float4(rgb, 1.f)); },
                    [&](Falcor::Texture::SharedPtr pTexture) { pStandardMaterial->setBaseColorTexture(pTexture); }
                );
                pStandardMaterial->setDoubleSided(true);
                pMaterial = pStandardMaterial;
            }
            else if (type == "coateddiffuse")
            {
                // Parameters:
                // SpectrumTexture reflectance
                // FloatTexture roughness, FloatTexture uroughness, FloatTexture vroughness, Bool remaproughness
                // FloatTexture thickness, Float|Spectrum eta, Int maxdepth, Int nsamples, FloatTexture g, SpectrumTexture albedo
                // FloatTexture displacement
                warnUnsupportedParameters(params, { "thickness", "eta", "maxdepth", "nsamples", "g", "albedo", "displacement" });

                auto reflectance = getSpectrumTexture(ctx, params, "reflectance", float3(0.5f), SpectrumType::Albedo);
                float roughness = getScalarRoughness(ctx, entity);

                auto pStandardMaterial = StandardMaterial::create(entity.name);
                pStandardMaterial->setMetallic(0.f);
                pStandardMaterial->setRoughness(roughness);
                assignSpectrumTexture(reflectance,
                    [&](float3 rgb) { pStandardMaterial->setBaseColor(float4(rgb, 1.f)); },
                    [&](Falcor::Texture::SharedPtr pTexture) { pStandardMaterial->setBaseColorTexture(pTexture); }
                );
                pStandardMaterial->setDoubleSided(true);
                pMaterial = pStandardMaterial;
            }
            else if (type == "conductor")
            {
                // Parameters:
                // SpectrumTexture eta, SpectrumTexture k, SpectrumTexture reflectance
                // FloatTexture roughness, FloatTexture uroughness, FloatTexture vroughness, Bool remaproughness
                // FloatTexture displacement
                warnUnsupportedParameters(params, { "displacement"});

                float3 specularAlbedo = getSpecularAlbedo(ctx, entity);
                float roughness = getScalarRoughness(ctx, entity);

                auto pStandardMaterial = StandardMaterial::create(entity.name);
                pStandardMaterial->setBaseColor(float4(specularAlbedo, 1.f));
                pStandardMaterial->setMetallic(1.f);
                pStandardMaterial->setRoughness(roughness);
                pStandardMaterial->setDoubleSided(true);
                pMaterial = pStandardMaterial;
            }
            else if (type == "coatedconductor")
            {
                warnUnsupported();
            }
            else if (type == "dielectric")
            {
                // Parameters:
                // Float|Spectrum eta
                // FloatTexture roughness, FloatTexture uroughness, FloatTexture vroughness, Bool remaproughness
                // FloatTexture displacement
                warnUnsupportedParameters(params, { "displacement" });

                float roughness = getScalarRoughness(ctx, entity);
                float eta = getScalarEta(ctx, entity);

                auto pStandardMaterial = StandardMaterial::create(entity.name);
                pStandardMaterial->setMetallic(0.f);
                pStandardMaterial->setRoughness(roughness);
                pStandardMaterial->setIndexOfRefraction(eta);
                pStandardMaterial->setSpecularTransmission(1.f);
                pMaterial = pStandardMaterial;
            }
            else if (type == "thindielectric")
            {
                // Parameters:
                // Float|Spectrum eta
                // FloatTexture displacement
                warnUnsupportedParameters(params, { "displacement" });

                float eta = getScalarEta(ctx, entity);

                auto pStandardMaterial = StandardMaterial::create(entity.name);
                pStandardMaterial->setMetallic(0.f);
                pStandardMaterial->setRoughness(0.f);
                pStandardMaterial->setIndexOfRefraction(eta);
                pStandardMaterial->setSpecularTransmission(1.f);
                pStandardMaterial->setThinSurface(true);
                pMaterial = pStandardMaterial;
            }
            else if (type == "diffusetransmission")
            {
                // Parameters:
                // SpectrumTexture reflectance, SpectrumTexture transmittance, Float scale
                // FloatTexture displacement
                warnUnsupportedParameters(params, { "displacement", "scale" });

                auto reflectance = getSpectrumTexture(ctx, params, "reflectance", float3(0.25f), SpectrumType::Albedo);
                auto transmission = getSpectrumTexture(ctx, params, "transmission", float3(0.25f), SpectrumType::Albedo);

                auto pStandardMaterial = StandardMaterial::create(entity.name);
                pStandardMaterial->setMetallic(0.f);
                pStandardMaterial->setRoughness(1.f);
                pStandardMaterial->setDiffuseTransmission(0.5f);
                assignSpectrumTexture(reflectance,
                    [&](float3 rgb) { pStandardMaterial->setBaseColor(float4(rgb, 1.f)); },
                    [&](Falcor::Texture::SharedPtr pTexture) { pStandardMaterial->setBaseColorTexture(pTexture); }
                );
                assignSpectrumTexture(transmission,
                    [&](float3 rgb) { pStandardMaterial->setTransmissionColor(rgb); },
                    [&](Falcor::Texture::SharedPtr pTexture) { pStandardMaterial->setTransmissionTexture(pTexture); }
                );
                pStandardMaterial->setDoubleSided(true);
                pMaterial = pStandardMaterial;
            }
            else if (type == "hair")
            {
                warnUnsupported();
            }
            else if (type == "measured")
            {
                warnUnsupported();
            }
            else if (type == "subsurface")
            {
                warnUnsupported();
            }
            else if (type == "mix")
            {
                warnUnsupported();
            }
            else
            {
                throwError(entity.loc, "Unknown material type '{}'.", type);
            }

            // Load normal map.
            if (pMaterial)
            {
                auto normalmap = params.getString("normalmap", "");
                if (!normalmap.empty())
                {
                    auto pNormalMap = Texture::createFromFile(ctx.resolver(normalmap), true, false);
                    pMaterial->setTexture(Material::TextureSlot::Normal, pNormalMap);
                }
            }

            return pMaterial;
        }

        Medium createMedium(BuilderContext& ctx, const MediumSceneEntity& entity)
        {
            const auto& type = entity.params.getString("type", "");
            const auto& params = entity.params;

            auto warnUnsupported = [&]() { warnUnsupportedType(entity.loc, "Medium", type); };

            Medium medium;

            if (type == "homogeneous")
            {
                warnUnsupported();
            }
            else if (type == "uniformgrid")
            {
                warnUnsupported();
            }
            else if (type == "rgbgrid")
            {
                warnUnsupported();
            }
            else if (type == "cloud")
            {
                warnUnsupported();
            }
            else if (type == "nanovdb")
            {
                warnUnsupported();
            }

            return medium;
        }

        void createAreaLight(BuilderContext& ctx, const SceneEntity& entity, const Falcor::Material::SharedPtr& pMaterial)
        {
            auto warnUnsupported = [&]() { warnUnsupportedType(entity.loc, "Area light", entity.name); };

            const auto& type = entity.name;
            const auto& params = entity.params;

            if (type == "diffuse")
            {
                warnUnsupportedParameters(params, { "twosided", "power", "filename" });

                auto L = getSpectrumAsRGB(ctx, params, "L", float3(1.f), SpectrumType::Illuminant);
                auto scale = params.getFloat("scale", 1.f);

                if (auto pStandardMaterial = std::dynamic_pointer_cast<Falcor::StandardMaterial>(pMaterial))
                {
                    pStandardMaterial->setEmissiveColor(L);
                    pStandardMaterial->setEmissiveFactor(1.f);
                }
                else
                {
                    logWarning(entity.loc, "Area lights are only supported for shapes that have a standard material.");
                }
            }
            else
            {
                throwError(entity.loc, "Unknown area light type '{}'.", type);
            }
        }

        Shape createShape(BuilderContext& ctx, const ShapeSceneEntity& entity)
        {
            auto warnUnsupported = [&]() { warnUnsupportedType(entity.loc, "Shape", entity.name); };

            const auto& type = entity.name;
            const auto& params = entity.params;

            warnUnsupportedParameters(params, { "alpha" });

            Shape shape;

            if (type == "sphere")
            {
                // Parameters:
                // Float radius, Float zmin, Float zmax, Float phimax
                warnUnsupportedParameters(params, { "zmin", "zmax", "phimax" });

                auto radius = params.getFloat("radius", 1.f);

                shape.pTriangleMesh = Falcor::TriangleMesh::createSphere(radius);
                shape.pTriangleMesh->setName("sphere");
                shape.transform = entity.transform;
            }
            else if (type == "cylinder")
            {
                // Parameters:
                // Float radius, Float zmin, Float zmax, Float phimax
                warnUnsupported();
            }
            else if (type == "disk")
            {
                // Parameters:
                // Float radius, Float height, Float innerradius, Float phimax
                warnUnsupportedParameters(params, { "innerradius", "phimax" });

                auto radius = params.getFloat("radius", 1.f);
                auto height = params.getFloat("height", 0.f);

                shape.pTriangleMesh = Falcor::TriangleMesh::createDisk(radius);
                shape.pTriangleMesh->setName("disk");
                glm::mat4 transform = glm::translate(float3(0.f, 0.f, height)) * kYtoZ;
                shape.pTriangleMesh->applyTransform(transform);
            }
            else if (type == "bilinearmesh")
            {
                // Parameters:
                // Int[] indices, Point3[] P, Point2[] uv, Normal3[] N, Int[] faceIndices, String emissionfilename
                warnUnsupported();
            }
            else if (type == "curve")
            {
                // Parameters:
                // Float width, Float width0, Float width1, Int degree, String basis,
                // Point3[] P, String type, Normal3[] N, Int splitdepth
                // PBRT scenes typically contain thousands of curve shapes (each shape is just a segment) so we skip warnings.
                ctx.curveCount++;
                if (ctx.curveCount < 10)
                {
                    warnUnsupported();
                }
                else if (ctx.curveCount == 10)
                {
                    Falcor::logWarning("Skipping additional warnings on unsupported curves.");
                }
            }
            else if (type == "trianglemesh")
            {
                // Parameters:
                // Int[] indices, Point3[] P, Point2[] uv, Vector3[] S, Normal3[] N, Int[] faceIndices
                warnUnsupportedParameters(params, { "S", "faceIndices" });

                auto indices = params.getIntArray("indices");
                auto P = params.getPoint3Array("P");
                auto N = params.getNormalArray("N");
                auto uv = params.getPoint2Array("uv");

                if (indices.empty())
                {
                    if (P.size() == 3)
                    {
                        indices = { 0, 1, 2 };
                    }
                    else
                    {
                        logWarning(entity.loc, "Vertex indices 'indices' missing. Skipping.");
                        return {};
                    }
                }
                if (indices.size() % 3 != 0)
                {
                    logWarning(entity.loc, "Number of vertex indices {} is not a multiple of 3. Discarding {} indices.", indices.size(), indices.size() % 3);
                    while (indices.size() % 3 != 0) indices.pop_back();
                }
                if (P.empty())
                {
                    logWarning(entity.loc, "Vertex positions 'positions' missing. Skipping.");
                    return {};
                }
                if (!uv.empty() && uv.size() != P.size())
                {
                    logWarning(entity.loc, "Number of 'uv' elements must match number of 'P' elements. Discarding 'uv'.");
                    uv = {};
                }
                if (!N.empty() && N.size() != P.size())
                {
                    logWarning(entity.loc, "Number of 'N' elements must match number of 'P' elements. Discarding 'N'.");
                    N = {};
                }
                for (auto i : indices)
                {
                    if (i < 0 || i >= P.size())
                    {
                        logWarning(entity.loc, "Vertex index {} is out of bounds. Skipping.", i);
                        return {};
                    }
                }

                Falcor::TriangleMesh::VertexList vertexList(P.size());
                for (size_t i = 0; i < P.size(); ++i)
                {
                    auto& vertex = vertexList[i];
                    vertex.position = P[i];
                    vertex.normal = N.empty() ? float3(0.f) : N[i];
                    vertex.texCoord = uv.empty() ? float2(0.f) : uv[i];
                }
                Falcor::TriangleMesh::IndexList indexList(indices.size());
                for (size_t i = 0; i < indices.size(); ++i)
                {
                    indexList[i] = indices[i];
                }

                shape.pTriangleMesh = Falcor::TriangleMesh::create(std::move(vertexList), std::move(indexList));
                shape.transform = entity.transform;
            }
            else if (type == "plymesh")
            {
                // Parameters:
                // String filename, Texture displacement, Float displacement.edgelength,
                warnUnsupportedParameters(params, { "displacement", "displacement.edgelength" });

                auto filename = params.getString("filename", "");
                auto path = ctx.resolver(filename);

                shape.pTriangleMesh = Falcor::TriangleMesh::createFromFile(path.string());
                if (shape.pTriangleMesh) shape.pTriangleMesh->setName(filename);
                shape.transform = entity.transform;
            }
            else if (type == "loopsubdiv")
            {
                // Parameters:
                // Int levels, Int[] indices, Point3[] P
                // String scheme (also not supported in pbrt-v4)
                warnUnsupportedParameters(params, { "scheme" });

                auto levels = params.getInt("levels", 3);
                auto indices = params.getIntArray("indices");
                auto P = params.getPoint3Array("P");

                if (indices.empty()) throwError(entity.loc, "Missing vertex indices in 'indices'.");
                if (P.empty()) throwError(entity.loc, "Missing vertex positions in 'P'.");

                auto result = loopSubdivide(levels, P, fstd::span<const uint32_t>(reinterpret_cast<const uint32_t*>(indices.data()), indices.size()));
                Falcor::TriangleMesh::VertexList vertexList(result.positions.size());
                for (size_t i = 0; i < result.positions.size(); ++i)
                {
                    auto& vertex = vertexList[i];
                    vertex.position = result.positions[i];
                    vertex.normal = result.normals[i];
                    vertex.texCoord = float3(0.f);
                }

                shape.pTriangleMesh = Falcor::TriangleMesh::create(vertexList, result.indices);
                shape.pTriangleMesh->setName("loopsubdiv");
                shape.transform = entity.transform;
            }
            else
            {
                throwError(entity.loc, "Unknown shape type '{}'.", type);
            }

            // Reverse orientation.
            if (entity.reverseOrientation && shape.pTriangleMesh)
            {
                shape.pTriangleMesh->setFrontFaceCW(!shape.pTriangleMesh->getFrontFaceCW());
            }

            // Get the material.
            shape.pMaterial = ctx.getMaterial(entity.materialRef);

            // Create area light.
            if (entity.lightIndex != -1)
            {
                std::string nameSuffix = fmt::format("Emissive{}", entity.lightIndex);

                // Create a new material as we may already use it for other shapes with no area light attached to it.
                if (!std::holds_alternative<std::monostate>(entity.materialRef))
                {
                    shape.pMaterial = createMaterial(ctx, ctx.scene.getMaterial(entity.materialRef));
                    shape.pMaterial->setName(shape.pMaterial->getName() + "_" + nameSuffix);
                }
                else
                {
                    auto pStandardMaterial = Falcor::StandardMaterial::create(nameSuffix);
                    pStandardMaterial->setBaseColor(float4(0.f, 0.f, 0.f, 1.f));
                    pStandardMaterial->setRoughness(0.f);
                    shape.pMaterial = pStandardMaterial;
                }
                const SceneEntity& areaLightEntity = ctx.scene.getAreaLight(entity.lightIndex);
                createAreaLight(ctx, areaLightEntity, shape.pMaterial);
            }

            return shape;
        }

        InstanceDefinition createInstanceDefinition(BuilderContext& ctx, const InstanceDefinitionSceneEntity& entity)
        {
            InstanceDefinition instanceDefinition;

            for (const auto& shapeEntity : entity.shapes)
            {
                auto shape = createShape(ctx, shapeEntity);
                if (shape.pTriangleMesh)
                {
                    auto meshID = ctx.builder.addTriangleMesh(shape.pTriangleMesh, shape.pMaterial);
                    instanceDefinition.meshes.emplace_back(meshID, shape.transform);
                }
            }

            return instanceDefinition;
        }

        void buildScene(BuilderContext& ctx)
        {
            // Load float textures.
            for (const auto& [name, entity] : ctx.scene.getFloatTextures())
            {
                ctx.floatTextures.emplace(name, createFloatTexture(ctx, entity));
            }

            for (const auto& [name, entity] : ctx.scene.getSpectrumTextures())
            {
                ctx.spectrumTextures.emplace(name, createSpectrumTexture(ctx, entity));
            }

            // Create media.
            for (const auto& entity : ctx.scene.getMedia())
            {
                ctx.media.emplace(entity.name, createMedium(ctx, entity));
            }

            // Create named materials.
            for (const auto& [name, entity] : ctx.scene.getNamedMaterials())
            {
                ctx.namedMaterials.emplace(name, createMaterial(ctx, entity));
            }

            // Create unnamed materials.
            for (const auto& entity : ctx.scene.getMaterials())
            {
                ctx.materials.push_back(createMaterial(ctx, entity));
            }

            // Create camera.
            auto camera = createCamera(ctx, ctx.scene.getCamera());
            if (camera.pCamera)
            {
                auto nodeID = ctx.builder.addNode({ "camera", camera.transform * kInvertZ });
                camera.pCamera->setNodeID(nodeID);
                ctx.builder.addCamera(camera.pCamera);
            }

            // Create lights.
            for (const auto& entity : ctx.scene.getLights())
            {
                auto light = createLight(ctx, entity);
                if (light.pLight)
                {
                    ctx.builder.addLight(light.pLight);
                }
                if (light.pEnvMap)
                {
                    if (ctx.builder.getEnvMap() == nullptr)
                    {
                        ctx.builder.setEnvMap(light.pEnvMap);
                    }
                    else
                    {
                        logWarning(entity.loc, "No support for multiple infinite light. Discarding this light.");
                    }
                }
            }

            // Create shapes.
            for (const auto& entity : ctx.scene.getShapes())
            {
                auto shape = createShape(ctx, entity);
                if (shape.pTriangleMesh)
                {
                    auto nodeID = ctx.builder.addNode({ entity.name, shape.transform });
                    auto meshID = ctx.builder.addTriangleMesh(shape.pTriangleMesh, shape.pMaterial);
                    ctx.builder.addMeshInstance(nodeID, meshID);
                }
            }

            auto getInstanceDefinition = [&ctx](const InstanceSceneEntity& entity)
            {
                auto it = ctx.instanceDefinitions.find(entity.name);
                if (it == ctx.instanceDefinitions.end())
                {
                    auto it2 = ctx.scene.getInstanceDefinitions().find(entity.name);
                    if (it2 == ctx.scene.getInstanceDefinitions().end())
                    {
                        throwError(entity.loc, "Object instance '{}' not defined.", entity.name);
                    }
                    ctx.instanceDefinitions.emplace(entity.name, createInstanceDefinition(ctx, it2->second));
                    it = ctx.instanceDefinitions.find(entity.name);
                }
                return it->second;
            };

            // Create instanced shapes.
            for (const auto& entity : ctx.scene.getInstances())
            {
                const auto& instanceDefinition = getInstanceDefinition(entity);
                auto instanceTransform = entity.transform;

                // Instantiate meshes.
                for (const auto& [meshID, transform] : instanceDefinition.meshes)
                {
                    auto nodeID = ctx.builder.addNode({ "instance", instanceTransform * transform });
                    ctx.builder.addMeshInstance(nodeID, meshID);
                }
            }
        }
    }

    void PBRTImporter::import(const std::filesystem::path& path, SceneBuilder& builder, const SceneBuilder::InstanceMatrices& instances, const Dictionary& dict)
    {
        if (!instances.empty())
        {
            throw ImporterError(path, "PBRT importer does not support instancing.");
        }

        std::filesystem::path fullPath;
        if (!findFileInDataDirectories(path, fullPath))
        {
            throw ImporterError(path, "File not found.");
        }

        try
        {
            TimeReport timeReport;
            pbrt::BasicScene pbrtScene(fullPath.parent_path());
            pbrt::BasicSceneBuilder pbrtBuilder(pbrtScene);
            pbrt::parseFile(pbrtBuilder, fullPath);
            timeReport.measure("Parsing pbrt scene");

            pbrt::BuilderContext ctx { pbrtScene, builder };
            pbrt::buildScene(ctx);
            timeReport.measure("Building pbrt scene");
            timeReport.printToLog();

        }
        catch (const RuntimeError& e)
        {
            throw ImporterError(path, e.what());
        }
    }

    FALCOR_REGISTER_IMPORTER(
        PBRTImporter,
        Importer::ExtensionList({
            "pbrt"
        })
    )
}
