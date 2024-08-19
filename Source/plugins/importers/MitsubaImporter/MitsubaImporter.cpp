/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
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
#include "MitsubaImporter.h"
#include "Parser.h"
#include "Tables.h"
#include "Utils/Math/Common.h"
#include "Utils/Math/FalcorMath.h"
#include "Utils/Math/MathHelpers.h"
#include "Scene/Material/PBRT/PBRTDiffuseMaterial.h"
#include "Scene/Material/PBRT/PBRTDielectricMaterial.h"
#include "Scene/Material/PBRT/PBRTConductorMaterial.h"

#include <pybind11/pybind11.h>

#include <unordered_map>

namespace Falcor
{
namespace Mitsuba
{
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

Transform transformFromMatrix4x4(const float4x4& m)
{
    float3 scale, translation, skew;
    quatf rotation;
    float4 perspective;
    decompose(m, scale, rotation, translation, skew, perspective);

    Transform t;
    t.setScaling(scale);
    t.setRotation(rotation);
    t.setTranslation(translation);
    return t;
}

struct BuilderContext
{
    SceneBuilder& builder;
    std::unordered_map<std::string, XMLObject>& instances;
    std::unordered_set<std::string> warnings;

    void forEachReference(const XMLObject& inst, Class cls, std::function<void(const XMLObject&)> func)
    {
        for (const auto& [name, id] : inst.props.getNamedReferences())
        {
            const auto& child = instances[id];
            if (child.cls == cls)
                func(child);
        }
    }

    template<typename... Args>
    void logWarningOnce(const std::string_view fmtString, Args&&... args)
    {
        auto msg = fmt::format(fmtString, std::forward<Args>(args)...);
        auto it = warnings.find(msg);
        if (it == warnings.end())
        {
            warnings.insert(msg);
            Falcor::logWarning("MitsubaImporter: {}", msg);
        }
    }

    void unsupportedParameter(const std::string& name) { logWarningOnce("Parameter '{}' is not supported.", name); }

    void unsupportedType(const std::string& name) { logWarningOnce("Type '{}' is not supported.", name); }
};

struct ShapeInfo
{
    ref<TriangleMesh> pMesh;
    float4x4 transform;
    ref<Material> pMaterial;
};

struct SensorInfo
{
    ref<Camera> pCamera;
    float4x4 transform;
};

struct EmitterInfo
{
    ref<EnvMap> pEnvMap;
    ref<Light> pLight;
    float4x4 transform;
};

struct TextureInfo
{
    float4 value;
    ref<Texture> pTexture;
    float4x4 transform;
};

struct BSDFInfo
{
    ref<Material> pMaterial;
};

struct MediumInfo
{
    struct Homogeneous
    {
        using SharedPtr = std::shared_ptr<Homogeneous>;
        static SharedPtr create() { return SharedPtr(new Homogeneous()); }
        float3 sigmaS = float3(0.f);
        float3 sigmaA = float3(0.f);
    };

    Homogeneous::SharedPtr pHomogeneous;
};

float lookupIOR(const Properties& props, const std::string& name, const std::string& defaultIOR)
{
    if (props.hasFloat(name))
    {
        return props.getFloat(name);
    }
    else
    {
        return lookupIOR(props.getString(name, defaultIOR));
    }
}

TextureInfo buildTexture(BuilderContext& ctx, const XMLObject& inst)
{
    FALCOR_ASSERT(inst.cls == Class::Texture);

    const auto& props = inst.props;

    // Common properties.
    auto toUV = props.getTransform("to_uv", float4x4::identity());
    toUV = inverse(toUV);

    TextureInfo texture;

    if (inst.type == "bitmap")
    {
        auto filename = props.getString("filename");
        auto raw = props.getBool("raw", false);

        if (props.hasString("filter_type"))
            ctx.unsupportedParameter("filter_type");
        if (props.hasString("wrap_mode"))
            ctx.unsupportedParameter("wrap_mode");

        texture.pTexture = Texture::createFromFile(ctx.builder.getDevice(), filename, true, !raw);
        texture.transform = toUV;
    }
    else if (inst.type == "checkerboard")
    {
        auto color0 = props.getColor3("color0", Color3(0.4f));
        auto color1 = props.getColor3("color1", Color3(0.2f));

        const uint32_t kSize = 512;
        std::vector<float4> pixels(kSize * kSize);
        for (uint32_t y = 0; y < kSize; ++y)
        {
            for (uint32_t x = 0; x < kSize; ++x)
            {
                float4 c(1.f);
                c.xyz() = (x < kSize / 2) ^ (y < kSize / 2) ? color1 : color0;
                pixels[y * kSize + x] = c;
            }
        }

        auto pDevice = ctx.builder.getDevice();
        texture.pTexture = pDevice->createTexture2D(kSize, kSize, ResourceFormat::RGBA32Float, 1, Resource::kMaxPossible, pixels.data());
        texture.transform = toUV;
    }
    else
    {
        // Unsupported: mesh_attribute, volume
        ctx.unsupportedType(inst.type);
    }

    return texture;
}

TextureInfo lookupTexture(BuilderContext& ctx, const Properties& props, const std::string& name, float4 defaultValue)
{
    if (props.hasFloat(name))
    {
        return {float4(props.getFloat(name))};
    }
    else if (props.hasColor3(name))
    {
        return {props.getColor3(name)};
    }
    else if (props.hasNamedReference(name))
    {
        const auto& inst = ctx.instances[props.getNamedReference(name)];
        if (inst.cls != Class::Texture)
            FALCOR_THROW("Parameter '{}' needs to be a color or texture.", name);
        return buildTexture(ctx, inst);
    }
    else
    {
        return {defaultValue};
    }
}

void setMicrofacetProperties(ref<StandardMaterial> pMaterial, BuilderContext& ctx, const Properties& props, float defaultAlpha = 0.1f)
{
    if (props.hasString("distribution"))
        ctx.unsupportedParameter("distribution");
    if (props.hasBool("sample_visible"))
        ctx.unsupportedParameter("sample_visible");
    auto alpha = lookupTexture(ctx, props, "alpha", float4(defaultAlpha));
    if (alpha.pTexture)
        ctx.logWarningOnce("Microfacet alpha texture is not supported.");
    pMaterial->setRoughness(std::sqrt(alpha.pTexture ? defaultAlpha : alpha.value.x));
    // TODO: set roughness texture
}

BSDFInfo buildBSDF(BuilderContext& ctx, const XMLObject& inst)
{
    FALCOR_ASSERT(inst.cls == Class::BSDF);

    const auto& props = inst.props;

    ref<Material> pMaterial;

    if (inst.type == "diffuse")
    {
        auto pPBRTMaterial = PBRTDiffuseMaterial::create(ctx.builder.getDevice(), inst.id);
        auto reflectance = lookupTexture(ctx, props, "reflectance", float4(0.5f));
        if (reflectance.pTexture)
        {
            pPBRTMaterial->setTexture(Material::TextureSlot::BaseColor, reflectance.pTexture);
            pPBRTMaterial->setTextureTransform(transformFromMatrix4x4(reflectance.transform));
        }
        else
        {
            pPBRTMaterial->setBaseColor(reflectance.value);
        }

        pMaterial = pPBRTMaterial;
    }
    else if (inst.type == "dielectric" || inst.type == "roughdielectric")
    {
        auto pStandardMaterial = StandardMaterial::create(ctx.builder.getDevice(), inst.id);
        auto intIOR = lookupIOR(props, "int_ior", "bk7");
        auto extIOR = lookupIOR(props, "ext_ior", "air");

        if (props.hasFloat("specular_reflectance"))
            ctx.unsupportedParameter("specular_reflectance");
        if (props.hasFloat("specular_transmittance"))
            ctx.unsupportedParameter("specular_transmittance");

        pStandardMaterial->setSpecularTransmission(1.f);
        pStandardMaterial->setDoubleSided(true);
        pStandardMaterial->setRoughness(0.f);
        pStandardMaterial->setIndexOfRefraction(intIOR / extIOR);

        if (inst.type == "roughdielectric")
            setMicrofacetProperties(pStandardMaterial, ctx, props);

        pMaterial = pStandardMaterial;
    }
    else if (inst.type == "thindielectric")
    {
        auto pStandardMaterial = StandardMaterial::create(ctx.builder.getDevice(), inst.id);
        auto intIOR = lookupIOR(props, "int_ior", "bk7");
        auto extIOR = lookupIOR(props, "ext_ior", "air");

        if (props.hasFloat("specular_reflectance"))
            ctx.unsupportedParameter("specular_reflectance");
        if (props.hasFloat("specular_transmittance"))
            ctx.unsupportedParameter("specular_transmittance");

        pStandardMaterial->setSpecularTransmission(1.f);
        pStandardMaterial->setDoubleSided(true);
        pStandardMaterial->setThinSurface(true);
        pStandardMaterial->setRoughness(0.f);
        pStandardMaterial->setIndexOfRefraction(intIOR / extIOR);

        pMaterial = pStandardMaterial;
    }
    else if (inst.type == "conductor" || inst.type == "roughconductor")
    {
        auto pPBRTMaterial = PBRTConductorMaterial::create(ctx.builder.getDevice(), inst.id);
        if (props.hasFloat("specular_reflectance"))
            ctx.unsupportedParameter("specular_reflectance");

        if (props.hasColor3("eta") && props.hasColor3("k"))
        {
            auto eta = props.getColor3("eta");
            pPBRTMaterial->setBaseColor(float4(eta.r, eta.g, eta.b, 1.f));
            pPBRTMaterial->setTransmissionColor(props.getColor3("k"));
        }

        pPBRTMaterial->setRoughness(float2(0.f));
        pPBRTMaterial->setDoubleSided(true);

        if (inst.type == "roughconductor")
        {
            const float defaultAlpha = 0.1f;
            auto alpha = lookupTexture(ctx, props, "alpha", float4(defaultAlpha));
            if (alpha.pTexture)
                ctx.logWarningOnce("Microfacet alpha texture is not supported.");
            pPBRTMaterial->setRoughness(alpha.pTexture ? float2(defaultAlpha) : alpha.value.xy());
        }

        pMaterial = pPBRTMaterial;
    }
    else if (inst.type == "plastic" || inst.type == "roughplastic")
    {
        auto pStandardMaterial = StandardMaterial::create(ctx.builder.getDevice(), inst.id);
        auto diffuseReflectance = lookupTexture(ctx, props, "diffuse_reflectance", float4(0.5f));
        if (diffuseReflectance.pTexture)
        {
            pStandardMaterial->setTexture(Material::TextureSlot::BaseColor, diffuseReflectance.pTexture);
            pStandardMaterial->setTextureTransform(transformFromMatrix4x4(diffuseReflectance.transform));
        }
        else
        {
            pStandardMaterial->setBaseColor(diffuseReflectance.value);
        }

        auto intIOR = lookupIOR(props, "int_ior", "polypropylene");
        auto extIOR = lookupIOR(props, "ext_ior", "air");

        if (props.hasBool("nonlinear"))
            ctx.unsupportedParameter("nonlinear");
        if (props.hasFloat("specular_reflectance"))
            ctx.unsupportedParameter("specular_reflectance");

        pStandardMaterial->setRoughness(0.f);
        pStandardMaterial->setIndexOfRefraction(intIOR / extIOR);

        if (inst.type == "roughplastic")
            setMicrofacetProperties(pStandardMaterial, ctx, props);

        pMaterial = pStandardMaterial;
    }
    else if (inst.type == "twosided")
    {
        ref<Material> pInnerMaterial = nullptr;
        for (const auto& [name, id] : props.getNamedReferences())
        {
            const auto& child = ctx.instances[id];
            if (child.cls == Class::BSDF)
            {
                if (pInnerMaterial)
                    FALCOR_THROW("'twosided' BSDF can only have one nested BSDF.");
                pInnerMaterial = buildBSDF(ctx, child).pMaterial;
                if (pInnerMaterial)
                    pMaterial->setDoubleSided(true);
            }
        }

        pMaterial = pInnerMaterial;
    }
    else
    {
        ctx.unsupportedType(inst.type);
        pMaterial = nullptr;
    }

    return {pMaterial};
}

MediumInfo buildMedium(BuilderContext& ctx, const XMLObject& inst)
{
    FALCOR_ASSERT(inst.cls == Class::Medium);

    const auto& props = inst.props;

    MediumInfo medium;

    if (inst.type == "homogeneous")
    {
        auto scale = props.getFloat("scale", 1.f);

        if (props.hasString("material"))
        {
            ctx.unsupportedParameter("material");
        }
        else if (props.hasColor3("sigma_s") && props.hasColor3("sigma_a"))
        {
            float3 sigmaS = props.getColor3("sigma_s");
            float3 sigmaA = props.getColor3("sigma_a");
            medium.pHomogeneous = MediumInfo::Homogeneous::create();
            medium.pHomogeneous->sigmaS = scale * sigmaS;
            medium.pHomogeneous->sigmaA = scale * sigmaA;
        }
        else if (props.hasColor3("albedo") && props.hasColor3("sigma_t"))
        {
            float3 albedo = props.getColor3("albedo");
            float3 sigmaT = props.getColor3("sigma_s");
            medium.pHomogeneous = MediumInfo::Homogeneous::create();
            medium.pHomogeneous->sigmaS = scale * (albedo * sigmaT);
            medium.pHomogeneous->sigmaA = scale * (sigmaT - medium.pHomogeneous->sigmaS);
        }
    }
    else
    {
        // Unsupported: heterogeneous
        ctx.unsupportedType(inst.type);
    }

    return medium;
}

ShapeInfo buildShape(BuilderContext& ctx, const XMLObject& inst)
{
    FALCOR_ASSERT(inst.cls == Class::Shape);

    const auto& props = inst.props;

    // Common properties.
    auto toWorld = props.getTransform("to_world", float4x4::identity());
    auto flipNormals = props.getBool("flip_normals", false);
    if (props.hasBool("flip_normals"))
        ctx.unsupportedParameter("flip_normals");

    const float4x4 transformYtoZ({1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f});

    ShapeInfo shape;

    if (inst.type == "obj" || inst.type == "ply")
    {
        auto filename = props.getString("filename");
        auto faceNormals = props.getBool("face_normals", false);
        auto flipTexCoords = props.getBool("flip_tex_coords", true);

        if (props.hasBool("flip_tex_coords"))
            ctx.unsupportedParameter("flip_tex_coords");

        TriangleMesh::ImportFlags flags = TriangleMesh::ImportFlags::None;
        if (faceNormals)
        {
            flags = TriangleMesh::ImportFlags::JoinIdenticalVertices;
        }
        else
        {
            // Recommend `faceNormals=false` for inverse/differentiable rendering to avoid vertex duplication.
            flags = TriangleMesh::ImportFlags::GenSmoothNormals | TriangleMesh::ImportFlags::JoinIdenticalVertices;
        }

        shape.pMesh = TriangleMesh::createFromFile(filename, flags);
        if (shape.pMesh)
            shape.pMesh->setName(inst.id);
        shape.transform = toWorld;
    }
    else if (inst.type == "sphere")
    {
        auto center = props.getFloat3("center", float3(0.f));
        auto radius = props.getFloat("radius", 1.f);

        shape.pMesh = TriangleMesh::createSphere(radius);
        shape.pMesh->setName(inst.id);
        shape.transform = mul(toWorld, math::matrixFromTranslation(center));
    }
    else if (inst.type == "disk")
    {
        shape.pMesh = TriangleMesh::createDisk(1.f);
        shape.pMesh->setName(inst.id);
        shape.transform = mul(toWorld, transformYtoZ);
    }
    else if (inst.type == "rectangle")
    {
        shape.pMesh = TriangleMesh::createQuad(float2(2.f));
        shape.pMesh->setName(inst.id);
        shape.transform = mul(toWorld, transformYtoZ);
    }
    else if (inst.type == "cube")
    {
        shape.pMesh = TriangleMesh::createCube(float3(2.f));
        shape.pMesh->setName(inst.id);
        shape.transform = toWorld;
    }
    else
    {
        ctx.unsupportedType(inst.type);
    }

    // Look for nested BSDF.
    for (const auto& [name, id] : props.getNamedReferences())
    {
        const auto& child = ctx.instances[id];
        if (child.cls == Class::BSDF)
        {
            if (shape.pMaterial)
                FALCOR_THROW("Shape can only have one BSDF.");
            auto bsdf = buildBSDF(ctx, child);
            shape.pMaterial = bsdf.pMaterial;
        }
    }

    // Create default material.
    if (!shape.pMaterial)
        shape.pMaterial = StandardMaterial::create(ctx.builder.getDevice());

    // Look for interior medium.
    for (const auto& [name, id] : props.getNamedReferences())
    {
        const auto& child = ctx.instances[id];
        if (child.cls == Class::Medium && name == "interior")
        {
            auto medium = buildMedium(ctx, child);
            if (medium.pHomogeneous)
            {
                auto pBasicMaterial = static_cast<BasicMaterial*>(shape.pMaterial.get());
                pBasicMaterial->setVolumeScattering(medium.pHomogeneous->sigmaS);
                pBasicMaterial->setVolumeAbsorption(medium.pHomogeneous->sigmaA);
            }
        }
    }

    // Look for nested area emitter.
    for (const auto& [name, id] : props.getNamedReferences())
    {
        const auto& child = ctx.instances[id];
        if (child.cls == Class::Emitter && child.type == "area")
        {
            float4 baseColor;
            if (shape.pMaterial)
            {
                if (shape.pMaterial->getType() != MaterialType::PBRTDiffuse)
                {
                    FALCOR_THROW("Shape with area emitter must have a diffuse material.");
                }

                // Store base color.
                auto pBasicMaterial = static_cast<BasicMaterial*>(shape.pMaterial.get());
                baseColor = pBasicMaterial->getBaseColor();

                shape.pMaterial = nullptr;
            }

            // Create a new StandardMaterial with emissive properties.
            shape.pMaterial = StandardMaterial::create(ctx.builder.getDevice());
            float3 radiance = child.props.getColor3("radiance");
            float factor = std::max(radiance.x, std::max(radiance.y, radiance.z));
            if (factor > 0.f)
                radiance /= factor;

            auto pMaterial = static_cast<StandardMaterial*>(shape.pMaterial.get());
            pMaterial->setEmissiveColor(radiance);
            pMaterial->setEmissiveFactor(factor);
            pMaterial->setMetallic(0.f);
            pMaterial->setRoughness(1.f);
            pMaterial->setBaseColor(baseColor);
        }
    }

    return shape;
}

SensorInfo buildSensor(BuilderContext& ctx, const XMLObject& inst)
{
    FALCOR_ASSERT(inst.cls == Class::Sensor)

    const auto& props = inst.props;

    // Common properties.
    auto toWorld = props.getTransform("to_world", float4x4::identity());

    // Check for film to get resolution.
    uint32_t width = 768;
    uint32_t height = 576;
    for (const auto& [name, id] : props.getNamedReferences())
    {
        const auto& child = ctx.instances[id];
        if (child.cls == Class::Film)
        {
            width = (uint32_t)child.props.getInt("width", 768);
            height = (uint32_t)child.props.getInt("height", 576);
        }
    }

    SensorInfo sensor;

    if (inst.type == "perspective" || inst.type == "thinlens")
    {
        if (props.hasFloat("focal_length") && props.hasFloat("fov"))
        {
            FALCOR_THROW("Cannot specify both 'focal_length' and 'fov'.");
        }
        float focalLength = props.getFloat("focal_length", 50.f);
        if (props.hasFloat("fov"))
        {
            float filmWidth = (24.f / height) * width;
            focalLength = fovYToFocalLength(math::radians(props.getFloat("fov")), filmWidth);
        }

        if (props.hasString("fov_axis"))
            ctx.unsupportedParameter("fov_axis");

        // TODO handle fov_axis
        auto pCamera = Camera::create();
        pCamera = Camera::create();
        pCamera->setFocalLength(focalLength);
        // pCamera->setFrameHeight(24.f);
        pCamera->setNearPlane(props.getFloat("near_clip", 1e-2f));
        pCamera->setFarPlane(props.getFloat("far_clip", 1e4f));
        pCamera->setFocalDistance(props.getFloat("focus_distance", 1.f));
        pCamera->setApertureRadius(props.getFloat("aperture_radius", 0.f));

        const float4x4 flipZ({1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, -1.f, 0.f, 0.f, 0.f, 0.f, 1.f});

        sensor.pCamera = pCamera;
        sensor.transform = mul(toWorld, flipZ);
    }
    else
    {
        // Unsupported: orthographic, radiancemeter, irradiancemeter, distant, batch
        ctx.unsupportedType(inst.type);
    }

    return sensor;
}

EmitterInfo buildEmitter(BuilderContext& ctx, const XMLObject& inst)
{
    FALCOR_ASSERT(inst.cls == Class::Emitter);

    const auto& props = inst.props;

    // Common properties.
    auto toWorld = props.getTransform("to_world", float4x4::identity());

    EmitterInfo emitter;

    if (inst.type == "area")
    {
        FALCOR_THROW("'area' emitter needs to be nested in a shape.");
    }
    else if (inst.type == "constant")
    {
        auto radiance = props.getColor3("radiance");
        float4 data = radiance;
        auto pDevice = ctx.builder.getDevice();
        auto pTexture = pDevice->createTexture2D(1, 1, ResourceFormat::RGBA32Float, 1, Texture::kMaxPossible, &data);
        auto pEnvMap = EnvMap::create(ctx.builder.getDevice(), pTexture);
        emitter.pEnvMap = pEnvMap;
    }
    else if (inst.type == "envmap")
    {
        auto filename = props.getString("filename");
        auto scale = props.getFloat("scale", 1.f);
        auto pEnvMap = EnvMap::createFromFile(ctx.builder.getDevice(), filename);
        if (pEnvMap)
        {
            const float4x4 flipZ({1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, -1.f, 0.f, 0.f, 0.f, 0.f, 1.f});
            toWorld = mul(toWorld, flipZ);

            pEnvMap->setIntensity(scale);
            float3 rotation;
            extractEulerAngleXYZ(toWorld, rotation.x, rotation.y, rotation.z);
            pEnvMap->setRotation(math::degrees(rotation));
        }
        emitter.pEnvMap = pEnvMap;
    }
    else if (inst.type == "point")
    {
        auto intensity = props.getColor3("intensity", Color3(1.f));
        // TODO optional position prop

        if (props.hasFloat3("position"))
        {
            if (props.hasTransform("to_world"))
                FALCOR_THROW("Either 'to_world' or 'position' can be specified, not both.");
            toWorld = math::matrixFromTranslation(props.getFloat3("position"));
        }

        ctx.unsupportedType(inst.type);
    }
    else
    {
        // Unsupported: spot, projector, directional, directionalarea
        ctx.unsupportedType(inst.type);
    }

    return emitter;
}

void buildScene(BuilderContext& ctx, const XMLObject& inst)
{
    FALCOR_ASSERT(inst.cls == Class::Scene);

    const auto& props = inst.props;

    for (const auto& [name, id] : props.getNamedReferences())
    {
        const auto& child = ctx.instances[id];

        switch (child.cls)
        {
        case Class::Sensor:
        {
            auto sensor = buildSensor(ctx, child);

            if (sensor.pCamera)
            {
                SceneBuilder::Node node{id, sensor.transform};
                auto nodeID = ctx.builder.addNode(node);
                sensor.pCamera->setNodeID(nodeID);
                ctx.builder.addCamera(sensor.pCamera);
            }
        }
        break;

        case Class::Emitter:
        {
            auto emitter = buildEmitter(ctx, child);

            if (emitter.pEnvMap)
            {
                if (ctx.builder.getEnvMap() != nullptr)
                    FALCOR_THROW("Only one envmap can be added.");
                ctx.builder.setEnvMap(emitter.pEnvMap);
            }
            else if (emitter.pLight)
            {
                SceneBuilder::Node node{id, emitter.transform};
                auto nodeID = ctx.builder.addNode(node);
                emitter.pLight->setNodeID(nodeID);
                ctx.builder.addLight(emitter.pLight);
            }
        }
        break;

        case Class::Shape:
        {
            auto shape = buildShape(ctx, child);

            if (shape.pMesh && shape.pMaterial)
            {
                SceneBuilder::Node node{id, shape.transform};
                auto nodeID = ctx.builder.addNode(node);
                auto meshID = ctx.builder.addTriangleMesh(shape.pMesh, shape.pMaterial);
                ctx.builder.addMeshInstance(nodeID, meshID);
            }
        }
        break;
        }
    }
}

} // namespace Mitsuba

std::unique_ptr<Importer> MitsubaImporter::create()
{
    return std::make_unique<MitsubaImporter>();
}

void MitsubaImporter::importScene(
    const std::filesystem::path& path,
    SceneBuilder& builder,
    const std::map<std::string, std::string>& materialToShortName
)
{
    if (!path.is_absolute())
        throw ImporterError(path, "Path must be absolute.");

    try
    {
        pugi::xml_document doc;
        auto result = doc.load_file(path.c_str(), pugi::parse_default | pugi::parse_comments);
        if (!result)
            throw ImporterError(path, "Failed to parse XML: {}", result.description());

        Mitsuba::XMLSource src{path.string(), doc};
        Mitsuba::XMLContext ctx;
        ctx.resolver.append(std::filesystem::path(path).parent_path());
        Mitsuba::Properties props;
        pugi::xml_node root = doc.document_element();
        size_t argCounter = 0;
        auto sceneID = Mitsuba::parseXML(src, ctx, root, Mitsuba::Tag::Invalid, props, argCounter).second;

        Mitsuba::BuilderContext builderCtx{builder, ctx.instances};
        Mitsuba::buildScene(builderCtx, builderCtx.instances[sceneID]);
    }
    catch (const RuntimeError& e)
    {
        throw ImporterError(path, e.what());
    }
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<Importer, MitsubaImporter>();
}

} // namespace Falcor
