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
#pragma once
#include "Core/Macros.h"
#include "Core/Object.h"
#include "Core/Program/Program.h"
#include "Core/API/Raytracing.h"

#include <sigs/sigs.h>

namespace Falcor
{

struct AABB;
class EnvMap;
class ILightCollection;
class Camera;
class RenderContext;
class Light;
class MaterialSystem;
class Sampler;
struct ShaderVar;

class FALCOR_API IScene : public Object
{
public:
    /** Flags indicating if and what was updated in the scene.
     */
    enum class UpdateFlags
    {
        None = 0x0,                      ///< Nothing happened.
        GeometryMoved = 0x1,             ///< Geometry moved.
        CameraMoved = 0x2,               ///< The camera moved.
        CameraPropertiesChanged = 0x4,   ///< Some camera properties changed, excluding position.
        CameraSwitched = 0x8,            ///< Selected a different camera.
        LightsMoved = 0x10,              ///< Lights were moved.
        LightIntensityChanged = 0x20,    ///< Light intensity changed.
        LightPropertiesChanged = 0x40,   ///< Other light changes not included in LightIntensityChanged and LightsMoved.
        SceneGraphChanged = 0x80,        ///< Any transform in the scene graph changed.
        LightCollectionChanged = 0x100,  ///< Light collection changed (mesh lights).
        MaterialsChanged = 0x200,        ///< Materials changed.
        EnvMapChanged = 0x400,           ///< Environment map changed.
        EnvMapPropertiesChanged = 0x800, ///< Environment map properties changed (check EnvMap::getChanges() for more specific information).
        LightCountChanged = 0x1000,      ///< Number of active lights changed.
        RenderSettingsChanged = 0x2000,  ///< Render settings changed.
        GridVolumesMoved = 0x4000,       ///< Grid volumes were moved.
        GridVolumePropertiesChanged = 0x8000,  ///< Grid volume properties changed.
        GridVolumeGridsChanged = 0x10000,      ///< Grid volume grids changed.
        GridVolumeBoundsChanged = 0x20000,     ///< Grid volume bounds changed.
        CurvesMoved = 0x40000,                 ///< Curves moved.
        CustomPrimitivesMoved = 0x80000,       ///< Custom primitives moved.
        GeometryChanged = 0x100000,            ///< Scene geometry changed (added/removed).
        DisplacementChanged = 0x200000,        ///< Displacement mapping parameters changed.
        SDFGridConfigChanged = 0x400000,       ///< SDF grid config changed.
        SDFGeometryChanged = 0x800000,         ///< SDF grid geometry changed.
        MeshesChanged = 0x1000000,             ///< Mesh data changed (skinning or vertex animations).
        SceneDefinesChanged = 0x2000000,       ///< Scene defines changed. All programs that access the scene must be updated!
        TypeConformancesChanged = 0x4000000,   ///< Type conformances changed. All programs that access the scene must be updated!
        ShaderCodeChanged = 0x8000000,         ///< Shader code changed. All programs that access the scene must be updated!
        EmissiveMaterialsChanged = 0x10000000, ///< Emissive materials changed.

        /// Flags indicating that programs that access the scene need to be recompiled.
        /// This is needed if defines, type conformances, and/or the shader code has changed.
        /// The goal is to minimize changes that require recompilation, as it can be costly.
        RecompileNeeded = SceneDefinesChanged | TypeConformancesChanged | ShaderCodeChanged,

        All = -1
    };

    enum class TypeConformancesKind
    {
        None = 0,
        Material = (1 << 0),
        Geometry = (1 << 1),
        Other = (1 << 2),
        All = -1
    };

    /** Render settings determining how the scene is rendered.
        This is used primarily by the path tracer renderers.
    */
    struct RenderSettings
    {
        bool useEnvLight = true;        ///< Enable lighting from environment map.
        bool useAnalyticLights = true;  ///< Enable lighting from analytic lights.
        bool useEmissiveLights = true;  ///< Enable lighting from emissive lights.
        bool useGridVolumes = true;     ///< Enable rendering of grid volumes.

        // DEMO21
        float diffuseAlbedoMultiplier = 1.f;    ///< Fixed multiplier applied to material diffuse albedo.

        bool operator==(const RenderSettings& other) const
        {
            return (useEnvLight == other.useEnvLight) &&
                (useAnalyticLights == other.useAnalyticLights) &&
                (useEmissiveLights == other.useEmissiveLights) &&
                (useGridVolumes == other.useGridVolumes);
        }

        bool operator!=(const RenderSettings& other) const { return !(*this == other); }
    };

    static_assert(std::is_trivially_copyable<RenderSettings>() , "RenderSettings needs to be trivially copyable");

    virtual ~IScene() = default;

    virtual const ref<Device>& getDevice() const = 0;

    using UpdateFlagsSignal = sigs::Signal<void(IScene::UpdateFlags)>;

    virtual UpdateFlagsSignal::Interface getUpdateFlagsSignal() = 0;

    // All defines required by the Scene
    virtual void getShaderDefines(DefineList& defines) const = 0;
    // All type conformances required by the Scene
    virtual void getTypeConformances(TypeConformanceList& conformances, TypeConformancesKind kind = TypeConformancesKind::All) const = 0;
    // All shader modules required by the Scene
    virtual void getShaderModules(ProgramDesc::ShaderModuleList& shaderModuleList) const = 0;
    /// Assign all required variables into the Scene slang object, except TLAS and RayTypeCount.
    /// Pass in the ShaderVar, the Scene will bind to the correct global var name.
    virtual void bindShaderData(const ShaderVar& sceneVar) const = 0;

    /// On-demand creates TLAS and binds and the associated rayTypeCount to the scene.
    /// If rayTypeCount is not specified (== 0), will bind any available existing TLAS (creating it for rayTypeCount == 1 if no TLAS exists).
    /// Otherwise will bind TLAS for the specified rayTypeCount. Also calls bindShaderData().
    virtual void bindShaderDataForRaytracing(RenderContext* renderContext, const ShaderVar& sceneVar, uint32_t rayTypeCount = 0) = 0;

    /// TODO: Replace this with more generic thing
    virtual const RenderSettings& getRenderSettings() const = 0;

    /// Return the RtPipelineFlags relevant for the scene.
    virtual RtPipelineFlags getRtPipelineFlags() const = 0;

    /// Get AABB for the currently updated scene.
    virtual const AABB& getSceneBounds() const = 0;

    /// True when emissive lights are both enablee and present in the scene.
    /// Emissive lights are meshes with emissive material.
    /// TODO: Rename to GeometricLights (all lights are by definition emissive...)
    virtual bool useEmissiveLights() const = 0;

    /// True when environment map is both enabled and present in the scene.
    virtual bool useEnvLight() const = 0;

    virtual bool useAnalyticLights() const = 0;

    virtual const ref<EnvMap>& getEnvMap() const = 0;
    /// TODO: Remove the `renderContext` when not needed
    /// TODO: Ideally this wouldn't be non-const and build-on-demand (as Scene1 does)
    virtual ref<ILightCollection> getILightCollection(RenderContext* renderContext) = 0;

    /// Returns list of active analytic lights.
    /// TODO: This implementation requires lights to be tightly repacked every time activity
    /// changes, which might be suboptimal for editing, but better for rendering. Needs to be investigated.
    virtual const std::vector<ref<Light>>& getActiveAnalyticLights() const = 0;

    virtual const ref<Camera>& getCamera() const = 0;

    /// Used to get type conformances for materials.
    virtual const MaterialSystem& getMaterialSystem() const = 0;

    /// Allows changing the default texture sampler based on the pass requirements.
    /// This will be applied to all materials, old and new.
    virtual void setDefaultTextureSampler(const ref<Sampler>& pSampler) = 0;

public: /// Compatibility calls
    /// Convenience function when the Scene wants to do something besides just calling raytrace.
    /// TODO: Remove when no longer useful
    virtual void raytrace(RenderContext* renderContext, Program* pProgram, const ref<RtProgramVars>& pVars, uint3 dispatchDims);
};

FALCOR_ENUM_CLASS_OPERATORS(IScene::UpdateFlags);
FALCOR_ENUM_CLASS_OPERATORS(IScene::TypeConformancesKind);
} // namespace Falcor
