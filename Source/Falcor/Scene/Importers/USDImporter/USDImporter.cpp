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
#include "USDImporter.h"
#include "USDHelpers.h"
#include "ImporterContext.h"
#include "Core/Platform/OS.h"
#include "Utils/Timing/TimeReport.h"
#include "Utils/Settings.h"
#include "Scene/Importer.h"

#include <glm/gtx/transform.hpp>

BEGIN_DISABLE_USD_WARNINGS
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/ar/resolver.h>
#include <pxr/usd/usdGeom/bboxCache.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdGeom/scope.h>
#include <pxr/usd/usdLux/distantLight.h>
#include <pxr/usd/usdLux/domeLight.h>
#include <pxr/usd/usdLux/rectLight.h>
#include <pxr/usd/usdLux/sphereLight.h>
#include <pxr/usd/usdLux/diskLight.h>
#include <pxr/usd/usdGeom/basisCurves.h>
#include <pxr/usd/usdSkel/root.h>
END_DISABLE_USD_WARNINGS

namespace Falcor
{
    bool checkPrim(const UsdPrim& prim)
    {
        std::string primName = prim.GetPath().GetString();
        if (gpFramework->getSettings().getAttribute(primName, "usdImporter:dropPrim", false))
            return false;

        return true;
    }

    // Traverse scene graph, converting supported prims from USD to Falcor equivalents
    void traversePrims(const UsdPrim& rootPrim, ImporterContext& ctx)
    {
        Usd_PrimFlagsPredicate pred = UsdPrimDefaultPredicate;
        if (ctx.useInstanceProxies)
        {
            // Treat instances as if they were unique prims (primarily for debugging)
            pred.TraverseInstanceProxies(true);
        }

        UsdPrimRange range = UsdPrimRange::PreAndPostVisit(rootPrim, pred);

        for (auto it = range.begin(); it != range.end(); ++it)
        {
            UsdPrim prim = *it;
            std::string primName = prim.GetPath().GetString();

            if (!it.IsPostVisit())
            {
                // Pre visits

                // If this prim has an xform associated with it, push it onto the xform stack
                if (prim.IsA<UsdGeomXformable>())
                {
                    ctx.pushNode(UsdGeomXformable(prim));
                }

                if (prim.IsA<UsdGeomImageable>() && !isRenderable(UsdGeomImageable(prim)))
                {
                    logDebug("Pruning non-renderable prim '{}'.", primName);
                    it.PruneChildren();
                    continue;
                }

                if (prim.IsInstance() && !ctx.useInstanceProxies)
                {
                    if (!checkPrim(prim)) continue;

                    const UsdPrim protoPrim(prim.GetMaster());

                    if (protoPrim.IsValid())
                    {
                        logDebug("Adding instance '{}' of '{}'.", primName, protoPrim.GetPath().GetString());
                        PrototypeInstance protoInst = { primName, protoPrim, ctx.nodeStack.back() };

                        ctx.addPrototypeInstance(protoInst);
                    }
                    else
                    {
                        logError("No valid prototype prim for instance '{}'.", primName);
                        it.PruneChildren();
                        continue;
                    }
                }
                else if (prim.IsA<UsdGeomPointInstancer>())
                {
                    if (!checkPrim(prim)) continue;

                    logDebug("Processing point instancer '{}'.", primName);
                    ctx.createPointInstances(prim);
                    it.PruneChildren();
                }
                else if (prim.IsA<UsdGeomMesh>())
                {
                    if (!checkPrim(prim)) continue;

                    logDebug("Adding mesh '{}'.", primName);
                    ctx.addMesh(prim);
                    rmcv::mat4 bindXform = ctx.getGeomBindTransform(prim);
                    ctx.addGeomInstance(primName, prim, rmcv::mat4(1.f), bindXform);
                }
                else if (prim.IsA<UsdGeomBasisCurves>())
                {
                    if (!checkPrim(prim)) continue;

                    logDebug("Adding curve '{}' for linear swept sphere tessellation.", primName);
                    ctx.addCurve(prim);

                    // TODO: Add support for curve instancing
                    // Now we assume each curve has only one instance.
                    ctx.addCurveInstance(primName, prim, rmcv::mat4(1.f), ctx.nodeStack.back());
                }
                else if (prim.IsA<UsdSkelRoot>())
                {
                    logDebug("Processing Skeleton '{}'.", primName);
                    ctx.createSkeleton(prim);
                }
                else if (prim.IsA<UsdLuxDistantLight>())
                {
                    logDebug("Processing distant light '{}'.", primName);
                    ctx.createDistantLight(prim);
                }
                else if (prim.IsA<UsdLuxRectLight>())
                {
                    logDebug("Processing rect light '{}'.", primName);
                    ctx.createRectLight(prim);
                }
                else if (prim.IsA<UsdLuxSphereLight>())
                {
                    logDebug("Processing sphere light '{}'.", primName);
                    ctx.createSphereLight(prim);
                }
                else if (prim.IsA<UsdLuxDiskLight>())
                {
                    logDebug("Processing disk light '{}'.", primName);
                    if (gpFramework->getSettings().getOption("usdImporter:meshDiskLight", false))
                    {
                        ctx.createMeshedDiskLight(prim);
                    }
                    else
                    {
                        ctx.createDiskLight(prim);
                    }
                }
                else if (prim.IsA<UsdLuxDomeLight>())
                {
                    logDebug("Processing dome light '{}'.", primName);
                    ctx.createEnvMap(prim);
                }
                else if (prim.IsA<UsdGeomCamera>())
                {
                    logDebug("Processing camera '{}'.", primName);
                    ctx.createCamera(prim);
                }
                else if (prim.IsA<UsdGeomXform>())
                {
                    logDebug("Processing xform '{}'.", primName);
                    // Processing of this UsdGeomXformable performed above
                }
                else if (prim.IsA<UsdShadeMaterial>() ||
                    prim.IsA<UsdShadeShader>() ||
                    prim.IsA<UsdGeomSubset>())
                {
                    // No processing to do; ignore without issuing a warning.
                    it.PruneChildren();
                }
                else if (prim.IsA<UsdGeomScope>())
                {
                    logDebug("Processing scope '{}'.", primName);
                }
                else if (!prim.GetTypeName().GetString().empty())
                {
                    logWarning("Ignoring prim '{}' of unsupported type {}.", primName, prim.GetTypeName().GetString());
                    it.PruneChildren();
                }
            }
            else
            {
                // Post visits
                if (prim.IsA<UsdGeomXformable>())
                {
                    ctx.popNode();
                }
            }
        }
    }

    template <typename T>
    T getMetadata(VtDictionary& renderDict, const std::string& key, T defaultValue)
    {
        const auto& iter = renderDict.find(key);
        if (iter != renderDict.end())
        {
            pxr::VtValue& vtval = iter->second;
            if (vtval.GetTypeid() != typeid(T))
            {
                // If the type of the value stored in the USD dictionary is not the same as we are expecting, cast to the proper type if possible.
                if (!vtval.CanCastToTypeid(typeid(T)))
                {
                    logWarning(
                        "USD metadata parameter '{}' is of type '{}', which is not compatible with the equivalent Falcor parameter of type '{}'. Using default value instead.",
                        key, vtval.GetTypeid().name(), typeid(T).name()
                    );
                    return defaultValue;
                }
                else
                {
                    vtval = vtval.CastToTypeid(typeid(T));
                }
            }
            return vtval.Get<T>();
        }
        return defaultValue;
    }

    // Convert entries in the USD renderSettings dictionary to their Falcor equivalents, if any.
    Scene::Metadata createMetadata(UsdStageRefPtr& pStage)
    {
        VtDictionary customLayerDict;
        if (!pStage->GetMetadata(TfToken("customLayerData"), &customLayerDict))
        {
            // Not custom layer metadata
            return {};
        }

        if (customLayerDict.find("renderSettings") == customLayerDict.end())
        {
            // No render settings dictionary
            return {};
        }

        // Found an OV custom rendering dictionary. Convert relevant parameters to Falcor equivalents.
        // If a param value isn't explicitly set in the renderSettings dictionary, use the default OV/Create value.

        VtDictionary renderDict = customLayerDict["renderSettings"].Get<VtDictionary>();
        Scene::Metadata meta;

        //                                                     Create parameter name                                Default Create value
        meta.fNumber =                getMetadata(renderDict, "rtx:post:tonemap:fNumber",                           5.f);
        meta.filmISO =                getMetadata(renderDict, "rtx:post:tonemap:filmIso",                           100.f);
        meta.shutterSpeed =           getMetadata(renderDict, "rtx:post:tonemap:cameraShutter",                     50.f);
        meta.samplesPerPixel =        getMetadata(renderDict, "rtx:pathtracing:spp",                                (uint32_t)1);
        meta.maxDiffuseBounces =      getMetadata(renderDict, "rtx:pathtracing:maxBounces",                         (uint32_t)4);
        meta.maxSpecularBounces =     getMetadata(renderDict, "rtx:pathtracing:maxSpecularAndTransmissionBounces",  (uint32_t)6);
        meta.maxTransmissionBounces = getMetadata(renderDict, "rtx:pathtracing:maxSpecularAndTransmissionBounces",  (uint32_t)6);
        meta.maxVolumeBounces =       getMetadata(renderDict, "rtx:pathtracing:maxVolumeBounces",                   (uint32_t)4);

        // Falcor's "0 bounce" includes GBuffer output (i.e, primary visibility), while Create's does not.
        // Further, each Falcor bounce includes NEE, while Create's path tracer's NEE requires an extra bounce.
        //
        // Taken together, a decent, although not entirely accurate, mapping from Create to Falcor bounce parameters
        // subtracts 2: one for the extra GBuffer visibility "bounce", and one for the extra NEE bounce.
        //
        // Also, in Create, non-diffuse transport-mode-specific bounce counts are the maximum of maxBounces/maxDiffuseBounces and the specified count.
        // Take care not to underflow the uint32_t values.
        meta.maxDiffuseBounces = std::max(2U, meta.maxDiffuseBounces.value()) - 2U;
        meta.maxSpecularBounces = std::max(meta.maxDiffuseBounces.value(), std::max(2U, meta.maxSpecularBounces.value()) - 2U);
        meta.maxTransmissionBounces = std::max(meta.maxDiffuseBounces.value(), std::max(2U, meta.maxTransmissionBounces.value()) - 2U);
        meta.maxVolumeBounces = std::max(meta.maxDiffuseBounces.value(), std::max(2U, meta.maxVolumeBounces.value()) - 2U);

        return meta;
    }

    void USDImporter::import(const std::filesystem::path& path, SceneBuilder& builder, const SceneBuilder::InstanceMatrices& instances, const Dictionary& dict)
    {
        TimeReport timeReport;

        if (!instances.empty())
        {
            throw ImporterError(path, "USD importer does not support instancing.");
        }

        DiagDelegate diagnosticDelegate;
        TfDiagnosticMgr::GetInstance().AddDelegate(&diagnosticDelegate);

        // Remove the diagnostic delegate from the TfDiagnosticMgr upon return.
        ScopeGuard removeDiagDelegate{ [&]() { TfDiagnosticMgr::GetInstance().RemoveDelegate(&diagnosticDelegate); } };

        std::filesystem::path fullPath;
        if (findFileInDataDirectories(path, fullPath) == false)
        {
            throw ImporterError(path, "File not found.");
        }

        ArGetResolver().ConfigureResolverForAsset(fullPath.string());

        UsdStageRefPtr pStage = UsdStage::Open(fullPath.string());
        if (!pStage)
        {
            throw ImporterError(path, "Failed to open USD stage.");
        }

        timeReport.measure("Open stage");

        ImporterContext ctx(path, pStage, builder, dict, timeReport);

        // Falcor uses meter scene unit; scale if necessary. Note that Omniverse uses cm by default.
        ctx.metersPerUnit = float(UsdGeomGetStageMetersPerUnit(pStage));

        ctx.timeCodesPerSecond = pStage->GetTimeCodesPerSecond();

        // Instantiate a bbox cache and compute stage dimensions in order to set default camera params to reasonable values
        // Don't include prims that are marked as guides or proxies.
        TfTokenVector purposes(pxr::UsdGeomImageable::GetOrderedPurposeTokens());
        purposes.erase(std::remove(purposes.begin(), purposes.end(), UsdGeomTokens->guide), purposes.end());
        purposes.erase(std::remove(purposes.begin(), purposes.end(), UsdGeomTokens->proxy), purposes.end());
        UsdGeomBBoxCache bboxCache(pxr::UsdTimeCode().EarliestTime(), purposes);

        UsdPrim rootPrim = pStage->GetPseudoRoot();
        GfBBox3d stageBound = bboxCache.ComputeWorldBound(rootPrim);
        GfVec3d stageCenter = stageBound.GetRange().GetMidpoint();
        GfVec3d stageSize = stageBound.GetRange().GetSize();

        float stageDiagonal = (float)sqrt(stageSize[0] * stageSize[0] + stageSize[1] * stageSize[1] + stageSize[2] * stageSize[2]);
        if (isfinite(stageDiagonal) && stageDiagonal > 0.f)
        {
            ctx.builder.setCameraSpeed(0.025f * stageDiagonal * ctx.metersPerUnit);
        }

        Scene::Metadata metadata = createMetadata(pStage);
        ctx.builder.setMetadata(metadata);

        timeReport.measure("Load scene settings");


        if (!ctx.useInstanceProxies)
        {
            // Create prototypes for all prototype prims.
            ctx.pushNodeStack();
            std::vector<UsdPrim> prototypes(pStage->GetMasters());
            for (const UsdPrim& rootPrim : prototypes)
            {
                if (!checkPrim(rootPrim)) continue;
                ctx.createPrototype(rootPrim);
            }
            ctx.popNodeStack();
            FALCOR_ASSERT(ctx.getNodeStackDepth() == 0);
        }

        // Initialize stage-to-Falcor transformation based on specified stage up and unit scaling which
        // accounts for any differences in scene units and 'up' orientation between the stage and Falcor
        rmcv::mat4 rootXform = rmcv::scale(float3(ctx.metersPerUnit));
        if (UsdGeomGetStageUpAxis(pStage) == UsdGeomTokens->z)
        {
            rootXform = rmcv::eulerAngleX(glm::radians(-90.0f)) * rootXform;
        }
        else
        {
            FALCOR_ASSERT(UsdGeomGetStageUpAxis(pStage) == UsdGeomTokens->y);
        }

        // A root prim in USD doesn't have an associated xform, so we must manually set the root transform we have computed.
        ctx.setRootXform(rootXform);

        // Traverse the stage, converting USD prims to Falcor equivalents
        traversePrims(rootPrim, ctx);

        // Only the stage root xform should remain.
        FALCOR_ASSERT(ctx.getNodeStackDepth() == 1);

        timeReport.measure("Traverse prims");

        ctx.finalize();

        if (ctx.builder.getCameras().empty())
        {
            // No camera specified; attempt to create a reasonable default
            float3 viewDir = normalize(float3(-1.f, -1.f, -1.f));
            float3 target = toGlm(stageCenter * ctx.metersPerUnit);
            float3 pos = target - (viewDir * stageDiagonal * 1.5f * ctx.metersPerUnit);
            float3 up(0.f, 1.f, 0.f);

            Camera::SharedPtr pCamera = Camera::create("Default");
            pCamera->setPosition(pos);
            pCamera->setTarget(target);
            pCamera->setUpVector(up);
            pCamera->setFocalLength(18.f);
            pCamera->setDepthRange(0.001f, 4.f * stageDiagonal * ctx.metersPerUnit);
            ctx.builder.addCamera(pCamera);
        }

        timeReport.printToLog();
    }

    FALCOR_REGISTER_IMPORTER(
        USDImporter,
        Importer::ExtensionList({
            "usd",
            "usda",
            "usdc"
        })
    )
}
