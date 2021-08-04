/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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
#include "TestRtProgram.h"

const char* TestRtProgram::kDesc = "Test pass for RtProgram";

namespace
{
    const char kShaderFilename[] = "RenderPasses/TestPasses/TestRtProgram.rt.slang";

    // Ray tracing program settings. Set as small values as possible.
    const uint32_t kMaxPayloadSizeBytes = 16;
    const uint32_t kMaxAttributeSizeBytes = 8;
    const uint32_t kMaxRecursionDepth = 1;

    const char kOutput[] = "output";

    std::mt19937 rng;
}

void TestRtProgram::registerScriptBindings(pybind11::module& m)
{
    pybind11::class_<TestRtProgram, RenderPass, TestRtProgram::SharedPtr> pass(m, "TestRtProgram");
    pass.def("addCustomPrimitive", &TestRtProgram::addCustomPrimitive);
    pass.def("removeCustomPrimitive", &TestRtProgram::removeCustomPrimitive);
    pass.def("moveCustomPrimitive", &TestRtProgram::moveCustomPrimitive);
}

TestRtProgram::SharedPtr TestRtProgram::create(RenderContext* pRenderContext, const Dictionary& dict)
{
    return SharedPtr(new TestRtProgram());
}

Dictionary TestRtProgram::getScriptingDictionary()
{
    return Dictionary();
}

RenderPassReflection TestRtProgram::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    reflector.addOutput(kOutput, "Output image").bindFlags(Resource::BindFlags::UnorderedAccess).format(ResourceFormat::RGBA32Float);
    return reflector;
}

void TestRtProgram::setScene(RenderContext* pRenderContext, const Scene::SharedPtr& pScene)
{
    mpScene = pScene;

    mRT.pProgram = nullptr;
    mRT.pVars = nullptr;

    if (mpScene)
    {
        sceneChanged();
    }
}

void TestRtProgram::sceneChanged()
{
    assert(mpScene);

    //
    // Example creating a ray tracing program using the new interfaces.
    //
    uint32_t geometryCount = mpScene->getGeometryCount();

    RtProgram::Desc desc;
    desc.addShaderLibrary(kShaderFilename);
    desc.addDefines(mpScene->getSceneDefines());
    desc.setMaxTraceRecursionDepth(kMaxRecursionDepth);
    desc.setMaxPayloadSize(kMaxPayloadSizeBytes);
    desc.setMaxAttributeSize(kMaxAttributeSizeBytes);

    auto sbt = RtBindingTable::create(2, 2, mpScene->getGeometryCount());

    // Create miss shaders.
    sbt->setRayGen(desc.addRayGen("rayGen"));
    sbt->setMiss(0, desc.addMiss("miss0"));
    sbt->setMiss(1, desc.addMiss("miss1"));

    // Create hit group shaders.
    auto defaultMtl0 = desc.addHitGroup("closestHitMtl0", "anyHit", "");
    auto defaultMtl1 = desc.addHitGroup("closestHitMtl1", "anyHit", "");

    auto greenMtl = desc.addHitGroup("closestHitGreen", "", "");
    auto redMtl = desc.addHitGroup("closestHitRed", "", "");

    auto sphereDefaultMtl0 = desc.addHitGroup("closestHitSphereMtl0", "", "intersectSphere");
    auto sphereDefaultMtl1 = desc.addHitGroup("closestHitSphereMtl1", "", "intersectSphere");

    auto spherePurple = desc.addHitGroup("closestHitSpherePurple", "", "intersectSphere");
    auto sphereYellow = desc.addHitGroup("closestHitSphereYellow", "", "intersectSphere");

    // Assign default hit groups to all geometries.
    sbt->setHitGroupByType(0, mpScene, Scene::GeometryType::TriangleMesh, defaultMtl0);
    sbt->setHitGroupByType(1, mpScene, Scene::GeometryType::TriangleMesh, defaultMtl1);

    sbt->setHitGroupByType(0, mpScene, Scene::GeometryType::Custom, sphereDefaultMtl0);
    sbt->setHitGroupByType(1, mpScene, Scene::GeometryType::Custom, sphereDefaultMtl1);

    // Override specific hit groups for some geometries.
    for (uint geometryID = 0; geometryID < geometryCount; geometryID++)
    {
        auto type = mpScene->getGeometryType(geometryID);

        if (type == Scene::GeometryType::TriangleMesh)
        {
            if (geometryID == 1)
            {
                sbt->setHitGroup(0, geometryID, greenMtl);
                sbt->setHitGroup(1, geometryID, redMtl);
            }
            else if (geometryID == 3)
            {
                sbt->setHitGroup(0, geometryID, redMtl);
                sbt->setHitGroup(1, geometryID, greenMtl);
            }
        }
        else if (type == Scene::GeometryType::Custom)
        {
            uint32_t index = mpScene->getCustomPrimitiveIndex(geometryID);
            uint32_t userID = mpScene->getCustomPrimitive(index).userID;

            // Use non-default material for custom primitives with even userID.
            if (userID % 2 == 0)
            {
                sbt->setHitGroup(0, geometryID, spherePurple);
                sbt->setHitGroup(1, geometryID, sphereYellow);
            }
        }
    }

    mRT.pProgram = RtProgram::create(desc);
    mRT.pVars = RtProgramVars::create(mRT.pProgram, sbt);
}

void TestRtProgram::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    const uint2 frameDim = renderData.getDefaultTextureDims();

    auto pOutput = renderData[kOutput]->asTexture();
    pRenderContext->clearUAV(pOutput->getUAV().get(), float4(0, 0, 0, 1));

    if (!mpScene) return;

    // Check for scene geometry changes.
    // Such changes require us to re-create the raytracing binding table and vars.
    if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::GeometryChanged))
    {
        sceneChanged();
    }

    auto var = mRT.pVars->getRootVar()["gTestProgram"];
    var["frameDim"] = frameDim;
    var["output"] = pOutput;

    mpScene->raytrace(pRenderContext, mRT.pProgram.get(), mRT.pVars, uint3(frameDim, 1));
}

void TestRtProgram::renderUI(Gui::Widgets& widget)
{
    if (!mpScene)
    {
        widget.text("No scene loaded!");
        return;
    }

    auto primCount = mpScene->getCustomPrimitiveCount();
    widget.text("Custom primitives: " + std::to_string(primCount));

    mSelectedIdx = std::min(mSelectedIdx, primCount - 1);
    widget.text("\nSelected primitive:");
    widget.var("##idx", mSelectedIdx, 0u, primCount - 1);

    if (mSelectedIdx != mPrevSelectedIdx)
    {
        mPrevSelectedIdx = mSelectedIdx;
        mSelectedAABB = mpScene->getCustomPrimitiveAABB(mSelectedIdx);
    }

    if (widget.button("Add"))
    {
        addCustomPrimitive();
    }

    if (primCount > 0)
    {
        if (widget.button("Remove", true))
        {
            removeCustomPrimitive(mSelectedIdx);
        }

        if (widget.button("Random move"))
        {
            moveCustomPrimitive();
        }

        bool modified = false;
        modified |= widget.var("Min", mSelectedAABB.minPoint);
        modified |= widget.var("Max", mSelectedAABB.maxPoint);
        if (widget.button("Update"))
        {
            mpScene->updateCustomPrimitive(mSelectedIdx, mSelectedAABB);
        }
    }
}

void TestRtProgram::addCustomPrimitive()
{
    if (!mpScene)
    {
        logWarning("No scene! Ignoring call to addCustomPrimitive()");
        return;
    }

    std::uniform_real_distribution<float> u(0.f, 1.f);
    float3 c = { 4.f * u(rng) - 2.f, u(rng), 4.f * u(rng) - 2.f };
    float r = 0.5f * u(rng) + 0.5f;

    mpScene->addCustomPrimitive(mUserID++, AABB(c - r, c + r));
}

void TestRtProgram::removeCustomPrimitive(uint32_t index)
{
    if (!mpScene)
    {
        logWarning("No scene! Ignoring call to removeCustomPrimitive()");
        return;
    }

    if (index >= mpScene->getCustomPrimitiveCount())
    {
        logWarning("Custom primitive index is out of range. Ignoring call to removeCustomPrimitive()");
        return;
    }

    mpScene->removeCustomPrimitive(index);
}

void TestRtProgram::moveCustomPrimitive()
{
    if (!mpScene)
    {
        logWarning("No scene! Ignoring call to moveCustomPrimitive()");
        return;
    }

    uint32_t primCount = mpScene->getCustomPrimitiveCount();
    if (primCount == 0)
    {
        logWarning("Scene has no custom primitives. Ignoring call to moveCustomPrimitive()");
        return;
    }

    std::uniform_real_distribution<float> u(0.f, 1.f);
    uint32_t index = std::min((uint32_t)(u(rng) * primCount), primCount - 1);

    AABB aabb = mpScene->getCustomPrimitiveAABB(index);
    float3 d = float3(u(rng), u(rng), u(rng)) * 2.f - 1.f;
    aabb.minPoint += d;
    aabb.maxPoint += d;
    mpScene->updateCustomPrimitive(index, aabb);
}
