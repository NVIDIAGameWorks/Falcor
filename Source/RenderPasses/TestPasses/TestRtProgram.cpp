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
#include "TestRtProgram.h"
#include <random>

const RenderPass::Info TestRtProgram::kInfo { "TestRtProgram", "Test pass for RtProgram." };

namespace
{
    const char kShaderFilename[] = "RenderPasses/TestPasses/TestRtProgram.rt.slang";

    // Ray tracing program settings. Set as small values as possible.
    const uint32_t kMaxPayloadSizeBytes = 16;
    const uint32_t kMaxAttributeSizeBytes = 8;
    const uint32_t kMaxRecursionDepth = 1;

    const char kMode[] = "mode";
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
    return SharedPtr(new TestRtProgram(dict));
}

TestRtProgram::TestRtProgram(const Dictionary& dict)
    : RenderPass(kInfo)
{
    for (const auto& [key, value] : dict)
    {
        if (key == kMode) mMode = value;
        else logWarning("Unknown field '{}' in TestRtProgram dictionary.", key);
    }
    if (mMode > 1) throw RuntimeError("mode has to be 0 or 1");
}

Dictionary TestRtProgram::getScriptingDictionary()
{
    Dictionary dict;
    dict[kMode] = mMode;
    return dict;
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
    FALCOR_ASSERT(mpScene);
    const uint32_t geometryCount = mpScene->getGeometryCount();

    //
    // Example creating a ray tracing program using the new interfaces.
    //

    RtProgram::Desc desc;
    desc.addShaderModules(mpScene->getShaderModules());
    desc.addShaderLibrary(kShaderFilename);
    desc.setMaxTraceRecursionDepth(kMaxRecursionDepth);
    desc.setMaxPayloadSize(kMaxPayloadSizeBytes);
    desc.setMaxAttributeSize(kMaxAttributeSizeBytes);

    RtBindingTable::SharedPtr sbt;

    if (mMode == 0)
    {
        // In this mode we test having two different ray types traced against
        // both triangles and custom primitives using intersection shaders.

        sbt = RtBindingTable::create(2, 2, geometryCount);

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
        sbt->setHitGroup(0, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), defaultMtl0);
        sbt->setHitGroup(1, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), defaultMtl1);

        sbt->setHitGroup(0, mpScene->getGeometryIDs(Scene::GeometryType::Custom), sphereDefaultMtl0);
        sbt->setHitGroup(1, mpScene->getGeometryIDs(Scene::GeometryType::Custom), sphereDefaultMtl1);

        // Override specific hit groups for some geometries.
        for (uint geometryID = 0; geometryID < geometryCount; geometryID++)
        {
            auto type = mpScene->getGeometryType(GlobalGeometryID{ geometryID });

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
                uint32_t index = mpScene->getCustomPrimitiveIndex(GlobalGeometryID{ geometryID });
                uint32_t userID = mpScene->getCustomPrimitive(index).userID;

                // Use non-default material for custom primitives with even userID.
                if (userID % 2 == 0)
                {
                    sbt->setHitGroup(0, geometryID, spherePurple);
                    sbt->setHitGroup(1, geometryID, sphereYellow);
                }
            }
        }

        // Add global type conformances.
        desc.addTypeConformances(mpScene->getTypeConformances());
    }
    else
    {
        // In this mode we test specialization of a hit group using two different
        // sets of type conformances. This functionality is normally used for specializing
        // a hit group for different materials types created with createDynamicObject().

        sbt = RtBindingTable::create(2, 1, geometryCount);

        // Create type conformances.
        Program::TypeConformanceList typeConformances0 = Program::TypeConformanceList{ {{"Mtl0", "IMtl"}, 0u} };
        Program::TypeConformanceList typeConformances1 = Program::TypeConformanceList{ {{"Mtl1", "IMtl"}, 1u} };
        Program::TypeConformanceList typeConformances2 = Program::TypeConformanceList{ {{"Mtl2", "IMtl"}, 2u} };

        // Create hit group shaders.
        // These are using the same entry points but are specialized using different type conformances.
        // For each specialization we add a name suffix so that each generated entry point has a unique name.
        RtProgram::ShaderID mtl[3];
        mtl[0] = desc.addHitGroup("closestHit", "anyHit", "", typeConformances0, "Mtl0");
        mtl[1] = desc.addHitGroup("closestHit", "anyHit", "", typeConformances1, "Mtl1");
        mtl[2] = desc.addHitGroup("closestHit", "anyHit", "", typeConformances2, "Mtl2");

        // Assign hit groups to all triangle geometries.
        for (auto geometryID : mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh))
        {
            // Select hit group shader ID based on geometry ID.
            // This will ensure that we use the correct specialized shader for each geometry.
            auto shaderID = mtl[geometryID.get() % 3];
            sbt->setHitGroup(0 /* rayType*/, geometryID, shaderID);
        }
    }

    // Create raygen and miss shaders.
    sbt->setRayGen(desc.addRayGen("rayGen"));
    sbt->setMiss(0, desc.addMiss("miss0"));
    sbt->setMiss(1, desc.addMiss("miss1"));

    Program::DefineList defines = mpScene->getSceneDefines();
    defines.add("MODE", std::to_string(mMode));

    // Create program and vars.
    mRT.pProgram = RtProgram::create(desc, defines);
    mRT.pVars = RtProgramVars::create(mRT.pProgram, sbt);
}

void TestRtProgram::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    const uint2 frameDim = renderData.getDefaultTextureDims();

    auto pOutput = renderData.getTexture(kOutput);
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

    widget.text("Test mode: " + std::to_string(mMode));

    if (mMode == 0)
    {
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
