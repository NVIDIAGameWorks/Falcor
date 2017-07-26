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
#include "RTContext.h"
#include <optixu/optixu.h>
#include <io.h>
#include "Data/VertexAttrib.h"

#pragma warning(disable : 4996)

using namespace Falcor;
using namespace Falcor::RT;
using namespace optix;
using namespace std;

RTContext::RTContext() : mpContext(nullptr)
{
    mpContext = Context::create();

    // Setup context
    mpContext->setRayTypeCount(1);
    mpContext->setEntryPointCount(1);
    mpContext->setExceptionEnabled(RT_EXCEPTION_ALL, false);
    mpContext->setPrintEnabled(false);
    //mpContext->setPrintLaunchIndex(0, 0);
    //mpContext->setPrintBufferSize(4096);
    mpContext->setGPUPagingForcedOff(1);

    // Load default programs
    mMeshIntersectionRtn    = createRoutine("mesh", {{0, "mesh_intersect"}});
    mMeshBoundRtn           = createRoutine("mesh", {{0, "mesh_bounds"}});
    mDefaultExceptRtn       = createRoutine("mesh", {{0, "exception"}});
}

RTContext::~RTContext()
{
    try
    {
        mMeshIntersectionRtn.reset();
        mMeshBoundRtn.reset();
        mDefaultExceptRtn.reset();
        for(auto tex : mOGLSharedTextures)
        {
            tex.second->unregisterGLTexture();
            tex.second->destroy();
        }
        mOGLSharedTextures.clear();
        for(auto buf : mOGLSharedBuffers)
        {
            buf.second->unregisterGLBuffer();
            buf.second->destroy();
        }
        mOGLSharedBuffers.clear();
        mpContext->destroy();
        mpContext = 0;
    }
    catch(const optix::Exception& e)
    {
        Logger::log(Logger::Level::Error, e.getErrorString(), true);
    }
}

RTContext::SharedPtr RTContext::create()
{
    SharedPtr pCtx = SharedPtr(new RTContext());
    return pCtx;
}

RoutineHandle RTContext::createRoutine(const string& fileName, const std::initializer_list<std::pair<int, std::string>>& entryPoints)
{
    std::vector<std::tuple<int, std::string, std::string>> tuples;
    for (const auto& entryPoint : entryPoints)
    {
        tuples.push_back(std::make_tuple(entryPoint.first, fileName, entryPoint.second));
    }
    return createRoutineInternal(tuples);
}

Falcor::RT::RoutineHandle Falcor::RT::RTContext::createRoutine(const std::initializer_list<std::tuple<int, std::string, std::string>>& entryPoints)
{
    return createRoutineInternal(entryPoints);
}

void Falcor::RT::RTContext::resetRoutines()
{
    for(uint32_t i = 0;i < mpContext->getEntryPointCount();++i)
    {
        rtContextSetRayGenerationProgram(mpContext->getContext()->get(), i, nullptr);
        rtContextSetMissProgram(mpContext->getContext()->get(), i, nullptr);
    }
}

Falcor::RT::RoutineHandle Falcor::RT::RTContext::createRoutineInternal(const std::vector<std::tuple<int, std::string, std::string>>& entryPoints)
{
    RoutineHandle prg(new Routine());
    string fileName;
    try
    {
        for (const auto& entryPoint : entryPoints)
        {
            fileName = std::get<1>(entryPoint);
            string fullPath;
            if (findFileInDataDirectories(fileName + ".ptx", fullPath) == false)
                Logger::log(Logger::Level::Error, "Routine file does not exist: " + fileName, true);
            prg->mPrograms.push_back({ 
                std::get<0>(entryPoint), 
                mpContext->createProgramFromPTXFile(fullPath, std::get<2>(entryPoint)), 
                fullPath, 
                getFileModifiedTime(fullPath),
                std::get<2>(entryPoint) });
        }
    }
    catch (const optix::Exception& e)
    {
        Logger::log(Logger::Level::Error, "Failed to load routine '" + fileName + "'\n" + e.getErrorString(), true);
    }

    return prg;
}

void RTContext::reloadRoutine(RoutineHandle& routine)
{
    for(auto& program : routine->mPrograms)
    {
        try
        {
            /* Check if the file was actually modified */
            if(program.program && getFileModifiedTime(program.sourceFilePath) == program.sourceFileTimeStamp)
            {
                continue;
            }
            program.program = mpContext->createProgramFromPTXFile(program.sourceFilePath, program.entryPoint);
            if(program.program)
                program.sourceFileTimeStamp = getFileModifiedTime(program.sourceFilePath);
        }
        catch(const optix::Exception& e)
        {
            Logger::log(Logger::Level::Error, "Failed to reload routine '" + program.sourceFilePath + "', entry point: '" + program.entryPoint + "'\n" + e.getErrorString(), true);
        }
    }
}

void RTContext::setIntersectionTest(ObjectHandle object, RoutineHandle intersectionRtn)
{
    if(object && intersectionRtn && intersectionRtn->mPrograms.size() > 0)
    {
        for(unsigned int i = 0; i < object->getChildCount(); i++)
        {
            optix::GeometryInstance child = object->getChild(i);
            child->getGeometry()->setIntersectionProgram(intersectionRtn->mPrograms[0].program);
        }
    }
}

void RTContext::setSceneClosestHitRoutine(RoutineHandle hitRtn)
{
    if(!mCurrentScene)
    {
        Logger::log(Logger::Level::Error, "No scene loaded when calling setSceneClosestHitRoutine()");
        return;
    }

    for(unsigned int i = 0; i < mStaticGeometry->getChildCount(); i++)
    {
        GeometryInstance inst = mStaticGeometry->getChild(i);
        assert(inst->getMaterialCount() == 1);
        optix::Material mat = inst->getMaterial(0);

        if(hitRtn)
        {
            for(auto& rtn : hitRtn->mPrograms)
                mat->setClosestHitProgram(rtn.index, rtn.program);
        }
        else // Set a null program
        {
            for(unsigned int j = 0; j < mpContext->getRayTypeCount(); j++)
                rtMaterialSetClosestHitProgram(mat->get(), j, nullptr);
        }
    }

    // Dynamic objects
    for(auto& instance : mInstances)
    {
        if(instance.transformObject)
        {
            GeometryInstance inst = instance.transformObject->getChild<ObjectHandle>()->getChild(0);
            assert(inst->getMaterialCount() == 1);
            optix::Material mat = inst->getMaterial(0);

            if(hitRtn)
            {
                for(auto& rtn : hitRtn->mPrograms)
                    mat->setClosestHitProgram(rtn.index, rtn.program);
            }
            else // Set a null program
            {
                for(unsigned int j = 0; j < mpContext->getRayTypeCount(); j++)
                    rtMaterialSetClosestHitProgram(mat->get(), j, nullptr);
            }
        }
    }
}

void RTContext::setSceneAnyHitRoutine(RoutineHandle anyhitRtn)
{
    if(!mCurrentScene)
    {
        Logger::log(Logger::Level::Error, "No scene loaded when calling setSceneClosestHitRoutine()");
        return;
    }

    for(unsigned int i = 0; i < mStaticGeometry->getChildCount(); i++)
    {
        GeometryInstance inst = mStaticGeometry->getChild(i);
        assert(inst->getMaterialCount() == 1);
        optix::Material mat = inst->getMaterial(0);

        if(anyhitRtn)
        {
            for(auto& rtn : anyhitRtn->mPrograms)
                mat->setAnyHitProgram(rtn.index, rtn.program);
        }
        else // Set a null program
        {
            for(unsigned int j = 0; j < mpContext->getRayTypeCount(); j++)
                rtMaterialSetAnyHitProgram(mat->get(), j, nullptr);
        }
    }

    // Dynamic objects
    for(auto& instance : mInstances)
    {
        if(instance.transformObject)
        {
            GeometryInstance inst = instance.transformObject->getChild<ObjectHandle>()->getChild(0);
            assert(inst->getMaterialCount() == 1);
            optix::Material mat = inst->getMaterial(0);

            if(anyhitRtn)
            {
                for(auto& rtn : anyhitRtn->mPrograms)
                    mat->setAnyHitProgram(rtn.index, rtn.program);
            }
            else // Set a null program
            {
                for(unsigned int j = 0; j < mpContext->getRayTypeCount(); j++)
                    rtMaterialSetAnyHitProgram(mat->get(), j, nullptr);
            }
        }
    }
}

void RTContext::updateTransforms()
{
    mCurrentScene->getAcceleration()->markDirty();
    mSceneTransformDirty = true;
}

SceneHandle RTContext::newScene()
{
    // Clean up first
    for(auto tex : mOGLSharedTextures)
    {
        tex.second->unregisterGLTexture();
        tex.second->destroy();
    }
    mOGLSharedTextures.clear();
    for(auto buf : mOGLSharedBuffers)
    {
        buf.second->unregisterGLBuffer();
        buf.second->destroy();
    }
    mOGLSharedBuffers.clear();
    if(mCurrentScene)
    {
        mCurrentScene->destroy();
        mCurrentScene = 0;
    }
    mInstances.clear();

    // Clear caches
    for(auto& it : mSharedBuffers)
    {
        auto& buf = it.second;
        if(buf.PBOBuffer)
        {
            buf.PBOBuffer->unregisterGLBuffer();
            buf.PBOBuffer->destroy();
            buf.PBOBuffer = nullptr;
        }
        if(buf.PBO)
        {
            gl_call(glDeleteBuffers(1, &buf.PBO));
            buf.PBO = 0;
        }
    }
    mSharedBuffers.clear();
    mCachedLights.clear();

    mCurrentScene = mpContext->createGroup();
    mStaticGeometry = mpContext->createGeometryGroup();

    // Acceleration structure used between objects, a simple one is fine.
    Acceleration accel = mpContext->createAcceleration("Bvh", "Bvh");
    mCurrentScene->setAcceleration(accel);
    
    mStaticGeometryAcceleration = mpContext->createAcceleration("Trbvh", "Bvh"); // The fastest?
    mStaticGeometryAcceleration->setProperty("vertex_buffer_name", "gInstPositions");
    mStaticGeometryAcceleration->setProperty("index_buffer_name", "gInstIndices");
    mStaticGeometry->setAcceleration(mStaticGeometryAcceleration);
    mCurrentScene->setChildCount(1);
    mCurrentScene->setChild(0, mStaticGeometry);

    mSceneDirty = true;

    return mCurrentScene;
}

SceneHandle RTContext::newScene(const Scene::SharedPtr& scene, RoutineHandle& shadingRtn, RoutineHandle& anyHitRtn, RoutineHandle& intersectionRtn)
{
    SceneHandle rtScene = newScene();
    for(uint32_t iModel=0;iModel<scene->getModelCount();++iModel)
    {
        const auto& model = scene->getModel(iModel);
        if(scene->getModelInstanceCount(iModel) != 1)
        {
            Logger::log(Logger::Level::Error, "A model '" + model->getName() + "' has multiple instances (unsupported).");
        }
        // TODO: add instanced geometry
        //for(uint32_t iInst = 0;iInst<scene->getModelInstanceCount(iModel);++iInst)
        {
            addObject(model, scene->getModelInstance(iModel, 0), shadingRtn, anyHitRtn, intersectionRtn);
        }
    }

    mSceneDirty = true;
    mSceneRadius = scene->getRadius();
    return rtScene;
}

void RTContext::registerOptiXTexture(MaterialValue& v, const Sampler::SharedPtr& sampler)
{
    v.texture.ptr = 0;
    if(v.texture.pTexture)
    {
        try{
            optix::TextureSampler tex;
            if(mOGLSharedTextures.find(v.texture.pTexture->getApiHandle()) == mOGLSharedTextures.end())
            {
                tex = mpContext->createTextureSamplerFromGLImage(v.texture.pTexture->getApiHandle(), RT_TARGET_GL_TEXTURE_2D);
                tex->setReadMode(isSrgbFormat(v.texture.pTexture->getFormat()) ? RT_TEXTURE_READ_NORMALIZED_FLOAT_SRGB : RT_TEXTURE_READ_NORMALIZED_FLOAT);
                tex->setIndexingMode(RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);
                if(sampler)
                {
                    // Override sampler settings
                    tex->setWrapMode(0, sampler->getAddressModeU() == Sampler::AddressMode::Clamp ? RT_WRAP_CLAMP_TO_EDGE : RT_WRAP_REPEAT);
                    tex->setWrapMode(1, sampler->getAddressModeV() == Sampler::AddressMode::Clamp ? RT_WRAP_CLAMP_TO_EDGE : RT_WRAP_REPEAT);
                    tex->setWrapMode(2, sampler->getAddressModeW() == Sampler::AddressMode::Clamp ? RT_WRAP_CLAMP_TO_EDGE : RT_WRAP_REPEAT);
                    tex->setMaxAnisotropy((float)sampler->getMaxAnisotropy());
                    tex->setFilteringModes(sampler->getMinFilter() == Sampler::Filter::Linear ? RT_FILTER_LINEAR : RT_FILTER_NEAREST, 
                                            sampler->getMagFilter() == Sampler::Filter::Linear ? RT_FILTER_LINEAR : RT_FILTER_NEAREST, 
                                            sampler->getMipFilter() == Sampler::Filter::Linear ? RT_FILTER_LINEAR : RT_FILTER_NEAREST);
                    tex->setMipLevelBias(sampler->getLodBias());
                    tex->setMipLevelClamp(sampler->getMinLod(), sampler->getMaxLod());
                }
                else
                {
                    // Trilinear sampling by default
                    tex->setWrapMode(0, RT_WRAP_REPEAT);
                    tex->setWrapMode(1, RT_WRAP_REPEAT);
                    tex->setWrapMode(2, RT_WRAP_REPEAT);
                    tex->setMaxAnisotropy(1.0f);
                    tex->setFilteringModes(RT_FILTER_LINEAR, RT_FILTER_LINEAR, RT_FILTER_LINEAR);
                    tex->setMipLevelClamp(0, 100);
                }
                tex->validate();
                mOGLSharedTextures[v.texture.pTexture->getApiHandle()] = tex;
            }
            else
                tex = mOGLSharedTextures[v.texture.pTexture->getApiHandle()];

            v.texture.ptrLoHi[0] = tex->getId();
        }
        catch(const optix::Exception& e) {
            Logger::log(Logger::Level::Error, "Texture " + v.texture.pTexture->getSourceFilename() + "Error: " + e.getErrorString(), true);
        }
    }
}

RT::BufferHandle RTContext::createSharedBuffer(Falcor::BufferHandle glApiHandle, RTformat format, size_t elementCount, bool writable /*= false*/)
{
    BufferHandle buf = mpContext->createBufferFromGLBO(writable ? RT_BUFFER_INPUT_OUTPUT : RT_BUFFER_INPUT, glApiHandle);
    buf->setFormat(format);
    buf->setSize(elementCount);
    return buf;
}

Falcor::RT::SamplerHandle Falcor::RT::RTContext::createSharedTexture(Texture::SharedConstPtr& pTexture, bool bufferIndexing /*= false*/)
{
    SamplerHandle sampler = mpContext->createTextureSamplerFromGLImage(pTexture->getApiHandle(), RT_TARGET_GL_TEXTURE_2D);
    sampler->setReadMode(isSrgbFormat(pTexture->getFormat()) ? RT_TEXTURE_READ_NORMALIZED_FLOAT_SRGB : RT_TEXTURE_READ_NORMALIZED_FLOAT);
    sampler->setIndexingMode(bufferIndexing ? RT_TEXTURE_INDEX_ARRAY_INDEX : RT_TEXTURE_INDEX_NORMALIZED_COORDINATES);

    // By default, point sampling settings
    sampler->setWrapMode(0, RT_WRAP_CLAMP_TO_EDGE);
    sampler->setWrapMode(1, RT_WRAP_CLAMP_TO_EDGE);
    sampler->setWrapMode(2, RT_WRAP_CLAMP_TO_EDGE);
    sampler->setFilteringModes(RT_FILTER_NEAREST, RT_FILTER_NEAREST, RT_FILTER_NEAREST);

    return sampler;
}

RT::BufferHandle RTContext::_createSceneSharedBuffer(Falcor::BufferHandle glApiHandle, RTformat format, size_t elementCount)
{
    auto& buf = mOGLSharedBuffers[glApiHandle];
    if(!buf)
    {
        buf = mpContext->createBufferFromGLBO(RT_BUFFER_INPUT, glApiHandle);
        buf->setFormat(format);
        buf->setSize(elementCount);
    }
    else
    {
        assert(buf->getFormat() == format);
        RTsize sz = 0; buf->getSize(sz);
        assert(sz == elementCount);
    }
    return buf;
}

BufPtr Falcor::RT::RTContext::getBufferPtr(BufferHandle& buffer)
{
    BufPtr ptr;
    if(buffer)
    {
        ptr.ptrLoHi[0] = buffer->getId();
    }
    return ptr;
}

ObjectHandle RTContext::addGeometry(const Model::SharedPtr& model, const Scene::ModelInstance& modelInstance, RoutineHandle& shadingRtn, RoutineHandle& anyHitRtn, RoutineHandle& intersectionRtn)
{
    mSceneDirty = true;

    ObjectHandle group = mpContext->createGeometryGroup();

    // Add all submeshes
    for(uint32_t i = 0; i < model->getMeshCount(); ++i)
    {
        const Mesh::SharedPtr& mesh = model->getMesh(i);
        const Vao::SharedPtr vao = mesh->getVao();
        Buffer::SharedConstPtr ib = vao->getIndexBuffer();
        const size_t vtxCount = mesh->getVertexCount();
        const size_t triCount = mesh->getPrimitiveCount();

        // Validate the Mesh
        if(mesh->getTopology() != RenderContext::Topology::TriangleList)
        {
            Logger::log(Logger::Level::Error, "Submesh of a model '" + model->getName() + "' has unsupported geometry topology (only triangle list is supported)");
            continue;
        }
        if(vao->getVertexBuffer(0)->getSize() % vao->getVertexBufferStride(0) != 0 ||
            vao->getVertexBuffer(0)->getSize() / vao->getVertexBufferStride(0) != vtxCount)
        {
            Logger::log(Logger::Level::Error, "Vertex buffer of a submesh in model '" + model->getName() + "' had wrong stride or size");
            continue;
        }
        if(ib->getSize() % (sizeof(uint32_t) * 3) != 0 || triCount != ib->getSize() / (sizeof(uint32_t) * 3))
        {
            Logger::log(Logger::Level::Error, "Index buffer of a submesh in model '" + model->getName() + "' had wrong stride or size");
            continue;
        }

        // Create model instance
        mInstances.push_back(MeshInstance());
        MeshInstance& inst = mInstances.back();
        inst.meshId = (int32_t)mInstances.size() - 1;
        assert(inst.meshId == mesh->getId());

        // Figure out vertex layout of vertex attributes
        bool hasTg = false, hasUv = false;
        // Find positions
        const int32_t posIdx = vao->getElementIndexByLocation(VERTEX_POSITION_LOC).vbIndex;
        if(posIdx == Vao::ElementDesc::kInvalidIndex)
        {
            Logger::log(Logger::Level::Error, "A submesh in model '" + model->getName() + "' has no positions or normals!");
            continue;
        }
        assert(vao->getVertexBufferLayout(posIdx)->getElementFormat(0) == ResourceFormat::RGB32Float);  // Assuming float3 for positions
        const int32_t nrmIdx = vao->getElementIndexByLocation(VERTEX_NORMAL_LOC).vbIndex;
        assert(nrmIdx != Vao::ElementDesc::kInvalidIndex);

        // Find tangent frames
        const int32_t tgIdx = vao->getElementIndexByLocation(VERTEX_TANGENT_LOC).vbIndex;
        const int32_t btIdx = vao->getElementIndexByLocation(VERTEX_BITANGENT_LOC).vbIndex;
        {
            if(tgIdx != Vao::ElementDesc::kInvalidIndex)
            {
                if(btIdx == Vao::ElementDesc::kInvalidIndex)
                    Logger::log(Logger::Level::Error, "A submesh in model '" + model->getName() + "' has no tangents but no bitangents!");
                else
                    assert(vao->getVertexBufferLayout(btIdx)->getElementFormat(0) == ResourceFormat::RGB32Float);  // Assuming float3 for bitangents
                assert(vao->getVertexBufferLayout(tgIdx)->getElementFormat(0) == ResourceFormat::RGB32Float);  // Assuming float3 for tangents
            }
            hasTg = tgIdx != Vao::ElementDesc::kInvalidIndex && btIdx != Vao::ElementDesc::kInvalidIndex;
        }
        // Find texture coordinates
        const int32_t uvIdx = vao->getElementIndexByLocation(VERTEX_TEXCOORD_LOC).vbIndex;
        {
            hasUv = uvIdx != Vao::ElementDesc::kInvalidIndex;
            if(hasUv)
                assert(vao->getVertexBufferLayout(uvIdx)->getElementFormat(0) == ResourceFormat::RGB32Float);  // Assuming at least float2 for texcoords
        }

        // Share index buffer
        inst.geo.indices = _createSceneSharedBuffer(ib->getApiHandle(), RT_FORMAT_INT3, triCount);

        // Share vertex buffers, or create if needed
        inst.geo.positions = _createSceneSharedBuffer(vao->getVertexBuffer(posIdx)->getApiHandle(), RT_FORMAT_FLOAT3, vtxCount);
        inst.geo.normals = _createSceneSharedBuffer(vao->getVertexBuffer(nrmIdx)->getApiHandle(), RT_FORMAT_FLOAT3, vtxCount);

        // Initialize tangent frame and texcoord buffers with default values if needed
        {
            vec3* tcr_data = nullptr;
            vec3* tg_data = nullptr, *bt_data = nullptr;
            vector<vec3> nrmData;

            if(hasTg)
            {
                inst.geo.tangents = _createSceneSharedBuffer(vao->getVertexBuffer(tgIdx)->getApiHandle(), RT_FORMAT_FLOAT3, vtxCount);
                inst.geo.bitangents = _createSceneSharedBuffer(vao->getVertexBuffer(btIdx)->getApiHandle(), RT_FORMAT_FLOAT3, vtxCount);
            }
            else
            {
                inst.geo.tangents = mpContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, vtxCount);
                inst.geo.bitangents = mpContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, vtxCount);
                tg_data = (vec3*)inst.geo.tangents->map();
                bt_data = (vec3*)inst.geo.bitangents->map();
                nrmData.resize(vao->getVertexBuffer(VERTEX_NORMAL_LOC)->getSize() / sizeof(vec3));
                vao->getVertexBuffer(VERTEX_NORMAL_LOC)->readData(nrmData.data(), 0, nrmData.size() * sizeof(vec3));
            }
            if(hasUv)
                inst.geo.texcoord = _createSceneSharedBuffer(vao->getVertexBuffer(uvIdx)->getApiHandle(), RT_FORMAT_FLOAT3, vtxCount);
            else
            {
                inst.geo.texcoord = mpContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, vtxCount);
                tcr_data = (vec3*)inst.geo.texcoord->map();
            }

            // Create default values in case if uv mapping / tangents do not exist
            if(!hasTg || !hasUv)
            {
                for(size_t i = 0; i < vtxCount; ++i)
                {
                    if(!hasTg)
                    {
                        // Generate tangents on the fly
                        const vec3 normal = nrmData[i];
                        vec3 bitangent, tangent;
                        if(abs(normal.x) > abs(normal.y))
                            bitangent = v3(normal.z, 0.f, -normal.x) / length(v2(normal.x, normal.z));
                        else
                            bitangent = v3(0.f, normal.z, -normal.y) / length(v2(normal.y, normal.z));
                        tangent = cross(bitangent, normal);

                        tg_data[i] = tangent;
                        bt_data[i] = bitangent;
                    }
                    if(!hasUv)
                        tcr_data[i] = vec3(0.5f, 0.5f, 0.f);
                }
            }
            if(!hasTg)
            {
                inst.geo.tangents->unmap();
                inst.geo.bitangents->unmap();
            }
            if(!hasUv)
                inst.geo.texcoord->unmap();
        }

        inst.geo.transform = mesh->getInstanceMatrix(0) * modelInstance.transformMatrix;

        // Create OptiX mesh
        Geometry geo = mpContext->createGeometry();

        // Set mesh properties
        geo->setPrimitiveCount((uint32_t)triCount);
        geo->setPrimitiveIndexOffset(0);

        if(intersectionRtn)
        {
            geo->setIntersectionProgram(intersectionRtn->mPrograms[0].program);
        }
        else
        {
            geo->setIntersectionProgram(mMeshIntersectionRtn->mPrograms[0].program);
        }

        geo->setBoundingBoxProgram(mMeshBoundRtn->mPrograms[0].program);
        geo->markDirty();

        try {
            geo->validate();
        }
        catch (const optix::Exception& e) {
            Logger::log(Logger::Level::Error, e.getErrorString(), true);
            continue;
        }

        // Create geo instance
        GeometryInstance optInst = mpContext->createGeometryInstance();
        optInst->setGeometry(geo);
        inst.instance = optInst;
        inst.object = group;

        // Create material data
        if(const Material::SharedPtr& meshMat = mesh->getMaterial())
        {
            inst.material = meshMat->getData();
            MaterialData& m = inst.material;
            registerOptiXTexture(m.values.alphaMap, meshMat->getSamplerOverride());
            registerOptiXTexture(m.values.normalMap, meshMat->getSamplerOverride());
            registerOptiXTexture(m.values.heightMap, meshMat->getSamplerOverride());
            for(int i = 0; i < MatMaxLayers; ++i)
            {
                registerOptiXTexture(m.values.layers[i].albedo, meshMat->getSamplerOverride());
                registerOptiXTexture(m.values.layers[i].roughness, meshMat->getSamplerOverride());
                registerOptiXTexture(m.values.layers[i].extraParam, meshMat->getSamplerOverride());
            }
        }
        else
        {
            // Default material
            Logger::log(Logger::Level::Warning, "Submesh of a mesh " + model->getName() + " has no material. Assigning default");
            static const MaterialData m;
            inst.material = m;
        }

        // Create a bindless instance
        BindlessGeoInstance gpuInst;
        gpuInst.indices = inst.geo.indices->getId();
        gpuInst.positions = inst.geo.positions->getId();
        gpuInst.normals = inst.geo.normals->getId();
        gpuInst.tangents = inst.geo.tangents->getId();
        gpuInst.bitangents = inst.geo.bitangents->getId();
        gpuInst.texcoord = inst.geo.texcoord->getId();
        gpuInst.transform = inst.geo.transform;
        gpuInst.invTrTransform = mat4(glm::transpose(glm::inverse(glm::mat3(gpuInst.transform))));
        memcpy(&gpuInst.material, &inst.material, sizeof(inst.material));
        optInst["gInstance"]->setUserData(sizeof(gpuInst), &gpuInst);
        memset(&gpuInst, 0x00, sizeof(gpuInst));

        optInst["gInstanceId"]->setInt(inst.meshId);

        // Set direct instance buffers
        optInst["gInstIndices"]->setBuffer(inst.geo.indices);
        optInst["gInstPositions"]->setBuffer(inst.geo.positions);
        optInst["gInstTexcoords"]->setBuffer(inst.geo.texcoord);
        optInst["gInstNormals"]->setBuffer(inst.geo.normals);
        optInst["gInstTangents"]->setBuffer(inst.geo.tangents);
        optInst["gInstBitangents"]->setBuffer(inst.geo.bitangents);
        optInst["gInstMaterial"]->setUserData(sizeof(inst.material), &inst.material);

        // Create material shader
        optix::Material mat = mpContext->createMaterial();
        if(shadingRtn)
        {
            for(auto& rtn : shadingRtn->mPrograms)
                mat->setClosestHitProgram(rtn.index, rtn.program);
        }

        if(anyHitRtn)
        {
            for(auto& rtn : anyHitRtn->mPrograms)
                mat->setAnyHitProgram(rtn.index, rtn.program);
        }

        optInst->setMaterialCount(1);
        optInst->setMaterial(0, mat);

        // Attach to the scene
        if(inst.geo.transform == mat4())
        {
            group->addChild(optInst);
        }
        else
        {
            // Create a transform
            ObjectHandle newGroup = mpContext->createGeometryGroup();
            Acceleration accel = mpContext->createAcceleration("Trbvh", "Bvh");
            accel->setProperty("vertex_buffer_name", "gInstPositions");
            accel->setProperty("index_buffer_name", "gInstIndices");
            newGroup->setAcceleration(accel);
            accel->markDirty();
            Transform transform = mpContext->createTransform();
            newGroup->addChild(optInst);
            transform->setChild(newGroup);
            transform->setMatrix(true, &inst.geo.transform[0][0], nullptr);
            mCurrentScene->addChild(transform);

            inst.transformObject = transform;
        }
    }

    return group;
}

Falcor::RT::DynamicObjectHandle Falcor::RT::RTContext::enableDynamicObject(const Mesh::SharedPtr& mesh)
{
    if(!mCurrentScene)
    {
        Logger::log(Logger::Level::Error, "Please load a scene first");
        return DynamicObjectHandle();
    }

    const uint32_t id = mesh->getId();
    assert(id < mInstances.size());
    // Early return if the object is just transformed, but not dynamic
    if(mInstances[id].transformObject)
    {
        return mInstances[id].transformObject;
    }
    assert(mInstances[id].instance);
    assert(mInstances[id].object);

    // Remove from static scene
    mStaticGeometry->removeChild(mStaticGeometry->getChildIndex(mInstances[id].instance));
    mStaticGeometryAcceleration->markDirty();   // Rebuild Bvh

    // Add to dynamic objects
    ObjectHandle newGroup = mpContext->createGeometryGroup();
    Acceleration accel = mpContext->createAcceleration("Trbvh", "Bvh");
    accel->setProperty("vertex_buffer_name", "gInstPositions");
    accel->setProperty("index_buffer_name", "gInstIndices");
    newGroup->setAcceleration(accel);
    accel->markDirty();
    Transform transform = mpContext->createTransform();
    newGroup->addChild(mInstances[id].instance);
    transform->setChild(newGroup);
    mCurrentScene->addChild(transform);
    mCurrentScene->getAcceleration()->markDirty();
    mInstances[id].transformObject = transform;
    mInstances[id].dynamic = true;
    mSceneDirty = true;

    return mInstances[id].transformObject;
}   

void Falcor::RT::RTContext::disableDynamicObject(const Mesh::SharedPtr& mesh)
{
    if(!mCurrentScene)
    {
        Logger::log(Logger::Level::Error, "Please load a scene first");
        return;
    }

    const uint32_t id = mesh->getId();
    assert(id < mInstances.size());
    assert(mInstances[id].transformObject);
    
    // Early return if the object is just transformed, but not dynamic
    if(!mInstances[id].dynamic)
    {
        return;
    }
    
    assert(mInstances[id].instance);
    assert(mInstances[id].object);

    ObjectHandle group = mInstances[id].object;

    // Remove from dynamic objects
    mCurrentScene->removeChild(mCurrentScene->getChildIndex(mInstances[id].transformObject));
    mInstances[id].transformObject->getChild<ObjectHandle>()->destroy();
    mInstances[id].transformObject->destroy();
    mInstances[id].transformObject = nullptr;
    mInstances[id].dynamic = false;

    // Add back to static scene
    mStaticGeometry->addChild(mInstances[id].instance);
    mStaticGeometry->setAcceleration(mStaticGeometryAcceleration);
    mStaticGeometryAcceleration->markDirty();
    mCurrentScene->getAcceleration()->markDirty();
    mSceneDirty = true;
}

ObjectHandle RTContext::addObject(const Model::SharedPtr& model, const Scene::ModelInstance& instance, RoutineHandle& shadingRtn, RoutineHandle& anyHitRtn, RoutineHandle& intersectionRtn)
{
    if(!mCurrentScene)
    {
        Logger::log(Logger::Level::Error, "Please create a new scene first");
        return ObjectHandle();
    }

    ObjectHandle group = addGeometry(model, instance, shadingRtn, anyHitRtn, intersectionRtn);

    // Static geometry, to avoid performance issues, shares the same group.
    // This avoid huge scenes (i.e. San Miguel) to be unbearably slow.
    for(unsigned int i = 0; i < group->getChildCount(); i++)
    {
        mStaticGeometry->addChild(group->getChild(i));
    }
    mStaticGeometryAcceleration->markDirty();   // Rebuild Bvh
    return group;
}

DynamicObjectHandle RTContext::addDynamicObject(const Model::SharedPtr& model, RoutineHandle& shadingRtn, RoutineHandle& anyHitRtn /*= RoutineHandle()*/, RoutineHandle& intersectionRtn /*= RoutineHandle()*/)
{
    if(!mCurrentScene)
    {
        Logger::log(Logger::Level::Error, "Please create a new scene first");
        return DynamicObjectHandle();
    }

    ObjectHandle group = addGeometry(model, Scene::ModelInstance(), shadingRtn, anyHitRtn, intersectionRtn);

    // We need to duplicate the acceleration structure, since we are creating a separate child of the main group.
    // (The acceleration is tied to the geometry in Optix.)
    Acceleration accel = mpContext->createAcceleration("Trbvh", "Bvh");
    accel->setProperty("vertex_buffer_name", "gInstPositions");
    accel->setProperty("index_buffer_name", "gInstIndices");
    group->setAcceleration(accel);
    accel->markDirty();

    // We add the transform as an intermediate node between the GeometryGroup and the scene node.
    Transform transform = mpContext->createTransform();
    transform->setChild(group);
    mCurrentScene->addChild(transform);
    mCurrentScene->getAcceleration()->markDirty();

    return transform;
}

ObjectHandle RTContext::getObject(DynamicObjectHandle dynamic_object)
{
    if(dynamic_object->getChildType() == RT_OBJECTTYPE_GEOMETRY_GROUP)
    {
        return dynamic_object->getChild<GeometryGroup>();
    }
    else
    {
        Logger::log(Logger::Level::Error, "Dynamic object");
        return ObjectHandle();
    }
}

void RTContext::transformStaticObject(ObjectHandle object, const glm::mat4x3& mx)
{
    for(uint32_t i = 0;i < object->getChildCount();++i)
    {
        GeometryInstance inst = object->getChild(i);
        optix::Buffer vbuffer = inst["gInstPositions"]->getBuffer();
        vec3* pos = (vec3*)vbuffer->map();
        RTsize sz; vbuffer->getSize(sz);
        for(uint32_t j = 0;j < sz;++j)
            pos[j] = mx * vec4(pos[j], 1.f);
        vbuffer->unmap();
    }
    if(object->getAcceleration())
        object->getAcceleration()->markDirty();
    mCurrentScene->getAcceleration()->markDirty();
}

void RTContext::setLights(const std::initializer_list<Light*>& list)
{
    // Create lights buffer
    if(!mLights)
    {
        mLights = mpContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
        mLights->setElementSize(sizeof(LightData));
    }

    // Set all lights
    mLights->setSize(list.size());
    LightData* lights = (LightData*)mLights->map();
    for(auto elem : list)
    {
        memcpy(lights, &elem->getData(), sizeof(LightData));
        lights++;
    }
    mLights->unmap();
    mpContext["gLights"]->set(mLights);
}

void RTContext::setLights(const std::vector<Light::SharedPtr>& list)
{
	// Create lights buffer
	if (!mLights)
	{
		mLights = mpContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
		mLights->setElementSize(sizeof(LightData));
	}

	// Set all lights
	mLights->setSize(list.size());
	LightData* lights = (LightData*)mLights->map();
	for (auto& elem : list)
	{
		// Prepare GPU data
		elem->prepareGPUData();

        // Patch buffer pointers
        if(mCachedLights.find(elem) == mCachedLights.end())
        {
            LightData& cachedInst = mCachedLights[elem];
            cachedInst = elem->getData();

			switch (elem->getType())
			{
				case LightArea:
				{
					const AreaLight::SharedPtr& pAreaLight = reinterpret_cast<const AreaLight::SharedPtr&>(elem);
					if (pAreaLight)
					{
						cachedInst.indexPtr.ptrLoHi[0] = _createSceneSharedBuffer(pAreaLight->getIndexBuffer()->getApiHandle(), RT_FORMAT_INT3, pAreaLight->getIndexBuffer()->getSize() / sizeof(glm::ivec3))->getId();
						cachedInst.vertexPtr.ptrLoHi[0] = _createSceneSharedBuffer(pAreaLight->getPositionsBuffer()->getApiHandle(), RT_FORMAT_FLOAT3, pAreaLight->getPositionsBuffer()->getSize() / sizeof(glm::vec3))->getId();
						if (pAreaLight->getTexCoordBuffer())
						{
							cachedInst.texCoordPtr.ptrLoHi[0] = _createSceneSharedBuffer(pAreaLight->getTexCoordBuffer()->getApiHandle(), RT_FORMAT_FLOAT3, pAreaLight->getTexCoordBuffer()->getSize() / sizeof(glm::vec3))->getId();
							assert(pAreaLight->getTexCoordBuffer()->getSize() % sizeof(glm::vec3) == 0);
						}
						cachedInst.meshCDFPtr.ptrLoHi[0] = _createSceneSharedBuffer(pAreaLight->getMeshCDFBuffer()->getApiHandle(), RT_FORMAT_FLOAT, pAreaLight->getMeshCDFBuffer()->getSize() / sizeof(float))->getId();
					}
				}
				break;
			}
        }
		else
        {
			// Update light data if modified 
            auto& cLight = mCachedLights[elem];
            auto oldLightPtrs = cLight;
            const auto& newData = elem->getData();
            memcpy(&cLight, &newData, sizeof(LightData));

            // Restore pointers
            cLight.indexPtr = oldLightPtrs.indexPtr;
            cLight.vertexPtr = oldLightPtrs.vertexPtr;
            cLight.texCoordPtr = oldLightPtrs.texCoordPtr;
            cLight.meshCDFPtr = oldLightPtrs.meshCDFPtr;
        }

        memcpy(lights, &(mCachedLights[elem]), sizeof(*lights));
		lights++;
	}

	mLights->unmap();
	mpContext["gLights"]->set(mLights);
}

static inline RTformat glToOptixFormat(GLint sizedGLFormat)
{
    RTformat format = RT_FORMAT_UNSIGNED_BYTE4;

    switch(sizedGLFormat)
    {
        // unnormalized integer formats
    case GL_R8I:
        format = RT_FORMAT_BYTE;
        break;
    case GL_R8UI:
        format = RT_FORMAT_UNSIGNED_BYTE;
        break;

    case GL_R16I:
        format = RT_FORMAT_SHORT;
        break;
    case GL_R16UI:
        format = RT_FORMAT_UNSIGNED_SHORT;
        break;

    case GL_R32I:
        format = RT_FORMAT_INT;
    case GL_R32UI:
        format = RT_FORMAT_UNSIGNED_INT;

    case GL_RG8I:
        format = RT_FORMAT_BYTE2;
        break;
    case GL_RG8UI:
        format = RT_FORMAT_UNSIGNED_BYTE2;
        break;

    case GL_RG16I:
        format = RT_FORMAT_SHORT2;
        break;
    case GL_RG16UI:
        format = RT_FORMAT_UNSIGNED_SHORT2;
        break;

    case GL_RG32I:
        format = RT_FORMAT_INT2;
        break;
    case GL_RG32UI:
        format = RT_FORMAT_UNSIGNED_INT2;
        break;

    case GL_RGBA8I:
        format = RT_FORMAT_BYTE4;
        break;
    case GL_RGBA8:
    case GL_RGBA8UI:
    case GL_SRGB8_ALPHA8:
        format = RT_FORMAT_UNSIGNED_BYTE4;
        break;

    case GL_RGBA16I:
        format = RT_FORMAT_SHORT4;
        break;
    case GL_RGBA16:
    case GL_RGBA16UI:
        format = RT_FORMAT_UNSIGNED_SHORT4;
        break;

    case GL_RGBA32I:
        format = RT_FORMAT_INT4;
        break;
    case GL_RGBA32UI:
        format = RT_FORMAT_UNSIGNED_INT4;
        break;

        // float formats
    case GL_R16F:
        format = RT_FORMAT_HALF;
        break;
    case GL_RG16F:
        format = RT_FORMAT_HALF2;
        break;
    case GL_RGB16F:
        format = RT_FORMAT_HALF3;
        break;
	case GL_RGBA16F:
		format = RT_FORMAT_HALF4;
		break;

    case GL_R32F:
        format = RT_FORMAT_FLOAT;
        break;
    case GL_RG32F:
        format = RT_FORMAT_FLOAT2;
        break;
    case GL_RGBA32F:
        format = RT_FORMAT_FLOAT4;
        break;

        // not supported
    default:
        throw std::exception("Unsupported GL target texture format.");
    }

    return format;
}

bool RTContext::updateTempBuffer(const Fbo::SharedPtr& target)
{
    // Create buffer for FBO
    if(!mFrameBuffers)
    {
        mFrameBuffers = mpContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_INT);
        mFrameBuffers->setSize(Fbo::getMaxColorTargetCount());
    }

    // Garbage collection
    for(auto it = mSharedBuffers.begin();it != mSharedBuffers.end();)
    {
        auto& buf = it->second;
        if(buf.sourceTexture.expired())
        {
            if(buf.PBOBuffer)
            {
                buf.PBOBuffer->unregisterGLBuffer();
                buf.PBOBuffer->destroy();
                buf.PBOBuffer = nullptr;
            }
            if(buf.PBO)
            {
                gl_call(glDeleteBuffers(1, &buf.PBO));
                buf.PBO = 0;
            }
            it = mSharedBuffers.erase(it);
        }
        else
            ++it;
    }

    int* frameBuffers = (int*)mFrameBuffers->map();
    for(uint32_t i = 0;i<Fbo::getMaxColorTargetCount();++i)
    {
        Texture::SharedConstPtr& pTex = target->getColorTexture(i);
        if(!pTex)
            break;
        const uint32_t w = pTex->getWidth(), h = pTex->getHeight();
        // Create persistent default frame buffer
        try{
            SharedTextureBuffer& buf = mSharedBuffers[pTex->getApiHandle()];

            RTsize w_old = 0, h_old = 0;
            if(buf.PBOBuffer)
                buf.PBOBuffer->getSize(w_old, h_old);
            
            const RTformat tgtFmt = glToOptixFormat(getGlSizedFormat(pTex->getFormat()));
            size_t element_size = getFormatBytesPerBlock(pTex->getFormat());
            //mpContext->checkError(rtuGetSizeForRTformat(tgtFmt, &element_size));
            // Reallocate if needed
            if(buf.PBO == 0 || !buf.PBOBuffer || element_size * w * h != buf.PBOBuffer->getElementSize() * w_old * h_old)
            {
                if(buf.PBOBuffer)
                {
                    buf.PBOBuffer->unregisterGLBuffer();
                    buf.PBOBuffer->destroy();
                    buf.PBOBuffer = nullptr;
                }
                if(buf.PBO)
                {
                    gl_call(glDeleteBuffers(1, &buf.PBO));
                    buf.PBO = 0;
                }

                gl_call(glGenBuffers(1, &buf.PBO));
                gl_call(glBindBuffer(GL_ARRAY_BUFFER, buf.PBO));
                gl_call(glBufferData(GL_ARRAY_BUFFER, element_size * w * h, 0, GL_DYNAMIC_COPY));
                gl_call(glBindBuffer(GL_ARRAY_BUFFER, 0));
                buf.PBOBuffer = mpContext->createBufferFromGLBO(RT_BUFFER_INPUT_OUTPUT, buf.PBO);
                buf.PBOBuffer->setFormat(tgtFmt);
                buf.PBOBuffer->setSize(w, h);
                buf.sourceTexture = pTex;
            }
            assert(!buf.sourceTexture.expired() && buf.sourceTexture.lock() == pTex);

            // Set buffer's id
            frameBuffers[i] = buf.PBOBuffer->getId();
        }
        catch(const optix::Exception& e) {
            Logger::log(Logger::Level::Error, e.getErrorString(), true);
            return  false;
        }
    }
    mFrameBuffers->unmap();
    mpContext["gFrameBuffers"]->set(mFrameBuffers);

    return true;
}

void RTContext::updateScene()
{
    if(!mSceneDirty && !mSceneTransformDirty)
        return;

    if(!mSceneInstancesBuffer)
    {
        mSceneInstancesBuffer = mpContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
        mSceneInstancesBuffer->setElementSize(sizeof(BindlessGeoInstance));
    }

    mSceneInstancesBuffer->setSize(mInstances.size());
    BindlessGeoInstance* gpuInstances = (BindlessGeoInstance*)mSceneInstancesBuffer->map();
    size_t instId = 0;
    for(auto& inst : mInstances)
    {
        assert(instId == inst.meshId);
        BindlessGeoInstance& gpuInst = gpuInstances[instId++];

        gpuInst.indices = inst.geo.indices->getId();
        gpuInst.positions = inst.geo.positions->getId();
        gpuInst.normals = inst.geo.normals->getId();
        gpuInst.tangents = inst.geo.tangents->getId();
        gpuInst.bitangents = inst.geo.bitangents->getId();
        gpuInst.texcoord = inst.geo.texcoord->getId();

        if(!inst.transformObject)
        {
            gpuInst.transform = inst.geo.transform;
        }
        else
        {
            mat4 mx;
            inst.transformObject->getMatrix(true, &mx[0][0], nullptr);
            gpuInst.transform = mx;
        }
        gpuInst.invTrTransform = mat4(glm::transpose(glm::inverse(glm::mat3(gpuInst.transform))));
        memcpy(&gpuInst.material, &inst.material, sizeof(inst.material));
    }
    mSceneInstancesBuffer->unmap();
    mpContext["gSceneInstances"]->set(mSceneInstancesBuffer);

    // Set scene
    if(mSceneDirty)
    {
        mpContext["top_object"]->set(mCurrentScene);
        mpContext["top_shadower"]->set(mCurrentScene);

        float scene_epsilon = 1e-5f * mSceneRadius;
        mpContext["scene_epsilon"]->setFloat(scene_epsilon);
    }

    mSceneDirty = false;
    mSceneTransformDirty = false;
}

void RTContext::render(const Fbo::SharedPtr& target, const Camera::SharedPtr& camera, RoutineHandle& raygenFn, RoutineHandle& missFn, const int activeEntryPoint, const std::initializer_list<uint32_t>& launchGrid)
{
    if(!mCurrentScene)
    {
        Logger::log(Logger::Level::Error, "Please create a new scene first");
        return;
    }

    if(!updateTempBuffer(target))
        return;

    if(!mCameras)
    {
        mCameras = mpContext->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_USER);
        mCameras->setElementSize(sizeof(CameraData));
        mpContext["gCams"]->set(mCameras);
    }

    // Update the scene
    updateScene();

    // Set camera
    if(camera)
    {
        mCameras->setSize(1);
        CameraData* cameras = (CameraData*)mCameras->map();
        cameras[0] = camera->getData();
        mCameras->unmap();
    }

    // Set miss program
    if(missFn)
    {
        for(const auto& rtn : missFn->mPrograms)
            mpContext->setMissProgram(rtn.index, rtn.program);
    }

    for(const auto& rtn : raygenFn->mPrograms)
        mpContext->setExceptionProgram(rtn.index, mDefaultExceptRtn->mPrograms[0].program);

    // Set ray generation program
    for(auto& rtn : raygenFn->mPrograms)
        mpContext->setRayGenerationProgram(rtn.index, rtn.program);

    const uint32_t w = target->getColorTexture(0)->getWidth(), h = target->getColorTexture(0)->getHeight();
    try{
        // Prepare to run 
        mpContext->validate();
        mpContext->compile();

        // Run
        if(launchGrid.size() > 0)
        {
            auto idx = launchGrid.begin();
            uint32_t dims[3] = { 0 };
            switch(launchGrid.size())
            {
            case 1: 
            {
                dims[0] = *idx++;
                mpContext->launch(activeEntryPoint, dims[0]); 
                break;
            }
            case 2:
            {
                dims[0] = *idx++; 
                dims[1] = *idx++;
                mpContext->launch(activeEntryPoint, dims[0], dims[1]);
                break;
            }
            case 3: 
            {
                dims[0] = *idx++;
                dims[1] = *idx++;
                dims[2] = *idx++;
                mpContext->launch(activeEntryPoint, dims[0], dims[1], dims[2]); 
                break;
            }
            default: Logger::log(Logger::Level::Error, "Too many dimensions in the launch grid");
            }
        }
        else
        {
            // Launch an image grid by default
            mpContext->launch(activeEntryPoint, w, h);
        }
    }
    catch(const optix::Exception& e) {
        Logger::log(Logger::Level::Error, e.getErrorString(), true);
        return;
    }

    // Transfer output to OpenGL texture
    for(uint32_t i = 0;i<Fbo::getMaxColorTargetCount();++i)
    {
        Texture::SharedConstPtr& pTex = target->getColorTexture(i);
        if(!pTex)
            break;
        SharedTextureBuffer& buf = mSharedBuffers[pTex->getApiHandle()];
        gl_call(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buf.PBO));        // Bind to the optix buffer
        gl_call(glPixelStorei(GL_UNPACK_ALIGNMENT, 1));
        // Copy the OptiX results into a GL texture
        const ResourceFormat fmt = pTex->getFormat();
        gl_call(glTextureSubImage2D(pTex->getApiHandle(), 0, 0, 0, w, h, getGlBaseFormat(fmt),       getGlFormatType(fmt), 0));
    }
    gl_call(glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0));
}

void Routine::destroy()
{
    for(auto& rtn : mPrograms)
        rtn.program->destroy();
    mPrograms.clear();
}

Routine::~Routine()
{
    destroy();
}
