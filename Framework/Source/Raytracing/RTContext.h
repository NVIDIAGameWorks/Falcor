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
#pragma once
#include "Falcor.h"
#include "Core/RenderContext.h"
#include <optix.h>
#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include <initializer_list>
#include <list>


namespace Falcor
{
namespace RT
{
    typedef void* Handle;

    typedef optix::Group            SceneHandle;
    typedef optix::Context          ContextHandle;

    typedef optix::GeometryGroup    ObjectHandle;
    typedef optix::Transform        DynamicObjectHandle;
    typedef optix::Buffer           BufferHandle;
    typedef optix::TextureSampler   SamplerHandle;

    class Routine
    {
        friend class RTContext;
    public:
        class Handle : public std::shared_ptr<Routine>
        {
        public:
            Handle() : std::shared_ptr<Routine>() {}
            Handle(Routine* ptr) : std::shared_ptr<Routine>(ptr) {}
        };
        ~Routine();
    protected:
        void destroy();
        Routine() {}
        struct Program
        {
            int             index;
            optix::Program  program;
            std::string     sourceFilePath;
            time_t          sourceFileTimeStamp;
            std::string     entryPoint;
        };
        std::vector<Program>    mPrograms;          //< List of programs indexed by entry point number
    };

    typedef Routine::Handle          RoutineHandle;

    class RTContext
    {
    public:
        using SharedPtr = std::shared_ptr<RTContext>;
        using SharedConstPtr = std::shared_ptr<const RTContext>;

        /** 
        Creates a new instance.
        */
        static SharedPtr create();

        RTContext();
        ~RTContext();

        /**
            Creates a RoutineHandle with entry points specified in a given filename. entryPoints pairs a entry point with a function
            in the file.

            As an example, createRoutine("camera", { {0, "pinhole"} }) creates a routine with a single entry point, from the function
            RT_PROGRAM void pinhole(void) in camera.cu.
        */
        RoutineHandle createRoutine(const std::string& fileName, const std::initializer_list<std::pair<int, std::string>>& entryPoints);

        /**
            Creates a routine with functions from multiple files. Each tuple is specified as follows:
            {int entry_point, string fileName, string function}.
            
            As an example, createRoutine({std::make_tuple(0, "camera", "pinhole")}) creates a routine with a single entry point, from the function
            RT_PROGRAM void pinhole(void) in camera.cu.
        */
        RoutineHandle createRoutine(const std::initializer_list<std::tuple<int, std::string, std::string>>& entryPoints);
        void reloadRoutine(RoutineHandle& routine);

        /**
            Creates a new scene, either empty or from a predefined Falcor scene. Note that from a Falcor scene we import the geometry only.
        */
        SceneHandle newScene();
        SceneHandle newScene(const Scene::SharedPtr& scene, RoutineHandle& shadingRtn, RoutineHandle& anyHitRtn = RoutineHandle(), RoutineHandle& intersectionRtn = RoutineHandle());

        /**
            Adds a static object to a scene. Scene-graph wise, all static geometry is placed in the same geometric node. 
        */
        ObjectHandle addObject(const Model::SharedPtr& model, const Scene::ModelInstance& instance, RoutineHandle& shadingRtn, RoutineHandle& anyHitRtn = RoutineHandle(), RoutineHandle& intersectionRtn = RoutineHandle());
        
        /**
            Adds a dynamic object. The returned handle can be used to apply affine transformations to the objects with the method setMatrix.
            There is no performance penalty in applying a rigid transformation (i.e. the Bvh is not rebuilt), but take care of the world/object
            semantic in routines (e.g. camera routines operate in world space, intersection in object space, see 4.1.6. Program Variable Transformation
            of the OptiX programming guide). 
            If you update a transform, you will need to call the updateTransforms() method to see the results.
        */
        DynamicObjectHandle addDynamicObject(const Model::SharedPtr& mesh, RoutineHandle& shadingRtn, RoutineHandle& anyHitRtn = RoutineHandle(), RoutineHandle& intersectionRtn = RoutineHandle());
        ObjectHandle getObject(DynamicObjectHandle dynamic_object);

        DynamicObjectHandle enableDynamicObject(const Mesh::SharedPtr& mesh);
        void disableDynamicObject(const Mesh::SharedPtr& mesh);

        void transformStaticObject(ObjectHandle object, const glm::mat4x3& mx);

        void setLights(const std::initializer_list<Light*>& list);

		/**
		    Set lights and pass it to GPU
		*/
		void setLights(const std::vector<Light::SharedPtr>& list);

        void render(const Fbo::SharedPtr& target, const Camera::SharedPtr& camera, RoutineHandle& raygenFn, RoutineHandle& missFn = RoutineHandle(), const int activeEntryPoint = 0, const std::initializer_list<uint32_t>& launchGrid = std::initializer_list<uint32_t>());

        /**
            Resets the context, removing the references to all routines
        */
        void resetRoutines();

        /**
            Returns a handle to the underlying context. Be aware that this function can be deprecated and removed, so use at your own risk.
        */
        inline ContextHandle& getCtx()  { return mpContext; }

        /**
            Overrides an intersection routine for a particular object
        */
        void setIntersectionTest(ObjectHandle object, RoutineHandle intersectionRtn);

        /**
            Overrides a hit routine for all objects in the scene
        */
        void setSceneClosestHitRoutine(RoutineHandle hitRtn);

        /**
            Overrides a miss routine for all objects in the scene
        */
        void setSceneAnyHitRoutine(RoutineHandle missRtn);

        void updateTransforms();

        /**************************************************************************
		    Variable setters
		**************************************************************************/
        template<typename T>    void set(const char* name, const T& value) { mpContext[name]->setUserData(sizeof(T), &value); }
        template<>              void set<float>(const char* name, const float& value) { mpContext[name]->setFloat(value); }
        template<>              void set<int32_t>(const char* name, const int32_t& value) { mpContext[name]->setInt(value); }
        template<>              void set<uint32_t>(const char* name, const uint32_t& value) { mpContext[name]->setUint(value); }
        template<>              void set<BufferHandle>(const char* name, const BufferHandle& value) { mpContext[name]->setBuffer(value); }
        
        /**************************************************************************
		    Interop
		**************************************************************************/

        /**
            Creates a shared OptiX buffer between OpenGL and OptiX
        */
        BufferHandle createSharedBuffer(Falcor::BufferHandle glApiHandle, RTformat format, size_t elementCount, bool writable = false);
        
        /**
            Creates a shared OptiX sampler for an OpenGL texture
        */
        SamplerHandle createSharedTexture(Texture::SharedConstPtr& pTexture, bool bufferIndexing = false);

        /**
            Returns a bindless pointer to the buffer
        */
        BufPtr getBufferPtr(BufferHandle& buffer);

    protected:

        void registerOptiXTexture(struct MaterialValue& v, const Sampler::SharedPtr& sampler);

        bool updateTempBuffer(const Fbo::SharedPtr& target);

        /**
            Extracts geometry from the Falcor model and inserts it into the given geometry group. 
        */
        ObjectHandle addGeometry(const Model::SharedPtr& mesh, const Scene::ModelInstance& instance, RoutineHandle& shadingRtn, RoutineHandle& anyHitRtn, RoutineHandle& intersectionRtn);
        
        RoutineHandle createRoutineInternal(const std::vector<std::tuple<int, std::string, std::string>>& entryPoints);

        /**
            Updates the scene, adds global indexing buffer for instances and materials. 
        */
        void updateScene();
        
        /**
            Creates a shared OptiX buffer between OpenGL and OptiX
        */
        BufferHandle _createSceneSharedBuffer(Falcor::BufferHandle glApiHandle, RTformat format, size_t elementCount);
        
        /**
            Creates a shared OptiX sampler for an OpenGL texture
        */
        SamplerHandle _createSceneSharedTexture(Texture::SharedConstPtr& pTexture, bool bufferIndexing = false);

    protected:
        ContextHandle            mpContext;

        SceneHandle              mCurrentScene;
        BufferHandle             mLights;
        BufferHandle             mCameras;

        // Static geometry default objects
        ObjectHandle             mStaticGeometry;
        optix::Acceleration      mStaticGeometryAcceleration;

        // Scene data
        struct ModelGeometry
        {
            BufferHandle        indices;
            BufferHandle        positions;
            BufferHandle        normals;
            BufferHandle        tangents;
            BufferHandle        bitangents;
            BufferHandle        texcoord;
            mat4                transform;
        };
        struct BindlessGeoInstance
        {
            int             indices;
            int             positions;
            int             normals;
            int             tangents;
            int             bitangents;
            int             texcoord;
            int             _pad[2];
            mat4            transform;
            mat4            invTrTransform;
            MaterialData    material;
        };
        struct MeshInstance
        {
            int32_t                     meshId = -1;
            ModelGeometry               geo;
            MaterialData                material;
            optix::GeometryInstance     instance;
            ObjectHandle                object;
            DynamicObjectHandle         transformObject;
            bool                        dynamic = false;
        };
        std::vector<MeshInstance>  mInstances;
        bool                     mSceneDirty = true;
        bool                     mSceneTransformDirty = true;

        float                    mSceneRadius = -1.0f;

        // Global access buffer for the scene
        BufferHandle             mSceneInstancesBuffer;

        // Default routines
        RoutineHandle            mMeshIntersectionRtn;
        RoutineHandle            mMeshBoundRtn;
        RoutineHandle            mDefaultExceptRtn;

        // Cache for sharing of textures and FBOs with OptiX
        struct SharedTextureBuffer
        {
            GLuint                  PBO = 0;
            BufferHandle            PBOBuffer;
            Texture::WeakConstPtr   sourceTexture;
        };
        std::map<uint32_t, SharedTextureBuffer> mSharedBuffers;
        BufferHandle             mFrameBuffers; ///< Used to bind (multiple) render targets

        // Cache shared lights
        std::map<Light::SharedPtr, LightData>    mCachedLights;

        // Registered OptiX texture 
        std::map<GLuint, SamplerHandle> mOGLSharedTextures;
        std::map<GLuint, BufferHandle>  mOGLSharedBuffers;
    };
}
}
