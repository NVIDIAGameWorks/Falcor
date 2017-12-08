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

#include "Framework.h"
#include "API/RenderContext.h"
#include "Data/Effects/ParticleData.h"

namespace Falcor
{
    class Gui;

    class ParticleSystem
    {
    public:
        static const char* kVertexShader;           ///< Filename for the vertex shader
        static const char* kSortShader;             ///< Filename for the sorting compute shader
        static const char* kEmitShader;             ///< Filename for the emit compute shader
        static const char* kDefaultPixelShader;     ///< Filename for the default pixel shader
        static const char* kDefaultSimulateShader;  ///< Filename for the particle update/simulation compute shader

        using SharedPtr = std::shared_ptr<ParticleSystem>;

        /** Creates a new particle system
            \params[in] pCtx The render context
            \params[in] maxParticles The max number of particles allowed at once, emits will be blocked if the system is maxxed out 
            \params[in] drawPixelShader The pixel shader used to draw the particles
            \params[in] simulateComputeShader The compute shader used to update the particles
            \params[in] sorted Whether or not the particles should be sorted by depth before render
        */
        static SharedPtr create(RenderContext* pCtx, uint32_t maxParticles, uint32_t maxEmitPerFrame,
            std::string drawPixelShader = kDefaultPixelShader,
            std::string simulateComputeShader = kDefaultSimulateShader,
            bool sorted = true);

        /** Updates the particle system, emitting if it's time to do so and simulating particles 
        */
        void update(RenderContext* pCtx, float dt, glm::mat4 view);

        /** Render the particle system, sorting if necessary and drawing the particles
        */
        void render(RenderContext* pCtx, glm::mat4 view, glm::mat4 proj);

        /** Render UI controls for this particle system.
            \param[in] pGui GUI instance to render UI elements with
        */
        void renderUi(Gui* pGui);

        /** Gets the graphics vars for drawing.
        */
        GraphicsVars::SharedPtr getDrawVars() { return mDrawResources.pVars; }

        /** Gets the simulation shader program
        */
        ComputeProgram::SharedPtr getSimulateProgram() { return mSimulateResources.pState->getProgram(); }

        /** Get the graphics vars used for the particle simulation shader
        */
        ComputeVars::SharedPtr getSimulateVars() { return mSimulateResources.pVars; }

        /** Sets how long a particle will remain alive after spawning.
            \params[in] dur The new base duration 
            \params[in] offset The new random offset to be applied. final value is base + randRange(-offset, offset)
        */
        void setParticleDuration(float dur, float offset);

        /** Returns the particle emitter's current duration.
        */
        float getParticleDuration() { return mEmitter.duration; }

        /** Sets data associated with the emitting of particles
            \params[in] emitCount The new base emit count
            \params[in] emitCountOffset The new random offset to be applied. final value is base + randRange(-offset, offset)
            \params[in] emitFrequency The frequency at which particles should be emitted
        */
        void setEmitData(uint32_t emitCount, uint32_t emitCountOffset, float emitFrequency);

        /** Sets particles' spawn position.
            \params[in] spawnPos The new base spawn position
            \params[in] offset The new random offset to be applied. final value is base + randRange(-offset, offset)
        */
        void setSpawnPos(vec3 spawnPos, vec3 offset);

        /** Sets the velocity particles spawn with.
            \params[in] velocity The new base velocity
            \params[in] offset The new random offset to be applied. final value is base + randRange(-offset, offset)
        */
        void setVelocity(vec3 velocity, vec3 offset);

        /** Sets the acceleration particles spawn with.
            \params[in] accel The new base acceleration
            \params[in] offset The new random offset to be applied. final value is base + randRange(-offset, offset)
        */
        void setAcceleration(vec3 accel, vec3 offset);

        /** Sets the scale particles spawn with.
            \params[in] scale The new base scale
            \params[in] offset The new random offset to be applied. final value is base + randRange(-offset, offset)
        */
        void setScale(float scale, float offset);

        /** Sets the rate of change of the particles' scale.
            \params[in] growth The new base growth
            \params[in] offset The new random offset to be applied. final value is base + randRange(-offset, offset)
        */
        void setGrowth(float growth, float offset);

        /** Sets the rotation particles spawn with.
            \params[in] rot The new base rotation in radians
            \params[in] offset The new random offset to be applied. final value is base + randRange(-offset, offset)
        */
        void setBillboardRotation(float rot, float offset);

        /** Sets the the rate of change of the particles' rotation.
            \params[in] rotVel The new base rotational velocity in radians/second
            \params[in] offset The new random offset to be applied. final value is base + randRange(-offset, offset)
        */
        void setBillboardRotationVelocity(float rotVel, float offset);

    private:
        ParticleSystem() = delete;
        ParticleSystem(RenderContext* pCtx, uint32_t maxParticles, uint32_t maxEmitPerFrame,
            std::string drawPixelShader, std::string simulateComputeShader, bool sorted);
        void emit(RenderContext* pCtx, uint32_t num);

        struct EmitterData
        {
            EmitterData() : duration(3.f), durationOffset(0.f), emitFrequency(0.1f), emitCount(32),
                emitCountOffset(0), spawnPos(0.f, 0.f, 0.f), spawnPosOffset(0.f, 0.5f, 0.f),
                vel(0, 5, 0), velOffset(2, 1, 2), accel(0, -3, 0), accelOffset(0.f, 0.f, 0.f),
                scale(0.2f), scaleOffset(0.f), growth(-0.05f), growthOffset(0.f), billboardRotation(0.f),
                billboardRotationOffset(0.25f), billboardRotationVel(0.f), billboardRotationVelOffset(0.f) {}
            float duration;
            float durationOffset; 
            float emitFrequency;
            int32_t emitCount;
            int32_t emitCountOffset;
            vec3 spawnPos;
            vec3 spawnPosOffset;
            vec3 vel;
            vec3 velOffset;
            vec3 accel;
            vec3 accelOffset;
            float scale;
            float scaleOffset;
            float growth;
            float growthOffset;
            float billboardRotation;
            float billboardRotationOffset;
            float billboardRotationVel;
            float billboardRotationVelOffset;
        } mEmitter;

        struct EmitResources
        {
            ComputeVars::SharedPtr pVars;
            ComputeState::SharedPtr pState;
        } mEmitResources;

        struct SimulateResources
        {
            ComputeVars::SharedPtr pVars;
            ComputeState::SharedPtr pState;
        } mSimulateResources;

        struct DrawResources
        {
            GraphicsVars::SharedPtr pVars;
            GraphicsState::SharedPtr pState;
            Vao::SharedPtr pVao;
        } mDrawResources;

        struct
        {
            ProgramReflection::BindLocation simulateCB;
            ProgramReflection::BindLocation drawCB;
            ProgramReflection::BindLocation emitCB;
        } mBindLocations;

        uint32_t mMaxParticles;
        uint32_t mMaxEmitPerFrame;
        uint32_t mSimulateThreads;
        float mEmitTimer = 0.f;

        //buffers
        StructuredBuffer::SharedPtr mpParticlePool;
        StructuredBuffer::SharedPtr mpEmitList;
        StructuredBuffer::SharedPtr mpDeadList;
        StructuredBuffer::SharedPtr mpAliveList;
        //for draw (0 - Verts Per Instance, 1 - Instance Count, 
        //2 - start vertex offset, 3 - start instance offset)
        StructuredBuffer::SharedPtr mpIndirectArgs;

        //Data for sorted systems
        void initSortResources();
        bool mShouldSort;
        std::vector<SortData> mSortDataReset;
        struct SortResources
        {
            StructuredBuffer::SharedPtr pSortIterationCounter;
            ComputeState::SharedPtr pState;
            ComputeVars::SharedPtr pVars;
        } mSortResources;
    };
}
