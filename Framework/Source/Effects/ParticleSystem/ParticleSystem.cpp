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
#include "ParticleSystem.h"
#include "API/Resource.h"
#include "glm/gtc/random.hpp"
#include <algorithm>
#include "Utils/Gui.h"
#include <limits>

namespace Falcor
{
    const char* ParticleSystem::kVertexShader = "Effects/ParticleVertex.vs.slang";
    const char* ParticleSystem::kSortShader = "Effects/ParticleSort.cs.slang";
    const char* ParticleSystem::kEmitShader = "Effects/ParticleEmit.cs.slang";
    const char* ParticleSystem::kDefaultPixelShader = "Effects/ParticleTexture.ps.slang";
    const char* ParticleSystem::kDefaultSimulateShader = "Effects/ParticleSimulate.cs.slang";

    ParticleSystem::SharedPtr ParticleSystem::create(RenderContext* pCtx, uint32_t maxParticles, uint32_t maxEmitPerFrame,
        std::string drawPixelShader, std::string simulateComputeShader, bool sorted)
    {
        return ParticleSystem::SharedPtr(
            new ParticleSystem(pCtx, maxParticles, maxEmitPerFrame, drawPixelShader, simulateComputeShader, sorted));
    }

    ParticleSystem::ParticleSystem(RenderContext* pCtx, uint32_t maxParticles, uint32_t maxEmitPerFrame,
        std::string drawPixelShader, std::string simulateComputeShader, bool sorted)
    {
        mShouldSort = sorted;
        mMaxEmitPerFrame = maxEmitPerFrame;

        //Data that is different if system is sorted
        Program::DefineList defineList;
        if (mShouldSort)
        {
            mMaxParticles = (uint32_t)pow(2, (std::ceil(log2((float)maxParticles))));
            initSortResources();
            defineList.add("_SORT");
        }
        else
        {
            mMaxParticles = maxParticles;
        }
        //compute cs
        ComputeProgram::SharedPtr pSimulateCs = ComputeProgram::createFromFile(simulateComputeShader, defineList);
  
        //get num sim threads, required as a define for emit cs
        uvec3 simThreads;

        simThreads = pSimulateCs->getActiveVersion()->getReflector()->getThreadGroupSize();
        mSimulateThreads = simThreads.x * simThreads.y * simThreads.z;

        //Emit cs
        Program::DefineList emitDefines;
        emitDefines.add("_SIMULATE_THREADS", std::to_string(mSimulateThreads));
        ComputeProgram::SharedPtr pEmitCs = ComputeProgram::createFromFile(kEmitShader, emitDefines);

        //draw shader
        GraphicsProgram::SharedPtr pDrawProgram = GraphicsProgram::createFromFile(kVertexShader, drawPixelShader, defineList);

        //ParticlePool
        mpParticlePool = StructuredBuffer::create(pEmitCs, "particlePool", mMaxParticles);

        //emitList
        mpEmitList = StructuredBuffer::create(pEmitCs, "emitList", mMaxEmitPerFrame);

        //Dead List
        mpDeadList = StructuredBuffer::create(pEmitCs, "deadList", mMaxParticles);

        // Init data in dead list buffer
        mpDeadList->getUAVCounter()->updateData(&mMaxParticles, 0, sizeof(uint32_t));
        std::vector<uint32_t> indices;
        indices.resize(mMaxParticles);
        uint32_t counter = 0;
        std::generate(indices.begin(), indices.end(), [&counter] {return counter++; });
        mpDeadList->setBlob(indices.data(), 0, indices.size() * sizeof(uint32_t));

        // Alive list
        mpAliveList = StructuredBuffer::create(pSimulateCs, "aliveList", mMaxParticles);

        // Indirect args
        Resource::BindFlags indirectBindFlags = Resource::BindFlags::IndirectArg | Resource::BindFlags::UnorderedAccess;
        mpIndirectArgs = StructuredBuffer::create(pSimulateCs, "drawArgs", 1, indirectBindFlags);

        //initialize the first member of the args, vert count per instance, to be 4 for particle billboards
        uint32_t vertexCountPerInstance = 4;
        mpIndirectArgs->setBlob(&vertexCountPerInstance, 0, sizeof(uint32_t));

        //Vars
        //emit
        mEmitResources.pVars = ComputeVars::create(pEmitCs->getActiveVersion()->getReflector());
        mEmitResources.pVars->setStructuredBuffer("deadList", mpDeadList);
        mEmitResources.pVars->setStructuredBuffer("particlePool", mpParticlePool);
        mEmitResources.pVars->setStructuredBuffer("emitList", mpEmitList);
        mEmitResources.pVars->setRawBuffer("numAlive", mpAliveList->getUAVCounter());
        //simulate
        mSimulateResources.pVars = ComputeVars::create(pSimulateCs->getActiveVersion()->getReflector());
        mSimulateResources.pVars->setStructuredBuffer("deadList", mpDeadList);
        mSimulateResources.pVars->setStructuredBuffer("particlePool", mpParticlePool);
        mSimulateResources.pVars->setStructuredBuffer("drawArgs", mpIndirectArgs);
        mSimulateResources.pVars->setStructuredBuffer("aliveList", mpAliveList);
        mSimulateResources.pVars->setRawBuffer("numDead", mpDeadList->getUAVCounter());
        if (mShouldSort)
        {
            mSimulateResources.pVars->setStructuredBuffer("sortIterationCounter", mSortResources.pSortIterationCounter);
            //sort
            mSortResources.pVars->setStructuredBuffer("sortList", mpAliveList);
            mSortResources.pVars->setStructuredBuffer("iterationCounter", mSortResources.pSortIterationCounter);
        }

        //draw
        mDrawResources.pVars = GraphicsVars::create(pDrawProgram->getActiveVersion()->getReflector());
        mDrawResources.pVars->setStructuredBuffer("aliveList", mpAliveList);
        mDrawResources.pVars->setStructuredBuffer("particlePool", mpParticlePool);

        //State
        mEmitResources.pState = ComputeState::create();
        mEmitResources.pState->setProgram(pEmitCs);
        mSimulateResources.pState = ComputeState::create();
        mSimulateResources.pState->setProgram(pSimulateCs);
        mDrawResources.pState = GraphicsState::create();
        mDrawResources.pState->setProgram(pDrawProgram);

        //Create empty vbo for draw 
        Vao::BufferVec bufferVec;
        VertexLayout::SharedPtr pLayout = VertexLayout::create();
        Vao::Topology topology = Vao::Topology::TriangleStrip;
        mDrawResources.pState->setVao(Vao::create(topology, pLayout, bufferVec));

        // Save bind locations for resourced updated during draw
        mBindLocations.simulateCB = pSimulateCs->getActiveVersion()->getReflector()->getDefaultParameterBlock()->getResourceBinding("PerFrame");
        mBindLocations.drawCB = pDrawProgram->getActiveVersion()->getReflector()->getDefaultParameterBlock()->getResourceBinding("PerFrame");
        mBindLocations.emitCB = pEmitCs->getActiveVersion()->getReflector()->getDefaultParameterBlock()->getResourceBinding("PerEmit");
    }

    void ParticleSystem::emit(RenderContext* pCtx, uint32_t num)
    {
        std::vector<Particle> emittedParticles;
        emittedParticles.resize(num);
        for (uint32_t i = 0; i < num; ++i)
        {
            Particle p;  
            p.pos = mEmitter.spawnPos + glm::linearRand(-mEmitter.spawnPosOffset, mEmitter.spawnPosOffset);
            p.vel = mEmitter.vel + glm::linearRand(-mEmitter.velOffset, mEmitter.velOffset);
            p.accel = mEmitter.accel + glm::linearRand(-mEmitter.accelOffset, mEmitter.accelOffset);
            //total scale of the billboard, so the amount to actually move to billboard corners is half scale. 
            p.scale = 0.5f * mEmitter.scale + glm::linearRand(-mEmitter.scaleOffset, mEmitter.scaleOffset);
            p.growth = 0.5f * mEmitter.growth + glm::linearRand(-mEmitter.growthOffset, mEmitter.growthOffset);
            p.life = mEmitter.duration + glm::linearRand(-mEmitter.growthOffset, mEmitter.growthOffset);
            p.rot = mEmitter.billboardRotation + glm::linearRand(-mEmitter.billboardRotationOffset, mEmitter.billboardRotationOffset);
            p.rotVel = mEmitter.billboardRotationVel + glm::linearRand(-mEmitter.billboardRotationVelOffset, mEmitter.billboardRotationVelOffset);
            emittedParticles[i] = p;
        }
        //Fill emit data
        EmitData emitData;
        emitData.numEmit = num;
        emitData.maxParticles = mMaxParticles;
        //update emitted particles list
        mpEmitList->setBlob(emittedParticles.data(), 0, emittedParticles.size() * sizeof(Particle));

        //Send vars and call
        pCtx->pushComputeState(mEmitResources.pState);
        mEmitResources.pVars->getDefaultBlock()->getConstantBuffer(mBindLocations.emitCB, 0)->setBlob(&emitData, 0u, sizeof(EmitData));
        pCtx->pushComputeVars(mEmitResources.pVars);
        uint32_t numGroups = (uint32_t)std::ceil((float)num / EMIT_THREADS);
        pCtx->dispatch(1, numGroups, 1);
        pCtx->popComputeVars();
        pCtx->popComputeState();
    }

    void ParticleSystem::update(RenderContext* pCtx, float dt, glm::mat4 view)
    {
        //emit
        mEmitTimer += dt;
        if (mEmitTimer >= mEmitter.emitFrequency)
        {
            mEmitTimer -= mEmitter.emitFrequency;
            emit(pCtx, max(mEmitter.emitCount + glm::linearRand(-mEmitter.emitCountOffset, mEmitter.emitCountOffset), 0));
        }

        //Simulate
        if (mShouldSort)
        {
            SimulateWithSortPerFrame perFrame;
            perFrame.view = view;
            perFrame.dt = dt;
            perFrame.maxParticles = mMaxParticles;
            mSimulateResources.pVars->getDefaultBlock()->getConstantBuffer(mBindLocations.simulateCB, 0)->setBlob(&perFrame, 0u, sizeof(SimulateWithSortPerFrame));
            mpAliveList->setBlob(mSortDataReset.data(), 0, sizeof(SortData) * mMaxParticles);
        }
        else
        {
            SimulatePerFrame perFrame;
            perFrame.dt = dt;
            perFrame.maxParticles = mMaxParticles;
            mSimulateResources.pVars->getDefaultBlock()->getConstantBuffer(mBindLocations.simulateCB, 0)->setBlob(&perFrame, 0u, sizeof(SimulatePerFrame));
        }

        //reset alive list counter to 0
        uint32_t zero = 0;
        mpAliveList->getUAVCounter()->updateData(&zero, 0, sizeof(uint32_t));

        pCtx->pushComputeState(mSimulateResources.pState);
        pCtx->pushComputeVars(mSimulateResources.pVars);
        pCtx->dispatch(max(mMaxParticles / mSimulateThreads, 1u), 1, 1);
        pCtx->popComputeVars();
        pCtx->popComputeState();
    }

    void ParticleSystem::render(RenderContext* pCtx, glm::mat4 view, glm::mat4 proj)
    {
        //sorting
        if (mShouldSort)
        {
            pCtx->pushComputeState(mSortResources.pState);
            pCtx->pushComputeVars(mSortResources.pVars);

            pCtx->dispatch(1, 1, 1);

            pCtx->popComputeVars();
            pCtx->popComputeState();
        }

        //Draw cbuf
        VSPerFrame cbuf;
        cbuf.view = view;
        cbuf.proj = proj;
        mDrawResources.pVars->getDefaultBlock()->getConstantBuffer(mBindLocations.drawCB, 0)->setBlob(&cbuf, 0, sizeof(cbuf));

        //particle draw uses many of render context's existing state's properties 
        GraphicsState::SharedPtr state = pCtx->getGraphicsState();
        mDrawResources.pState->setBlendState(state->getBlendState());
        mDrawResources.pState->setFbo(state->getFbo());
        mDrawResources.pState->setRasterizerState(state->getRasterizerState());
        mDrawResources.pState->setDepthStencilState(state->getDepthStencilState());

        pCtx->pushGraphicsState(mDrawResources.pState);
        pCtx->pushGraphicsVars(mDrawResources.pVars);
        pCtx->drawIndirect(mpIndirectArgs.get(), 0);
        pCtx->popGraphicsVars();
        pCtx->popGraphicsState();
    }

    void ParticleSystem::renderUi(Gui* pGui)
    {
        float floatMax = std::numeric_limits<float>::max();
        pGui->addFloatVar("Duration", mEmitter.duration, 0.f);
        pGui->addFloatVar("DurationOffset", mEmitter.durationOffset, 0.f);
        pGui->addFloatVar("Frequency", mEmitter.emitFrequency, 0.01f);
        int32_t emitCount = mEmitter.emitCount;
        pGui->addIntVar("EmitCount", emitCount, 0, mMaxEmitPerFrame);
        mEmitter.emitCount = emitCount;
        pGui->addIntVar("EmitCountOffset", mEmitter.emitCountOffset, 0);
        pGui->addFloat3Var("SpawnPos", mEmitter.spawnPos, -floatMax, floatMax);
        pGui->addFloat3Var("SpawnPosOffset", mEmitter.spawnPosOffset, 0.f, floatMax);
        pGui->addFloat3Var("Velocity", mEmitter.vel, -floatMax, floatMax);
        pGui->addFloat3Var("VelOffset", mEmitter.velOffset, 0.f, floatMax);
        pGui->addFloat3Var("Accel", mEmitter.accel, -floatMax, floatMax);
        pGui->addFloat3Var("AccelOffset", mEmitter.accelOffset, 0.f, floatMax);
        pGui->addFloatVar("Scale", mEmitter.scale, 0.001f);
        pGui->addFloatVar("ScaleOffset", mEmitter.scaleOffset, 0.001f);
        pGui->addFloatVar("Growth", mEmitter.growth);
        pGui->addFloatVar("GrowthOffset", mEmitter.growthOffset, 0.001f);
        pGui->addFloatVar("BillboardRotation", mEmitter.billboardRotation);
        pGui->addFloatVar("BillboardRotationOffset", mEmitter.billboardRotationOffset);
        pGui->addFloatVar("BillboardRotationVel", mEmitter.billboardRotationVel);
        pGui->addFloatVar("BillboardRotationVelOffset", mEmitter.billboardRotationVelOffset);
    }

    void ParticleSystem::initSortResources()
    {
        //Shader
        ComputeProgram::SharedPtr pSortCs = ComputeProgram::createFromFile(kSortShader);

        //iteration counter buffer
        mSortResources.pSortIterationCounter = StructuredBuffer::create(pSortCs, "iterationCounter", 2);

        //Sort data reset buffer
        SortData resetData;
        resetData.index = (uint32_t)(-1);
        resetData.depth = std::numeric_limits<float>::max();
        mSortDataReset.resize(mMaxParticles, resetData);

        //Vars and state
        mSortResources.pVars = ComputeVars::create(pSortCs->getActiveVersion()->getReflector());
        mSortResources.pState = ComputeState::create();
        mSortResources.pState->setProgram(pSortCs);
    }

    void ParticleSystem::setParticleDuration(float dur, float offset)
    { 
        mEmitter.duration = dur; 
        mEmitter.durationOffset = offset; 
    }

    void ParticleSystem::setEmitData(uint32_t emitCount, uint32_t emitCountOffset, float emitFrequency)
    {
        mEmitter.emitCount = (int32_t)emitCount;
        mEmitter.emitCountOffset = (int32_t)emitCountOffset;
        mEmitter.emitFrequency = emitFrequency;
    }

    void ParticleSystem::setSpawnPos(vec3 spawnPos, vec3 offset)
    {
        mEmitter.spawnPos = spawnPos;
        mEmitter.spawnPosOffset = offset;
    }

    void ParticleSystem::setVelocity(vec3 velocity, vec3 offset)
    {
        mEmitter.vel = velocity;
        mEmitter.velOffset = offset;
    }

    void ParticleSystem::setAcceleration(vec3 accel, vec3 offset)
    {
        mEmitter.accel = accel;
        mEmitter.accelOffset = offset;
    }

    void ParticleSystem::setScale(float scale, float offset)
    {
        mEmitter.scale = scale;
        mEmitter.scaleOffset = offset;
    }

    void ParticleSystem::setGrowth(float growth, float offset)
    {
        mEmitter.growth = growth;
        mEmitter.growthOffset = offset;
    }

    void ParticleSystem::setBillboardRotation(float rot, float offset)
    {
        mEmitter.billboardRotation = rot;
        mEmitter.billboardRotationOffset = offset;
    }
    
    void ParticleSystem::setBillboardRotationVelocity(float rotVel, float offset)
    {
        mEmitter.billboardRotationVel = rotVel;
        mEmitter.billboardRotationVelOffset = offset;
    }
}
