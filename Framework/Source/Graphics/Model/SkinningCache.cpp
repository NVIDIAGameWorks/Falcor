/***************************************************************************
# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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
#include "Framework.h"
#include "SkinningCache.h"
#include "API/Device.h"
#include "Data/VertexAttrib.h"
#include "Graphics/Model/Model.h"

namespace Falcor
{
    static const char* kShaderFilenameSkinning = "Data/Framework/Shaders/ComputeSkinning.cs.slang";
    static const char* kPerModelCbName = "PerModelCB";
    static const char* kPerMeshCbName = "PerMeshCB";

    static const uint32_t kGroupSize = 256;     // threads per group

    SkinningCache::SharedPtr SkinningCache::create()
    {
        SharedPtr ptr = SharedPtr(new SkinningCache());
        return ptr->init() ? ptr : nullptr;
    }

    bool SkinningCache::update(const Model* pModel)
    {
        bool changed = false;
        if (pModel->hasBones())
        {
            RenderContext::SharedPtr pRenderContext = gpDevice->getRenderContext();
            pRenderContext->pushComputeState(mSkinningPass.pState);
            pRenderContext->pushComputeVars(mSkinningPass.pVars);

            setPerModelData(pModel);

            for (uint32_t meshId = 0; meshId < pModel->getMeshCount(); meshId++)
            {
                const Mesh* pMesh = pModel->getMesh(meshId).get();
                if (pMesh->hasBones())
                {
                    createVertexBuffers(pMesh);

                    // Bind resources
                    setPerMeshData(pMesh);

                    // Execute
                    // TODO: Using 1D dispatch for simplicity, which limits us to 64k x 256 = 16M vertices with 256 in group size. Fix if needed.
                    assert(pMesh->getVertexCount() <= 16*1024*1024);
                    uint32_t numGroups = (pMesh->getVertexCount() + kGroupSize - 1) / kGroupSize;
                    pRenderContext->dispatch(numGroups, 1, 1);

                    changed = true;
                }
            }

            pRenderContext->popComputeVars();
            pRenderContext->popComputeState();
        }
        return changed;
    }

    Vao::SharedPtr SkinningCache::getVao(const Mesh* pMesh) const
    {
        auto it = mSkinnedBuffers.find(pMesh);
        if (it != mSkinnedBuffers.end())
        {
            return it->second.pVao;
        }
        return nullptr;
    }

    bool SkinningCache::init()
    {
        // Create shaders
        mSkinningPass.pProgram = ComputeProgram::createFromFile(kShaderFilenameSkinning, "main");
        assert(mSkinningPass.pProgram);
        mSkinningPass.pVars = ComputeVars::create(mSkinningPass.pProgram->getReflector());

        // Create state
        mSkinningPass.pState = ComputeState::create();
        mSkinningPass.pState->setProgram(mSkinningPass.pProgram);

        const ParameterBlockReflection* pBlock = mSkinningPass.pProgram->getReflector()->getDefaultParameterBlock().get();
        initVariableOffsets(pBlock);
        initMeshBufferLocations(pBlock);

        return true;
    }

    void SkinningCache::initVariableOffsets(const ParameterBlockReflection* pBlock)
    {
        if (mVariableOffsets.bonesOffset == ConstantBuffer::kInvalidOffset)
        {
            const ReflectionVar* pVar = pBlock->getResource(kPerModelCbName).get();

            if (pVar != nullptr)
            {
                assert(pVar->getType()->asResourceType()->getType() == ReflectionResourceType::Type::ConstantBuffer);
                const ReflectionType* pType = pVar->getType().get();

                assert(pType->findMember("gBoneMat[0]")->getType()->asBasicType()->isRowMajor() == false); // We copy into CBs as column-major
                assert(pType->findMember("gInvTransposeBoneMat[0]")->getType()->asBasicType()->isRowMajor() == false);
                assert(pType->findMember("gBoneMat")->getType()->getTotalArraySize() >= MAX_BONES);
                assert(pType->findMember("gInvTransposeBoneMat")->getType()->getTotalArraySize() >= MAX_BONES);

                mVariableOffsets.bonesOffset = pType->findMember("gBoneMat[0]")->getOffset();
                mVariableOffsets.bonesInvTransposeOffset = pType->findMember("gInvTransposeBoneMat[0]")->getOffset();
            }
        }
    }

    void SkinningCache::initMeshBufferLocations(const ParameterBlockReflection* pBlock)
    {
        // Input
        mMeshBufferLocations.position = pBlock->getResourceBinding("gPositions");
        mMeshBufferLocations.normal = pBlock->getResourceBinding("gNormals");
        mMeshBufferLocations.bitangent = pBlock->getResourceBinding("gBitangents");
        mMeshBufferLocations.boneWeights = pBlock->getResourceBinding("gBoneWeights");
        mMeshBufferLocations.boneIds = pBlock->getResourceBinding("gBoneIds");
        // Output
        mMeshBufferLocations.positionOut = pBlock->getResourceBinding("gSkinnedPositions");
        mMeshBufferLocations.prevPositionOut = pBlock->getResourceBinding("gSkinnedPrevPositions");
        mMeshBufferLocations.normalOut = pBlock->getResourceBinding("gSkinnedNormals");
        mMeshBufferLocations.bitangentOut = pBlock->getResourceBinding("gSkinnedBitangents");
    }

    static Buffer::SharedPtr createVertexBuffer(uint32_t vertexLoc, const Vao* pVao, std::vector<Buffer::SharedPtr>& pVBs)
    {
        Buffer::SharedPtr pBuffer = nullptr;
        const auto& elemDesc = pVao->getElementIndexByLocation(vertexLoc);
        if (elemDesc.vbIndex != Vao::ElementDesc::kInvalidIndex)
        {
            assert(elemDesc.vbIndex < pVao->getVertexBuffersCount());
            size_t size = pVao->getVertexBuffer(elemDesc.vbIndex)->getSize();
            pBuffer = Buffer::create(size, Resource::BindFlags::Vertex | Resource::BindFlags::ShaderResource | Resource::BindFlags::UnorderedAccess, Buffer::CpuAccess::None);
            pVBs[elemDesc.vbIndex] = pBuffer;
        }
        return pBuffer;
    }

    // Create vertex buffers for the skinned vertices of a Mesh if they do not already exist.
    void SkinningCache::createVertexBuffers(const Mesh* pMesh)
    {
        auto it = mSkinnedBuffers.find(pMesh);
        if (it == mSkinnedBuffers.end())
        {
            const Vao* pVao = pMesh->getVao().get();
            const uint32_t bufferCount = pVao->getVertexBuffersCount();

            // Create buffers for skinned vertices
            std::vector<Buffer::SharedPtr> pVBs(bufferCount + 1);

            Buffer::SharedPtr pPosBuffer = createVertexBuffer(VERTEX_POSITION_LOC, pVao, pVBs);
            createVertexBuffer(VERTEX_NORMAL_LOC, pVao, pVBs);
            createVertexBuffer(VERTEX_BITANGENT_LOC, pVao, pVBs);

            // Copy non-skinned buffers from the original VAO.
            for (uint32_t i = 0; i < bufferCount; i++)
            {
                if (pVBs[i] == nullptr)
                {
                    pVBs[i] = pVao->getVertexBuffer(i);
                }
            }

            // Create duplicate of position buffer to hold positions for the previous frame
            pVBs[bufferCount] = Buffer::create(pPosBuffer->getSize(), pPosBuffer->getBindFlags(), pPosBuffer->getCpuAccess());

            VertexBufferLayout::SharedPtr pVbLayout = VertexBufferLayout::create();
            pVbLayout->addElement(VERTEX_PREV_POSITION_NAME, 0, ResourceFormat::RGB32Float, 1, VERTEX_PREV_POSITION_LOC);

            // Create new vertex layout including the additional buffer
            VertexLayout::SharedPtr pLayout = VertexLayout::create();
            assert(pVao->getVertexLayout()->getBufferCount() == bufferCount);
            for (uint32_t i = 0; i < bufferCount; i++)
            {
                pLayout->addBufferLayout(i, pVao->getVertexLayout()->getBufferLayout(i));
            }
            pLayout->addBufferLayout(bufferCount, pVbLayout);

            // Create VAO for skinned mesh.
            VertexBuffers buffers;
            buffers.pVao = Vao::create(pVao->getPrimitiveTopology(), pLayout, pVBs, pVao->getIndexBuffer(), pVao->getIndexBufferFormat());

            mSkinnedBuffers[pMesh] = buffers;
        }
    }

    void SkinningCache::setPerModelData(const Model* pModel)
    {
        // Set bones
        assert(pModel->hasBones());
        ConstantBuffer::SharedPtr pCB = mSkinningPass.pVars->getConstantBuffer(kPerModelCbName);
        if (pCB)
        {
            assert(pModel->getBoneCount() <= MAX_BONES);
            pCB->setVariableArray(mVariableOffsets.bonesOffset, pModel->getBoneMatrices(), pModel->getBoneCount());
            pCB->setVariableArray(mVariableOffsets.bonesInvTransposeOffset, pModel->getBoneInvTransposeMatrices(), pModel->getBoneCount());
        }
    }

    static bool setVertexBuffer(ParameterBlockReflection::BindLocation bindLocation, uint32_t vertexLoc, const Vao* pVao, ProgramVars* pVars, ResourceFormat expectedFormat = ResourceFormat::Unknown)
    {
        assert(bindLocation.setIndex != ProgramReflection::kInvalidLocation);
        const auto& elemDesc = pVao->getElementIndexByLocation(vertexLoc);
        if (elemDesc.elementIndex == Vao::ElementDesc::kInvalidIndex)
        {
            pVars->getDefaultBlock()->setSrv(bindLocation, 0, nullptr);
        }
        else
        {
            assert(elemDesc.elementIndex == 0);
            assert(elemDesc.vbIndex != Vao::ElementDesc::kInvalidIndex);
            assert(expectedFormat == ResourceFormat::Unknown || pVao->getVertexLayout()->getBufferLayout(elemDesc.vbIndex)->getElementFormat(elemDesc.elementIndex) == expectedFormat);
            pVars->getDefaultBlock()->setSrv(bindLocation, 0, pVao->getVertexBuffer(elemDesc.vbIndex)->getSRV());
            return true;
        }
        return false;
    }

    // TODO: avoid duplication, the functions only differ by Srv/Uav
    static bool setVertexBufferUAV(ParameterBlockReflection::BindLocation bindLocation, uint32_t vertexLoc, const Vao* pVao, ProgramVars* pVars, ResourceFormat expectedFormat = ResourceFormat::Unknown)
    {
        assert(bindLocation.setIndex != ProgramReflection::kInvalidLocation);
        const auto& elemDesc = pVao->getElementIndexByLocation(vertexLoc);
        if (elemDesc.elementIndex == Vao::ElementDesc::kInvalidIndex)
        {
            pVars->getDefaultBlock()->setUav(bindLocation, 0, nullptr);
        }
        else
        {
            assert(elemDesc.elementIndex == 0);
            assert(elemDesc.vbIndex != Vao::ElementDesc::kInvalidIndex);
            assert(expectedFormat == ResourceFormat::Unknown || pVao->getVertexLayout()->getBufferLayout(elemDesc.vbIndex)->getElementFormat(elemDesc.elementIndex) == expectedFormat);
            pVars->getDefaultBlock()->setUav(bindLocation, 0, pVao->getVertexBuffer(elemDesc.vbIndex)->getUAV());
            return true;
        }
        return false;
    }

    void SkinningCache::setPerMeshData(const Mesh* pMesh)
    {
        ProgramVars* pVars = mSkinningPass.pVars.get();

        // Set constants
        assert(pMesh->hasBones());
        ConstantBuffer::SharedPtr pCB = pVars->getConstantBuffer(kPerMeshCbName);
        if (pCB)
        {
            pCB["gNumVertices"] = pMesh->getVertexCount();
        }

        // Bind input vertex buffers
        const Vao* pVao = pMesh->getVao().get();
        bool hasPos = setVertexBuffer(mMeshBufferLocations.position, VERTEX_POSITION_LOC, pVao, pVars, ResourceFormat::RGB32Float);
        bool hasNormal = setVertexBuffer(mMeshBufferLocations.normal, VERTEX_NORMAL_LOC, pVao, pVars, ResourceFormat::RGB32Float);
        bool hasBitangent = setVertexBuffer(mMeshBufferLocations.bitangent, VERTEX_BITANGENT_LOC, pVao, pVars, ResourceFormat::RGB32Float);
        bool hasBoneWeight = setVertexBuffer(mMeshBufferLocations.boneWeights, VERTEX_BONE_WEIGHT_LOC, pVao, pVars, ResourceFormat::RGBA32Float);
        bool hasBoneId = setVertexBuffer(mMeshBufferLocations.boneIds, VERTEX_BONE_ID_LOC, pVao, pVars, ResourceFormat::RGBA8Uint);
        assert(hasPos && hasBoneWeight && hasBoneId);

        // Bind output vertex buffers. Note some of the buffers may be nullptr.
        const auto& it = mSkinnedBuffers.find(pMesh);        
        assert(it != mSkinnedBuffers.end());

        const Vao* pVaoOut = it->second.pVao.get();
        setVertexBufferUAV(mMeshBufferLocations.positionOut, VERTEX_POSITION_LOC, pVaoOut, pVars);
        setVertexBufferUAV(mMeshBufferLocations.prevPositionOut, VERTEX_PREV_POSITION_LOC, pVaoOut, pVars);
        setVertexBufferUAV(mMeshBufferLocations.normalOut, VERTEX_NORMAL_LOC, pVaoOut, pVars);
        setVertexBufferUAV(mMeshBufferLocations.bitangentOut, VERTEX_BITANGENT_LOC, pVaoOut, pVars);

        if (hasNormal) mSkinningPass.pProgram->addDefine("HAS_NORMAL");
        else mSkinningPass.pProgram->removeDefine("HAS_NORMAL");

        if (hasBitangent) mSkinningPass.pProgram->addDefine("HAS_BITANGENT");
        else mSkinningPass.pProgram->removeDefine("HAS_BITANGENT");

        if (!it->second.valid) mSkinningPass.pProgram->addDefine("FIRST_FRAME");
        else mSkinningPass.pProgram->removeDefine("FIRST_FRAME");

        it->second.valid = true;
    }
}