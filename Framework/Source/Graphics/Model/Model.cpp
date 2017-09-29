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
#include "Framework.h"
#include "Model.h"
#include "Loaders/AssimpModelImporter.h"
#include "Loaders/BinaryModelImporter.h"
#include "Loaders/BinaryModelExporter.h"
#include "Utils/OS.h"
#include "mesh.h"
#include "glm/geometric.hpp"
#include "AnimationController.h"
#include "Animation.h"
#include "API/Buffer.h"
#include "API/Texture.h"
#include "Graphics/TextureHelper.h"
#include "Utils/StringUtils.h"
#include "Graphics/Camera/Camera.h"
#include "API/VAO.h"
#include <set>

namespace Falcor
{

    uint32_t Model::sModelCounter = 0;
    const char* Model::kSupportedFileFormatsStr = "Supported Formats\0*.obj;*.bin;*.dae;*.x;*.md5mesh;*.ply;*.fbx;*.3ds;*.blend;*.ase;*.ifc;*.xgl;*.zgl;*.dxf;*.lwo;*.lws;*.lxo;*.stl;*.x;*.ac;*.ms3d;*.cob;*.scn;*.3d;*.mdl;*.mdl2;*.pk3;*.smd;*.vta;*.raw;*.ter\0\0";

    // Method to sort meshes
    bool compareMeshes(const Mesh::SharedPtr& p1, const Mesh::SharedPtr& p2)
    {
        // This relies on the fact that Model keeps only unique copies of materials, so same material == same address.
        // See GetOrAddMaterial() for more info
        return p1->getMaterial() < p2->getMaterial();
    }

    Model::Model() : mId(sModelCounter++)
    {

    }

    Model::Model(const Model& other) : mId(sModelCounter++)
    {
        mBoundingBox = other.mBoundingBox;
        mRadius = other.mRadius;
        mVertexCount = other.mVertexCount;
        mIndexCount = other.mIndexCount;
        mPrimitiveCount = other.mPrimitiveCount;
        mMeshInstanceCount = other.mMeshInstanceCount;
        mBufferCount = other.mBufferCount;
        mMaterialCount = other.mMaterialCount;
        mTextureCount = other.mTextureCount;

        mMeshes = other.mMeshes;
        if(other.mpAnimationController)
        {
            mpAnimationController = AnimationController::create(*other.mpAnimationController);
        }

        mName = other.mName + "_copy";
        mFilename = other.mFilename;
    }

    Model::~Model() = default;

	void Model::exportAOMeshes(std::string& filename)
	{
		FILE* file = NULL;
		fopen_s(&file, filename.c_str(), "wb");
		if (file)
		{
			uint64_t numVertices = this->getVertexCount();
			uint64_t numIndices = this->getIndexCount();

			//write number of meshes
			uint32_t numMeshes = this->getMeshCount();
			fwrite(&numMeshes, sizeof(numMeshes), 1, file);

			//for each mesh
			for (uint32_t meshID = 0; meshID < this->getMeshCount(); meshID++)
			{
				Mesh::SharedPtr pMesh = mMeshes[meshID][0]->getObject();
				Vao::SharedPtr vAO = pMesh->getVao();
				VertexLayout::SharedConstPtr vLayout= vAO->getVertexLayout();

				//only support triangle list
				uint32 vertex_count = 0;
				uint32 index_count = 0;
				uint32 mesh_identifier = meshID;
				fwrite(&mesh_identifier, sizeof(mesh_identifier), 1, file);
				if (vAO->getPrimitiveTopology() != Vao::Topology::TriangleList)
				{
					fwrite(&vertex_count, sizeof(vertex_count), 1, file);
					fwrite(&index_count, sizeof(index_count), 1, file);
				}
				else
				{
					vertex_count = pMesh->getVertexCount();
					index_count = pMesh->getIndexCount();
					fwrite(&vertex_count, sizeof(vertex_count), 1, file);
					fwrite(&index_count, sizeof(index_count), 1, file);

					bool hasPosition = false;
					bool hasNormal = false;
					for (int bi = 0; bi < vLayout->getBufferCount(); bi++)
					{
						VertexBufferLayout::SharedConstPtr bLayout = vLayout->getBufferLayout(bi);

						if (bLayout->getElementName(0) == VERTEX_POSITION_NAME)
						{
							Buffer::SharedPtr vb = vAO->getVertexBuffer(bi);
							float3 *positions = (float3 *)vb->map(Falcor::Buffer::MapType::Read);
							fwrite(positions, sizeof(float3), vertex_count, file);
							hasPosition = true;
						}
						else if(bLayout->getElementName(0) == VERTEX_NORMAL_NAME)
						{
							Buffer::SharedPtr vb = vAO->getVertexBuffer(bi);
							float3 *normals = (float3 *)vb->map(Falcor::Buffer::MapType::Read);
							fwrite(normals, sizeof(float3), pMesh->getVertexCount(), file);
							hasNormal = true;
						}
					}

					assert(hasPosition && hasNormal);

					//write indices
					Buffer::SharedPtr ib = vAO->getIndexBuffer();
					uint32 *indices = (uint32 *)ib->map(Falcor::Buffer::MapType::Read);
					fwrite(indices, sizeof(uint32), pMesh->getIndexCount(), file);

				}
			}

			fclose(file);
		}
	}

	void Model::importAO(std::string& filename)
	{
		FILE* file = NULL;
		fopen_s(&file, filename.c_str(), "rb");
		if (file)
		{
			uint64_t numInstances = 0;
			uint64_t numVertices = 0;

			// read header
			fread(&numInstances, sizeof(numInstances), 1, file);
			fread(&numVertices, sizeof(numVertices), 1, file);

			mAO.mInstances.resize(numInstances);

			// read instances
			for (size_t i = 0; i < numInstances; i++)
			{
				uint64_t identifier = 0;
				uint64_t vertexOffset = 0;
				uint64_t numVerticesPerInstance = 0;

				fread(&identifier, sizeof(identifier), 1, file);
				fread(&vertexOffset, sizeof(vertexOffset), 1, file);
				fread(&numVerticesPerInstance, sizeof(numVerticesPerInstance), 1, file);
				mAO.mInstances[i].mIdentifier = identifier;
				mAO.mInstances[i].mVertexOffset = vertexOffset;
				mAO.mInstances[i].mNumVertices = numVerticesPerInstance;
			}

			//read vertices
			float *AOValues = new float[numVertices];
			if (numVertices > 0)
			{
				fread(AOValues, sizeof(float)*numVertices, 1, file);
			}

			for (uint32_t meshID = 0; meshID < this->getMeshCount(); meshID++)
			{
				Mesh::SharedPtr pMesh = mMeshes[meshID][0]->getObject();
				Falcor::Vao::SharedPtr vAO = pMesh->getVao();
				VertexLayout::SharedPtr pLayout = vAO->getVertexLayoutForModify();

				for (uint32_t instance_id = 0; instance_id < numInstances; instance_id++)
				{
					if (mAO.mInstances[instance_id].mIdentifier == meshID)
					{
						uint32_t numVerticesMesh = pMesh->getVertexCount();
						assert(numVerticesMesh == mAO.mInstances[instance_id].mNumVertices);
						float *aoValues = &AOValues[mAO.mInstances[instance_id].mVertexOffset];

						bool has_color = false;
						for (int bi = 0; bi < pLayout->getBufferCount(); bi++)
						{

							if (pLayout->getBufferLayout(bi)->getElementName(0) == VERTEX_DIFFUSE_COLOR_NAME)
							{
								Buffer::SharedPtr aoBuffer = vAO->getVertexBuffer(bi);
								float4 *aoValuesmesh = new float4[numVerticesMesh];
								float4 *originalColors = (float4 *)aoBuffer->map(Buffer::MapType::Read);
								for (unsigned int i = 0; i < numVerticesMesh; i++)
								{
									aoValuesmesh[i][0] = originalColors[i][0];
									aoValuesmesh[i][1] = originalColors[i][1];
									aoValuesmesh[i][2] = originalColors[i][2];
									aoValuesmesh[i][3] = aoValues[i];
								}
								aoBuffer->updateData(aoValuesmesh, 0, numVerticesMesh * sizeof(float4));
								has_color = true;
								break;
							}
						}

						if (!has_color)
						{
							VertexBufferLayout::SharedPtr pBufferLayout = VertexBufferLayout::create();
							pLayout->addBufferLayout((uint32_t)(pLayout->getBufferCount()), pBufferLayout);

							const std::string falcorName = VERTEX_DIFFUSE_COLOR_NAME;
							ResourceFormat falcorFormat = ResourceFormat::RGBA32Float;
							uint32_t shaderLocation = VERTEX_DIFFUSE_COLOR_LOC;
							pBufferLayout->addElement(falcorName, 0, falcorFormat, 1, shaderLocation);

							float4 *aoValuesmesh = new float4[numVerticesMesh];
							for (unsigned int i = 0; i < numVerticesMesh; i++)
							{
								aoValuesmesh[i][0] = aoValuesmesh[i][1] = aoValuesmesh[i][2] = aoValuesmesh[i][3] = max(aoValues[i], 0.0f);
							}

							Buffer::SharedPtr aoBuffer = Buffer::create(sizeof(float4)*numVerticesMesh, Buffer::BindFlags::Vertex, Buffer::CpuAccess::None, aoValuesmesh);
							vAO->addVertexBuffer(aoBuffer);
							delete[] aoValuesmesh;
							break;
						}

					}
				}
			}

			delete[] AOValues;

			fclose(file);

		}
	}

    Model::SharedPtr Model::createFromFile(const char* filename, LoadFlags flags)
    {
        SharedPtr pModel = SharedPtr(new Model());
        bool res;

        if(hasSuffix(filename, ".bin", false))
        {
            res = BinaryModelImporter::import(*pModel, filename, flags);
        }
        else
        {
            res = AssimpModelImporter::import(*pModel, filename, flags);
        }

        if(res)
        {
            pModel->calculateModelProperties();
            pModel->setFilename(filename);

            std::string name = getFilenameFromPath(filename);
            size_t extPos = name.find_last_of('.');
            name = (extPos == std::string::npos) ? name : name.substr(0, extPos);
            pModel->setName(name);
        }
        else
        {
            pModel = nullptr;
        }

        return pModel;
    }

    Model::SharedPtr Model::create()
    {
        return SharedPtr(new Model());
    }

    void Model::exportToBinaryFile(const std::string& filename)
    {
        if(hasSuffix(filename, ".bin", false) == false)
        {
            logWarning("Exporting model to binary file, but extension is not '.bin'. This will cause error when loading the file");
        }

        BinaryModelExporter::exportToFile(filename, this);
    }

    void Model::calculateModelProperties()
    {
        mVertexCount = 0;
        mIndexCount = 0;
        mPrimitiveCount = 0;
        mMeshInstanceCount = 0;
        mBufferCount = 0;
        mMaterialCount = 0;
        mTextureCount = 0;

        std::set<const Material*> uniqueMaterials;
        std::set<const Texture*> uniqueTextures;
        std::set<const Buffer*> uniqueBuffers;

        // Sort the meshes
        sortMeshes();

        vec3 modelMin = vec3(1e25f), modelMax = vec3(-1e25f);

        for(const auto& meshInstances : mMeshes)
        {
            // If we have a vector for a mesh, there should be at least one instance of it
            assert(meshInstances.size() > 0);

            const auto& pMesh = meshInstances[0]->getObject();
            const uint32_t instanceCount = (uint32_t)meshInstances.size();
            
            mVertexCount += pMesh->getVertexCount() * instanceCount;
            mIndexCount += pMesh->getIndexCount() * instanceCount;
            mPrimitiveCount += pMesh->getPrimitiveCount() * instanceCount;
            mMeshInstanceCount += instanceCount;

            const Material* pMaterial = pMesh->getMaterial().get();

            // Track material
            uniqueMaterials.insert(pMaterial);

            // Track all of the material's textures
            for (uint32_t i = 0; i < pMaterial->getNumLayers(); i++)
            {
                uniqueTextures.insert(pMaterial->getLayer(i).pTexture.get());
            }

            uniqueTextures.insert(pMaterial->getNormalMap().get());
            uniqueTextures.insert(pMaterial->getAlphaMap().get());
            uniqueTextures.insert(pMaterial->getAmbientOcclusionMap().get());
            uniqueTextures.insert(pMaterial->getHeightMap().get());

            // Track the material's buffers
            const auto& pVao = pMesh->getVao();
            for (uint32_t i = 0; i < (uint32_t)pVao->getVertexBuffersCount(); i++)
            {
                if (pVao->getVertexBuffer(i) != nullptr)
                {
                    uniqueBuffers.insert(pVao->getVertexBuffer(i).get());
                }
            }

            if (pVao->getIndexBuffer() != nullptr)
            {
                uniqueBuffers.insert(pVao->getIndexBuffer().get());
            }

            // Expand bounding box
            for(uint32_t i = 0 ; i < instanceCount; i++)
            {
                const BoundingBox& meshBox = meshInstances[i]->getBoundingBox();

                vec3 meshMin = meshBox.center - meshBox.extent;
                vec3 meshMax = meshBox.center + meshBox.extent;

                modelMin = min(modelMin, meshMin);
                modelMax = max(modelMax, meshMax);
            }
        }

        // Don't count nullptrs
        uniqueTextures.erase(nullptr);
        uniqueMaterials.erase(nullptr);
        uniqueBuffers.erase(nullptr);

        mTextureCount = (uint32_t)uniqueTextures.size();
        mMaterialCount = (uint32_t)uniqueMaterials.size();
        mBufferCount = (uint32_t)uniqueBuffers.size();

        mBoundingBox = BoundingBox::fromMinMax(modelMin, modelMax);
        mRadius = glm::length(modelMin - modelMax) * 0.5f;
    }

    void Model::animate(double currentTime)
    {
        if(mpAnimationController)
        {
            mpAnimationController->animate(currentTime);
        }
    }

    bool Model::hasAnimations() const
    {
        return (getAnimationsCount() != 0);
    }

    uint32_t Model::getAnimationsCount() const
    {
        return mpAnimationController ? (mpAnimationController->getAnimationCount()) : 0;
    }

    uint32_t Model::getActiveAnimation() const
    {
        return mpAnimationController->getActiveAnimation();
    }

    const std::string& Model::getAnimationName(uint32_t animationID) const
    {
        assert(mpAnimationController);
        assert(animationID < getAnimationsCount());
        return mpAnimationController->getAnimationName(animationID);
    }

    void Model::setBindPose()
    {
        if(mpAnimationController)
        {
            mpAnimationController->setActiveAnimation(AnimationController::kBindPoseAnimationId);
        }
    }

    void Model::setActiveAnimation(uint32_t animationID)
    {
        assert(animationID < getAnimationsCount() || animationID == AnimationController::kBindPoseAnimationId);
        mpAnimationController->setActiveAnimation(animationID);
    }

    bool Model::hasBones() const
    {
        return (getBonesCount() != 0);
    }

    uint32_t Model::getBonesCount() const
    {
        return mpAnimationController ? mpAnimationController->getBoneCount() : 0;
    }

    const glm::mat4* Model::getBonesMatrices() const
    {
        assert(mpAnimationController);
        return mpAnimationController->getBoneMatrices();
    }

    void Model::bindSamplerToMaterials(const Sampler::SharedPtr& pSampler)
    {
        // Go over materials for all meshes and bind the sampler
        for(auto& meshInstances : mMeshes)
        {
            meshInstances[0]->getObject()->getMaterial()->setSampler(pSampler);
        }
    }

    void Model::setAnimationController(AnimationController::UniquePtr pAnimController)
    {
        mpAnimationController = std::move(pAnimController);
    }

    void Model::addMeshInstance(const Mesh::SharedPtr& pMesh, const glm::mat4& baseTransform)
    {
        int32_t meshID = -1;

        // Linear search from the end. Instances are usually added in order by mesh
        for (int32_t i = (int32_t)mMeshes.size() - 1; i >= 0; i--)
        {
            if (mMeshes[i][0]->getObject() == pMesh)
            {
                meshID = i;
                break;
            }
        }

        // If mesh not found, new mesh, add new instance vector
        if (meshID == -1)
        {
            mMeshes.push_back(MeshInstanceList());
            meshID = (int32_t)mMeshes.size() - 1;
        }

        mMeshes[meshID].push_back(MeshInstance::create(pMesh, baseTransform));
    }

    void Model::sortMeshes()
    {
        // Sort meshes by material ptr
        auto matSortPred = [](MeshInstanceList& lhs, MeshInstanceList& rhs) 
        {
            return lhs[0]->getObject()->getMaterial()->getId() < rhs[0]->getObject()->getMaterial()->getId();
        };
        
        std::sort(mMeshes.begin(), mMeshes.end(), matSortPred);
    }

    template<typename T>
    void removeNullElements(std::vector<T>& Vec)
    {
        auto Pred = [](T& t) {return t == nullptr; };
        auto& NewEnd = std::remove_if(Vec.begin(), Vec.end(), Pred);
        Vec.erase(NewEnd, Vec.end());
    }

    void Model::deleteCulledMeshInstances(MeshInstanceList& meshInstances, const Camera *pCamera)
    {
        for (auto& instance : meshInstances)
        {
            if (pCamera->isObjectCulled(instance->getBoundingBox()))
            {
                // Remove mesh ptr reference
                instance->mpObject = nullptr;
            }
        }

        // Remove culled instances
        auto instPred = [](MeshInstance::SharedPtr& instance) { return instance->getObject() == nullptr; };
        auto& instEnd = std::remove_if(meshInstances.begin(), meshInstances.end(), instPred);
        meshInstances.erase(instEnd, meshInstances.end());
    }

    void Model::deleteCulledMeshes(const Camera* pCamera)
    {
        std::map<const Material*, bool> usedMaterials;
        std::map<const Buffer*, bool> usedBuffers;

        // Loop over all the meshes and remove its instances
        for(auto& meshInstances : mMeshes)
        {
            deleteCulledMeshInstances(meshInstances, pCamera);

            if(meshInstances.size() > 0)
            {
                // Mark the mesh's objects as used
                usedMaterials[meshInstances[0]->getObject()->getMaterial().get()] = true;
                const auto pVao = meshInstances[0]->getObject()->getVao();
                usedBuffers[pVao->getIndexBuffer().get()] = true;

                for(uint32_t i = 0 ; i < pVao->getVertexBuffersCount() ; i++)
                {
                    usedBuffers[pVao->getVertexBuffer(i).get()] = true;
                }
            }
        }

        // Remove unused meshes from the vector
        auto pred = [](MeshInstanceList& meshInstances) { return meshInstances.size() == 0; };
        auto& meshesEnd = std::remove_if(mMeshes.begin(), mMeshes.end(), pred);
        mMeshes.erase(meshesEnd, mMeshes.end());

        calculateModelProperties();
    }

    void Model::resetGlobalIdCounter()
    {
        sModelCounter = 0;
        Mesh::resetGlobalIdCounter();
    }
}
