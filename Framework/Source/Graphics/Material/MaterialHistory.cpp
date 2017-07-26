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
#include "Graphics/Material/MaterialHistory.h"

#include "Graphics/Model/Mesh.h"
#include "Graphics/Model/Model.h"

namespace Falcor
{
    void MaterialHistory::replace(Mesh* pMesh, const Material::SharedPtr& pNewMaterial)
    {
        if (hasOverride(pMesh))
        {
            const Material* pCurrMaterial = pMesh->getMaterial().get();

            // pMesh's material will be changed, clean up history
            auto& meshes = mMaterialToMeshes[pCurrMaterial];
            meshes.erase(pMesh);
            
            if (meshes.empty())
            {
                mMaterialToMeshes.erase(pCurrMaterial);
            }
        }
        else
        {
            // Save original material
            mOriginalMaterials[pMesh] = pMesh->getMaterial();
        }

        pMesh->setMaterial(pNewMaterial);

        if (pNewMaterial == mOriginalMaterials[pMesh])
        {
            // If replacing pMesh's material with it's original, clean up history
            mOriginalMaterials.erase(pMesh);
        }
        else
        {
            // If pNewMaterial is an override, add to tracker
            mMaterialToMeshes[pNewMaterial.get()].insert(pMesh);
        }
    }

    void MaterialHistory::revert(Mesh* pMesh)
    {
        if (hasOverride(pMesh) == false)
        {
            return;
        }

        replace(pMesh, mOriginalMaterials[pMesh]);
    }

    bool MaterialHistory::hasOverride(const Mesh* pMesh) const
    {
        return mOriginalMaterials.count(pMesh) > 0;
    }

    void MaterialHistory::onModelRemoved(const Model* pModel)
    {
        for (uint32_t i = 0; i < pModel->getMeshCount(); i++)
        {
            revert(pModel->getMesh(i).get());
        }
    }

    void MaterialHistory::onMaterialRemoved(const Material* pMaterial)
    {
        if (mMaterialToMeshes.count(pMaterial) > 0)
        {
            // Revert all meshes
            auto& meshList = mMaterialToMeshes.at(pMaterial);
            for (auto& pMesh : meshList)
            {
                bool lastMesh = meshList.size() == 1;

                revert(pMesh);

                // Manually exit because revert() will have cleaned up meshList, so the loop cannot reference it anymore
                if (lastMesh)
                {
                    break;
                }
            }
        }
    }
}
