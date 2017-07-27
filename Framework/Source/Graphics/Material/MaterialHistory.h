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

#pragma once

#include "Graphics/Material/Material.h"
#include <unordered_set>

namespace Falcor
{
    class Mesh;
    class Model;

    /** Helper class for the Scene and Scene Editor to track how custom material overrides are applied to meshes.
    */
    class MaterialHistory
    {
    public:

        using SharedPtr = std::shared_ptr<MaterialHistory>;
        using SharedConstPtr = std::shared_ptr<MaterialHistory>;

        /** Create an instance.
        */
        static SharedPtr create() { return SharedPtr(new MaterialHistory()); }

        /** Sets a new material on a mesh, and saves the original
            \param[in] pMesh Mesh to override material of
            \param[in] pNewMaterial Material to set on the mesh
        */
        void replace(Mesh* pMesh, const Material::SharedPtr& pNewMaterial);

        /** Restores a mesh's material to its original. Does nothing if the mesh's material is not currently overridden
            \param[in] pMesh Mesh to restore original material to
        */
        void revert(Mesh* pMesh);

        /** Check whether a mesh's material has been overridden
        */
        bool hasOverride(const Mesh* pMesh) const;

        /** Restores materials on all meshes with overridden materials. Does not affect meshes without overrides.
        */
        void onModelRemoved(const Model* pModel);

        /** Restores materials on all meshes currently using an overriding material.
            \param[in] pMaterial Overriding material to remove from meshes
        */
        void onMaterialRemoved(const Material* pMaterial);

    private:

        MaterialHistory() {};

        // Stores original materials for overridden meshes so user can revert them
        std::unordered_map<const Mesh*, Material::SharedPtr> mOriginalMaterials;

        // Map of materials from the scene and what meshes they have overridden
        std::unordered_map<const Material*, std::unordered_set<Mesh*>> mMaterialToMeshes;
    };

}