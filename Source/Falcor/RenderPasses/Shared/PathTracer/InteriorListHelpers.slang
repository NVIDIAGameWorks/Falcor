/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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
 **************************************************************************/
import Scene.Scene;
__exported import RenderPasses.Shared.PathTracer.InteriorList;

/** Extra ray payload and helper functions to track nested dielectrics in raytracing kernels.
    Note: This is currently limited to handling a maximum of 1 false intersection.
*/
struct NestedRayData
{
    uint topNestedPriority;     ///< Current top nested priority on interior list.
    float falseT;               ///< Ray distance of false intersection.
    uint falseMaterialID;       ///< MaterialID of false intersection (MSB stores "entering" flag).

    /** Initialize the payload data.
        \param[in] interiorList Interior list.
        \param[in] maxT Max ray distance.
    */
    [mutating]
    void init(InteriorList interiorList, float maxT)
    {
        topNestedPriority = interiorList.getTopNestedPriority();
        falseT = maxT;
        falseMaterialID = kNoMaterial;
    }

    /** Function to be called from the any hit shader. This keeps track of the closest false intersections.
        If this function returns true, the current intersection is a false intersection and the caller
        should call IgnoreHit() to ignore it.
        \param[in] materialID Material ID at the current intersection.
        \param[in] faceNormalW Face normal at the current intersection.
        \return Returns true if this is a false intersection.
    */
    [mutating]
    bool handleAnyHit(uint materialID, float3 faceNormalW)
    {
        const uint nestedPriority = gScene.materials[materialID].getNestedPriority();
        const float currentT = RayTCurrent();

        // Keep track of closest false intersection.
        if (nestedPriority < topNestedPriority && currentT <= falseT)
        {
            falseT = currentT;
            uint entering = dot(WorldRayDirection(), faceNormalW) < 0.f ? 1 : 0;
            falseMaterialID = materialID | (entering << 31);
            return true;
        }

        return false;
    }

    /** Function to be called from the closest hit shader. This function invalidates any recorded
        false intersection that are further away than the closest hit.
    */
    [mutating]
    void handleClosestHit()
    {
        const float currentT = RayTCurrent();

        if (falseT >= currentT)
        {
            falseMaterialID = kNoMaterial;
        }
    }

    /** Commits the false intersections, if any, to the interior list.
        \paramp[in,out] interiorList Interior list.
    */
    void commit(inout InteriorList interiorList)
    {
        if (falseMaterialID != kNoMaterial)
        {
            const uint materialID = falseMaterialID & 0x7fffffff;
            bool entering = falseMaterialID >> 31;
            interiorList.handleIntersection(materialID, gScene.materials[materialID].getNestedPriority(), entering);
        }
    }
};


/** Compute relative index of refraction at intersection using interior list.
    \param[in] interiorList Interior list.
    \param[in] materialID Material ID of intersected material.
    \param[in] entering True if material is entered, false if material is left.
    \return Returns the relative index of refraction.
*/
float computeRelativeIoR(InteriorList interiorList, uint materialID, bool entering)
{
    if (entering)
    {
        float toIoR = gScene.materials[materialID].IoR;
        float fromIoR = 1.f;
        uint topMaterialID = interiorList.getTopMaterialID();
        if (topMaterialID != kNoMaterial) fromIoR = gScene.materials[topMaterialID].IoR;
        return fromIoR / toIoR;
    }
    else
    {
        float fromIoR = gScene.materials[materialID].IoR;
        float toIoR = 1.f;
        uint topMaterialID = interiorList.getTopMaterialID();
        if (topMaterialID == materialID) topMaterialID = interiorList.getNextMaterialID();
        if (topMaterialID != kNoMaterial) toIoR = gScene.materials[topMaterialID].IoR;
        return fromIoR / toIoR;
    }
}
