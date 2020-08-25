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
import Scene.Scene;
__exported import RenderPasses.Shared.PathTracer.InteriorList;

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
        if (topMaterialID != InteriorList::kNoMaterial) fromIoR = gScene.materials[topMaterialID].IoR;
        return fromIoR / toIoR;
    }
    else
    {
        float fromIoR = gScene.materials[materialID].IoR;
        float toIoR = 1.f;
        uint topMaterialID = interiorList.getTopMaterialID();
        if (topMaterialID == materialID) topMaterialID = interiorList.getNextMaterialID();
        if (topMaterialID != InteriorList::kNoMaterial) toIoR = gScene.materials[topMaterialID].IoR;
        return fromIoR / toIoR;
    }
}
