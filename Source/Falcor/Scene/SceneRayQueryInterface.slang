/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
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
__exported import Scene.HitInfo;
__exported import Utils.Math.Ray;

// TODO: Use default arguments when slang supports them:
// bool traceRay(const RayDesc ray, out HitInfo hit, out float hitT, uint rayFlags = RAY_FLAG_NONE, uint instanceInclusionMask = 0xff);
// bool traceVisibilityRay(const RayDesc ray, uint rayFlags = RAY_FLAG_NONE, uint instanceInclusionMask = 0xff);

/** Interface for scene ray queries.
*/
interface ISceneRayQuery
{
    /** Trace a ray against the scene and return the closest hit point.
        \param[in] ray Ray.
        \param[out] hitT Hit distance (only valid if function returns true).
        \param[in] rayFlags Optional ray flags OR'ed with queries flags.
        \param[in] instanceInclusionMask Includes/rejects geometry based on instance mask.
        \return Returns Hit info, only valid when there was a hit.
    */
    HitInfo traceRay(const Ray ray, out float hitT, uint rayFlags = RAY_FLAG_NONE, uint instanceInclusionMask = 0xff, bool isPrevFrame = false);

    /** Trace a visibility ray against the scene.
        \param[in] ray Ray.
        \param[in] rayFlags Optional ray flags OR'ed with queries flags.
        \param[in] instanceInclusionMask Includes/rejects geometry based on instance mask.
        \return Returns true if the ray endpoints are mutually visible (i.e. the ray does NOT intersect the scene).
    */
    bool traceVisibilityRay(const Ray ray, uint rayFlags = RAY_FLAG_NONE, uint instanceInclusionMask = 0xff, bool isPrevFrame = false);
};
