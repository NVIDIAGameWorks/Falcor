/***************************************************************************
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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
#include "SingleShaderProgram.h"
#include "HitProgram.h"

namespace Falcor
{
    class RtProgram : public std::enable_shared_from_this<RtProgram>
    {
    public:
        using SharedPtr = std::shared_ptr<RtProgram>;
        using SharedConstPtr = std::shared_ptr<const RtProgram>;
        using MissProgramList = std::vector<MissProgram::SharedPtr>;
        using HitProgramList = std::vector<HitProgram::SharedPtr>;

        static RtProgram::SharedPtr create(RayGenProgram::SharedPtr pRayGenProgram, const MissProgramList& missPrograms, const HitProgramList& hitPrograms);

        // Ray-gen
        RayGenProgram::SharedPtr getRayGenProgram() const { return mpRayGenProgram; }

        // Hit
        uint32_t getHitProgramCount() const { return (uint32_t)mHitProgs.size(); }
        HitProgram::SharedPtr getHitProgram(uint32_t rayIndex) const { return mHitProgs[rayIndex]; }

        // Miss
        uint32_t getMissProgramCount() const { return (uint32_t)mMissProgs.size(); }
        MissProgram::SharedPtr getMissProgram(uint32_t rayIndex) const { return mMissProgs[rayIndex]; }

        void addDefine(const std::string& name, const std::string& value = "");
        void removeDefine(const std::string& name);

    private:
        RtProgram(RayGenProgram::SharedPtr pRayGenProgram, const MissProgramList& missPrograms, const HitProgramList& hitPrograms) : mHitProgs(hitPrograms), mMissProgs(missPrograms), mpRayGenProgram(pRayGenProgram) {}
        HitProgramList mHitProgs;
        MissProgramList mMissProgs;
        RayGenProgram::SharedPtr mpRayGenProgram;
    };
}