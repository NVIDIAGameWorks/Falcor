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
#include "Falcor.h"
#include "ShaderStage.h"
#include <string>
#include <map>
#include <vector>
#include <set>
#include <random>
#include <algorithm>    // std::random_shuffle
#include <ctime>        // std::time
#include <cstdlib>      // std::rand, std::srand

using namespace Falcor;

class ShaderProgramMaker
{
public:

    //  Program Desc.
    struct ProgramDesc
    {
        //  Shader Stages.
        bool hasVSStage = false;
        bool hasHSStage = false;
        bool hasDSStage = false;
        bool hasGSStage = false;
        bool hasPSStage = false;
        bool hasCSStage = false;

    };


    //  Allocation Range Struct.
    struct AllocationRange
    {
        bool isArray = false;
        std::vector<uint32_t> dimensions;
        bool isAttachmentPointExplicit;
        bool isAttachmentSubpointExplicit;
        uint32_t attachmentPoint;
        uint32_t attachmentSubpoint;
    };

    ShaderProgramMaker(const ProgramDesc & programDesc);

    //  Return the Vertex Shader Stage Maker
    virtual std::shared_ptr<VertexShaderStage> getVSStage();

    //  Return the Hull Shader Stage Maker.
    virtual std::shared_ptr<HullShaderStage> getHSStage();

    //  Return Domain Shader Stage Maker.
    virtual std::shared_ptr<DomainShaderStage> getDSStage();

    //  Return the Pixel Shader Stage Maker.
    virtual std::shared_ptr<PixelShaderStage> getPSStage();

    //  Return the Compute Shader Stage Maker.
    virtual std::shared_ptr<ComputeShaderStage> getCSStage();

protected:

    //  Vertex Shader Stage.
    std::shared_ptr<VertexShaderStage> mVSStageMaker;

    //  Hull Shader Stage.
    std::shared_ptr<HullShaderStage> mHSStageMaker;

    //  Domain Shader Stage.
    std::shared_ptr<DomainShaderStage> mDSStageMaker;

    //  Geometry Shader Stage.
    std::shared_ptr<GeometryShaderStage> mGSStageMaker;

    //  Pixel Shader Stage.
    std::shared_ptr<PixelShaderStage> mPSStageMaker;

    //  Compute Shader Stage.
    std::shared_ptr<ComputeShaderStage> mCSStageMaker;

};