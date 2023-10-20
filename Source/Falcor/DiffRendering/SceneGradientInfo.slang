/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
import Scene.SceneTypes;

enum GradientMode
{
    None,         // No gradients are propagated (in either direction) for this pass.
    Scene,        // Pass gradients to scene parameters normally.
    ForwardDebug, // Use forward-mode gradients to debug.
};

// Create a differentiable flag for higher-order differentiation.
// It is mainly used for:
// 1. Set dFlag = { GradientMode.None } to zero out gradients from/to scene parameters for the Jacobian computation.
// 2. Set dFlag = { GradientMode.ForwardDebug } to pass in forward-mode gradients for debugging.
struct SceneGradientFlag : IDifferentiable
{
    GradientMode gradMode;

    typealias Differential = SceneGradientFlag;

    [ForwardDerivative(__fwd_make)]
    [BackwardDerivative(__bwd_make)]
    static SceneGradientFlag make(GradientMode mode)
    {
        SceneGradientFlag info = { mode };
        return info;
    }

    static DifferentialPair<SceneGradientFlag> __fwd_make(GradientMode mode)
    {
        SceneGradientFlag flag = { mode };
        return diffPair(flag, flag);
    }

    static void __bwd_make(GradientMode mode, SceneGradientFlag.Differential dFlag)
    {
        // Do nothing.
    }

    [Differentiable]
    static SceneGradientFlag dadd(SceneGradientFlag a, SceneGradientFlag b)
    {
        // Default to all gradients.
        return SceneGradientFlag.make(GradientMode.Scene);
    }

    [Differentiable]
    static SceneGradientFlag dmul(SceneGradientFlag a, SceneGradientFlag b)
    {
        // Default to all gradients.
        return SceneGradientFlag.make(GradientMode.Scene);
    }

    [Differentiable]
    static SceneGradientFlag dzero()
    {
        // Default to all gradients.
        return SceneGradientFlag.make(GradientMode.Scene);
    }
};

struct SceneGradientInfo : IDifferentiable
{
    // Differentiable flag for higher-order differentiation.
    SceneGradientFlag flag;

    // Extra data.
    uint2 pixel;

    [Differentiable]
    __init(SceneGradientFlag _flag, uint2 _pixel)
    {
        flag = _flag;
        pixel = _pixel;
    }
};
