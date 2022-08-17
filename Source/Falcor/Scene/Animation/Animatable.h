/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
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
#pragma once
#include "Core/Macros.h"
#include "Scene/SceneIDs.h"
#include "Utils/Math/Matrix.h"
#include <memory>

namespace Falcor
{
    /** Represents an object that has a transform which can be animated using a scene graph node.
    */
    class FALCOR_API Animatable
    {
    public:
        // While this is an abstract base class, we still need a holder type (shared_ptr)
        // for pybind11 bindings to work on inherited types.
        using SharedPtr = std::shared_ptr<Animatable>;

        virtual ~Animatable() {}

        /** Set if object has animation data.
        */
        void setHasAnimation(bool hasAnimation) { mHasAnimation = hasAnimation; }

        /** Returns true if object has animation data.
        */
        bool hasAnimation() const { return mHasAnimation; }

        /** Enable/disable object animation.
        */
        void setIsAnimated(bool isAnimated) { mIsAnimated = isAnimated; }

        /** Returns true if object animation is enabled.
        */
        bool isAnimated() const { return mIsAnimated; }

        /** Sets the node ID of the animated scene graph node.
        */
        void setNodeID(NodeID nodeID) { mNodeID = nodeID; }

        /** Gets the node ID of the animated scene graph node.
        */
        NodeID getNodeID() const { return mNodeID; }

        /** Update the transform of the animatable object.
        */
        virtual void updateFromAnimation(const rmcv::mat4& transform) = 0;

    protected:
        bool mHasAnimation = false;
        bool mIsAnimated = true;
        NodeID mNodeID{ NodeID::Invalid() };
    };
}
