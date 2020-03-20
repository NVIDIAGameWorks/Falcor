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
#pragma once
#include <vector>

namespace Falcor
{
    class AnimationController;

    class dlldecl Animation
    {
    public:
        using SharedPtr = std::shared_ptr<Animation>;
        using ConstSharedPtrRef = const SharedPtr&;

        struct Keyframe
        {
            double time = 0;
            vec3 translation = vec3(0, 0, 0);
            vec3 scaling = vec3(1, 1, 1);
            quat rotation = quat(1, 0, 0, 0);
        };

        /** Create a new object
        */
        static SharedPtr create(const std::string& name, double durationInSeconds);

        /** Get the animation's name
        */
        const std::string& getName() const { return mName; }

        /** Add a new channel
        */
        size_t addChannel(size_t matrixID);

        /** Get the channel count
        */
        size_t getChannelCount() const { return mChannels.size(); }

        /** Add a keyframe.
            If there's already a keyframe at the requested time, this call will override the existing frame
        */
        void addKeyframe(size_t channelID, const Keyframe& keyframe);

        /** Get the keyframe from a specific time.
            If the keyframe doesn't exists, the function will throw an exception. If you don't want to handle exceptions, call doesKeyframeExist() first
        */
        const Keyframe& getKeyframe(size_t channelID, double time) const;

        /** Check if a keyframe exists in a specific time
        */
        bool doesKeyframeExists(size_t channelID, double time) const;

        /** Run the animation
            \param currentTime The current time in seconds. This can be larger then the animation time, in which case the animation will loop
            \param matrices The array of global matrices to update
        */
        void animate(double currentTime, std::vector<mat4>& matrices);

        /** Get the matrixID affected by a channel
        */
        size_t getChannelMatrixID(size_t channel) const { return mChannels[channel].matrixID; }
    private:
        Animation(const std::string& name, double durationInSeconds);

        struct Channel
        {
            Channel(size_t matID) : matrixID(matID) {};
            size_t matrixID;
            std::vector<Keyframe> keyframes;
            size_t lastKeyframeUsed = 0;
            double lastUpdateTime = 0;
        };

        std::vector<Channel> mChannels;
        const std::string mName;
        double mDurationInSeconds = 0;

        mat4 animateChannel(Channel& c, double time);
        size_t findChannelFrame(const Channel& c, double time) const;
        mat4 interpolate(const Keyframe& start, const Keyframe& end, double curTime) const;
    };
}
