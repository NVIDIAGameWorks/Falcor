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
#pragma once
#include <vector>

namespace Falcor
{
    class AnimationController;

    class dlldecl Animation
    {
    public:
        using SharedPtr = std::shared_ptr<Animation>;

        static const uint32_t kInvalidChannel = -1;

        enum class InterpolationMode
        {
            Linear,
            Hermite,
        };

        struct Keyframe
        {
            double time = 0;
            float3 translation = float3(0, 0, 0);
            float3 scaling = float3(1, 1, 1);
            glm::quat rotation = glm::quat(1, 0, 0, 0);
        };

        /** Create a new object
        */
        static SharedPtr create(const std::string& name, double durationInSeconds);

        /** Get the animation's name
        */
        const std::string& getName() const { return mName; }

        /** Add a new channel
        */
        uint32_t addChannel(uint32_t matrixID);

        /** Get the channel for a given matrix ID or kInvalidChannel if not available
        */
        uint32_t getChannel(uint32_t matrixID) const;

        /** Get the channel count
        */
        size_t getChannelCount() const { return mChannels.size(); }

        /** Add a keyframe.
            If there's already a keyframe at the requested time, this call will override the existing frame
        */
        void addKeyframe(uint32_t channelID, const Keyframe& keyframe);

        /** Get the keyframe from a specific time.
            If the keyframe doesn't exists, the function will throw an exception. If you don't want to handle exceptions, call doesKeyframeExist() first
        */
        const Keyframe& getKeyframe(uint32_t channelID, double time) const;

        /** Check if a keyframe exists in a specific time
        */
        bool doesKeyframeExists(uint32_t channelID, double time) const;

        /** Set the interpolation mode and enable/disable warping for a given channel.
        */
        void setInterpolationMode(uint32_t channelID, InterpolationMode mode, bool enableWarping);

        /** Run the animation
            \param currentTime The current time in seconds. This can be larger then the animation time, in which case the animation will loop
            \param matrices The array of global matrices to update
        */
        void animate(double currentTime, std::vector<glm::mat4>& matrices);

        /** Get the matrixID affected by a channel
        */
        uint32_t getChannelMatrixID(uint32_t channel) const { return mChannels[channel].matrixID; }

    private:
        Animation(const std::string& name, double durationInSeconds);

        struct Channel
        {
            Channel(uint32_t matrixID, InterpolationMode interpolationMode = InterpolationMode::Linear, bool enableWarping = true)
                : matrixID(matrixID)
                , interpolationMode(interpolationMode)
                , enableWarping(enableWarping)
            {};

            uint32_t matrixID;
            InterpolationMode interpolationMode;
            bool enableWarping;
            std::vector<Keyframe> keyframes;
            mutable size_t lastKeyframeUsed = 0;
            mutable double lastUpdateTime = 0;
        };

        std::vector<Channel> mChannels;
        const std::string mName;
        double mDurationInSeconds = 0;

        glm::mat4 animateChannel(const Channel& c, double time) const;
        size_t findChannelFrame(const Channel& c, double time) const;
        glm::mat4 interpolate(const Keyframe& start, const Keyframe& end, double curTime) const;
    };
}
