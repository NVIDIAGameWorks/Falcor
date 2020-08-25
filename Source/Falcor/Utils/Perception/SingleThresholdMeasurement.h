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
#include <map>

namespace Falcor
{
    namespace Perception
    {
        /** Psychophysics method types
        */
        enum class Method
        {
            DiscreteStaircase,
            BucketStaircase,
            MethodOfConstantStimuli
        };

        /** Struct for experimental design parameters: Contains all parameters for both 
            Staircase and Method of Constant Stimuli.
        */
        struct ExperimentalDesignParameter
        {
            Method mMeasuringMethod; // 0 = General staircase, 1 = Staircase with pre-determined stimLevels, 2 = Method of Constant Stimuli
            bool mIsDefault;
            float mInitLevel, mInitLevelRandomRange, mMinLevel, mMaxLevel, mInitLevelStepSize, mMinLevelStepSize; // for SC
            int32_t mNumUp, mNumDown, mMaxReversals, mMaxTotalTrialCount, mMaxLimitHitCount; // for SC
            int32_t mInitIndex, mInitIndexRandomRange, mInitIndexStepSize; // SC with pre-determined stimLevels. Some values are self-obvious (minIndexLevel = 0, maxIndexLevel = stimLevels.size(), minIndexStepSize = 1)
            std::vector<float> mStimLevels; // for SC with pre-determined stimLevels or Method of Constant Stimuli
            std::vector<int32_t> mMaxTrialCounts; // for Method of Constant Stimuli
        };

        /** Struct representing condition parameter
        */
        struct ConditionParameter
        {
            std::map<std::string, float>mParamList;
        };

        /** Response struct
        */
        struct Response
        {
            float mStimLevel;
            int32_t mResponse; // 1 = correct, 0 = wrong
            int32_t mReversalCount; // for debugging
        };

        /** Class to abstract single threshold measurement
        */
        class dlldecl SingleThresholdMeasurement
        {
        public:

            /** Initialize measurement
            */
            void initMeasurement(ConditionParameter initConditionParam, ExperimentalDesignParameter initExpParam); // return true if successfully initialized, false if not

            /** Get current level
            */
            float getCurrentLevel();

            /** Get current condition parameter
            */
            ConditionParameter getConditionParam();

            /** Process response
                \param[in] response response to process
            */
            void processResponse(int32_t response);

            /** Get progress ratio
            */
            float getProgressRatio();

            /** Check whether measurement is complete
            */
            bool isComplete();

            ExperimentalDesignParameter mExpParam;
            std::vector<Response> mResponses;
            float mCurrentLevel; // universal for both general SC and MCS
            float mLevelStepSize; // for general SC only
            int32_t mCurrentIndex, mIndexStepSize; // for SC with pre-determined stimulus levels
            int32_t mUpCount, mDownCount, mCurrentDirection, mReversalCount, mLimitHitCount; // for SC only
            std::vector<int32_t> mTrialCounts; // for MCS only
                                               // description of condition for the current measurement
            ConditionParameter mConditionParam;

            bool mIsInitialized = false;
        };


    }
}
