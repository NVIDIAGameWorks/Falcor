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

#include "SingleThresholdMeasurement.h"
#include <map>

namespace Falcor
{
    namespace Psychophysics
    {
        /** Description of an experiment: Any information that could be useful 
            in future, in case an experiment grows while piloting or when 
            between-experiment analysis becomes necessary later.
        */
        struct ExperimentDescription
        {
            std::string mName;
            std::map<std::string, float> mParamList;
            std::map<std::string, std::string> mDescList;
        };

        /** A class representing a psychophysical experiment
        */
        class Experiment
        {
        public:

            /** Provide experiment description: This is not a mandatory step for initialization, 
                but is strongly recommended because it could be useful in many cases.
                \param[in] newExpDesc New experiment description
            */
            void describeExperiment(ExperimentDescription newExpDesc);

            /** Add condition
                \param[in] newConditionParam New condition
                \param[in] newExpParam New experiment design parameter
            */
            void addCondition(ConditionParameter newConditionParam, ExperimentalDesignParameter newExpParam);

            /** Pick next condition
            */
            void chooseNextCondition();

            /** Get current condtion parameter
            */
            ConditionParameter getConditionParamForCurrentCondition();

            /** Get level for current trial
            */
            float getLevelForCurrentTrial();

            /** Process user response
                \param[in] response Integer indicating user response
            */
            void processResponse(int32_t response);

            /** Check whether experiment is complete
            */
            bool isComplete();

            /** Returns name of result file used
            */
            std::string generateResultFilename(std::string prefix);

            /** Write the result into a result file with a unique name
            */
            void printResult(std::string subjectInitials);

            /** Reset the experiment state
            */
            void clear();

        private:

            std::vector<std::string> mConditionParamNames;
            std::vector<float> mConditionParamValues;
            std::vector<SingleThresholdMeasurement> mMeasurements;
            std::vector<std::string> mRecordFieldNames;
            std::vector<std::vector <float>> mRecordFieldValues;
            int32_t mCurrentConditionIndex;
            int32_t mTrialCount = 0;

            ExperimentDescription mExpDesc;
        };
    }
}