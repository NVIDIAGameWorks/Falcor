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
#include "Experiment.h"
#include <fstream>
#include <ctime>

namespace Falcor
{
    namespace Psychophysics
    {
        void Experiment::describeExperiment(ExperimentDescription newExpDesc)
        {
            mExpDesc = newExpDesc;
        }

        void Experiment::addCondition(ConditionParameter newConditionParam, ExperimentalDesignParameter newExpParam)
        {
            SingleThresholdMeasurement m;
            m.initMeasurement(newConditionParam, newExpParam);
            mMeasurements.push_back(m);
            // initialize mRecordFieldNames if it was not defined already.
            if ((int32_t)mRecordFieldNames.size() == 0)
            {
                mRecordFieldNames.push_back("staircaseID");
                for (auto keyval : newConditionParam.mParamList)
                {
                    mRecordFieldNames.push_back(keyval.first);
                }
                mRecordFieldNames.push_back("stimLevel");
                mRecordFieldNames.push_back("response");
            }
        }

        void Experiment::chooseNextCondition()
        {
            // Choose any staircase whose progress ratio is minimum
            float minimumProgressRatio = 1;
            std::vector<int32_t> validIndex;
            // Find minimum progress ratio
            for (int32_t i = 0; i < (int32_t)mMeasurements.size(); i++)
            {
                if (mMeasurements[i].getProgressRatio() < minimumProgressRatio)
                {
                    minimumProgressRatio = mMeasurements[i].getProgressRatio();
                }
            }
            // Make a vector with all the measurement cells with minimum progress ratio
            for (int32_t i = 0; i < (int32_t)mMeasurements.size(); i++)
            {
                if (mMeasurements[i].getProgressRatio() == minimumProgressRatio)
                {
                    validIndex.push_back(i);
                }
            }
            // Now choose any one from validIndex
            mCurrentConditionIndex = validIndex[rand() % (int32_t)validIndex.size()];
            std::cout << "Next chosen staircase is: " << mCurrentConditionIndex << '\n';
        }

        ConditionParameter Experiment::getConditionParamForCurrentCondition()
        {
            return mMeasurements[mCurrentConditionIndex].getConditionParam();
        }

        float Experiment::getLevelForCurrentTrial()
        {
            return mMeasurements[mCurrentConditionIndex].getCurrentLevel();
        }

        void Experiment::processResponse(int32_t response)
        {
            // First record the condition and response and then process the response.
            // Recording...
            std::vector<float> newRecord;
            newRecord.push_back((float)mCurrentConditionIndex);
            for (auto keyval : getConditionParamForCurrentCondition().mParamList)
            {
                newRecord.push_back(keyval.second);
            }
            newRecord.push_back(getLevelForCurrentTrial());
            newRecord.push_back((float)response);
            mRecordFieldValues.push_back(newRecord);
            // now process the response
            mMeasurements[mCurrentConditionIndex].processResponse(response);
            mTrialCount++;
        }

        bool Experiment::isComplete() // did the experiment end?
        {
            bool allMeasurementComplete = true;
            for (int32_t i = 0; i < (int32_t)mMeasurements.size(); i++)
            {
                if (!mMeasurements[i].isComplete()) // if any one staircase is incomplete, set allSCCompleted to false and break.
                {
                    allMeasurementComplete = false;
                    break;
                }
            }
            return allMeasurementComplete;
        }

        std::string Experiment::generateResultFilename(std::string prefix)
        {
            // subject name.. later this should be received as an input
            // make a unique file name
            time_t t = std::time(nullptr);
            const std::tm* tmbuf = std::localtime(&t);
            char tmCharArray[17];
            std::strftime(tmCharArray, sizeof(tmCharArray), "%Y%m%d_%H%M%S", tmbuf);
            std::string timeStr(tmCharArray);
            std::string fileNameStr = "UserStudy_" + prefix + "_" + timeStr + ".csv";

            return fileNameStr;
        }

        void Experiment::printResult(std::string prefix) // write the result into a result file with a unique name
        {

            std::ofstream ResultFile(generateResultFilename(prefix));
            // // On the first row goes the description.
            // ResultFile << mExpDesc.mName.c_str() << std::endl;

            // // On the second row goes all the constant parameters
            //for (auto keyval : mExpDesc.mParamList)
            //{
            //    ResultFile << keyval.first.c_str() << ":" << keyval.second << ";";
            //}
            //for (auto keyval : mExpDesc.mDescList)
            //{
            //    ResultFile << keyval.first.c_str() << ":" << keyval.second.c_str() << ";";
            //}
            //ResultFile << std::endl;

            // Write field names in the third row.
            for (int32_t i = 0; i < (int32_t)mRecordFieldNames.size(); i++)
            {
                ResultFile << mRecordFieldNames[i].c_str();
                if (i < (int32_t)mRecordFieldNames.size() - 1)
                {
                    ResultFile << ',';
                }
                else
                {
                    ResultFile << std::endl;
                }
            }
            // Write down all the field values
            for (int32_t i = 0; i < (int32_t)mRecordFieldValues.size(); i++)
            {
                for (int32_t j = 0; j < (int32_t)mRecordFieldValues[i].size(); j++)
                {
                    ResultFile << mRecordFieldValues[i][j];
                    if (j < (int32_t)mRecordFieldValues[i].size() - 1)
                    {
                        ResultFile << ',';
                    }
                    else
                    {
                        ResultFile << std::endl;
                    }
                }
            }
            ResultFile.close();
        }

        void Experiment::clear()
        {
            mConditionParamNames.clear();
            mConditionParamValues.clear();
            mMeasurements.clear();
            mRecordFieldNames.clear();
            mRecordFieldValues.clear();
            mCurrentConditionIndex = 0;
            mTrialCount = 0;

            mExpDesc = ExperimentDescription();
        }

    }
}