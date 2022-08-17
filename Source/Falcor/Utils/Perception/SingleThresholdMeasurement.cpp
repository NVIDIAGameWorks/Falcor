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
#include "SingleThresholdMeasurement.h"
#include <iostream>

namespace Falcor
{
    namespace Perception
    {
        void SingleThresholdMeasurement::initMeasurement(ConditionParameter initConditionParam, ExperimentalDesignParameter initExpParam) // return true if successfully initialized, false if not
        {
            if (mIsInitialized)
            {
                // already initialized: print out an error message
            }
            else
            {
                mExpParam = initExpParam;
                mConditionParam = initConditionParam;
                if (mExpParam.mMeasuringMethod == Method::DiscreteStaircase)
                {
                    if (mExpParam.mIsDefault) // only mMinLevel, mMaxLevel, mMinLevelStepSize were defined
                    {
                        mExpParam.mInitLevelRandomRange = 0; // no random change
                        mExpParam.mInitLevel = mExpParam.mMaxLevel; // start with the easiest
                        mExpParam.mInitLevelStepSize = 4 * mExpParam.mMinLevelStepSize;
                        mExpParam.mNumUp = 1; // assuming 2AFC design, 1 up 2 down is standard
                        mExpParam.mNumDown = 2;
                        mExpParam.mMaxReversals = 50;
                        mExpParam.mMaxTotalTrialCount = 150;
                        mExpParam.mMaxLimitHitCount = 2;
                    }
                    // First, set the stimulus level perturbed within the initial random range
                    float perturbation;
                    if (mExpParam.mMinLevelStepSize == 0)
                    {
                        perturbation = 0;
                    }
                    else
                    {
                        int32_t numMinSteps = (int32_t)(mExpParam.mInitLevelRandomRange / mExpParam.mMinLevelStepSize);
                        int32_t randomSign = 2 * (rand() % 2) - 1;
                        int32_t stepsForPerturbation = randomSign * (rand() % numMinSteps);
                        perturbation = stepsForPerturbation * mExpParam.mMinLevelStepSize;
                    }
                    // set initial stim level
                    mCurrentLevel = mExpParam.mInitLevel + perturbation;
                    if (mCurrentLevel < mExpParam.mMinLevel)
                    {
                        mCurrentLevel = mExpParam.mMinLevel;
                    }
                    else if (mCurrentLevel > mExpParam.mMaxLevel)
                    {
                        mCurrentLevel = mExpParam.mMaxLevel;
                    }
                    // Initialize all other necessary values
                    mLevelStepSize = mExpParam.mInitLevelStepSize;
                    mUpCount = 0;
                    mDownCount = 0;
                    mCurrentDirection = 0;
                    mReversalCount = 0;
                    mLimitHitCount = 0;
                }
                else if (mExpParam.mMeasuringMethod == Method::BucketStaircase) // SC with pre-determined stimLevels
                {
                    if (mExpParam.mIsDefault) // only stimLevels were defined
                    {
                        mExpParam.mInitIndexRandomRange = 0; // no random change
                        mExpParam.mInitIndex = (int32_t)mExpParam.mStimLevels.size() - 1; // start with the easiest
                        mExpParam.mInitIndexStepSize = 4;
                        mExpParam.mNumUp = 1;
                        mExpParam.mNumDown = 2;
                        mExpParam.mMaxReversals = 15;
                        mExpParam.mMaxTotalTrialCount = 50;
                        mExpParam.mMaxLimitHitCount = 2;
                    }
                    // First, set the stimulus level perturbed within the initial random range
                    int32_t randomSign = 2 * (rand() % 2) - 1;
                    int32_t perturbation;
                    if (mExpParam.mInitIndexRandomRange == 0)
                    {
                        perturbation = 0;
                    }
                    else
                    {
                        perturbation = randomSign * (rand() % mExpParam.mInitIndexRandomRange);
                    }
                    // set initial stim level
                    mCurrentIndex = mExpParam.mInitIndex + perturbation;
                    if (mCurrentIndex < 0)
                    {
                        mCurrentIndex = 0;
                    }
                    else if (mCurrentIndex >= (int32_t)mExpParam.mStimLevels.size())
                    {
                        mCurrentIndex = (int32_t)mExpParam.mStimLevels.size() - 1;
                    }
                    mCurrentLevel = mExpParam.mStimLevels[mCurrentIndex];
                    // Initialize all other necessary values
                    mIndexStepSize = mExpParam.mInitIndexStepSize;
                    mUpCount = 0;
                    mDownCount = 0;
                    mCurrentDirection = 0;
                    mReversalCount = 0;
                    mLimitHitCount = 0;
                }
                else if (mExpParam.mMeasuringMethod == Method::MethodOfConstantStimuli) // MCS
                {
                    if (mExpParam.mIsDefault) // only stimLevels were defined
                    {
                        int32_t trialCount = (int32_t)(200/(int32_t)mExpParam.mStimLevels.size()); // let's do ~200 trials per each condition
                        for (int32_t i = 0; i < (int32_t)mExpParam.mStimLevels.size(); i++)
                        {
                            mExpParam.mMaxTrialCounts.push_back(trialCount);
                        }
                    }
                    // Set initial stimulus level
                    mCurrentLevel = mExpParam.mStimLevels[rand() % (int32_t)mExpParam.mStimLevels.size()];
                    // Initialize all other necessary values
                    for (int32_t i = 0; i < (int32_t)mExpParam.mStimLevels.size(); i++)
                    {
                        mTrialCounts.push_back(0);
                    }
                }
                mIsInitialized = true;
            }
        }

        float SingleThresholdMeasurement::getCurrentLevel()
        {
            return mCurrentLevel;
        }

        ConditionParameter SingleThresholdMeasurement::getConditionParam()
        {
            return mConditionParam;
        }

        void SingleThresholdMeasurement::processResponse(int32_t response)
        {
            // record current response
            Response res;
            res.mStimLevel = mCurrentLevel;
            res.mResponse = response;
            if ((mExpParam.mMeasuringMethod == Method::DiscreteStaircase) || (mExpParam.mMeasuringMethod == Method::BucketStaircase)) // SC. This is for debugging.
            {
                res.mReversalCount = mReversalCount;
            }
            mResponses.push_back(res);

            // select next stim level based on measuring strategy (SC or MCS)
            if (mExpParam.mMeasuringMethod == Method::DiscreteStaircase) // SC. Count reversals and select next stim level.
            {
                if (mResponses.back().mResponse == 0) // incorrect response
                {
                    mUpCount++; // increment up count by one
                    mDownCount = 0; // reset down count
                    if (mUpCount == mExpParam.mNumUp) // time to move up.
                    {
                        if (mCurrentDirection == -1) // Direction reversal. Increment reversal count. halve the step size.
                        {
                            mReversalCount++;
                            mLevelStepSize = mLevelStepSize / 2;
                            if (mLevelStepSize < mExpParam.mMinLevelStepSize) // step size too small
                            {
                                mLevelStepSize = mExpParam.mMinLevelStepSize;
                            }
                        }
                        mCurrentDirection = 1;
                        mCurrentLevel = mCurrentLevel + mCurrentDirection * mLevelStepSize; // move one step up.
                        if (mCurrentLevel > mExpParam.mMaxLevel)
                        {
                            mCurrentLevel = mExpParam.mMaxLevel;
                            mLimitHitCount++;
                            if (mLimitHitCount >= mExpParam.mMaxLimitHitCount)
                            {
                                mReversalCount++;
                                mLimitHitCount = 0;
                            }
                        }
                        else
                        {
                            mLimitHitCount = 0;
                        }
                        mUpCount = 0; // reset up count
                    }
                    std::cout << "Processed a response that was incorrect. Reversal count is: " << mReversalCount << "\n";
                }
                else // correct response
                {
                    mDownCount++; // increment down count by one
                    mUpCount = 0; // reset up count
                    if (mDownCount == mExpParam.mNumDown) // time to move down.
                    {
                        if (mCurrentDirection == 1) // Direction reversal. Increment reversal count. halve the step size.
                        {
                            mReversalCount++;
                            mLevelStepSize = mLevelStepSize / 2;
                            if (mLevelStepSize < mExpParam.mMinLevelStepSize) // step size too small
                            {
                                mLevelStepSize = mExpParam.mMinLevelStepSize;
                            }
                        }
                        mCurrentDirection = -1;
                        mCurrentLevel = mCurrentLevel + mCurrentDirection * mLevelStepSize; // move one step down.
                        if (mCurrentLevel < mExpParam.mMinLevel)
                        {
                            mCurrentLevel = mExpParam.mMinLevel;
                            mLimitHitCount++;
                            if (mLimitHitCount >= mExpParam.mMaxLimitHitCount)
                            {
                                mReversalCount++;
                                mLimitHitCount = 0;
                            }
                        }
                        else
                        {
                            mLimitHitCount = 0;
                        }
                        mDownCount = 0; // reset down count
                    }
                    std::cout << "Processed a response that was correct. Reversal count is: " << mReversalCount << "\n";

                }
            }
            else if (mExpParam.mMeasuringMethod == Method::BucketStaircase) // SC with pre-determined stimLevels
            {
                if (mResponses.back().mResponse == 0) // incorrect response
                {
                    mUpCount++; // increment up count by one
                    mDownCount = 0; // reset down count
                    if (mUpCount == mExpParam.mNumUp) // time to move up.
                    {
                        if (mCurrentDirection == -1) // Direction reversal. Increment reversal count. halve the step size.
                        {
                            mReversalCount++;
                            mIndexStepSize = mIndexStepSize / 2;
                            if (mIndexStepSize < 1) // step size too small
                            {
                                mIndexStepSize = 1;
                            }
                        }
                        mCurrentDirection = 1;
                        mCurrentIndex = mCurrentIndex + mCurrentDirection * mIndexStepSize; // move one step up.
                        if (mCurrentIndex >= (int32_t)mExpParam.mStimLevels.size())
                        {
                            mCurrentIndex = (int32_t)mExpParam.mStimLevels.size() - 1;
                            mLimitHitCount++;
                            if (mLimitHitCount >= mExpParam.mMaxLimitHitCount)
                            {
                                mReversalCount++;
                                mLimitHitCount = 0;
                            }
                        }
                        else
                        {
                            mLimitHitCount = 0;
                        }
                        mUpCount = 0; // reset up count
                    }
                    mCurrentLevel = mExpParam.mStimLevels[mCurrentIndex];
                    std::cout << "Processed a response that was incorrect. Reversal count is: " << mReversalCount << "\n";
                }
                else // correct response
                {
                    mDownCount++; // increment down count by one
                    mUpCount = 0; // reset up count
                    if (mDownCount == mExpParam.mNumDown) // time to move down.
                    {
                        if (mCurrentDirection == 1) // Direction reversal. Increment reversal count. halve the step size.
                        {
                            mReversalCount++;
                            mIndexStepSize = mIndexStepSize / 2;
                            if (mIndexStepSize < 1) // step size too small
                            {
                                mIndexStepSize = 1;
                            }
                        }
                        mCurrentDirection = -1;
                        mCurrentIndex = mCurrentIndex + mCurrentDirection * mIndexStepSize; // move one step down.
                        if (mCurrentIndex < 0)
                        {
                            mCurrentIndex = 0;
                            mLimitHitCount++;
                            if (mLimitHitCount >= mExpParam.mMaxLimitHitCount)
                            {
                                mReversalCount++;
                                mLimitHitCount = 0;
                            }
                        }
                        else
                        {
                            mLimitHitCount = 0;
                        }
                        mDownCount = 0; // reset down count
                    }
                    mCurrentLevel = mExpParam.mStimLevels[mCurrentIndex];
                    std::cout << "Processed a response that was correct. Reversal count is: " << mReversalCount << "\n";
                }
            }
            else if (mExpParam.mMeasuringMethod == Method::MethodOfConstantStimuli) // MCS. Count numTrials and select next stim level.
            {
                // count number of trials & calculate progress ratio per each stimuli level
                std::vector<float> progressRatio;
                float minimumProgressRatio = 1;
                for (int32_t i = 0; i < (int32_t)mTrialCounts.size(); i++)
                {
                    if (mExpParam.mStimLevels[i] == mCurrentLevel)
                    {
                        mTrialCounts[i]++;
                        break;
                    }
                    float pr = (float)mTrialCounts[i] / (float)mExpParam.mMaxTrialCounts[i];
                    if (pr < minimumProgressRatio)
                    {
                        minimumProgressRatio = pr;
                    }
                    progressRatio.push_back(pr);
                }
                // select one stimuli level from those with minimum progress ratio
                std::vector<int32_t> validIndex;
                for (int32_t i = 0; i<(int32_t)progressRatio.size(); i++)
                {
                    if (minimumProgressRatio == progressRatio[i])
                    {
                        validIndex.push_back(i);
                    }
                }
                // Now choose any one from validIndex
                int32_t chosenIndex = validIndex[rand() % (int32_t)validIndex.size()];
                mCurrentLevel = mExpParam.mStimLevels[chosenIndex];
                std::cout << "Next chosen MCS stim level is: " << mCurrentLevel << '\n';
            }
        }

        float SingleThresholdMeasurement::getProgressRatio()
        {
            if ((mExpParam.mMeasuringMethod == Method::DiscreteStaircase) || (mExpParam.mMeasuringMethod == Method::BucketStaircase)) // SC
            {
                // if maximum trial count reached, report 1
                if ((int)mResponses.size() >= mExpParam.mMaxTotalTrialCount)
                {
                    return 1.0f;
                }
                else
                {
                    return (float)mReversalCount / (float)mExpParam.mMaxReversals;
                }
            }
            else if (mExpParam.mMeasuringMethod == Method::MethodOfConstantStimuli) // MCS
            {
                int32_t totalCount = 0;
                int32_t totalMax = 0;
                for (int32_t i = 0; i<(int32_t)mExpParam.mStimLevels.size(); i++)
                {
                    totalCount = totalCount + mTrialCounts[i];
                    totalMax = totalMax + mExpParam.mMaxTrialCounts[i];
                }
                return (float)totalCount / (float)totalMax;
            }
            return 0; // shouldn't reach here...
        }

        bool SingleThresholdMeasurement::isComplete()
        {
            if ((mExpParam.mMeasuringMethod == Method::DiscreteStaircase) || (mExpParam.mMeasuringMethod == Method::BucketStaircase)) // SC
            {
                if ((mReversalCount >= mExpParam.mMaxReversals) || ((int32_t)mResponses.size() >= mExpParam.mMaxTotalTrialCount))
                {
                    return true;
                }
                else
                {
                    return false;
                }
            }
            else if (mExpParam.mMeasuringMethod == Method::MethodOfConstantStimuli) // MCS
            {
                int32_t totalCount = 0;
                int32_t totalMax = 0;
                for (int32_t i = 0; i<(int32_t)mExpParam.mStimLevels.size(); i++)
                {
                    totalCount = totalCount + mTrialCounts[i];
                    totalMax = totalMax + mExpParam.mMaxTrialCounts[i];
                }
                if (((float)totalCount / (float)totalMax) == 1)
                {
                    return true;
                }
                else
                {
                    return false;
                }
            }
            return false; // shouldn't reach here
        }

    }
}
