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
#include "Externals/RapidJson/include/rapidjson/document.h"
#include "Externals/RapidJson/include/rapidjson/stringbuffer.h"
#include "Externals/RapidJson/include/rapidjson/prettywriter.h"
#include "Falcor.h"

namespace Falcor
{
    /** Test framework layer for Falcor.
        Runs tests based on config files passed in by command line arguments
    */
    class SampleTest : public Sample
    {
    public:

        /** Initializes Test Task vectors based on command line arguments
        */
        void initializeTesting();

        /** Callback for anything the testing sample wants to initialize
        */
        virtual void onInitializeTesting() {};

        /** Testing actions that need to happen before the frame renders
        */
        void beginTestFrame();

        /** Callback for anything the testing sample wants to do before the frame renders
        */
        virtual void onBeginTestFrame() {};

        /** Testing actions that need to happen after the frame renders
        */
        void endTestFrame();

        /** Callback for anything the testing sample wants to do after the frame renders
        */
        virtual void onEndTestFrame() {};

        /** Callback for anything the testing sample wants to do right before shutdown
        */
        virtual void onTestShutdown() {};

    protected:

        /** Different ways test tasks can be triggered
        */
        enum class TriggerType
        {
            Frame,
            Time,
            None 
        };

        TriggerType mCurrentTriggerType = TriggerType::None;

        /** Types of operations used in testing
        */
        enum class TaskType
        {
            LoadTimeCheckTask,
            MemoryCheckTask,
            PerformanceCheckTask,
            ScreenCaptureTask,
            ShutdownTask
        };

        /** The Memory Check for one point. 
        */
        struct MemoryCheck
        {
            float time = 0;
            float effectiveTime = 0;
            uint64_t frame = 0;
            uint64_t totalVirtualMemory = 0;
            uint64_t totalUsedVirtualMemory = 0;
            uint64_t currentlyUsedVirtualMemory = 0;
        };



        //  
        struct FrameTask
        {
            //  Construct a new Task, with the appropriate Task Type and Trigger Type.
            FrameTask(TaskType newTaskType, uint32_t newStartFrame, uint32_t newEndFrame) : mTaskType(newTaskType), mStartFrame(newStartFrame), mEndFrame(mEndFrame) {};

            //  
            bool operator<(const FrameTask & rhs) const
            {
                return mStartFrame < rhs.mStartFrame;
            }

            //  
            virtual bool isActive(SampleTest * sampleTest) = 0;

            //  On Frame Begin.
            virtual void onFrameBegin(SampleTest * sampleTest) = 0;

            //  On Frame End.
            virtual void onFrameEnd(SampleTest * sampleTest)  = 0;

            //  Task Type.
            TaskType mTaskType;

            //  Start Frame.
            uint32_t mStartFrame = 0;
            
            //  End Frame.
            uint32_t mEndFrame = 0;

            //  Task Complete.
            bool mIsTaskComplete = false;
        };

        //  
        struct FrameTaskPtrCompare
        {
            bool operator()(const std::shared_ptr<FrameTask> & lhsTask, const std::shared_ptr<FrameTask> & rhsTask) 
            {
                return (*lhsTask) < (*rhsTask);
            }
        };

        //  The Frame Tasks.
        std::vector<std::shared_ptr<FrameTask>> mFrameTasks;
        
        //  The Current Frame Task Index.
        uint32_t mCurrentFrameTaskIndex = 0;



        struct LoadTimeCheckTask : public FrameTask
        {
            //  
            LoadTimeCheckTask() : FrameTask(TaskType::LoadTimeCheckTask, 2, 2) {};

            //  Basic Check.
            virtual bool isActive(SampleTest * sampleTest)
            {
                return sampleTest->getFrameID() == mStartFrame && !mIsTaskComplete;
            }

            //  On Frame Begin.
            virtual void onFrameBegin(SampleTest * sampleTest) 
            {
                mLoadTimeCheckResult = sampleTest->frameRate().getLastFrameTime();
            }

            //  On Frame End.
            virtual void onFrameEnd(SampleTest * sampleTest) 
            {
                //  Task is Complete!
                mIsTaskComplete = true;
            }

            //  
            float mLoadTimeCheckResult = 0;
        };


        struct MemoryCheckFrameTask: public FrameTask
        {
            //  
            MemoryCheckFrameTask(uint32_t newStartFrame, uint32_t newEndFrame) : FrameTask(TaskType::MemoryCheckTask, newStartFrame, newEndFrame) {};

            //  Basic Check.
            virtual bool isActive(SampleTest * sampleTest)
            {
                return sampleTest->getFrameID() == mStartFrame && !mIsTaskComplete;
            }

            //  On Frame Begin.
            virtual void onFrameBegin(SampleTest * sampleTest)
            {

            }

            //  On Frame End.
            virtual void onFrameEnd(SampleTest * sampleTest)
            {
                //  Task is Complete!
                mIsTaskComplete = true;
            }
            
        };

        struct PerformanceCheckFrameTask : public FrameTask
        {
            //  
            PerformanceCheckFrameTask(uint32_t newStartFrame, uint32_t newEndFrame) : FrameTask(TaskType::PerformanceCheckTask, newStartFrame, newEndFrame) {};


            //  On Frame Begin.
            virtual void onFrameBegin(SampleTest * sampleTest)
            {

            }

            //  On Frame End.
            virtual void onFrameEnd(SampleTest * sampleTest)
            {
                //  Task is Complete!
                mIsTaskComplete = true;
            }

            float mPerformanceCheckResults = 0;
        };


        struct ScreenCaptureFrameTask : public FrameTask
        {
            //  
            ScreenCaptureFrameTask(uint32_t captureFrame) : FrameTask(TaskType::ScreenCaptureTask, captureFrame, captureFrame), mCaptureFrame(captureFrame) {};

            //  Basic Check.
            virtual bool isActive(SampleTest * sampleTest)
            {
                return sampleTest->getFrameID() == mCaptureFrame && !mIsTaskComplete;
            }

            //  On Frame Begin.
            virtual void onFrameBegin(SampleTest * sampleTest)
            {
                sampleTest->toggleText(false);
            }

            //  On Frame End.
            virtual void onFrameEnd(SampleTest * sampleTest)
            {

                if (sampleTest->mHasSetDirectory)
                {
                    //  Capture the Screen.
                    sampleTest->captureScreen(sampleTest->mTestOutputDirectory, sampleTest->mTestOutputFilePrefix);
                }
                else
                {
                    //  Capture the Screen.
                    sampleTest->captureScreen(sampleTest->mTestOutputFilePrefix);
                }
                
                //  Toggle the Text Back.
                sampleTest->toggleText(true);

                //  Task is Complete!
                mIsTaskComplete = true;
            }

            //
            uint32_t mCaptureFrame = 0;
        };


        struct ShutdownFrameTask : public FrameTask
        {
            ShutdownFrameTask(uint32_t shutdownFrame) : FrameTask(TaskType::ShutdownTask, shutdownFrame, shutdownFrame), mShutdownFrame(shutdownFrame) {};

            //  Basic Check.
            virtual bool isActive(SampleTest * sampleTest)
            {
                return sampleTest->getFrameID() == mStartFrame && !mIsTaskComplete;
            }

            //  On Frame Begin.
            virtual void onFrameBegin(SampleTest * sampleTest){};
            
            //  On Frame End.
            virtual void onFrameEnd(SampleTest * sampleTest)
            {
                //  Write the json Test Results.
                sampleTest->writeJsonTestResults();
                
                //  Shutdown the App.
                sampleTest->shutdownApp();
                
                //  On Test Shutdown.
                sampleTest->onTestShutdown();
 
                //  Task is Complete!
                mIsTaskComplete = true;
            }

            //
            uint32_t mShutdownFrame = 0;

        };



        //
        struct TimeTask
        {
            //  Construct a new Task, with the appropriate Task Type and Trigger Type.
            TimeTask(TaskType newTaskType, float newStartTime, float newEndTime) : mTaskType(newTaskType), mStartTime(newStartTime), mEndTime(newEndTime) {};

            //  
            bool operator<(const TimeTask & rhs) const
            {
                return mStartTime < rhs.mStartTime;
            }

            //  
            virtual bool isActive(SampleTest * sampleTest) = 0;

            //  On Frame Begin.
            virtual void onFrameBegin(SampleTest * sampleTest) = 0;

            //  On Frame End.
            virtual void onFrameEnd(SampleTest * sampleTest) = 0;

            //  Task Type.
            TaskType mTaskType;

            //  Start Time.
            float mStartTime = 0;

            //  End Time.
            float  mEndTime = 0;

            //  Task Complete.
            bool mIsTaskComplete = false;
        };

        //  
        struct TimeTaskPtrCompare
        {
            bool operator()(const std::shared_ptr<TimeTask> & lhsTask, const std::shared_ptr<TimeTask> & rhsTask)
            {
                return (*lhsTask) < (*rhsTask);
            }
        };

        //  The Time Tasks.
        std::vector<std::shared_ptr<TimeTask>> mTimeTasks;

        //  The Current Time Task Index,
        uint32_t mCurrentTimeTaskIndex = 0;



        //  
        struct MemoryCheckTimeTask : public TimeTask
        {
            //  
            MemoryCheckTimeTask(float memoryCheckRangeBeginTime, float memoryCheckRangeBeginEnd) : TimeTask(TaskType::MemoryCheckTask, memoryCheckRangeBeginTime, memoryCheckRangeBeginEnd) {};

            //  On Frame Begin.
            virtual void onFrameBegin(SampleTest * sampleTest)
            {


            }

            //  On Frame End.
            virtual void onFrameEnd(SampleTest * sampleTest)
            {
                //  Task is Complete!
                mIsTaskComplete = true;
            }

        };

        //
        struct PerformanceCheckTimeTask : public TimeTask
        {
            //  
            PerformanceCheckTimeTask(float perfomanceCheckRangeBeginTime, float perfomanceCheckRangeBeginEnd) : TimeTask(TaskType::PerformanceCheckTask, perfomanceCheckRangeBeginTime, perfomanceCheckRangeBeginEnd) {};

            //  On Frame Begin.
            virtual void onFrameBegin(SampleTest * sampleTest)
            {


            }

            //  On Frame End.
            virtual void onFrameEnd(SampleTest * sampleTest)
            {
                //  Task is Complete!
                mIsTaskComplete = true;
            }

            float mPerformanceCheckResults = 0;
        };


        struct ScreenCaptureTimeTask : public TimeTask
        {
            //  
            ScreenCaptureTimeTask(float captureTime) : TimeTask(TaskType::PerformanceCheckTask, captureTime, captureTime), mCaptureTime(captureTime) {};

            //  Basic Check.
            virtual bool isActive(SampleTest * sampleTest)
            {
                return mCaptureTime <= sampleTest->mCurrentTime && !mIsTaskComplete;
            }

            //  On Frame Begin.
            virtual void onFrameBegin(SampleTest * sampleTest)
            {
                if (mCaptureTime <= sampleTest->mCurrentTime && !mIsTaskComplete)
                {
                    //  Sneakily set the time of the program! For perfect pictures.
                    sampleTest->mCurrentTime = mCaptureTime;

                    sampleTest->toggleText(false);
                }
            }

            //  On Frame End.
            virtual void onFrameEnd(SampleTest * sampleTest)
            {
                if (mCaptureTime <= sampleTest->mCurrentTime && !mIsTaskComplete)
                {
                    
                    if (sampleTest->mHasSetDirectory)
                    {
                        //  Capture the Screen.
                        sampleTest->captureScreen(sampleTest->mTestOutputDirectory, sampleTest->mTestOutputFilePrefix);
                    }
                    else
                    {
                        //  Capture the Screen.
                        sampleTest->captureScreen(sampleTest->mTestOutputFilePrefix);
                    }


                    //  Toggle the Text Back.
                    sampleTest->toggleText(true);

                    //  Task is Complete!
                    mIsTaskComplete = true;
                }
            }

            //  Capture Time.
            float mCaptureTime = 0;

        };


        //  
        struct ShutdownTimeTask : public TimeTask
        {
            ShutdownTimeTask(float shutdownTime) : TimeTask(TaskType::ShutdownTask, shutdownTime, shutdownTime), mShutdownTime(shutdownTime) {};

            //  Basic Check.
            virtual bool isActive(SampleTest * sampleTest)
            {
                return mShutdownTime <= sampleTest->mCurrentTime && !mIsTaskComplete;
            }

            //  On Frame Begin.
            virtual void onFrameBegin(SampleTest * sampleTest) {};

            //  On Frame End.
            virtual void onFrameEnd(SampleTest * sampleTest)
            {
                if (mShutdownTime <= sampleTest->mCurrentTime && !mIsTaskComplete)
                {
                    //  Write the json Test Results.
                    sampleTest->writeJsonTestResults();

                    //  Shutdown the App.
                    sampleTest->shutdownApp();

                    //  On Test Shutdown.
                    sampleTest->onTestShutdown();

                    //  Task is Complete!
                    mIsTaskComplete = true;
                }
            }

            //  Shutdown Time.
            float mShutdownTime = 0;
        };

        std::shared_ptr<LoadTimeCheckTask> mLoadTimeCheckTask;


        /*  Write JSON Literal.
        */
        template<typename T>
        void writeJsonLiteral(rapidjson::Value& jval, rapidjson::Document::AllocatorType& jallocator, const std::string& key, const T& value);

        /* Write JSON Array.
        */
        template<typename T>
        void writeJsonArray(rapidjson::Value& jval, rapidjson::Document::AllocatorType& jallocator, const std::string& key, const T& value);

        /* Write JSON Value.
        */
        void writeJsonValue(rapidjson::Value& jval, rapidjson::Document::AllocatorType& jallocator, const std::string& key, rapidjson::Value& value);

        /* Write JSON String.
        */
        void writeJsonString(rapidjson::Value& jval, rapidjson::Document::AllocatorType& jallocator, const std::string& key, const std::string& value);

        /* Write JSON Bool.
        */
        void writeJsonBool(rapidjson::Value& jval, rapidjson::Document::AllocatorType& jallocator, const std::string& key, bool isValue);

        /** Output Test Results.
        */
        void writeJsonTestResults();
        void writeJsonTestResults(rapidjson::Value& jval, rapidjson::Document::AllocatorType& jallocator);

        /** Write the Load Time Check Results.
        */
        void writeLoadTimeCheckResults(rapidjson::Value& jval, rapidjson::Document::AllocatorType& jallocator);

        /** Write the Memory Ranges Results.
        */
        void writeMemoryRangesResults(rapidjson::Value& jval, rapidjson::Document::AllocatorType& jallocator);

        /** Write the Performance Ranges Results.
        */
        void writePerformanceRangesResults(rapidjson::Value& jval, rapidjson::Document::AllocatorType& jallocator);

        /** Write the Screen Capture.
        */
        void writeScreenCaptureResults(rapidjson::Value& jval, rapidjson::Document::AllocatorType& jallocator);

        /** Initialize the Tests.
        */
        void initializeTests();
        
        /** Initialize the Frame Tests.
        */
        void initializeFrameTests();

        /** Initialize the Tests.
        */
        void initializeTimeTests();

        bool mHasSetDirectory = false;
        std::string mTestOutputDirectory = "";

        bool mHasSetFilename = false;
        std::string mTestOutputFilePrefix = "";



        // The Memory Check Between Frames.
        struct MemoryCheckRange
        {
            bool active = false;
            MemoryCheck startCheck;
            MemoryCheck endCheck;
        };

        // The List of Memory Check Ranges.
        MemoryCheckRange mMemoryFrameCheckRange;
        MemoryCheckRange mMemoryTimeCheckRange;

        /** Capture the Current Memory and write it to the provided memory check.
        */
        void getMemoryStatistics(MemoryCheck& memoryCheck);

        /** Write the Memory Check Range in terms of Time to a file. Outputs Difference, Start and End Frames.
        */
        void writeMemoryRange(const MemoryCheckRange& memoryCheckRange, bool frameTest = true);

        /** Capture the Memory Snapshot.
        */
        void captureMemory(uint64_t frameCount, float currentTime, bool frameTest = true, bool endRange = false);


    };


}