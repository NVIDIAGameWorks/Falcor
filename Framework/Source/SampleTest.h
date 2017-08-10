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
        /** Checks whether testing is enabled, returns true if either Test Task vector isn't empty
        */
        bool hasTests() const;

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
            Frame,  ///< Triggered by frame count
            Time,   ///< Triggered by time since application launch
            None,   ///< No tests will be run
        };
        TriggerType mCurrentTrigger = TriggerType::None;

        /** Types of operations used in testing
        */
        enum class TaskType
        {
            MemoryCheck,    ///< Records the application's current memory usage
            LoadTime,       ///< 
            MeasureFps,     ///< Records the current FPS
            ScreenCapture,  ///< Captures a screenshot
            Shutdown,       ///< Close the application
            Uninitialized   ///< Default value. All tasks must be initialized on startup
        };

        struct Task
        {
            Task() : mStartFrame(0u), mEndFrame(0u), mTask(TaskType::Uninitialized), mResult(0.f) {}
            Task(uint32_t startFrame, uint32_t endFrame, TaskType t) :
                mStartFrame(startFrame), mEndFrame(endFrame), mTask(t), mResult(0.f) {}
            bool operator<(const Task& rhs) { return mStartFrame < rhs.mStartFrame; }

            uint32_t mStartFrame;
            uint32_t mEndFrame;
            float mResult = 0;
            TaskType mTask;
        };

        std::vector<Task> mTestTasks;
        std::vector<Task>::iterator mCurrentFrameTest;

        struct TimedTask
        {
            TimedTask() : mStartTime(0.f), mTask(TaskType::Uninitialized), mStartFrame(0) {};
            TimedTask(float startTime, float endTime, TaskType t) : mStartTime(startTime), mEndTime(endTime), mTask(t), mStartFrame(0) {};
            bool operator<(const TimedTask& rhs) { return mStartTime < rhs.mStartTime; }

            float mStartTime;
            float mEndTime;
            float mResult = 0;
            //used to calc avg fps in a perf range
            uint mStartFrame = 0;
            TaskType mTask;
        };

        std::vector<TimedTask> mTimedTestTasks;
        std::vector<TimedTask>::iterator mCurrentTimeTest;


        //  A TestTask
        struct TestTask 
        {
            //  Construct a new Task, with the appropriate Task Type and Trigger Type.
            TestTask(TaskType newTaskType) : mTaskType(newTaskType) {};

            //  Execute the Task.
            virtual void onFrameBegin(SampleTest * currentSampleTest) = 0;
            virtual void onFrameEnd(SampleTest * currentSampleTest) = 0;

            //  The Task Type.
            TaskType mTaskType;
        };

        //  Frame-Based Task.
        struct FrameTask
        {
                //  Construct the Frame Task, with the appropriate Task Type and the Frame Trigger.
            FrameTask(uint32_t onFrameTrigger, std::shared_ptr<TestTask> newTestTask) 
                : mOnFrameTrigger(onFrameTrigger), mTestTask(newTestTask) {};

            //  Execute the Task at Frame Begin.
            virtual void onFrameBegin(SampleTest * sampleTest)
            {
                //  
                if (sampleTest->getFrameID() == mOnFrameTrigger)
                {
                    if (mTestTask != nullptr)
                    {
                        mTestTask->onFrameBegin(sampleTest);
                    }
                }
            }

            //  Execute the Task at Frame End.
            virtual void onFrameEnd(SampleTest * sampleTest)
            {
                //  
                if (sampleTest->getFrameID() == mOnFrameTrigger)
                {
                    if (mTestTask != nullptr)
                    {
                        mTestTask->onFrameEnd(sampleTest);
                    }
                }
            }

            //  Default Frame Trigger Type.
            TriggerType mTriggerType = TriggerType::Frame;

            //   On Frame Trigger.
            uint32_t mOnFrameTrigger;

            //  Test Task to Execute.
            std::shared_ptr<TestTask> mTestTask = nullptr;
        };

        //  Time-Based Task.
        struct TimeTask
        {
            //  Construct the Frame Task, with the appropriate Task Type and the Time Trigger.
            TimeTask(float newTimeTrigger, std::shared_ptr<TestTask> newTestTask) 
                : mTimeTrigger(newTimeTrigger), mTestTask(newTestTask) {};

            //  Execute the Task at Frame Begin.
            virtual void onFrameBegin(SampleTest * sampleTest)
            {
                //  
                if (sampleTest->mCurrentTime >= mTimeTrigger)
                {
                    if (mTestTask != nullptr)
                    {
                        mTestTask->onFrameBegin(sampleTest);
                    }
                }
            }

            //  Execute the Task at Frame End.
            virtual void onFrameEnd(SampleTest * sampleTest)
            {
                //  
                if (sampleTest->mCurrentTime >= mTimeTrigger)
                {
                    if (mTestTask != nullptr)
                    {
                        mTestTask->onFrameEnd(sampleTest);
                    }
                }
            }

            //  Default Time Trigger Type.
            TriggerType mTriggerType = TriggerType::Time;

            //   On Time Trigger.
            float mTimeTrigger;

            //  Test Task to Execute.
            std::shared_ptr<TestTask> mTestTask = nullptr;

        };


        //
        //  Frame Range Task.
        struct FrameRangeTask
        {
            FrameRangeTask(uint32_t newFrameBeginTrigger, uint32_t newFrameEndTrigger, std::shared_ptr<TestTask> newBeginTask, std::shared_ptr<TestTask> newEndTask) 
                : mFrameBeginTrigger(newFrameBeginTrigger), mFrameEndTrigger(newFrameEndTrigger), mBeginTask(newBeginTask), mEndTask(newEndTask){};

            //  Execute the Task at Frame Begin.
            virtual void onFrameBegin(SampleTest * sampleTest)
            {
                if (!isRangeComplete && !isRangeActive && sampleTest->mCurrentTime >= mFrameBeginTrigger)
                {
                    isRangeActive = true;
                    mBeginTask->onFrameBegin(sampleTest);
                }

                if (!isRangeComplete && sampleTest->mCurrentTime <= mFrameEndTrigger)
                {
                    mEndTask->onFrameBegin(sampleTest);
                }
            }

            //  Execute the Task at Frame Begin.
            virtual void onFrameEnd(SampleTest * sampleTest)
            {
                if (!isRangeComplete && isRangeActive && sampleTest->mCurrentTime >= mFrameBeginTrigger)
                {
                    mBeginTask->onFrameEnd(sampleTest);
                }

                if (!isRangeComplete && isRangeActive && sampleTest->mCurrentTime <= mFrameEndTrigger)
                {
                    isRangeActive = false;
                    isRangeComplete = true;
                    mEndTask->onFrameEnd(sampleTest);
                }
            }


            //  Begin and End Trigger Frames.
            uint32_t mFrameBeginTrigger = 0;
            uint32_t mFrameEndTrigger = 0;

            //  Begin and End Tasks.
            std::shared_ptr<TestTask> mBeginTask;
            std::shared_ptr<TestTask> mEndTask;

            //  Range Delimiters.
            bool isRangeActive = false;
            bool isRangeComplete = false;
        };


        //
        //  Time Range Task.
        struct TimeRangeTask
        {
            TimeRangeTask(float newTimeBeginTrigger, float newTimeEndTrigger, std::shared_ptr<TestTask> newBeginTask, std::shared_ptr<TestTask> newEndTask)
                : mTimeBeginTrigger(newTimeBeginTrigger), mTimeEndTrigger(newTimeEndTrigger), mBeginTask(newBeginTask), mEndTask(newEndTask) {};


            //  Execute the Task at Frame Begin.
            virtual void onFrameBegin(SampleTest * sampleTest)
            {
                if (!isRangeComplete && !isRangeActive && sampleTest->mCurrentTime >= mTimeBeginTrigger)
                {
                    isRangeActive = true;
                    mBeginTask->onFrameBegin(sampleTest);
                }

                if (!isRangeComplete && sampleTest->mCurrentTime <= mTimeEndTrigger)
                {
                    mEndTask->onFrameBegin(sampleTest);
                }
            }

            //  Execute the Task at Frame Begin.
            virtual void onFrameEnd(SampleTest * sampleTest)
            {
                if (!isRangeComplete && isRangeActive && sampleTest->mCurrentTime >= mTimeBeginTrigger)
                {
                    mBeginTask->onFrameEnd(sampleTest);
                }

                if (!isRangeComplete && isRangeActive && sampleTest->mCurrentTime <= mTimeEndTrigger)
                {
                    isRangeActive = false;
                    isRangeComplete = true;
                    mEndTask->onFrameEnd(sampleTest);
                }
            }
            
            //  Begin and End Trigger Times.
            float mTimeBeginTrigger = 0;
            float mTimeEndTrigger = 0;

            //  Begin and End Tasks.
            std::shared_ptr<TestTask> mBeginTask;
            std::shared_ptr<TestTask> mEndTask;

            //  Range Delimiters.
            bool isRangeActive = false;
            bool isRangeComplete = false;
        };

        //  
        struct RecurrentFrameRangeTask
        {
            RecurrentFrameRangeTask(uint32_t newFrameBeginTrigger, uint32_t newFrameEndTrigger, std::shared_ptr<TestTask> newRecurrentTask) 
                : mFrameBeginTrigger(newFrameBeginTrigger), mFrameEndTrigger(newFrameEndTrigger), mRecurrentTask(newRecurrentTask) {};

            //  
            virtual void onFrameBegin(SampleTest * sampleTest)
            {
                if (sampleTest->getFrameID() >= mFrameBeginTrigger && sampleTest->getFrameID() <= mFrameEndTrigger)
                {
                    mRecurrentTask->onFrameBegin(sampleTest);
                }
            }

            //  
            virtual void onFrameEnd(SampleTest * sampleTest)
            {
                if (sampleTest->getFrameID() >= mFrameBeginTrigger && sampleTest->getFrameID() <= mFrameEndTrigger)
                {
                    mRecurrentTask->onFrameEnd(sampleTest);
                }
            }

            uint32_t mFrameBeginTrigger = 0;

            uint32_t mFrameEndTrigger = 0;

            std::shared_ptr<TestTask> mRecurrentTask;
        };

        //
        struct RecurrentTimeRangeTask
        {
            RecurrentTimeRangeTask(float newTimeBeginTrigger, float newTimeEndTrigger, std::shared_ptr<TestTask> newRecurrentTask)
                : mTimeBeginTrigger(newTimeBeginTrigger), mTimeEndTrigger(newTimeEndTrigger), mRecurrentTask(newRecurrentTask) {};

            //  
            virtual void onFrameBegin(SampleTest * sampleTest)
            {
                if (sampleTest->mCurrentTime >= mTimeBeginTrigger && sampleTest->mCurrentTime <= mTimeEndTrigger)
                {
                    mRecurrentTask->onFrameBegin(sampleTest);
                }
            }
            
            //  
            virtual void onFrameEnd(SampleTest * sampleTest)
            {
                if (sampleTest->mCurrentTime >= mTimeBeginTrigger && sampleTest->mCurrentTime <= mTimeEndTrigger)
                {
                    mRecurrentTask->onFrameEnd(sampleTest);
                }
            }
              
            float mTimeBeginTrigger = 0;

            float mTimeEndTrigger = 0;

            std::shared_ptr<TestTask> mRecurrentTask;
        };

        //  Memory Check Task.
        struct MemoryCheckTask : public TestTask
        {
            //  
            MemoryCheckTask() 
                : TestTask(TaskType::MemoryCheck) {};
            
            //  
            virtual void onFrameBegin(SampleTest * currentSampleTest)
            {

            };

            //  
            virtual void onFrameEnd(SampleTest * currentSampleTest)
            {
                //  Get the Total Virtual Memory.
                uint64_t mTotalVirtualMemory = getTotalVirtualMemory();

                //  Get the Total Used Virtual Memory.
                uint64_t mTotalUsedVirtualMemory = getUsedVirtualMemory();

                //  Get the Currently Used Virtual Memory.
                uint64_t mCurrentlyUsedVirtualMemory = getProcessUsedVirtualMemory();
            };
            
            //  Total Virtual Memory.            
            uint64_t mTotalVirtualMemory;

            //  Total Used Virtual Memory.
            uint64_t mTotalUsedVirtualMemory;
            
            //  Currently Used Virtual Memory.
            uint64_t mCurrentlyUsedVirtualMemory;
        };
        


        //  Performance Check Task.
        struct PerformanceCheckTask : public TestTask
        {
            //  
            PerformanceCheckTask() 
                : TestTask(TaskType::MeasureFps) {};

            //  
            virtual void onFrameBegin(SampleTest * currentSampleTest)
            {

            };

            //  
            virtual void onFrameEnd(SampleTest * currentSampleTest)
            {

            };
        };
            


        struct ScreenCaptureTask : public TestTask
        {
            //  
            ScreenCaptureTask(std::string newScreenCaptureFile) 
                : TestTask(TaskType::ScreenCapture), mScreenCaptureFile(newScreenCaptureFile) {};

            //  
            virtual void onFrameBegin(SampleTest * currentSampleTest)
            {
                currentSampleTest->toggleText(false);
            };

            //  
            virtual void onFrameEnd(SampleTest * currentSampleTest)
            {
                currentSampleTest->captureScreen(mScreenCaptureFile);
                currentSampleTest->toggleText(true);
            };

            std::string mScreenCaptureFile = "";
        };
    

        //  Load Time Task.
        struct LoadTimeCheckTask : public TestTask
        {
            //
            LoadTimeCheckTask() 
                : TestTask(TaskType::LoadTime) {};

            //  
            virtual void onFrameBegin(SampleTest * currentSampleTest)
            {
                mLoadTimeResult = currentSampleTest->frameRate().getLastFrameTime();
            };

            //  
            virtual void onFrameEnd(SampleTest * currentSampleTest)
            {
            };

            float mLoadTimeResult = 0.0;
        };

        //  Shutdown Task.
        struct ShutdownTask : public TestTask
        {
            //  
            ShutdownTask() 
                : TestTask(TaskType::Shutdown) {};

            //  
            virtual void onFrameBegin(SampleTest * currentSampleTest)
            {

            };

            //  
            virtual void onFrameEnd(SampleTest * currentSampleTest)
            {
                currentSampleTest->shutdownApp();
                currentSampleTest->onTestShutdown();
                currentSampleTest->writeJsonTestResults();
            };
        };



        //  The Time Tasks.
        std::vector<std::shared_ptr<TimeTask>> mTimeTasks;
        
        //  The Frame Tasks.
        std::vector<std::shared_ptr<FrameTask>> mFrameTasks;

        //  The Frame Range Tasks.
        std::vector<std::shared_ptr<FrameRangeTask>> mFrameRangeTasks;

        //  The Time Range Tasks.
        std::vector<std::shared_ptr<TimeRangeTask>> mTimeRangeTasks;

        //  The Recurrent Range Tasks.
        std::vector<std::shared_ptr<RecurrentFrameRangeTask>> mRecurrentFrameRangeTasks;

        //  The Recurrent Time Tasks.
        std::vector<std::shared_ptr<RecurrentFrameRangeTask>> mRecurrentTimeRangeTasks;


        /** Outputs xml test results file
        */
        void outputXML();

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

        void writeLoadTime(rapidjson::Value& jval, rapidjson::Document::AllocatorType& jallocator);
        void writeMemoryRangesResults(rapidjson::Value& jval, rapidjson::Document::AllocatorType& jallocator);
        void writePerformanceRangesResults(rapidjson::Value& jval, rapidjson::Document::AllocatorType& jallocator);
        void writeScreenCaptureResults(rapidjson::Value& jval, rapidjson::Document::AllocatorType& jallocator); 

        //  Initialize the Tests.
        void initializeTests();
        void initializeFrameTests();
        void initializeTimeTests();

        /** Initializes tests that start based on frame number
        */
        void initFrameTests();


        /** Initializes tests that start based on time
        */
        void initTimeTests();

        /** Run tests that start based on frame number
        */
        void runFrameTests();

        /** Run tests that start based on time
        */
        void runTimeTests();

        // The Memory Check for one point.
        struct MemoryCheck
        {
            float time;
            float effectiveTime;
            uint64_t frame;
            uint64_t totalVirtualMemory;
            uint64_t totalUsedVirtualMemory;
            uint64_t currentlyUsedVirtualMemory;
        };

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