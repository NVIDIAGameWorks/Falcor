TESTING FRAMEWORK
---
C++ / Falcor Side
    SampleTest
        SampleTest is the base class for all system tests
        
        Functions
            InitializeTesting()
                Reads the command line and fills vectors with TestTasks 
                (based on frame) and TimedTestTasks (based on global time)
            onInitializeTesting() 
                Does nothing on its own, is a callback you can override if your
                specific sample needs to handle additional command line 
                arguments or do anything else specific at init time 
            beginTestFrame()
                Checks if its time / the frame to perform a test and sets 
                mCurrentTest in the SampleTest class. Also does any pre-render
                test stuff required, like overriding the current global time to
                get deterministic results for screen captures on a path 
            onBeginTestFrame()
                Does nothing on its own, is a callback you can override if your
                specific sample needs to do any pre-render testing stuff 
            endTestFrame() 
                Based on the mCurrentTest var set by beginTestFrame, runs frame
                based tests, time based tests, or no tests in a particular
                frame. The main test stuff is done in this function, like
                screen captures and shut down. Happens after rendering code 
            onEndTestFrame() 
                Does nothing on its own, is a callback you can override if your 
                specific sample needs to do any post-render testing stuff 
            onTestShutdown()
                Does nothing on its own, is a callback you can override if your 
                specific sample needs to do anything right before shutdown 
                
        Testing Arguments
            The SampleTest class knows to look for the following arguments in 
            InitializeTesting, any other test actions you want will need to be
            either added to sampleTest or handle in callbacks   
            -test 
                This is required for all testing, without this argument, 
                sampleTest will behave exactly like a sample. This just enables
                testing 
            -loadtime 
                Adds a frameTest to frame 2, gets the previous frame time, 
                which is the frame duration of frame 1, which is the load time
            -shutdown X
                Shuts down the app at frame X
            -ssframes X Y Z
                Takes a screenshot at frames X, Y, and Z. Any number of 
                screenshot frames can be supplied.
            -perfframes X Y ... A B 
                Measures performance between frames X and Y and between frames 
                A and B. Any number of frame ranges can be supplied.
            -memframes X-Y ... A-B
                Measures the change in Private Working Set Memory between the end of frame X and the end of frame Y, and then between time A and time B.
            -shutdowntime X 
                Shuts down the app after X seconds have passed 
            -memtimes X-Y ... A-B
                Measures the change in Private Working Set Memory between time X and time Y, and then between time A and time B.
            -sstimes X Y Z
                Takes a screenshot after X, Y, and Z seconds have passed. Any 
                number of screenshot times can be supplied. 
            -perftimes X Y ... A B
                Measures performance between times X and Y and between times A 
                and B. Any number of time ranges can be supplied 
                
        Integration into Existing Sample 
            To integrate testing into an existing sample, perform the following actions 
            Inherit from SampleTest rather than Sample 
            At the end of your sample's OnLoad(), call InitializeTesting()
            At the beginning of your sample's OnFrameRender(), call beginTestFrame()
            At the end of your sample's OnFrameRender(), call endTestFrame()
    
    TestBase 
        TestBase is the base class for low level tests. It lives in the 
        FalcorTest solution. Tests are functors that are added to a vector
        and then called in a try catch block (to recover from tests that crash)
        
        Functions
            addTestToList<T>()
                Adds a testing functor of type T to the testing vector 
            addTests()
                Virtual function that needs to be overriden by the derived
                class. Should call addTestToList for all the tests the 
                low level testing project wants to run
            init(bool initDevice = false) 
                Disables errors that require human intervention, calls 
                addTests() and optionally inits a dummy device if a device 
                is needed for the tests. Calls onInit() at the end. 
            onInit() 
                Does nothing on its own, is a callback you can override if your 
                specific project needs to do any unique init stuff 
            runTests()
                Calls all functors added to the testing vector by 
                addTestToList() in a try catch block to recover from crashes
                Returns a vector of testData structs (which include result, 
                test name, and optionally error message) 
            XMLFromTestResult()
                Takes a TestData struct and converts it to an xml string to be
                written to the output file 
            GenerateXml()
                Takes a vector of xml strings, outputs a testing xml file 
            run()
                calls runTests() to get a testData vector, passes structs in 
                the vector into GenerateXml to populate a vector of xml 
                strings, passes the vector of xml strings to GenerateXML() to
                make the result file 
                
        Macros 
            register_testing_func(x_)
                This is to do the header side declaration of testing functors.
                For example, in your header, you might have...
                    register_testing_func(TestBlending)
            testing_func(classname_, functorName_) 
                This is to do the cpp side declaration of testing functors
                For example, in your implementation file, you might have....
                    testing_func(BlendStateTest, TestBlending) 
            test_pass() 
                Creates a testData struct with the correct info for a passed test
            test_fail(errorMessage_)
                Creates a testdata struct with the info for a failed test
                For example...
                    return test_fail("State doesn't match desc used to create it")
        
        Subclasses 
            ResultSummary 
                Contains meta data about the tests, how many total tests were 
                run, how many successes, how many failures. Used to generate
                test result xml file 
            DummyWindowCallbacks 
                A window is required to create a dummy device, and callbacks 
                are required to create a window 
            TestFunction 
                The base class for test functors, you don't really need to 
                worry about this because it's automatically dealt with in the 
                macros  
        
        Notes
            The FalcorTest solution still has all the original configurations,
            so if everything is error'd out, you're probably not in DebugD3D12
            or ReleaseD3D12 

Python Side
    RunAllTests.py
        This script builds and runs tests then checks results against reference
        results
        
        Arguments 
            -nb (--nobuild) 
                Assumes all test exes are already built and runs without 
                building them, this was mostly used for testing the fw 
            -ss (--showsummary)
                Automatically opens the testSummary html file after testing
            -gr (--generatereference)
                Rather than comparing against reference, saves test results to
                the reference directory 
            -ref (--referencedir) X
                Sets the reference directory to check against to X
            -tests (--testlist) X 
                Sets the test list file to run to X 
        
    Test List File 
        The test list file read by RunAllTests.py includes a list of one or 
        more solutions and the test projects within each solution that should
        be built and run. It has the following format.... 
        
        Test List Structure 
            [
            pathToSolution1FromGitDir {config1 pathToConfig1ExeDir ... configN pathToConfigNExeDir}
            Project1 {-test ... other args,
                      -test ... other args (for another run of this project)} {config1 ... conifgN}
            Project2 {-test ... other args } {config1 ... configN}
            ]
            [
            pathToSolution2FromGitDir {config1 pathToConfig1ExeDir ... configN pathToConfigNExeDir}
            Project1 {-test ... other args } {config1 ... conifgN}
            Project2 {-test ... other args } {config1 ... configN}
            ]
        
        Example Test List 
            [
            ../Falcor.sln {Released3d12 ..\Bin\x64\Release\}
            ShaderBuffers {-test -ssframes 50 -shutdown 100} {released3d12}
            FeatureDemo {
            -test -loadscene san-miguel\SanMiguel.fscene -sstimes 10 20 30 -shutdowntime 35, 
            -test -loadscene classroom\Classroom.fscene -sstimes 7 15 -shutdowntime 20,
            -test -loadscene living_room\living_room.fscene -sstimes 10 20 30 -shutdowntime 35}
            {released3d12}
            ]
            [
            FalcorTest.sln {Released3d12 Bin\x64\Release\ Debugd3d12 Bin\x64\Debug\}
            BlendStateTest {} {released3d12}
            RasterizerStateTest {} {debugd3d12 released3d12}
            ]
    
    CallTestScript.py
        This script reads a test config file, and calls RunnAllTests.py on 
        each test set described in it. RunAllTests.py returns the results of
        each test set to this script, which then interprets/organizes the 
        results and sends an email summarizing them 
        
        Arguments 
            -config (--testconfig) X
                sets the test config file to X 
            -ne (--noemail) 
                runs without sending an email 
            -ss (--showsummary)
                for each test set, opens the testing summary after finishing 
            -gr (--generatereference)
                generates reference for each test set in the config file 
        
        Test Config File 
            The test config file read by CallTestScript.py includes a list of 
            one or more repository/branch pairs and information about where to 
            find relevant files and directories within each repository 
            It has the following format...
            
            Test Config Structure 
            {
                pathToReferenceDir, PathToTestDirFromRepoDir, TestListFile, urlToCloneFrom, DestinationDirToCloneTo, branchToClone 
            }
            {
                //same as above for another repository/branch pair 
            }
            
            Example Test Config 
            {
                \\netapp-wa02\public\Falcor\ReferenceResultsHub\, Test, TestList.txt, https://github.com/NVIDIA/Falcor.git, C:\Users\nvrgfxtest\Desktop\FalcorDest, master
            }
            {
                \\netapp-wa02\public\Falcor\ReferenceResultsHubDev\, Test, TestList.txt, https://github.com/NVIDIA/Falcor.git, C:\Users\nvrgfxtest\Desktop\FalcorDestDev, 2.0a4-Testing
            }
    
	OutputTestingHtml.py and TestingUtil.py
		These files are more or less headers for organizational purposes. They 
		are imported as modules in CallTestScript and RunAllTests. They should 
		not be called themselves. 