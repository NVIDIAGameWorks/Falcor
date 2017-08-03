The Tests Collections and Sets.

Test Collections:
    
    -   Test Collections are a list of the Test Sets. 
        This is to, especially for the daily, group together tests of different configurations and solutions for a single run.
    
    -   Test Collections can be used to clone and run from a specific repository and branch, to a particular destination.
        This is useful when trying to run the tests against a particular branch, in a separate location from the usual tests.
    
    -   Test Collections are run through the RunTestsCollection.py . 
        Since they are run from that directory, the paths for the Test Sets to use must be relative to that directory.

    -   The Test Results will be placed in the DestinationTarget folder.
        The result of each Test will be placed in a subfolder, relative to the "Destination Target/Branch Target/" folder, with its name being the same as the file.

    -   Defaults from RunTestsCollection.py:
        -   Not specifying the "Repository Target" will clone from the default location - "https://github.com/NVIDIAGameworks/Falcor.git".
        -   Not specifying the "Branch Target" will use the default branch - "master".
        -   Not specifying the "DestinationTarget" will use the default destination - "C:\\Falcor\\"