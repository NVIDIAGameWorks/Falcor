After cloning, please copy and rename TestConfigs to TestConfigsDev so any local changes you make are not pushed to GitHub.
This is also needed to ensure that the tests are looking in the right place for the TestConfigs.

Running the CheckInTest(D3D12 or VK) uses a relative path from the batch file to the main Falcor Directory (defaults to ../) and uses the TestConfigsDev by default.
The reference files that are used by default are defined in MachineConfigs.py

All Test Results from the CheckInTest are placed in the TestsResults\\local-results\\directory 


RunTestsCollection.py runs a TestCollection file from the configs folder.
Pulls from the Repository Target + Source Branch Target to the local Destination Target.
Uses the Compare Reference Target\\(name of the local machine)\\Compare Branch Target\\(Folder for each Test Collection (in the array!))\\


RunGenerateReferences.py runs a TestCollection file from the configs folder.
Pulls from the Repository Target + Source Branch Target to the local Destination Target.
Uses the Generate Reference Target\\(name of the local machine)\\Source Branch Target\\(Folder for each Test Collection (in the array!))\\



If you would like to make a stand-alone copy of the Tests, simply copy out the Tests Folder to a different directory.
Both RunTestsCollection.py and RunGenerateReferences.py will work, but the Arguments to the CheckInTest will have to be changed.
