After cloning, please copy and rename TestConfigs to TestConfigsDev so any local changes you make are not pushed to GitHub.
This is also needed to ensure that the tests are looking in the right place for the TestConfigs.

Running the CheckInTest(D3D12 or VK) uses a relative path from the batch file (defaults to ../) and uses the TestConfigsDev by default.
The reference files are used by default are defined in MachineConfigs.py


