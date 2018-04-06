#!/bin/sh
#Daily Test - Clones and Runs the Vulkan tests specified in TC_VK_Release
python3 RunTestsCollection.py -tc "TestConfigs/TC_VK_Release.json" -ne
