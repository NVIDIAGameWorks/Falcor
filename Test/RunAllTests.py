import subprocess
import shutil
import time
import os
from xml.dom import minidom
from xml.parsers.expat import ExpatError
import argparse
import sys
import ntpath
from datetime import date
#custom written modules
import OutputTestingHtml as htmlWriter
import TestingUtil as testingUtil

#relevant paths
gBuildBatchFile = 'BuildFalcor.ps1 '
gTestListFile = 'TestList.txt'
gResultsDirBase = 'TestResults'
gReferenceDir = 'VulkanReferenceResults'
gPowerShell = "C:\\WINDOWS\\system32\\WindowsPowerShell\\v1.0\\powershell.exe"

#default values
#percent
gDefaultFrameTimeMargin = 0.05
#seconds
gDefaultLoadTimeMargin = 10
#seconds
gDefaultHangTimeDuration = 60 * 5

#stores data for all tests in a particular solution
class TestSolution(object):
    def __init__(self, name):
        self.name = name
        self.systemResultList = []
        self.lowLevelResultList = []
        self.skippedList = []
        self.errorList = []
        dateStr = date.today().strftime("%m-%d-%y")
        self.resultsDir = gResultsDirBase + '\\' + dateStr
        testingUtil.makeDirIfDoesntExist(self.resultsDir)
        self.configDict = {}

#test info class stores test data and has functions that require test data, mostly getting necessary file paths
class TestInfo(object):
    def __init__(self, name, configName, slnInfo):
        self.Name = name
        self.ConfigName = configName
        self.LoadErrorMargin = gDefaultLoadTimeMargin
        self.FrameErrorMargin = gDefaultFrameTimeMargin
        self.Index = 0
        self.slnInfo = slnInfo

    def determineIndex(self, generateReference,):
        initialFilename = self.getResultsFile()
        if generateReference:
            while os.path.isfile(self.getReferenceFile()):
                self.Index += 1
        else:
            if os.path.isdir(self.getResultsDir()):
                while os.path.isfile(self.getResultsDir() + '\\' + self.getResultsFile()):
                    self.Index += 1
        if self.Index != 0:
            testingUtil.overwriteMove(initialFilename, self.getResultsFile())

    def getResultsFile(self):
        return self.Name + '_TestingLog_' + str(self.Index) + '.xml'
    def getResultsDir(self):
        return self.slnInfo.resultsDir + '\\' + self.ConfigName
    def getReferenceDir(self):
        return gReferenceDir + '\\' + self.ConfigName
    def getReferenceFile(self):
        return self.getReferenceDir() + '\\' + self.getResultsFile()
    def getFullName(self):
        return self.Name + '_' + self.ConfigName + '_' + str(self.Index)
    def getTestDir(self):
        try:
            return self.slnInfo.configDict[self.ConfigName]
        except:
            print 'Invalid config' + self.ConfigName + ' for testInfo obj named ' + self.Name
            return ''
    def getTestPath(self):
        return self.getTestDir() + '\\' + self.Name + '.exe'
    def getRenamedTestScreenshot(self, i):
        return self.getTestDir() + '\\' + self.Name + '_' + str(self.Index) + '_' + str(i) + '.png'
    def getInitialTestScreenshot(self, i):
        return self.getTestDir() + '\\' + self.Name + '.exe.'+ str(i) + '.png'
    def getReferenceScreenshot(self, i):
        return self.getReferenceDir() + '\\' + self.Name + '_'+ str(self.Index) + '_' + str(i) + '.png'

    def getRenamedFileForIndex(self, i, typefix = ""):
        if(typefix != ""):
            return self.getTestDir() + '\\' + self.Name + '_' + str(self.Index) + '_' + typefix + "_" + str(i)
        else:
            return self.getTestDir() + '\\' + self.Name + '_' + str(self.Index) + '_' + str(i)

    def getInitialFileForIndex(self, i, typefix = ""):
        if(typefix != ""):
            return self.getTestDir() + '\\' + self.Name + '.exe.' + typefix + '.' + str(i)
        else:
            return self.getTestDir() + '\\' + self.Name + '.exe.' + str(i)

    def getReferenceFileForIndex(self, i, typefix = ""):
        return self.getReferenceDir() + '\\' + self.Name + '_' + str(self.Index) + '_' + str(i)



class SystemResult(object):
    def __init__(self):
        self.Name = ''
        self.LoadTime = 0
        self.AvgFrameTime = 0
        self.RefLoadTime = 0
        self.RefAvgFrameTime = 0
        self.LoadErrorMargin = gDefaultLoadTimeMargin
        self.FrameErrorMargin = gDefaultFrameTimeMargin
        self.CompareImageResults = []
        self.CompareMemoryFrameResults = []
        self.CompareMemoryTimeResults = []

class LowLevelResult(object):
    def __init__(self):
        self.Name = ''
        self.Total = 0
        self.Passed = 0
        self.Failed = 0
        self.Crashed = 0
    def add(self, other):
        self.Total += other.Total
        self.Passed += other.Passed
        self.Failed += other.Failed
        self.Crashed += other.Crashed

def renameScreenshots(testInfo, numScreenshots):
    for i in range (0, numScreenshots):
        os.rename(testInfo.getInitialFileForIndex(i) + '.png', testInfo.getRenamedFileForIndex(i) + '.png')

def compareImages(resultObj, testInfo, numScreenshots, slnInfo):
    renameScreenshots(testInfo, numScreenshots)
    for i in range(0, numScreenshots):
        testScreenshot = testInfo.getRenamedFileForIndex(i) + '.png'
        refScreenshot = testInfo.getReferenceFileForIndex(i) + '.png'
        outFile = testInfo.Name + '_' + str(testInfo.Index) + '_' + str(i) + '_Compare.png'
        command = ['magick', 'compare', '-metric', 'MSE', '-compose', 'Src', '-highlight-color', 'White',
            '-lowlight-color', 'Black', testScreenshot, refScreenshot, outFile]
        p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
        result = p.communicate()[0]
        spaceIndex = result.find(' ')
        resultValStr = result[:spaceIndex]
        imagesDir = testInfo.getResultsDir() + '\\Images'
        try:
            resultVal = float(resultValStr)
        except:
            slnInfo.errorList.append(('For test ' + testInfo.getFullName() +
                ' failed to compare screenshot ' + testScreenshot + ' with ref ' + refScreenshot +
                ' instead of compare result, got \"' + resultValStr + '\" from larger string \"' + result + '\"'))
            testingUtil.makeDirIfDoesntExist(imagesDir)
            testingUtil.overwriteMove(testScreenshot, imagesDir)
            resultObj.CompareImageResults.append(-1)
            

        # Append the Result Value String.
        resultObj.CompareImageResults.append(resultVal)
        # Move images to results folder
        testingUtil.makeDirIfDoesntExist(imagesDir)
        testingUtil.overwriteMove(testScreenshot, imagesDir)
        testingUtil.overwriteMove(outFile, imagesDir)

        # If the images are sufficiently different, save them in test results.
        if resultVal > testingUtil.gDefaultImageCompareMargin:
            slnInfo.errorList.append(('For test ' + testInfo.getFullName() + ', screenshot ' +
                testScreenshot + ' differs from ' + refScreenshot + ' by ' + result +
                ' average difference per pixel. (Exceeds threshold .01)'))


def renameMemoryChecks(testInfo, countMemoryChecks, typefix=""):
    print typefix
    for i in range (0, countMemoryChecks):
        print testInfo.getInitialFileForIndex(i, typefix) + '.txt'
        os.rename(testInfo.getInitialFileForIndex(i, typefix) + '.txt', testInfo.getRenamedFileForIndex(i, typefix) + '.txt')


def compareMemoryChecks(resultObj, testInfo, countMemoryChecks, slnInfo, typefix):

    if(typefix == "MemoryFrameCheck" or typefix == "MemoryTimeCheck"):

        renameMemoryChecks(testInfo, countMemoryChecks, typefix)
        for i in range (0, countMemoryChecks):
            keystring = ""
            with open(testInfo.getRenamedFileForIndex(i, typefix) + '.txt') as f:
                content = f.readlines()
                keystring = content[0].strip()

            if (keystring != ""):
                values = keystring.split(' ')
                percentDifference = 0.0


                if(int(values[2]) == 0.0):
                    currentRange = [float(values[0]), float(values[1]), "NA", float(values[2]), int(values[3]), int(values[4])]

                else:
                    percentDifference = float(values[4]) / float(values[2]) * 100.0
                    currentRange = [float(values[0]), float(values[1]), percentDifference, float(values[2]), int(values[3]), int(values[4])]

                if(typefix == "MemoryFrameCheck"):
                    resultObj.CompareMemoryFrameResults.append(currentRange)

                if(typefix == "MemoryTimeCheck"):
                    resultObj.CompareMemoryTimeResults.append(currentRange)


                memoryDir =  testInfo.getResultsDir()
                if(percentDifference > testingUtil.gMemoryPercentCompareMargin):
                    memoryDir + "\\" + typefix
                    testingUtil.makeDirIfDoesntExist(memoryDir)
                    testingUtil.overwriteMove(testInfo.getRenamedFileForIndex(i, typefix) + '.txt', memoryDir)
                    slnInfo.errorList.append(('For test ' + testInfo.getFullName() + ' Percent Change in Memory Too High : ' + str(percentDifference) + ' %.'))

                else:
                    os.remove(testInfo.getRenamedFileForIndex(i, typefix) + '.txt')
    else:
        return

def addSystemTestReferences(testInfo, numScreenshots):
    renameScreenshots(testInfo, numScreenshots)
    referenceDir = testInfo.getReferenceDir()
    testingUtil.makeDirIfDoesntExist(referenceDir)
    testingUtil.overwriteMove(testInfo.getResultsFile(), referenceDir)
    for i in range(0, numScreenshots):
        testingUtil.overwriteMove(testInfo.getRenamedFileForIndex(i) + '.png', referenceDir)

#args should be action(build, rebuild, clean), solution, config(debug, release), and optionally project
#if no project given, performs action on entire solution

#Calls PowerShell Build Script
# Arguments should be - (Target Solution, Target Build Configuration, Build Type - Clean / Build / Rebuild)
def callBuildScript(batchArgs):
    numArgs = len(batchArgs)
    if numArgs == 3 :
        batchArgs[0] = batchArgs[0] + " "
        batchArgs[1] = batchArgs[1] + " "
        batchArgs[2] = batchArgs[2] + " "
        batchArgs.insert(0, " " + os.getcwd() + "\\" + gBuildBatchFile + " ")
        batchArgs.insert(0, gPowerShell)
        print ''.join(batchArgs)
        try:
            return subprocess.call(batchArgs)
        except (WindowsError, subprocess.CalledProcessError) as info:
            print 'Error with Build Script. Exception: ', info
            return 1
    else:
        print 'Wrong number of arguments provided. Expected 3, got ' + str(numArgs)
        return 1

def buildFail(slnName, configName, slnInfo):
    testingUtil.makeDirIfDoesntExist(slnInfo.resultsDir)
    buildLog = 'Solution_BuildFailLog.txt'
    testingUtil.overwriteMove(buildLog, slnInfo.resultsDir)
    errorMsg = 'failed to build one or more projects with config ' + configName + ' of solution ' + slnName  + ". Build log written to " + buildLog
    print 'Error: ' + errorMsg
    slnInfo.errorList.append(errorMsg)

def addCrash(test, slnInfo):
    slnInfo.skippedList.append(test, 'Unhandled Crash')

def getXMLTag(xmlFilename, tagName):
    try:
        referenceDoc = minidom.parse(xmlFilename)
    except ExpatError:
        return None
    tag = referenceDoc.getElementsByTagName(tagName)
    if len(tag) == 0:
        return None
    else:
        return tag

def addToLowLevelSummary(slnInfo, result):
    if not slnInfo.lowLevelResultList:
        resultSummary = LowLevelResult()
        resultSummary.Name = 'Summary'
        slnInfo.lowLevelResultList.append(resultSummary)
    slnInfo.lowLevelResultList[0].add(result)
    slnInfo.lowLevelResultList.append(result)

def processLowLevelTest(xmlElement, testInfo, slnInfo):
    newResult = LowLevelResult()
    newResult.Name = testInfo.getFullName()
    newResult.Total = int(xmlElement[0].attributes['TotalTests'].value)
    newResult.Passed = int(xmlElement[0].attributes['PassedTests'].value)
    newResult.Failed = int(xmlElement[0].attributes['FailedTests'].value)
    if newResult.Failed > 0:
        slnInfo.errorList.append(testInfo.getFullName() + ' had ' + str(newResult.Failed) + ' failed tests')
    newResult.Crashed = int(xmlElement[0].attributes['CrashedTests'].value)
    addToLowLevelSummary(slnInfo, newResult)
    slnInfo.lowLevelResultList.append(newResult)
    #result at index 0 is summary
    testingUtil.makeDirIfDoesntExist(testInfo.getResultsDir())
    testingUtil.overwriteMove(testInfo.getResultsFile(), testInfo.getResultsDir())

def processSystemTest(xmlElement, testInfo, slnInfo):

    # Get the System Test Result/
    newSysResult = SystemResult()
    newSysResult.Name = testInfo.getFullName()

    # Get the Load Time and the Average Frame Time.
    newSysResult.LoadTime = float(xmlElement[0].attributes['LoadTime'].value)
    newSysResult.AvgFrameTime = float(xmlElement[0].attributes['FrameTime'].value)

    # Get the Load Error Margin and the Frame Error Margin.
    newSysResult.LoadErrorMargin = testInfo.LoadErrorMargin
    newSysResult.FrameErrorMargin = testInfo.FrameErrorMargin

    # Get the Number of Screen Shots from the XML output.
    numScreenshots = int(xmlElement[0].attributes['NumScreenshots'].value)
    numMemoryFrameChecks = int(xmlElement[0].attributes['NumMemoryFrameChecks'].value)
    numMemoryTimeChecks = int(xmlElement[0].attributes['NumMemoryTimeChecks'].value)

    # Get the Reference and Result Files.
    referenceFile = testInfo.getReferenceFile()
    resultFile = testInfo.getResultsFile()

    print referenceFile
    print resultFile

    # Find the Reference File, and quit if we cannot.
    if not os.path.isfile(referenceFile):
        slnInfo.skippedList.append((testInfo.getFullName(), 'Could not find reference file ' + referenceFile + ' for comparison'))
        return
    refResults = getXMLTag(referenceFile, 'Summary')

    # Find the Results File, and quit if we cannot.
    if not refResults:
        slnInfo.skippedList.append((testInfo.getFullName(), 'Error getting xml data from reference file ' + referenceFile))
        return
    newSysResult.RefLoadTime = float(refResults[0].attributes['LoadTime'].value)
    newSysResult.RefAvgFrameTime = float(refResults[0].attributes['FrameTime'].value)

    # Check Average FPS
    if newSysResult.AvgFrameTime != 0 and newSysResult.RefAvgFrameTime != 0:
        if testingUtil.marginCompare(newSysResult.AvgFrameTime, newSysResult.RefAvgFrameTime, newSysResult.FrameErrorMargin) == 1:
            slnInfo.errorList.append((testInfo.getFullName() + ': average frame time ' +
            str(newSysResult.AvgFrameTime) + ' is larger than reference ' + str(newSysResult.RefAvgFrameTime) +
            ' considering error margin ' + str(newSysResult.FrameErrorMargin * 100) + '%'))

    # Check Load Time.
    if newSysResult.LoadTime != 0 and newSysResult.RefLoadTime != 0:
        if newSysResult.LoadTime > (newSysResult.RefLoadTime + newSysResult.LoadErrorMargin):
            slnInfo.errorList.append(testInfo.getFullName() + ': load time' + (str(newSysResult.LoadTime) +
            ' is larger than reference ' + str(newSysResult.RefLoadTime) + ' considering error margin ' +
            str(newSysResult.LoadErrorMargin) + ' seconds'))


    # Compare the images.
    compareImages(newSysResult, testInfo, numScreenshots, slnInfo)

    # Compare the Memory Checks.
    if(numMemoryFrameChecks != 0):
        compareMemoryChecks(newSysResult, testInfo, numMemoryFrameChecks, slnInfo, "MemoryFrameCheck")

    if(numMemoryTimeChecks != 0):
        compareMemoryChecks(newSysResult, testInfo, numMemoryTimeChecks, slnInfo, "MemoryTimeCheck")

    # Add it to the System Result List.
    slnInfo.systemResultList.append(newSysResult)

    # Make Directory if it does not exist and move the result file to that directory, overwriting any previous version.
    testingUtil.makeDirIfDoesntExist(testInfo.getResultsDir())
    testingUtil.overwriteMove(resultFile, testInfo.getResultsDir())

def readTestList(generateReference, buildTests, pullBranch):
    testFile = open(gTestListFile)
    contents = testFile.read()

    slnInfos = []

    slnDataStartIndex = contents.find('[')
    while slnDataStartIndex != -1:
        #get all data about testing this solution
        slnDataEndIndex = contents.find(']')
        solutionData = contents[slnDataStartIndex + 1 : slnDataEndIndex]
        slnEndIndex = solutionData.find(' ')
        slnName = testingUtil.cleanupString(solutionData[:slnEndIndex])

        #make sln name dir within date dir
        slnBaseName, extension = os.path.splitext(slnName)
        slnBaseName = ntpath.basename(slnBaseName)
        slnInfo = TestSolution(slnBaseName)
        slnInfos.append(slnInfo)
        if pullBranch:
            slnInfo.resultsDir += '\\' + slnBaseName + '_' + pullBranch
        else:
            slnInfo.resultsDir += '\\' + slnBaseName
        if os.path.isdir(slnInfo.resultsDir):
            testingUtil.cleanDir(slnInfo.resultsDir, None, None)
        else:
            os.makedirs(slnInfo.resultsDir)

        #parse solutiondata
        slnConfigStartIndex = solutionData.find('{')
        slnConfigEndIndex = solutionData.find('}')
        configData = solutionData[slnConfigStartIndex + 1 : slnConfigEndIndex]
        configDataList = configData.split(' ')
        for i in xrange(0, len(configDataList), 2):
            exeDir = configDataList[i + 1]
            configName = configDataList[i].lower()
            slnInfo.configDict[configName] = exeDir
            if buildTests:
                #delete bin dir
                if os.path.isdir(exeDir):
                    shutil.rmtree(exeDir)
                #returns 1 on fail
                if callBuildScript([slnName, configName, 'rebuild']):
                    buildFail(slnName, configName, slnInfo)
                #returns 1 on fail
                if callBuildScript([slnName, configName, 'build']):
                    buildFail(slnName, configName, slnInfo)
            else:
                testingUtil.cleanDir(exeDir, None, '.png')

        #move buffer to beginning of args
        solutionData = solutionData[slnConfigEndIndex + 1 :]

        #parse arg data
        argStartIndex = solutionData.find('{')
        while argStartIndex != -1:
            testName = testingUtil.cleanupString(solutionData[:argStartIndex])
            argEndIndex = solutionData.find('}')
            argString = testingUtil.cleanupString(solutionData[argStartIndex + 1 : argEndIndex])
            argsList = argString.split(',')
            solutionData = solutionData[argEndIndex + 1 :]
            configStartIndex = solutionData.find('{')
            configEndIndex = solutionData.find('}')
            configList = testingUtil.cleanupString(solutionData[configStartIndex + 1 : configEndIndex]).split(' ')
            #run test for each config and each set of args
            for config in configList:
                print 'Running ' + testName + ' in config ' + config
                testInfo = TestInfo(testName, config, slnInfo)
                if generateReference:
                    testingUtil.cleanDir(testInfo.getReferenceDir(), testName, '.png')
                    testingUtil.cleanDir(testInfo.getReferenceDir(), testName, '.xml')
                for argSet in argsList:
                    testInfo = TestInfo(testName, config, slnInfo)
                    runTest(testInfo, testingUtil.cleanupString(argSet), generateReference, slnInfo)
					
            #goto next set
            solutionData = solutionData[configEndIndex + 1 :]
            argStartIndex = solutionData.find('{')
        #goto next solution set
        contents = contents[slnDataEndIndex + 1 :]
        slnDataStartIndex = contents.find('[')
    return slnInfos

def runTest(testInfo, cmdLine, generateReference, slnInfo):
    # Don't generate the Low Level Reference.
    if generateReference and not cmdLine:
        return

    testPath = testInfo.getTestPath()
    if not os.path.exists(testPath):
        slnInfo.skippedList.append((testInfo.getFullName(), 'Unable to find ' + testPath))
        return
    try:
        p = subprocess.Popen(testPath + ' ' + cmdLine)
        # Run test until timeout or return.
        start = time.time()
        while p.returncode == None:
            p.poll()
            cur = time.time() - start
            if cur > gDefaultHangTimeDuration:
                p.kill()
                slnInfo.skippedList.append((testInfo.getFullName(), ('Test timed out ( > ' +
                    str(gDefaultHangTimeDuration) + ' seconds)')))
                return
        # Ensure results file exists
        if not os.path.isfile(testInfo.getResultsFile()):
            slnInfo.skippedList.append((testInfo.getFullName(), 'Failed to open test result file ' + testInfo.getResultsFile()))
            return
        # Check for name conflicts
        testInfo.determineIndex(generateReference)
        # Get xml from results file
        summary = getXMLTag(testInfo.getResultsFile(), 'Summary')
        if not summary:
            slnInfo.skippedList.append((testInfo.getFullName(), 'Error getting xml data from ' + testInfo.getResultsFile()))
            if generateReference:
                testingUtil.makeDirIfDoesntExist(testInfo.getReferenceDir())
                testingUtil.overwriteMove(testInfo.getResultsFile(), testInfo.getReferenceDir())
            else:
                testingUtil.makeDirIfDoesntExist(testInfo.getResultsDir())
                testingUtil.overwriteMove(testInfo.getResultsFile(), testInfo.getResultsDir())
            return
        # Gen system ref
        if cmdLine and generateReference:
            numScreenshots = int(summary[0].attributes['NumScreenshots'].value)
            addSystemTestReferences(testInfo, numScreenshots)
        # Process system
        elif cmdLine:
            processSystemTest(summary, testInfo, slnInfo)

        # Process Low Level
        else:
            processLowLevelTest(summary, testInfo, slnInfo)
    except subprocess.CalledProcessError:
        addCrash(testInfo.getFullName(), slnInfo)

def main(build, showSummary, generateReference, referenceDir, testList, pullBranch):
    global gReferenceDir
    global gTestListFile

    reference = testingUtil.cleanupString(referenceDir)
    if os.path.isdir(reference):
        gReferenceDir = reference
    elif not generateReference:
        print 'Fatal Error, Failed to find reference dir: ' + referenceDir
        sys.exit(1)

    if generateReference:
        if not os.path.isdir(gReferenceDir):
            try:
                os.makedirs(gReferenceDir)
            except:
                print 'Fatal Error, Failed to create reference dir.'
                sys.exit(1)

    if not os.path.exists(testList):
        print 'Fatal Error, Failed to find test list ' + testList
        sys.exit(1)
    else:
        gTestListFile = testList

    #make outer dir if need to
    testingUtil.makeDirIfDoesntExist(gResultsDirBase)

    #Read test list, run tests, and return results in a list of slnInfo classes
    slnInfos = readTestList(generateReference, build, pullBranch)
    if not generateReference:
        for sln in slnInfos:
            htmlWriter.outputHTML(showSummary, sln, pullBranch)

    #parse slninfo classes to fill testing result list to return to calling script
    testingResults = []
    for sln in slnInfos:
        slnResultPath =  os.getcwd() + '\\' + sln.resultsDir + '\\' + sln.name
        if pullBranch:
            slnResultPath += '_' + pullBranch
        resultSummary = slnResultPath + '_TestSummary.html'
        errorSummary = slnResultPath + '_ErrorSummary.txt'

        if len(sln.skippedList) > 0 or len(sln.errorList) > 0:
            errorStr = ''
            for name, skip in sln.skippedList:
                errorStr += name + ': ' + skip + '\n'
            for reason in sln.errorList:
                errorStr += reason + '\n'
            errorFile = open(errorSummary, 'w')
            errorFile.write(errorStr)
            errorFile.close()
            testingResults.append([sln.name, resultSummary, False])
        else:
            testingResults.append([sln.name, resultSummary, True])

    #return results back to calling script
    return testingResults

#Main is separated from this so other scripts (CallTestingScript.py) can import this script as a module and
#call main as a function. All this __main__ does is parse command line arguments and pass them into the main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-nb', '--nobuild', action='store_true', help='run without rebuilding Falcor and test apps')
    parser.add_argument('-ss', '--showsummary', action='store_true', help='opens testing summary upon completion')
    parser.add_argument('-gr', '--generatereference', action='store_true', help='generates reference testing logs and images')
    parser.add_argument('-ref', '--referencedir', action='store', help='Allows user to specify an existing reference dir')
    parser.add_argument('-tests', '--testlist', action='store', help='Allows user to specify the test list file')
    args = parser.parse_args()

    if args.referencedir:
        refDir = args.referencedir
    else:
        refDir = gReferenceDir

    if args.testlist:
        testListFile = args.testlist
    else:
        testListFile = gTestListFile

    #final arg is pull branch, just to name subdirectories in the same repo folder so results dont overwrite
    main(not args.nobuild, args.showsummary, args.generatereference, refDir, testListFile, '')
