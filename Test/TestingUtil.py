import subprocess
import os
import shutil
import stat

#average pixel color difference
gDefaultImageCompareMargin = 0.0
gMemoryPercentCompareMargin = 2.5

# -1 smaller, 0 same, 1 larger
def marginCompare(result, reference, margin):
    delta = result - reference
    if abs(delta) < reference * margin:
        return 0
    elif delta > 0:
        return 1
    else:
        return -1

def makeDirIfDoesntExist(dirPath):
    if not os.path.isdir(dirPath):
        os.makedirs(dirPath)

def overwriteMove(filename, newLocation):
    try:
        shutil.copy(filename, newLocation)
        os.remove(filename)
    except IOError, info:
        print 'Error moving ' + filename + ' to ' + newLocation + '. Exception: ', info

def cleanDir(cleanedDir, prefix, suffix):
    if os.path.isdir(cleanedDir):
        if prefix and suffix:
            deadFiles = [f for f in os.listdir(cleanedDir) if f.endswith(suffix) and f.startswith(prefix)]
        elif prefix:
            deadFiles = [f for f in os.listdir(cleanedDir) if f.startswith(prefix)]
        elif suffix:
            deadFiles = [f for f in os.listdir(cleanedDir) if f.endswith(suffix)]
        else:
            deadFiles = [f for f in os.listdir(cleanedDir)]
        for f in deadFiles:
            filepath = cleanedDir + '\\' + f
            if os.path.isdir(filepath):
                cleanDir(filepath, prefix, suffix)
            else:
                os.remove(filepath)

#this is more aggressive and less flexible than clean dir.
#this changes permissions as it traverses the dir tree to delete EVERYTHING in the dir
def removeDirTree(dirRoot):
    filesToDelete = [f for f in os.listdir(dirRoot)]
    batchArgs = []
    batchArgs.insert(0, "RemoveDirectoryTree.bat ")
    batchArgs.insert(1, dirRoot)


def cleanupString(string):
    string = string.replace('\t', '')
    return string.replace('\n', '').strip()