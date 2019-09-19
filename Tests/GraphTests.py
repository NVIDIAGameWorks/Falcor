import subprocess
import shutil
from pathlib import Path
import argparse
import CompareOutput as compareOutput
import TestConfig as testConfig
import Helpers as helpers
import os

def getMogwaiExe(config):
    return helpers.findExecutable(config, testConfig.mogwaiExe)

def compareDirs(outDir, refDir):
    if(refDir):
        return compareOutput.compare(str(outDir), str(refDir))
    return True

def writeFile(filename, data):
    file = open(filename, "w")
    file.write(data)
    file.close()

def generate_config_file(graphFile, outputDir, basename, scene):
    a = 'm.loadScene(r"' + str(scene) + '")\n'
    a += 'm.script(r"' + str(graphFile) + '")\n'
    a += 'm.ui(false)\n'
    a += 't.framerate(' + str(testConfig.framerate) + ')\n'
    a += 't.now(0)\n'
    a += 't.exitTime(' + str(testConfig.exitTime) + ')\n'
    a += 'fc.outputDir(r"' + str(outputDir) + '")\n'
    a += 'fc.baseFilename("' + str(basename) + '")\n'
    a += 'm.activeGraph().markOutput("*")\n'
    a += 'fc.frames(m.activeGraph(), ' + str(testConfig.frames) + ')\n'
    return a

def test_graph_with_scene(graphFile, outputDir, scene, buildConfig):
    graphFile = Path(graphFile).resolve()
    outputDir = Path(outputDir).resolve()
    scene = Path(scene)
    graphName = graphFile.stem
    sceneName = scene.stem
    basename = graphName + '.' + sceneName
    configFile = outputDir / (basename + ".config.py")
    s = generate_config_file(graphFile, outputDir, basename, scene)
    writeFile(configFile, s)
    helpers.runProcessAsync([getMogwaiExe(buildConfig), '-script', str(configFile)])


def test_graph(graphFile, outputDir, buildConfig, referenceDir):
    helpers.cleanDir(outputDir)
    for s in testConfig.scenes:
        test_graph_with_scene(graphFile, outputDir, s, buildConfig)
    return compareDirs(outputDir, referenceDir)

def print_all_graphs_msg(rootDir, dirFilter):
    msg = "Procesing folder " + rootDir
    if dirFilter:
        msg += " using the directory filter `" + dirFilter + "`\n"
    else:
        msg += ". No directory filter specified\n"
    print(msg)


def run_test_if_graph_file(file, dirName, outputDir, buildConfig, refDir):
    success = True
    file = Path(file)
    if(file.suffix == '.py'):
        subdir = os.path.splitext(file)[0]
        graphOutDir = Path(outputDir) / subdir
        file = Path(dirName) / file
        print("Testing " + str(file))
        if refDir:
            refDir = Path(refDir) / subdir
        success = test_graph(file, graphOutDir, buildConfig, refDir) and success
    return success

def test_all_graphs(rootDir, outputDir, dirFilter, buildConfig, refDir):
    success = True
    dirFilter = dirFilter.lower()
    rootDir = os.path.abspath(rootDir)
    print_all_graphs_msg(rootDir, dirFilter)

    for dirName, subDirs, files in os.walk(rootDir):
        if dirFilter and (Path(dirName).name.lower() != dirFilter):
            continue

        for f in files:
            success = run_test_if_graph_file(f, dirName, outputDir, buildConfig, refDir) and success
    return success


def getBuildConfig(args):
    if args.config:
        return args.config
    else:
        return testConfig.defaultConfig

def main():	
    # Argument Parser.
    parser = argparse.ArgumentParser()
    parser.add_argument('--graphsDir', action='store', help='Test all graph files found in a direcoty called `Testing`, which is a specific directory and its subfolders')
    parser.add_argument('--graphsDirFilter', action='store', help='Only run graphs found if the subfolder name matches this')
    parser.add_argument('-g', '--graphFile', action='store', help='Specify the graph file to test')
    parser.add_argument('-o', '--outputDir', action='store', help='The output directory for the output images', required=True)
    parser.add_argument('-r', '--reference', action='store', help='Reference images directory. If this arg is provided, the generated results will be compared against the reference.')
    parser.add_argument('--config', action='store', help='Build configuration')
    args = parser.parse_args()

    if (not args.graphsDir and not args.graphFile) or (args.graphsDir and args.graphFile):
        print ("Please specify either '--graphFile' or '--graphFile' within the script arguments. They are mutually exclusive, so don't use both")
        sys.exit(1)

    outputDir = Path(args.outputDir)
    buildConfig = getBuildConfig(args)

    success = True

    if(args.graphFile):
        success = test_graph(args.graphFile, outputDir, buildConfig, args.reference)

    if(args.graphsDir):
        success = test_all_graphs(args.graphsDir, outputDir, args.graphsDirFilter, buildConfig, args.reference)

    if success:
        print("Graph tests passed")
    else:
        print("Graph tests failed")

if __name__ == '__main__':
    main()
