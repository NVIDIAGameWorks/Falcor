import argparse 
import os
import sys
from random import randint
import ntpath
import xml.etree.ElementTree as ET


# Custom Error Output.
class QuickParser(argparse.ArgumentParser):
    def error(self, message):
        print("\n")
        print("Requires Five Arguments : ")
        print("solutionDirectory -> Solution Directory.")
        print("projectDirectory -> Project Directory. (Needs to contain the props file).")
        print("platformName -> Platform Name.")
        print("platformShortName -> Platform Short Name.")
        print("buildConfiguration -> Build Configuration.")
        print("\n")
        sys.exit(2)


def main():
    
    # Add the Arguments to Parse.
    parser = QuickParser('Change the User Macros in the Falcor.props file.')
    parser.add_argument('solutionDirectory', action='store', help = 'Solution Directory.')    
    parser.add_argument('projectDirectory', action='store', help='Project Directory. (Needs to contain the props file).')
    parser.add_argument('platformName', action='store', help='Platform Name.')
    parser.add_argument('platformShortName', action='store', help='Platform Short Name.')
    parser.add_argument('buildConfiguration', action='store', help='Build Configuration')
   
    args = parser.parse_args()

    # Backend Macro defaults to FALCOR_D3D12
    backendMacro = "FALCOR_D3D12"

    #  Check if we are using a D3D12 Configuration.   
    if(args.buildConfiguration == 'Debug' or args.buildConfiguration == 'DebugD3D12' or args.buildConfiguration == 'ReleaseD3D12'):
        backendMacro = "FALCOR_D3D12"

    # Check if we are using a Vulkan Configuration.
    if(args.buildConfiguration == 'DebugVK' or args.buildConfiguration == 'ReleaseVK'):
        backendMacro = "FALCOR_VK"

    # print (args.solutionDirectory)
    # print (args.projectDirectory)
    # print (args.platformName)
    # print (args.platformShortName)
    # print (args.buildConfiguration)
    # print (args.projectDirectory + 'Falcor.props')

    # Register the namespace, we need to do this to prevent "ns:0" being written.
    ET.register_namespace('', "http://schemas.microsoft.com/developer/msbuild/2003")

    # Read in the props file, which is really an XML file.
    xmltree = ET.parse(args.projectDirectory + 'Falcor.props')

    # Get the Root of the XML Tree.
    root = xmltree.getroot() 

    # Look for the FALCOR_BACKEND Property.
    backendproperty = root.find('.//' + '{http://schemas.microsoft.com/developer/msbuild/2003}' + 'FALCOR_BACKEND')

    # Set the FALCOR BACKEND Propety.
    backendproperty.text = backendMacro
    
    # Write back the file.
    xmltree.write(args.projectDirectory + 'Falcor.props')

if __name__ == '__main__':    
    main()
