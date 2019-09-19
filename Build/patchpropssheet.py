import sys
import os

def patchGroup(propSheet, group, val):
	groupStart = "<" + group + ">"
	groupEnd = "</" + group + ">"
	s = propSheet.find(groupStart)
	e = propSheet.find(groupEnd)
	if(s == -1 or e == -1):
		sys.exit("Can't find a `" + groupStart + "` section in the file. This is probably because someone deleted it for the property sheet. Revert the changes and try again.\n")

	if(s >= e):
		sys.exit("The property sheet is corrupted. `" + groupEnd + "` can't appear before `" + groupStart + "` \n")

	s += len(groupStart)
	propSheet = propSheet[:s] + val + propSheet[e:]
	return propSheet

if(len(sys.argv) != 4):
	sys.exit("Usage:\npatchpropssheet.py <Falcor Core Directory> <Current Solution Directory> <Backend [FALCOR_D3D12, FALCOR_VK]>")    

coreDir = sys.argv[1]
solutionDir = sys.argv[2]
backend = sys.argv[3]
propsFileName = coreDir + "\\Falcor\\falcor.props"
# Open and read the file
f = open(propsFileName)
propSheet = f.read()
f.close()

# Get a relative path from the Current Solution Directory to the Falcor Core Directory.
relcorepath = os.path.relpath(coreDir, solutionDir)
coreDir = "$(SolutionDir)\\" + relcorepath

propSheet = patchGroup(propSheet, "FALCOR_CORE_DIRECTORY", coreDir)
propSheet = patchGroup(propSheet, "FALCOR_BACKEND", backend)

# Save the file
f = open(propsFileName, "w")
f.write(propSheet)
f.close()