#!/usr/bin/python
import shutil
import sys
import os

if len(sys.argv) != 2:
	print 'Usage: make_new_project <NewProjectName>'
	sys.exit(1)

# copy the ProjectTemplate directory
Src = "ProjectTemplate"
Dst = sys.argv[1]

os.mkdir(Dst)

Files=[]
Files.append(".cpp")
Files.append(".h")
Files.append(".vcxproj")
Files.append(".vcxproj.filters")

for File in Files:
	#rename the File
	SrcFile = Src + '/' + Src + File
	DstFile = Dst + '/' + Dst + File

	# replace all occurences
	F = open(SrcFile)
	Content = F.read()
	F.close()
	F = open(DstFile, 'w')
	F.write(Content.replace(Src, Dst))
	F.close() 
