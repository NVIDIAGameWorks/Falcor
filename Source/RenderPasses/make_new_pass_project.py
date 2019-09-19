#!/usr/bin/python
import shutil
import sys
import os

if len(sys.argv) != 2:
	print ('Usage: make_new_project <NewProjectName>')
	sys.exit(1)

# copy the make_new_pass_project directory
Src = "PassLibraryTemplate"
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
	Content = Content.replace(Src, Dst);
	F.write(Content.replace("RenderPassTemplate", Dst))
	F.close() 
