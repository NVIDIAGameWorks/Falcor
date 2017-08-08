import argparse 
import os
import sys
from random import randint
import ntpath

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-model', '--modelFile', action='store', help='the name of the model to tile')
	parser.add_argument('-numX', '--numTilesX', action='store', help='the number of width tiles')
	parser.add_argument('-numZ', '--numTilesZ', action='store', help='the number of depth tiles')
	parser.add_argument('-sizeX', '--tileSizeX', action='store', help='the x distance between tiles')
	parser.add_argument('-sizeZ', '--tileSizeZ', action='store', help='the z distance between tiles')
	parser.add_argument('-randRotY', '--randRotationY', action='store_true', help='Optional. If set, applies a random 90 degree rot to tiles')
	parser.add_argument('-rotX', '--rotationX', action='store', help='Optional, applies a given x rotation to tiles.')
	args = parser.parse_args()

	if not args.modelFile:
		print 'ERROR: no model file name specified'
		sys.exit(1)

	if not args.tileSizeX:
		print 'WARNING: no x tile size given, using 1'
		xSize = 1
	else:
		try:
			xSize = float(args.tileSizeX)
		except:
			print 'ERROR: Unable to convert x size ' + args.tileSizeX + ' to float'
			sys.exit(1)

	if not args.tileSizeZ:
		print 'WARNING: no z size given, using 1'
		zSize = 1
	else:
		try:
			zSize = float(args.tileSizeZ)
		except:
			print 'ERROR: Unable to convert z size ' + args.tileSizeZ + ' to float'
			sys.exit(1)

	if not args.numTilesX:
		print 'WARNING: x tile count not given, using 1'
		numX = 1
	else:
		try:
			numX = int(args.numTilesX)
		except:
			print 'ERROR: Unable to convert num tiles X ' + args.numTilesX + ' to int'
			sys.exit(1)

	if not args.numTilesZ:
		print 'WARNING: z tile count not givem, using 1'
		numZ = 1
	else:
		try:
			numZ = int(args.numTilesZ)
		except:
			print 'ERROR: Unable to convert num tiles Z ' + args.numTilesZ + ' to int'
			sys.exit(1)

	if args.rotationX:
		try:
			rotX = float(args.rotationX)
		except:
			print 'ERROR: Unabled to convert rotation X ' + args.rotationX + ' to float'
	else:
		rotX = 0

	name, extension = os.path.splitext(args.modelFile)
	name = ntpath.basename(name)
	outfile = open(name + '_tiled.fscene', 'w')

	#global data
	outfile.write('{\n')
	outfile.write('\t\"version\": 2,\n')
	outfile.write('\t\"camera_speed\": 1.0,\n')
	outfile.write('\t\"lighting_scale\": 1.0,\n')
	outfile.write('\t\"active_camera\": \"Default\",\n')
	outfile.write('\t\"ambient_intensity\": [\n')
	outfile.write('\t\t0.0,\n')
	outfile.write('\t\t0.0,\n')
	outfile.write('\t\t0.0\n')
	outfile.write('\t],\n')

	#models
	outfile.write('\t\"models\": [\n')
	outfile.write('\t\t{\n')
	outfile.write('\t\t\t\"file\": \"' + args.modelFile + '\",\n')
	outfile.write('\t\t\t\"name\": \"' + name + '\",\n')
	outfile.write('\t\t\t\"instances\": [\n')

	#instance data
	xPos = (-0.5 * xSize) * (numX - 1)
	for i in range(0, numX):
		zPos = (-0.5 * zSize) * (numZ - 1)
		for j in range(0, numZ):
			outfile.write('\t\t\t\t{\n')
			outfile.write('\t\t\t\t\t\"name\": \"' + name + '_tile_' + str(i) + '_' + str(j) + '\",\n')
			outfile.write('\t\t\t\t\t\"translation\": [\n')
			outfile.write('\t\t\t\t\t\t' + str(xPos) + ',\n')
			outfile.write('\t\t\t\t\t\t0.0,\n')
			outfile.write('\t\t\t\t\t\t' + str(zPos) + '\n')
			outfile.write('\t\t\t\t\t],\n')
			outfile.write('\t\t\t\t\t\"scaling\": [\n')
			for k in range(0, 2):
				outfile.write('\t\t\t\t\t\t1.0,\n')
			outfile.write('\t\t\t\t\t\t1.0\n')
			outfile.write('\t\t\t\t\t],\n')
			outfile.write('\t\t\t\t\t\"rotation\": [\n')
			if args.randRotationY:
				randRotY = 90 * randint(0, 3)
			else:
				randRotY = 0.0
			#if it is Z up, actually want to rotate Z, not Y. 
			if args.rotationX:
				outfile.write('\t\t\t\t\t\t' + str(rotX) + ',\n')
				outfile.write('\t\t\t\t\t\t0.0,\n')
				outfile.write('\t\t\t\t\t\t' + str(randRotY) + '\n')
			else:
				outfile.write('\t\t\t\t\t\t' + str(randRotY) + ',\n')
				outfile.write('\t\t\t\t\t\t' + str(rotX) + ',\n')
				outfile.write('\t\t\t\t\t\t0.0\n')
			outfile.write('\t\t\t\t\t]\n')
			if(i == numX - 1 and j == numZ - 1):
				outfile.write('\t\t\t\t}\n')
			else:
				outfile.write('\t\t\t\t},\n')
			zPos += zSize
		xPos += xSize
	
	outfile.write('\t\t\t]\n')
	outfile.write('\t\t}\n')
	outfile.write('\t],\n')

	#light data
	outfile.write('\t\"lights\": [\n')
	outfile.write('\t\t{\n')
	outfile.write('\t\t\t\"name\": \"DirLight0\",\n')
	outfile.write('\t\t\t\"type\": \"dir_light\",\n')
	outfile.write('\t\t\t\"intensity\": [\n')
	outfile.write('\t\t\t\t0.388235640525818,\n')
	outfile.write('\t\t\t\t0.388235640525818,\n')
	outfile.write('\t\t\t\t0.388235640525818\n')
	outfile.write('\t\t\t],\n')
	outfile.write('\t\t\t\"direction\": [\n')
	outfile.write('\t\t\t\t0.2235,\n')
	outfile.write('\t\t\t\t-0.894,\n')
	outfile.write('\t\t\t\t-0.447\n')
	outfile.write('\t\t\t]\n')
	outfile.write('\t\t},\n')
	outfile.write('\t\t{\n')
	outfile.write('\t\t\t\"name\": \"DirLight1\",\n')
	outfile.write('\t\t\t\"type\": \"dir_light\",\n')
	outfile.write('\t\t\t\"intensity\": [\n')
	outfile.write('\t\t\t\t0.388235640525818,\n')
	outfile.write('\t\t\t\t0.388235640525818,\n')
	outfile.write('\t\t\t\t0.388235640525818\n')
	outfile.write('\t\t\t],\n')
	outfile.write('\t\t\t\"direction\": [\n')
	outfile.write('\t\t\t\t-0.2235,\n')
	outfile.write('\t\t\t\t-0.894,\n')
	outfile.write('\t\t\t\t0.447\n')
	outfile.write('\t\t\t]\n')
	outfile.write('\t\t}\n')
	outfile.write('\t],\n')

	#camera data
	outfile.write('\t\"cameras\": [\n')
	outfile.write('\t\t{\n')
	outfile.write('\t\t\t\"name\": \"Default\",\n')
	outfile.write('\t\t\t\"pos\": [\n')
	outfile.write('\t\t\t\t0.0,\n')
	posY = 0.5 * min(xSize, zSize)
	outfile.write('\t\t\t\t' + str(posY) + ',\n')
	outfile.write('\t\t\t\t0.0\n')
	outfile.write('\t\t\t],\n')
	outfile.write('\t\t\t\"target\": [\n')
	outfile.write('\t\t\t\t0.0,\n')
	outfile.write('\t\t\t\t' + str(posY) + ',\n')
	outfile.write('\t\t\t\t-0.5\n')
	outfile.write('\t\t\t],\n')
	outfile.write('\t\t\t\"up\": [\n')
	outfile.write('\t\t\t\t0.0,\n')
	outfile.write('\t\t\t\t1.0,\n')
	outfile.write('\t\t\t\t0.0\n')
	outfile.write('\t\t\t],\n')
	outfile.write('\t\t\t\"focal_length\": 21.0,\n')
	outfile.write('\t\t\t\"depth_range\": [\n')
	outfile.write('\t\t\t\t1.0,\n')
	depthMax = max(xSize * numX, zSize * numZ)
	outfile.write('\t\t\t\t' + str(depthMax) + '\n')
	outfile.write('\t\t\t],\n')
	outfile.write('\t\t\t\"aspect_ratio\": 1.777\n')
	outfile.write('\t\t}\n')
	outfile.write('\t]\n')
	outfile.write('}\n')

	outfile.close()

if __name__ == '__main__':
    main()