import argparse
import os
from bisect import bisect_left
from random import randint
from itertools import repeat
from itertools import accumulate
from math import sqrt
import json

parser = argparse.ArgumentParser()
parser.add_argument('models', nargs='+', action='store', help='A list of model filenames.')
parser.add_argument('tile_size', action='store', type=float, help='The side length of a tile.')
parser.add_argument('-e', '--edge_tile', action='store', help='Optional. Model filename of an edge tile that shares an origin with a regular tile, and connects to the -Z edge.')
parser.add_argument('-c', '--corner_tile', action='store', help='Optional. Model filename of a corner tile that shares an origin with a regular tile, and connects to the (-X, -Z) corner.')
parser.add_argument('-w', '--weights', nargs='+', action='store', type=float, help='Optional. Weights for each tile for random selection. If specified, number of weights must match number of tiles, otherwise all tiles will be selected with uniform randomness.')
parser.add_argument('-s', '--scene_size', action='store', nargs=2, type=int, default=[1,1], help='Optional. Size of scene to tile, specified in number of tiles in each dimension \"X Z\". Defaults to 1x1.')
parser.add_argument('-r', '--rotate', action='store_true', help='Optional. Randomly rotate tiles around the Y axis in 90 degree steps.')

args = parser.parse_args()

if args.weights is not None and len(args.models) != len(args.weights):
    print('Error: Number of weights does not match number of models.')
    quit()

model_count = len(args.models)
model_names = []
for model in args.models:
    model_names.append(os.path.splitext(model)[0])

# Calculate prefix sum of input weights
if args.weights is not None:
    summed_weights = list(accumulate(args.weights))

# Select tiles for each tile in the scene. Store results in buckets by model
# [x][z] origin top left
selected_tiles = [[] for i in range(len(args.models))]
for x in range(args.scene_size[0]):
    for z in range(args.scene_size[1]):
        if args.weights is None:
            model_id = randint(0, model_count - 1)
        else:
            # Pick a number inclusive between 1 and the cumulative sum of all the weights
            # Find where that number lands in the prefix summed array, that index is the selected tile
            model_id = bisect_left(summed_weights, randint(1, summed_weights[-1]))

        selected_tiles[model_id].append((x, z))

# Some scene size calculations
world_scene_dims = [args.scene_size[0] * args.tile_size,
                    args.scene_size[1] * args.tile_size]

scene_diag_dist = sqrt(world_scene_dims[0] ** 2 + world_scene_dims[1] ** 2)

# Start building JSON
fscene_json = {}

# Top level data
fscene_json['version'] = 2
fscene_json['camera_speed'] = (scene_diag_dist / 2.0) * 0.03 # Calculation taken from FeatureDemo
fscene_json['lighting_scale'] = 1.0
fscene_json['active_camera'] = "Default"
fscene_json['ambient_intensity'] = [0.0, 0.0, 0.0]

default_camera = {}
default_camera['name'] = fscene_json['active_camera']
default_camera['pos'] = [0.0, args.tile_size * 0.5, 0.0]
default_camera['target'] = [0.0, default_camera['pos'][1], -0.5]
default_camera['focal_length'] = 21.0
default_camera['aspect_ratio'] = 16.0 / 9.0
z_far = scene_diag_dist * 1.25
default_camera['depth_range'] = [max(0.0001, z_far / 10000.0), z_far]
fscene_json['cameras'] = [default_camera]

fscene_json['models'] = []

# Helper to fill tiles from a series of coordinates
def fill_tiles(x_coords, z_coords, y_rotations, name, instances):
    for x, z, y_rot in zip(x_coords, z_coords, y_rotations):
        x_pos = world_origin_x + args.tile_size * x
        z_pos = world_origin_z + args.tile_size * z

        edge_instance = {}
        edge_instance['name'] = name + '_' + str(len(instances)) + '_' + str(x) + '_' + str(z)
        edge_instance['translation'] = [x_pos, 0.0, z_pos]
        edge_instance['rotation'] = [y_rot, 0.0, 0.0] # Yaw, Pitch Roll = Y, X, Z
        edge_instance['scaling'] = [1.0, 1.0, 1.0]
        instances.append(edge_instance)

def rand_rot_gen():
    while True:
        yield 90.0 * float(randint(0, 3))

# Write grid tiles
world_origin_x = -((float(args.scene_size[0]) / 2) - 0.5) * args.tile_size
world_origin_z = -((float(args.scene_size[1]) / 2) - 0.5) * args.tile_size

for i, model_name in enumerate(model_names):

    if len(selected_tiles[i]) == 0:
        continue

    model_json_data = {}
    model_json_data['file'] = args.models[i]
    model_json_data['name'] = model_name
    model_json_data['instances'] = []

    if args.rotate:
        y_rot_gen = rand_rot_gen()
    else:
        y_rot_gen = repeat(0.0)

    # Split list of coordinates into lists of x and z components and generate tiles
    fill_tiles(*zip(*selected_tiles[i]), y_rot_gen, model_name, model_json_data['instances'])
    fscene_json['models'].append(model_json_data)

# Write edge and corner pieces if specified in arguments

if args.corner_tile is not None:
    corner_model_data = {}
    corner_model_data['file'] = args.corner_tile
    corner_model_data['name'] = os.path.splitext(args.corner_tile)[0]
    corner_model_data['instances'] = []

    name = corner_model_data['name']
    instances = corner_model_data['instances']

    # Create list of corner coordinates
    # [Top Left, Bottom Left, Bottom Right, Top Right]
    x_pos = ([0] * 2) + ([args.scene_size[0] - 1] * 2)
    z_pos = [0] + ([args.scene_size[1] - 1] * 2) + [0]
    y_rot = [90.0 * i for i in range(0, 4)]
    fill_tiles(x_pos, z_pos, y_rot, name, instances)

    fscene_json['models'].append(corner_model_data)

if args.edge_tile is not None:
    edge_model_data = {}
    edge_model_data['file'] = args.edge_tile
    edge_model_data['name'] = os.path.splitext(args.edge_tile)[0]
    edge_model_data['instances'] = []

    name = edge_model_data['name']
    instances = edge_model_data['instances']

    # Top Edge
    fill_tiles(range(0, args.scene_size[0]), repeat(0), repeat(0.0), name, instances)
    # Left Edge
    fill_tiles(repeat(0), range(0, args.scene_size[1]), repeat(90.0), name, instances)
    # Bottom Edge
    fill_tiles(range(0, args.scene_size[0]), repeat(args.scene_size[1] - 1), repeat(180.0), name, instances)
    # Right Edge
    fill_tiles(repeat(args.scene_size[0] - 1), range(0, args.scene_size[1]), repeat(270.0), name, instances)

    fscene_json['models'].append(edge_model_data)

print (json.dumps(fscene_json, indent=4))
