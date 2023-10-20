import sys
import time
import falcor
import argparse
import numpy as np

from PIL import Image
from pathlib import Path

# Create GPU device
device = falcor.Device()

# Build and parse command line
parser = argparse.ArgumentParser(description="Slang-based BC7 - mode 6 compressor")
parser.add_argument("input_path", help="Path to the input texture.")
parser.add_argument("-o", "--output_path", help="Optional path to save the decoded BC7 texture.")
parser.add_argument("-s", "--opt_steps", type=int, default=100, help="Number of optimization (gradient descene) steps.")
parser.add_argument("-b", "--benchmark", action="store_true", help="Run in benchmark mode to measure processing time.")
args = parser.parse_args()

# Load texture
try:
    img = Image.open(args.input_path).convert("RGBA")
    texture = np.asarray(img).astype(np.float32) / 255.0
    texture = np.clip(texture, 0, 1)
    w, h, _ = texture.shape
    print(f"\nTexture dimensions: {w}x{h}")
except Exception as e:
    print(f"\nError loading the texture: {e}")
    sys.exit(1)

input_tex = device.create_texture(
    format=falcor.ResourceFormat.RGBA32Float,
    width=w, height=h, mip_levels=1,
    bind_flags=falcor.ResourceBindFlags.ShaderResource)
input_tex.from_numpy(texture)

# Create output texture
decoded_tex = device.create_texture(
    format=falcor.ResourceFormat.RGBA32Float,
    width=w, height=h, mip_levels=1,
    bind_flags=falcor.ResourceBindFlags.UnorderedAccess,
)

# Create a compute pass for our BC7-mode6 texture encoder
encoder = falcor.ComputePass(
    device,
    file=Path(__file__).parent / "TinyBC.cs.slang", cs_entry="encoder",
    defines={"USE_ADAM": "true", "NUM_OPTIMIZATION_STEPS": str(args.opt_steps)}
)

# Set constants for the compute shader
encoder.globals.gInputTex = input_tex
encoder.globals.gDecodedTex = decoded_tex
encoder.globals.lr = 0.1
encoder.globals.adamBeta1 = 0.9
encoder.globals.adamBeta2 = 0.999
encoder.globals.textureDim = falcor.int2(w, h)

# When running in benchmark mode amortize overheads over many runs to measure more accurate GPU times
num_iters = 1000 if args.benchmark else 1

# Compress!
start_time = time.time()
for i in range(num_iters):
    # Compress input texture using BC7 mode 6, and output decompressed result
    encoder.execute(threads_x=w // 4, threads_y=h // 4)

# Calculate and print performance metrics
if args.benchmark:
    device.render_context.submit(True)
    comp_time_in_sec = (time.time() - start_time) / num_iters
    textures_per_sec = 1 / comp_time_in_sec
    giga_texels_per_sec = w * h * textures_per_sec / 1E9
    print(f"\nBenchmark mode:")
    print(f"  - Number of optimization passes: {args.opt_steps}")
    print(f"  - Compression time: {1E3 * comp_time_in_sec:.4g} ms --> {giga_texels_per_sec:.4g} GTexels/s")

# Calculate and print PSNR
decoded_texture = decoded_tex.to_numpy()
mse = np.mean((input_tex.to_numpy() - decoded_tex.to_numpy()) ** 2)
psnr = 20 * np.log10(1.0 / np.sqrt(mse))
print(f"\nPSNR: {psnr:.4g}")

# Output decoded texture
if args.output_path:
    img = Image.fromarray((255 * decoded_texture).astype(np.uint8), 'RGBA')
    img.save(args.output_path)
