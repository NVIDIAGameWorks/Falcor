import re
import glob
import argparse

# List of Python enum types that are replaced with strings.
# The pattern EnumName.Value is replaced with 'Value'.
ENUMS = [
    "CompositeMode",
    "ToneMapOp",
    "ExposureMode",
    "SceneDebuggerMode",
    "TexLODMode",
    "RayConeMode",
    "RayFootprintFilterMode",
    "ColorFormat",
    "MISHeuristic",
    "SchedulingMode",
    "EmissiveLightSamplerType",
    "OptixDenoiserModel",
    "NRDMethod",
    "SamplePattern",
    "FLIPToneMapperType",
    "OutputId",
    "DLSSProfile",
    "DLSSMotionVectorScale",
    "ColorMap",
    "BSDFViewerMode",
    "AccumulatePrecision",
    "AccumulateOverflowMode",
    "TransformNormalPassOp",
    "AdaptiveSamplerMode",
    "AdaptiveSamplerAnimMode",
    "AdaptiveSamplerClampMode",
    "DenoiserModel",
    "ExportPassFormat",
    "ExportPassOp",
    "ExportPassFreq",
    "IOSize",
    "SamplerFilter",
    "AddressMode",
    "ComparisonFunc",
    "SplitHeuristic",
    "SolidAngleBoundMethod",
    "RTXDIMode",
    "RTXDIBiasCorrection",
    "TransmittanceEstimator",
    "DistanceSampler",
    "ResourceFormat",
]

# Dictionary of Python enum values that are replaced with strings.
# This is used to convert enums that have different values in the Python bindings and the strings.
ENUM_MAP = {
    "CullMode.CullNone": "None",
    "CullMode.CullFront": "Front",
    "CullMode.CullBack": "Back",
}

SERIALIZABLE_STRUCTS = [
    "SplitSampleGeneratorOptions",
    "EmissiveUniformSamplerOptions",
    "LightBVHBuilderOptions",
    "LightBVHSamplerOptions",
    "RTXDIOptions",
    "GridVolumeSamplerOptions",
    "ScreenSpaceReSTIROptions",
    "PathTracerParams",
]

def is_render_script(text: str):
    return "from falcor import *" in text


def update_enums(text):
    for e in ENUMS:
        r = re.compile(rf"([^'\"]){e}\.(\w+)")
        if r.findall(text) != []:
            print(f"Replacing '{e}' enum with strings")
            text = r.sub(r"\1'\2'", text)
    for k, v in ENUM_MAP.items():
        r = re.compile(rf"([^'\"]){k}")
        if r.findall(text) != []:
            print(f"Replacing '{k}' enum value with string")
            text = r.sub(rf"\1'{v}'", text)
    return text


RE_ARG = re.compile(r"([a-zA-Z0-9]+)=([a-zA-Z0-9\.\"']+)")

def update_serializable_structs(text):
    for s in SERIALIZABLE_STRUCTS:
        r = re.compile(rf"{s}\(([^)]*)\)")
        for m in r.finditer(text):
            args = m[1]
            args = re.sub(r"([a-zA-Z0-9]+)=", r"'\1': ", args)
            args = "{" + args + "}"
            text = text.replace(m[0], args)

    return text


def run(args):
    files = list(glob.glob(args.path, recursive=True))

    for file in files:
        text = open(file, "r").read()
        if not is_render_script(text):
            if not args.force:
                print(f"Skipping '{file}' which does not seem to be a render script.")
                continue
        print(f"Checking file '{file}' ...")
        original = text
        text = update_enums(text)
        text = update_serializable_structs(text)
        if not args.dry_run and text != original:
            print(f"Writing file '{file}'")
            open(file, "w").write(text)


parser = argparse.ArgumentParser(description="Utility for fixing render scripts")
parser.add_argument("path", type=str, help="Glob pattern for searching files")
parser.add_argument(
    "-f",
    "--force",
    action="store_true",
    default=False,
    help="Force updating files even if not detected as render script",
)
parser.add_argument(
    "-d",
    "--dry-run",
    action="store_true",
    default=False,
    help="Run without writing files",
)

args = parser.parse_args()
run(args)
