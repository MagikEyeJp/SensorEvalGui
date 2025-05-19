# scripts/gen_dummy_stack.py
import numpy as np, tifffile, pathlib, argparse

parser = argparse.ArgumentParser()
parser.add_argument("outdir", type=pathlib.Path)
parser.add_argument("-n", "--num", type=int, default=4)
args = parser.parse_args()

args.outdir.mkdir(parents=True, exist_ok=True)
for i in range(args.num):
    img = np.full((120, 160), i * 2000, np.uint16)  # simple gradient
    tifffile.imwrite(args.outdir / f"{i}.tiff", img)
print("Dummy stack written:", args.outdir)
