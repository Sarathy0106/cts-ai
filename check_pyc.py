import os, time

files = [
    ("models/segmentation.py",
     "models/__pycache__/segmentation.cpython-310.pyc"),
    ("tools/segmentation_tool.py",
     "tools/__pycache__/segmentation_tool.cpython-310.pyc"),
]

for src, pyc in files:
    src_mtime = os.path.getmtime(src) if os.path.exists(src) else None
    pyc_mtime = os.path.getmtime(pyc) if os.path.exists(pyc) else None
    print(f"Source : {src}")
    print(f"  mtime: {time.ctime(src_mtime) if src_mtime else 'NOT FOUND'}")
    print(f"Cache  : {pyc}")
    print(f"  mtime: {time.ctime(pyc_mtime) if pyc_mtime else 'NOT FOUND'}")
    if src_mtime and pyc_mtime:
        stale = pyc_mtime < src_mtime
        print(f"  Stale pyc (pyc older than src): {stale}")
    print()

# Also read the magic number and source hash from the pyc header
import importlib.util, struct

for _, pyc in files:
    if not os.path.exists(pyc):
        continue
    with open(pyc, "rb") as f:
        magic   = f.read(4)
        bitfield = struct.unpack("<I", f.read(4))[0]
        mtime_in_pyc = struct.unpack("<I", f.read(4))[0]
        src_size_in_pyc = struct.unpack("<I", f.read(4))[0]
    print(f"{pyc}:")
    print(f"  mtime embedded in pyc : {time.ctime(mtime_in_pyc)}")
    print(f"  source size in pyc    : {src_size_in_pyc} bytes")
    actual_src = pyc.replace("__pycache__/", "").replace(".cpython-310.pyc", ".py").replace("\\", "/")
    actual_size = os.path.getsize(actual_src) if os.path.exists(actual_src) else "?"
    print(f"  actual source size    : {actual_size} bytes")
    print(f"  Size mismatch (stale) : {actual_size != src_size_in_pyc}")
    print()
