"""Run this ON THE JETRACER (not in simulation) during your next access window, to figure out
what you're actually working with, before I write the real deployment script against a guess.

Usage (on the Jetson, over SSH or directly):
    python3 jetracer_probe.py

Takes a few seconds, never crashes (every check is wrapped so one failure doesn't block the
rest), and prints a plain-text summary you can paste back into the conversation. Checks:

  1. which motor-control library is actually installed/importable (NVIDIA's jetracer package,
     Adafruit's PCA9685/ServoKit, or neither) -- this determines how the inference script talks
     to the motors, and I don't want to guess
  2. whether there are existing example/demo scripts already on the box that reveal the real
     API and calibration values (steering_gain, throttle_gain, channel numbers) -- often faster
     to read than documentation
  3. what camera interface is available (USB via /dev/video*, or CSI via gstreamer's
     nvarguscamerasrc) -- these need completely different capture code
  4. whether torch/opencv are importable and whether CUDA is actually available for inference
     (Jetson's PyTorch is a special NVIDIA build, not the regular pip one)

Doesn't touch the motors or drive anything -- pure read-only inspection, safe to run with the
car in any state.
"""

import glob
import importlib
import os
import subprocess
import sys


def _try_import(module_name: str):
    try:
        mod = importlib.import_module(module_name)
        return True, mod
    except Exception as e:
        return False, str(e)


def check_motor_libraries():
    print("\n=== motor control libraries ===")
    candidates = [
        "jetracer",
        "jetracer.nvidia_racecar",
        "adafruit_servokit",
        "Adafruit_PCA9685",
        "board",  # CircuitPython board module, usually present alongside adafruit_servokit
    ]
    found_any = False
    for name in candidates:
        ok, result = _try_import(name)
        if ok:
            found_any = True
            print(f"  [FOUND] import {name}  -- OK")
            if name == "jetracer.nvidia_racecar":
                try:
                    cls = result.NvidiaRacecar
                    print(f"          NvidiaRacecar class available: {cls}")
                    print(f"          (has .steering / .throttle properties if this is the standard package)")
                except Exception as e:
                    print(f"          (imported but NvidiaRacecar attribute check failed: {e})")
        else:
            print(f"  [ ]     import {name}  -- not available ({result.splitlines()[0] if result else 'unknown error'})")
    if not found_any:
        print("  No known motor-control library found importable. See the 'existing scripts' "
              "section below -- the kit's own example scripts likely import something not on "
              "this list, which will tell us the real module name.")


def check_existing_scripts():
    print("\n=== searching for existing kit scripts / driver code ===")
    search_roots = [os.path.expanduser("~"), "/opt", "/usr/local", "/home"]
    keywords = ["jetracer", "racecar", "waveshare", "pca9685", "servo", "motor"]
    self_path = os.path.abspath(__file__)
    hits = []
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            # don't wander into huge irrelevant trees -- keep this fast for a limited window
            dirnames[:] = [d for d in dirnames if d not in (
                ".git", "node_modules", "__pycache__", ".cache", "site-packages", "dist-packages"
            )]
            depth = dirpath[len(root):].count(os.sep)
            if depth > 4:
                dirnames[:] = []
                continue
            for fname in filenames:
                if not fname.endswith(".py"):
                    continue
                full_path = os.path.join(dirpath, fname)
                if os.path.abspath(full_path) == self_path:
                    continue  # don't report this probe script as a "found" candidate
                lower = fname.lower()
                if any(kw in lower for kw in keywords):
                    hits.append(full_path)
            for dname in list(dirnames):
                if any(kw in dname.lower() for kw in keywords):
                    hits.append(os.path.join(dirpath, dname) + "/  (directory)")

    if hits:
        print(f"  Found {len(hits)} candidate file(s)/dir(s) -- these likely show the real motor API:")
        for h in hits[:25]:
            print(f"    {h}")
        if len(hits) > 25:
            print(f"    ... and {len(hits) - 25} more")
        print("  Worth `cat`-ing a couple of these and pasting the content back -- example scripts "
              "that actually drive the car are the fastest way to confirm the calibration values.")
    else:
        print("  No obviously-named files found under "
              f"{', '.join(r for r in search_roots if os.path.isdir(r))}.")


def check_camera():
    print("\n=== camera interface ===")
    video_devices = sorted(glob.glob("/dev/video*"))
    if video_devices:
        print(f"  [FOUND] V4L2 device(s): {', '.join(video_devices)}")
        ok, cv2mod = _try_import("cv2")
        if ok:
            try:
                cap = cv2mod.VideoCapture(video_devices[0])
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        print(f"          cv2.VideoCapture({video_devices[0]}) -- grabbed a frame, "
                              f"shape={frame.shape}")
                    else:
                        print(f"          cv2.VideoCapture({video_devices[0]}) opened but couldn't "
                              f"read a frame (camera may need a gstreamer pipeline instead -- see below)")
                else:
                    print(f"          cv2.VideoCapture({video_devices[0]}) did not open")
                cap.release()
            except Exception as e:
                print(f"          cv2 capture attempt failed: {e}")
        else:
            print("  [ ]     cv2 not importable -- can't test frame capture")
    else:
        print("  [ ]     no /dev/video* devices found (normal for a CSI camera accessed purely "
              "via gstreamer, not a problem by itself)")

    # CSI cameras on Jetson often only work through a gstreamer nvarguscamerasrc pipeline,
    # not /dev/video* at all -- check whether that path is even available.
    try:
        result = subprocess.run(
            ["gst-inspect-1.0", "nvarguscamerasrc"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            print("  [FOUND] gstreamer nvarguscamerasrc plugin is available -- if this is a CSI "
                  "camera, capture will likely need a gstreamer pipeline string, not plain cv2.VideoCapture(0)")
        else:
            print("  [ ]     nvarguscamerasrc not available via gst-inspect-1.0")
    except FileNotFoundError:
        print("  [ ]     gst-inspect-1.0 not found on PATH (gstreamer tools not installed, or not on PATH)")
    except Exception as e:
        print(f"  [ ]     gst-inspect-1.0 check failed: {e}")


def check_compute_stack():
    print("\n=== torch / opencv / cuda ===")
    ok, torch = _try_import("torch")
    if ok:
        print(f"  [FOUND] torch {torch.__version__}")
        try:
            print(f"          cuda available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"          device: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            print(f"          cuda check failed: {e}")
    else:
        print(f"  [ ]     torch not importable ({torch.splitlines()[0] if torch else 'unknown error'})")

    ok, cv2mod = _try_import("cv2")
    print(f"  [{'FOUND' if ok else ' '}{'   ' if ok else '  '}] cv2 "
          f"{'(' + cv2mod.__version__ + ')' if ok else '-- not importable'}")

    print(f"\n  python: {sys.version.split()[0]}  ({sys.executable})")


def main():
    print("JetRacer hardware probe -- read-only, does not touch the motors")
    print("=" * 60)
    check_motor_libraries()
    check_existing_scripts()
    check_camera()
    check_compute_stack()
    print("\n" + "=" * 60)
    print("Done. Paste this whole output back into the conversation.")


if __name__ == "__main__":
    main()