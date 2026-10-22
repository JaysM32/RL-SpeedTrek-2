#!/usr/bin/env python3
"""Run ON THE JETRACER. Drives the car with the VAE + SAC policy exported by export_to_jetracer.py.

Written for the car's software: Python 3.6, torch 1.6 (CUDA), OpenCV 4.1, numpy, and the
`jetracer` package. Does NOT need Stable-Baselines3.

Pipeline (identical to training in envs/donkey_env.py):
    camera frame (BGR) -> RGB -> resize whole frame to 160x80 -> /255 -> CHW
      -> VAE encoder mean (32 numbers) -> SAC actor -> tanh -> scale to action space
      -> [steering in -1..1, throttle in 0..1] -> throttle capped by --max-throttle -> motors

Modes, safest first -- do them in this order:
    python3 drive.py --selftest            no camera, no motors: checks the maths matches SB3
    python3 drive.py --check               camera only: saves what the VAE sees, times inference
    python3 drive.py --dry-run             camera + policy, prints actions, motors never touched
    python3 drive.py --max-throttle 0.15   actually drives (wheels off the ground first!)

Stop at any time with Ctrl+C -- throttle and steering are zeroed on every exit path.
"""

from __future__ import print_function

import argparse
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_H, IMG_W = 80, 160  # configs/sac_vae.yaml env.image_size


# --------------------------------------------------------------------------- model
class Policy(object):
    def __init__(self, path, device="auto"):
        import torch
        import torch.nn as nn

        self.torch = torch
        w = np.load(path)
        self.latent_dim = int(w["latent_dim"])
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        def conv(i, o):
            return nn.Conv2d(i, o, kernel_size=4, stride=2, padding=1)

        def deconv(i, o):
            return nn.ConvTranspose2d(i, o, kernel_size=4, stride=2, padding=1)

        # Same layers as vae/model.py, rebuilt here so no pickled classes are needed.
        self.enc_conv = nn.Sequential(conv(3, 32), nn.ReLU(), conv(32, 64), nn.ReLU(),
                                      conv(64, 128), nn.ReLU(), conv(128, 256), nn.ReLU())
        self.enc_mu = nn.Linear(256 * 5 * 10, self.latent_dim)
        self._load(self.enc_conv, w, "vae.encoder.conv.")
        self._load(self.enc_mu, w, "vae.encoder.fc_mu.")
        self.enc_conv.to(self.device).eval()
        self.enc_mu.to(self.device).eval()

        self.dec_fc = None
        if "vae.decoder.fc.weight" in w.files:
            self.dec_fc = nn.Linear(self.latent_dim, 256 * 5 * 10)
            self.dec = nn.Sequential(deconv(256, 128), nn.ReLU(), deconv(128, 64), nn.ReLU(),
                                     deconv(64, 32), nn.ReLU(), deconv(32, 3), nn.Sigmoid())
            self._load(self.dec_fc, w, "vae.decoder.fc.")
            self._load(self.dec, w, "vae.decoder.deconv.")
            self.dec_fc.to(self.device).eval()
            self.dec.to(self.device).eval()

        # SAC actor in plain numpy (tiny MLP, faster than a GPU round-trip)
        self.layers = []
        i = 0
        while "actor_l%d_w" % i in w.files:
            self.layers.append((w["actor_l%d_w" % i], w["actor_l%d_b" % i]))
            i += 1
        self.mu_w, self.mu_b = w["actor_mu_w"], w["actor_mu_b"]
        self.low, self.high = w["act_low"], w["act_high"]
        self.ref = (w["ref_latent"], w["ref_action"]) if "ref_latent" in w.files else None

    def _load(self, module, w, prefix):
        import torch
        sd = {}
        for k in module.state_dict().keys():
            sd[k] = torch.from_numpy(np.ascontiguousarray(w[prefix + k]))
        module.load_state_dict(sd)

    def preprocess(self, frame_bgr):
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        return rgb  # uint8 HxWx3, exactly what the sim gave the VAE wrapper

    def encode(self, rgb_u8):
        x = rgb_u8.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None]
        with self.torch.no_grad():
            t = self.torch.from_numpy(np.ascontiguousarray(x)).to(self.device)
            mu = self.enc_mu(self.enc_conv(t).flatten(1))
        return mu.cpu().numpy()[0]

    def decode(self, latent):
        with self.torch.no_grad():
            z = self.torch.from_numpy(latent[None].astype(np.float32)).to(self.device)
            img = self.dec(self.dec_fc(z).view(-1, 256, 5, 10))
        return (img.cpu().numpy()[0].transpose(1, 2, 0) * 255).astype(np.uint8)

    def act(self, latent):
        """Same as SB3 SAC.predict(obs, deterministic=True)."""
        h = np.asarray(latent, dtype=np.float32)
        for W, b in self.layers:
            h = np.maximum(h.dot(W.T) + b, 0.0)
        a = np.tanh(h.dot(self.mu_w.T) + self.mu_b)
        return self.low + 0.5 * (a + 1.0) * (self.high - self.low)


# --------------------------------------------------------------------------- camera
def gst_pipeline(cap_w, cap_h, fps, out_w, out_h, flip):
    return ("nvarguscamerasrc sensor-id=0 ! "
            "video/x-raw(memory:NVMM), width=%d, height=%d, framerate=%d/1 ! "
            "nvvidconv flip-method=%d ! video/x-raw, width=%d, height=%d, format=BGRx ! "
            "videoconvert ! video/x-raw, format=BGR ! appsink drop=true max-buffers=1 sync=false"
            % (cap_w, cap_h, fps, flip, out_w, out_h))


class Camera(object):
    def __init__(self, flip=0):
        import cv2
        self.cv2 = cv2
        # Sensor is 3264x2464 (4:3); 1640x1232 is its full-FOV binned mode. nvvidconv shrinks
        # on the GPU so the CPU only ever sees a small frame.
        self.cap = cv2.VideoCapture(gst_pipeline(1640, 1232, 30, 320, 240, flip), cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open CSI camera. Try: sudo systemctl restart nvargus-daemon "
                               "(and make sure no notebook is holding the camera)")
        for _ in range(5):  # let auto-exposure settle
            self.cap.read()

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        self.cap.release()


# --------------------------------------------------------------------------- modes
def selftest(policy):
    print("device for VAE encoder:", policy.device)
    if policy.ref is None:
        print("no reference actions in the .npz (SB3 wasn't available at export) -- skipping")
    else:
        lat, ref = policy.ref
        got = np.stack([policy.act(l) for l in lat])
        err = float(np.abs(got - ref).max())
        print("actor vs SB3 predict(): max abs difference = %.2e" % err)
        print("PASS" if err < 1e-4 else "FAIL -- actor maths does not match training")
    dummy = (np.random.rand(IMG_H, IMG_W, 3) * 255).astype(np.uint8)
    for _ in range(5):
        policy.encode(dummy)
    t0 = time.time()
    n = 50
    for _ in range(n):
        policy.act(policy.encode(dummy))
    print("encoder + actor: %.1f ms per frame" % ((time.time() - t0) / n * 1000))


def check(policy, cam, n=5):
    import cv2
    out_dir = os.path.join(HERE, "check_frames")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    for i in range(n):
        frame = cam.read()
        if frame is None:
            print("camera returned no frame")
            continue
        t0 = time.time()
        rgb = policy.preprocess(frame)
        lat = policy.encode(rgb)
        a = policy.act(lat)
        ms = (time.time() - t0) * 1000
        print("frame %d: steering=%+.3f throttle=%.3f  latent|max|=%.2f  %.1f ms"
              % (i, a[0], a[1], np.abs(lat).max(), ms))
        panels = [cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)]
        if policy.dec_fc is not None:
            panels.append(cv2.cvtColor(policy.decode(lat), cv2.COLOR_RGB2BGR))
        big = cv2.resize(np.hstack(panels), None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(out_dir, "frame_%d.png" % i), big)
        time.sleep(0.5)
    print("saved to %s  (left: what the VAE gets, right: its reconstruction)" % out_dir)
    print("If the right side looks nothing like a track, the VAE doesn't recognise the real "
          "scene -- expect poor driving until it's fine-tuned on real frames.")


def drive(policy, cam, args):
    car = None
    if not args.dry_run:
        from jetracer.nvidia_racecar import NvidiaRacecar
        car = NvidiaRacecar()
        if args.steering_gain is not None:
            car.steering_gain = args.steering_gain
        if args.steering_offset is not None:
            car.steering_offset = args.steering_offset
        car.throttle = 0.0
        car.steering = 0.0
        print("steering_gain=%.3f steering_offset=%.3f throttle_gain=%.3f"
              % (car.steering_gain, car.steering_offset, car.throttle_gain))
        print("Starting in 3 s -- Ctrl+C to stop.")
        time.sleep(3)

    log_path = os.path.join(HERE, "drive_log_%s.csv" % time.strftime("%Y%m%d_%H%M%S"))
    f = open(log_path, "w")
    log = csv.writer(f)
    log.writerow(["t", "steering_raw", "throttle_raw", "steering_cmd", "throttle_cmd", "infer_ms"])

    period = 1.0 / args.hz
    t_start = time.time()
    last_frame_t = time.time()
    n = 0
    try:
        while True:
            loop_t = time.time()
            if args.duration and loop_t - t_start > args.duration:
                print("duration reached")
                break
            frame = cam.read()
            if frame is None:
                if time.time() - last_frame_t > 0.3:  # watchdog: no picture -> stop
                    print("camera stalled for >0.3 s -- stopping")
                    break
                if car:
                    car.throttle = 0.0
                continue
            last_frame_t = time.time()

            lat = policy.encode(policy.preprocess(frame))
            steer_raw, thr_raw = [float(v) for v in policy.act(lat)]
            infer_ms = (time.time() - loop_t) * 1000

            steer = -steer_raw if args.invert_steering else steer_raw
            thr = min(max(thr_raw, 0.0), 1.0) * args.max_throttle
            if args.reverse_throttle:
                thr = -thr

            if car:
                car.steering = float(np.clip(steer, -1, 1))
                car.throttle = float(thr)
            log.writerow(["%.3f" % (loop_t - t_start), "%.4f" % steer_raw, "%.4f" % thr_raw,
                          "%.4f" % steer, "%.4f" % thr, "%.1f" % infer_ms])
            n += 1
            if n % max(1, int(args.hz)) == 0 or args.dry_run:
                print("steer=%+.2f thr=%.2f (raw %.2f)  %.0f ms" % (steer, thr, thr_raw, infer_ms))

            sleep = period - (time.time() - loop_t)
            if sleep > 0:
                time.sleep(sleep)
    except KeyboardInterrupt:
        print("\nCtrl+C")
    finally:
        if car:
            car.throttle = 0.0
            car.steering = 0.0
        f.close()
        print("stopped. %d steps, log: %s" % (n, log_path))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.path.join(HERE, "jetracer_policy.npz"))
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--check", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-throttle", type=float, default=0.15,
                   help="policy throttle (0..1) is multiplied by this. Start low.")
    p.add_argument("--hz", type=float, default=20.0, help="control loop rate cap")
    p.add_argument("--duration", type=float, default=0, help="auto-stop after N seconds (0 = off)")
    p.add_argument("--steering-gain", type=float, default=None, help="override car.steering_gain")
    p.add_argument("--steering-offset", type=float, default=None, help="override car.steering_offset")
    p.add_argument("--invert-steering", action="store_true")
    p.add_argument("--reverse-throttle", action="store_true",
                   help="use if positive throttle drives the car backwards")
    p.add_argument("--flip", type=int, default=0, help="camera flip-method (2 = rotate 180)")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    if args.max_throttle > 0.5:
        print("refusing --max-throttle > 0.5 for safety; edit the script if you really mean it")
        sys.exit(1)

    policy = Policy(args.model, args.device)
    if args.selftest:
        selftest(policy)
        return

    cam = Camera(flip=args.flip)
    try:
        if args.check:
            check(policy, cam)
        else:
            drive(policy, cam, args)
    finally:
        cam.release()


if __name__ == "__main__":
    main()
