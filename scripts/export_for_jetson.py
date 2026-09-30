"""Placeholder / design notes for exporting the trained policy to JetRacer.

This is intentionally NOT a finished script -- the right export path depends on hardware
specifics (Jetson model, JetPack/TensorRT version, camera driver) you won't know until
you have the physical car in front of you. Fill this in then. What follows is the plan.

1. Export the VAE encoder and SAC policy as TorchScript (torch.jit.trace) or ONNX
   (torch.onnx.export). The SAC policy's `actor.forward` (deterministic action, no
   sampling) is what you want at deployment, not the full SAC object with its critics/
   replay buffer -- those are training-only.

2. On the Jetson, convert ONNX -> TensorRT for faster inference if needed. For a small
   MLP policy head + a 4-layer conv VAE encoder, plain PyTorch/ONNX Runtime inference may
   already be fast enough (single-digit milliseconds) -- profile before adding the TensorRT
   step, it's extra complexity you may not need.

3. Camera calibration: JetRacer's CSI camera has a different FOV/resolution/lens distortion
   than the Unity sim's virtual camera. At minimum, resize/crop real frames to match the
   aspect ratio the VAE was trained on, and expect reconstruction quality to be visibly worse
   on real frames until you either fine-tune the VAE on real images or apply domain
   randomization during sim training (randomized lighting/textures/camera noise).

4. Action calibration: map the policy's [-1, 1] steering/throttle outputs to your JetRacer's
   actual PWM ranges. Steering will likely need a trim offset (mechanical zero point rarely
   matches "PWM midpoint"), and throttle should be clipped to a conservative max speed for
   initial tests. Do this mapping in one place (e.g. a `JetRacerActuator` class) so it's easy
   to adjust without touching the policy code.

5. Safety: implement a hardware or RC-transmitter kill switch that overrides the policy
   before the first autonomous run. Start in a small, obstacle-free, low-speed test area.

6. If sim-trained performance doesn't transfer, the standard next step is a short phase of
   on-car fine-tuning: collect a small amount of real driving data (teleop or early policy
   rollouts), and either fine-tune the VAE on it, fine-tune the SAC policy with a few
   real-world episodes (this is exactly the "minutes" of real training in Raffin's original
   project), or both.
"""

if __name__ == "__main__":
    print(__doc__)
