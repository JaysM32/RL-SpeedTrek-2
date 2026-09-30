# DonkeyCar RL → JetRacer: Project Roadmap

## Goal

Train a reinforcement learning agent to drive around a track in the [gym-donkeycar](https://github.com/tawnkramer/gym-donkeycar) simulator, using an architecture that is deliberately chosen so the trained policy can later be carried over to a physical JetRacer AI Kit car (Jetson Nano/Orin, CSI camera, differential steering/throttle via PWM). The two goals — "get something training in sim" and "make it portable to real hardware" — pull in different directions if you're not careful, so the algorithm choice below is picked with the second goal in mind from day one rather than as an afterthought.

## Why gym-donkeycar needs to run on your own machine

The DonkeyCar simulator is a Unity application with a graphical window; gym-donkeycar is a thin Gym/Gymnasium wrapper that talks to it over a local socket. Neither piece can run inside this cloud sandbox, which is headless and has no GPU display. Everything in this plan assumes you're running the simulator and training loop on your own laptop or a machine with a GPU (a GPU is strongly recommended for the image-based training in Stage 2, though Stage 1 setup and the VAE/agent code will run, just slowly, on CPU too).

## Recommended approach: VAE + SAC, not end-to-end CNN+PPO

There are two common ways people apply RL to DonkeyCar:

The simpler route trains an on-policy algorithm like PPO directly on raw camera frames using a convolutional policy network (this is what Stable-Baselines3's `CnnPolicy` gives you out of the box). It's easy to set up but needs a lot of simulated driving to converge, and the resulting policy is a fairly large CNN that has to run in real time on the car.

The route this plan recommends instead — and the one your code scaffold is built around — separates *seeing* from *driving*. A variational autoencoder (VAE) is trained first, unsupervised, to compress camera frames (e.g. 80×160×3) down to a small latent vector (32–64 dimensions). A much smaller RL agent (SAC) then learns to drive using that latent vector as its observation instead of raw pixels. This is essentially the approach Antonin Raffin used in his "Learning to Drive Smoothly in Minutes" DonkeyCar project, which trained on a real RC car in about 5–20 minutes of driving time and is one of the few published RL-on-DonkeyCar projects that was actually validated on physical hardware rather than just in sim.

This matters for your JetRacer goal in three concrete ways. Sample efficiency: SAC is off-policy and reuses data via a replay buffer, so it needs far fewer environment steps than PPO to reach a good policy — important once you eventually want to fine-tune on the real car, where every second of driving costs a battery and a human supervisor. Inference cost: at deployment time you only need the VAE encoder (a small CNN, run once per frame) plus a small MLP policy head, which is far cheaper than a full end-to-end CNN policy and comfortably real-time on a Jetson Nano. Debuggability: because the VAE is trained separately, you can visually inspect its reconstructions to see whether the model is actually "looking at" the track, and you can reuse the same VAE for a completely different downstream controller (even a classical one) if the RL part doesn't work out.

The scaffold also includes a PPO+CNN baseline script so you can compare the two approaches directly if your assignment wants that kind of ablation — it's a smaller amount of extra work once the environment wrapper exists.

## Milestones

| Stage | What you're doing | Rough effort |
|---|---|---|
| 0. Environment setup | Install the Unity DonkeyCar simulator binary, `gym-donkeycar`, Python 3.10+, and the scaffold's `requirements.txt` (PyTorch, Stable-Baselines3, OpenCV). Confirm you can launch the sim and step the env with random actions. | 1 evening |
| 1. Sanity baseline | Drive with a random-action agent, then with a trivial scripted/PID agent, logging episode reward and crashes. This confirms the env, reward, and reset logic all work before any learning is involved. | 1–2 hours |
| 2. Collect frames + train VAE | Drive around (random policy is fine) collecting a few thousand camera frames, then train the VAE to reconstruct them. Check reconstructions visually — blurry-but-recognizable track edges are fine, unrecognizable output means something's wrong upstream. | 1 evening |
| 3. Train SAC in latent space | Wrap the env so observations are VAE latent vectors (optionally concatenated with speed), define the reward (track progress, centering, penalize off-track), and train SAC via Stable-Baselines3. Watch reward curves and periodically render an episode to sanity-check behavior. | Several sessions, iterate on reward |
| 4. Evaluate & tune | Formal evaluation episodes (mean reward, laps completed, crash rate) across a couple of tracks if available. Tune reward shaping, VAE latent size, SAC hyperparameters. | Ongoing |
| 5. (Optional) PPO+CNN baseline | Run the baseline script for comparison — useful if your assignment wants you to justify the algorithm choice empirically rather than just by argument. | 1 evening |
| 6. Sim-to-real prep for JetRacer | See below — this is mostly planning and small code changes now, executed later once you have the physical car. | Ongoing background thought |

## Reward design

A reasonable starting reward, and the one wired up as a placeholder in `utils/reward.py`, combines three terms: a positive term proportional to forward progress along the track centerline (or simply forward speed while roughly centered, if you don't have centerline telemetry from the sim), a penalty proportional to cross-track error (distance from the lane center), and a large negative terminal penalty plus episode end when the car goes off-track or flips. Getting this right takes iteration — if the agent seems to "give up" and drive slowly in circles, the progress term is usually too weak relative to the safety penalty, and vice versa if it's reckless.

## Sim-to-real: what will actually break on JetRacer

This is worth thinking about now even though you won't touch the physical car until later. The Unity simulator's camera has a different field of view, resolution, lens distortion, and lighting model than a real CSI camera in a real room — a policy trained purely in sim will very likely need either (a) domain randomization during sim training (randomize lighting, textures, camera noise) or (b) a period of fine-tuning on real frames, or ideally both. The action space also won't map 1:1: the sim's throttle/steering are normalized floats, but JetRacer's actual PWM duty cycle range, steering trim, and motor deadzone are specific to your hardware and will need calibration. Latency matters more on real hardware than in sim — budget your inference time (VAE encode + policy forward pass) against your control loop rate and camera capture latency; this is exactly the constraint the VAE+SAC choice above is optimizing for. Finally, plan for a safety story before the car ever drives autonomously near anything breakable: a kill switch, a low speed cap for early tests, and a way to hand control back to a human/RC transmitter instantly.

## If this needs to read as a course assignment deliverable

Loosely-scoped course projects still tend to get graded on a few recognizable things: a clear problem statement and why RL (vs. e.g. classical control) is an interesting choice here, a described and justified method (this is where the VAE+SAC vs. PPO+CNN comparison earns its keep), quantitative results (reward curves, success/crash rate, maybe a short video of the trained agent), and an honest discussion of limitations — sim-to-real gap is a legitimate and expected limitation to name rather than something to hide. Worth keeping a running log of what you tried and why as you go, since "we tried X, it didn't work because Y, so we tried Z" is usually a stronger discussion section than a plan that worked on the first try ever produces.

## What's in the code scaffold

The accompanying `donkey-rl-scaffold.zip` has a working directory structure and starter implementations for the environment wrapper, VAE, frame collection, SAC training (latent-space), and the PPO+CNN baseline, plus a README with the local setup steps (installing the simulator binary, `gym-donkeycar`, and Python deps). It's meant to run on your own machine, not in this sandbox — see that README for the exact install commands.
