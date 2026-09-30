# DonkeyCar RL Simulation

Starter code for training an RL driving agent in the `gym-donkeycar` simulator, structured
so the trained policy is a good fit for later deployment on a JetRacer AI Kit car.
See `PROJECT_PLAN.md` (shipped alongside this zip) for the reasoning behind the approach.

**This code will not run inside a cloud sandbox.** The DonkeyCar simulator is a Unity app
with a GUI window; run everything here on your own machine (a GPU is recommended for
Stage 2/3 but not required to get started).

## 1. Install the simulator binary

Download the prebuilt DonkeyCar simulator for your OS from the gym-donkeycar releases page:
https://github.com/tawnkramer/gym-donkeycar/releases

Unzip it somewhere and note the path to the executable
(e.g. `donkey_sim.exe`, `donkey_sim.x86_64`, or the `.app` on macOS) — you'll pass it as
`--sim-path` / set it in `configs/sac_vae.yaml`.

## 2. Python environment

```bash
python3 -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

`gym-donkeycar` itself isn't on PyPI under a stable release in all versions — if
`pip install gym-donkeycar` fails, install it from source instead:

```bash
git clone https://github.com/tawnkramer/gym-donkeycar
pip install -e gym-donkeycar
```

## 3. Smoke test

Launch the simulator binary once by hand to confirm it opens, then run:

```bash
python -m envs.donkey_env --sim-path /path/to/donkey_sim --steps 200
```

This drives with random actions for 200 steps and prints reward/done — if the Unity window
opens and the car moves (even badly), the env wiring is correct.

## 4. Collect frames and train the VAE


python -m vae.collect_frames --sim-path "/Users/jaysm/Desktop/donkey_sim.app/Contents/MacOS/donkey_sim" --out data/frames --n-frames 5000 --env-id donkey-circuit-launch-track-v0


```bash
python -m vae.collect_frames --sim-path /path/to/donkey_sim --out data/frames --n-frames 5000
python -m vae.train_vae --data data/frames --out checkpoints/vae.pt --latent-dim 32
```

Check `checkpoints/vae_reconstructions.png` after training (written automatically) —
you want recognizable-but-blurry track edges, not noise.

## 5. Train SAC in the VAE latent space

```bash
python -m agents.train_sac \
    --sim-path "/Users/jaysm/Desktop/donkey_sim.app/Contents/MacOS/donkey_sim" \
    --vae-checkpoint checkpoints/vae.pt \
    --config configs/sac_vae.yaml
```

## 5.1 train from checkpoint

python -m agents.train_sac \
    --sim-path "/Users/jaysm/Desktop/donkey_sim.app/Contents/MacOS/donkey_sim" \
    --vae-checkpoint checkpoints/vae.pt \
    --config configs/sac_vae.yaml \
    --resume auto

TensorBoard logs go to `runs/`; `tensorboard --logdir runs` to watch reward curves live.

## 5.2 check transitions (states, actions, rewards)

python -m scripts.inspect_transitions --sim-path "/Users/jaysm/Desktop/donkey_sim.app/Contents/MacOS/donkey_sim" --vae-checkpoint checkpoints/vae.pt --steps 300


## 6. (Optional) PPO + CNN baseline for comparison

```bash
python -m agents.train_ppo_baseline --sim-path /path/to/donkey_sim
```

## 7. Evaluate a trained policy

```bash
python -m scripts.evaluate --sim-path /path/to/donkey_sim --model checkpoints/sac_donkey.zip \
    --vae-checkpoint checkpoints/vae.pt --episodes 10
```

python -m scripts.time_trial --sim-path "/Users/jaysm/Desktop/donkey_sim.app/Contents/MacOS/donkey_sim" --model checkpoints/sac_donkey.zip --vae-checkpoint checkpoints/vae.pt --laps 3 --attempts 5

## Repo layout

```
configs/        hyperparameter/config files
envs/           gym-donkeycar wrapper (obs preprocessing, VAE-latent observation wrapper)
vae/            VAE model, frame collection, VAE training
agents/         SAC (main) and PPO+CNN (baseline) training scripts
utils/          reward shaping, SB3 callbacks
scripts/        evaluation and the JetRacer export placeholder
```

## Notes on moving to JetRacer later

`scripts/export_for_jetson.py` is a placeholder that documents (rather than fully
implements) the export path: TorchScript/ONNX export of the VAE encoder + SAC policy,
notes on TensorRT conversion, and the calibration steps you'll need for JetRacer's real
camera and motor controller. Fill it in once you have the physical car in hand — trying to
build it blind now would just be guessing at hardware specifics.
