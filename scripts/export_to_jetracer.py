import argparse
import io
import json
import zipfile

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sac", required=True)
    p.add_argument("--vae", required=True)
    p.add_argument("--out", default="jetracer_policy.npz")
    args = p.parse_args()

    out = {}

    # ---- SAC actor ----
    z = zipfile.ZipFile(args.sac)
    sd = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu")
    data = json.loads(z.read("data").decode())
    linear_ids = sorted({int(k.split(".")[2]) for k in sd if k.startswith("actor.latent_pi.")})
    for i, idx in enumerate(linear_ids):
        out["actor_l%d_w" % i] = sd["actor.latent_pi.%d.weight" % idx].numpy()
        out["actor_l%d_b" % i] = sd["actor.latent_pi.%d.bias" % idx].numpy()
    out["actor_mu_w"] = sd["actor.mu.weight"].numpy()
    out["actor_mu_b"] = sd["actor.mu.bias"].numpy()

    def parse_vec(s):
        return np.array([float(x) for x in s.strip("[]").split()], dtype=np.float32)

    out["act_low"] = parse_vec(data["action_space"]["low"])
    out["act_high"] = parse_vec(data["action_space"]["high"])

    ck = torch.load(args.vae, map_location="cpu")
    vsd = ck["state_dict"]
    out["latent_dim"] = np.array(ck["latent_dim"])
    for k, v in vsd.items():
        if k.startswith("encoder.fc_logvar"):
            continue
        out["vae." + k] = v.numpy()

    rng = np.random.RandomState(0)
    ref_latent = rng.randn(4, int(ck["latent_dim"])).astype(np.float32)
    try:
        from stable_baselines3 import SAC
        model = SAC.load(args.sac, device="cpu")
        ref_action, _ = model.predict(ref_latent, deterministic=True)
        out["ref_latent"] = ref_latent
        out["ref_action"] = ref_action.astype(np.float32)
        print("reference actions from SB3:\n", ref_action)
    except Exception as e:  
        print("WARNING: could not load with SB3 for the self-test reference (%s)" % e)

    np.savez(args.out, **out)
    n_params = sum(v.size for k, v in out.items() if k.startswith(("actor_", "vae.encoder")))
    print("wrote %s  (encoder+actor params used for driving: %d)" % (args.out, n_params))
    print("action low/high:", out["act_low"], out["act_high"])


if __name__ == "__main__":
    main()
