# Mirage Technical Note

## Thesis

Mirage tests whether a small continuous-control agent can improve by learning an internal simulator and training inside imagined futures. The implementation is deliberately modest: it favors inspectability, reproducibility, and local execution over benchmark-scale performance.

## Environment

The primary task is continuous rocket landing. The wrapper tries to use Gymnasium `LunarLander-v3` with `continuous=True`. If Box2D is unavailable, Mirage falls back to a deterministic NumPy lander with the same state/action contract:

- observation: 8D vector `(x, y, vx, vy, angle, angular_velocity, left_contact, right_contact)`;
- action: 2D vector in `[-1, 1]`, interpreted as main thrust and side thrust;
- reward: shaped landing objective with distance, velocity, angle, fuel, landing, and crash terms.

The fallback is intentional because local Box2D installation can be brittle. It keeps the research loop runnable on a clean Mac while preserving the continuous-control structure.

## Dreamer-Lite Model

The world model has four parts:

- encoder: maps physics state to latent state;
- recurrent dynamics: predicts the next latent from current latent and action;
- decoder: reconstructs next physics state;
- reward and continuation heads: predict reward and non-terminal probability.

For a replay sequence, Mirage optimizes:

```text
L = MSE(decoded_next_state, next_state)
  + MSE(predicted_reward, reward)
  + 0.2 * BCE(predicted_continuation, 1 - done)
  + latent_regularization
```

The policy is trained inside imagined rollouts. Starting from replay observations, the actor samples actions, the world model advances the latent state, and the reward head supplies imagined rewards. The critic learns discounted imagined returns. The actor maximizes predicted return with a small entropy term.

## Continual Runner

Each cycle does:

```text
act -> remember -> train world model -> dream/train policy -> evaluate -> checkpoint
```

The runner writes the complete state needed to resume:

- model and optimizer checkpoint;
- replay buffer;
- runner counters;
- metrics log;
- dashboard state;
- real and dreamed rollout videos.

This is the intended meaning of "never shut down": the agent can stop using compute and later resume with memory intact.

## Dashboard

The dashboard is a local FastAPI app. It reads `dashboard_state.json`, serves the latest real and dream videos, and polls `/api/state` every two seconds. It is not a hosted service and does not require cloud infrastructure.

## Evaluation

Mirage v1 reports:

- average evaluation return;
- landing success rate;
- replay size;
- world-model prediction loss;
- reward prediction loss;
- imagined rollout value;
- dream-to-real state error;
- wall-clock seconds;
- parameter count.

The default smoke config is for validation only. Serious results should use `configs/local_mps.yaml` or `configs/research.yaml`.
