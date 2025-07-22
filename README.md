# Stealing That Free Lunch: Exposing the Limits of Dyna-Style Reinforcement Learning
This repository contains an efficient JAX-based implementation of MBPO, enabling fast and reproducible experiments for studying the limits of Dyna-style reinforcement learning, as presented in [Stealing That Free Lunch: Exposing the Limits of Dyna-Style Reinforcement Learning (Brett Barkley and David Fridovich-Keil)](https://arxiv.org/pdf/2412.14312).

# STFL
The codebase is mostly based on [jaxrl](https://github.com/ikostrikov/jaxrl) and [high_replay_ratio_continuous_control](https://github.com/proceduralia/high_replay_ratio_continuous_control) which provided the following features:
- **Accessible implementations for off-policy RL**: Inherithed from jaxrl, but focused on the online setting (SAC and DDPG)
- **More efficient code for high replay ratio**: Additional jit compilation for speed increases at high replay ratios
- **Parallelization over multiple seeds on a single GPU**: Multiple seeds are run in parallel by sequentially collecting data from each seed's environment, while actions and updates are processed in parallel on the GPU
- **Off-the-shelf checkpointing**: a simple checkpointing and loading mechanism is provided

The modifications to the original repository include:
- **Simplfied Installation** The original jaxrl and high_replay_ratio_continuous_control repositories don't pip install out of the box due to deprecated versioning, so some effort was made to update their dependencies and simplify the requirements.txt
- **Modernization of Jax and Flax dependencies** The jaxrl and high_replay_ratio_continuous_control repositories are reliant on a few deprecated features of Jax and Flax, so we removed them
- **Removal of Brax interfaces** Brax interfaces are unneeded for our purposes, so we removed them

Additional features we have developed include:
- **Development of High-Throughput MBPO Code** We develop a fast MBPO implementation that can run multiple seeds of independent training in parallel and achieves an order-of-magnitude speedup over existing MBPO implementations.
- **Configurable Integration of Regularization Techniques for High Replay Ratio** We include configurable access to layer norms and resets of both the ensemble and base actor critic as presented in STFL.

## Setup 
```
Notes:
Need to install mujoco: wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
Then tar -xvf it 
Then follow directions here https://github.com/openai/mujoco-py?tab=readme-ov-file
Need to add two lines to bashrc:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/beb3238/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

Run pip install "cython<3" to cythonize properly
Run sudo apt-get install libglew-dev (or ensure GL/glew.h is present on the system)
Run sudo apt-get install patchelf

```
To install, clone the repository and run the following (Python versions >3.10 may be incompatible with gym and pygame):

```bash 
conda create -n STFL python=3.9.17
conda activate STFL
git submodule update --init --recursive
# see notes above and ensure everything is done
pip install -r requirements.txt
```

Note: We only tested up to 6 seeds in parallel extensively. Out of memory errors likely will be triggered for parallelization beyond that point.
Example usage (mbpo is turned on by default):

`python train_parallel.py --env_name cheetah-run --num_seeds 6 --max_steps 500000`

The following table reports the approximate running times on an NVIDIA RTX 4500 Ada gpu for running 6 seeds per task from the DMC15-500k and Gym6-300k with a replay ratio of 20.

| **Gym Tasks**        | **Running Time (hrs:min)** |
|----------------------|----------------------------|
| InvertedPendulum-v2  | 4:34                       |
| Hopper-v3            | 4:53                       |
| HalfCheetah-v3       | 5:00                       |
| Walker2d-v3          | 5:09                       |
| Ant-v2               | 8:34                       |
| Humanoid-v3          | 16:58                      |

| **DMC Tasks**        | **Running Time (hrs:min)** |
|----------------------|----------------------------|
| hopper-hop           | 11:44                      |
| hopper-stand         | 11:47                      |
| humanoid-walk        | 19:41                      |
| humanoid-stand       | 19:42                      |
| quadruped-run        | 19:48                      |
| quadruped-walk       | 20:35                      |



## Checkpointing
Agent and buffer are logged into `FLAGS.save_dir`. By default, training starts from scratch. If a checkpoint exists in FLAGS.save_dir, it will be automatically loaded.

## Citation
If you find this repository useful, feel free to cite our paper using the following bibtex.

```
@inproceedings{barkley2025stealing,
  title     = {Stealing That Free Lunch: Exposing the Limits of Dyna‑Style Reinforcement Learning},
  author    = {Barkley, Brett and Fridovich‑Keil, David},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year      = {2025},
  note      = {To appear},
  url       = {https://arxiv.org/abs/2412.14312}
}
```

