# T2TL
Exploiting Transformer in Reinforcement Learning for Interpretable Temporal Logic Motion Planning (Under-review)

The completed code will come soon.

## Installation instructions

We recommend using Python 3.7 to run this code.

1. `pip install -r requirements.txt`
2. Install [Spot-2.9](https://spot.lrde.epita.fr/install.html)
    - Follow the installation instructions at the link. Spot should be installed in `/usr/local/lib/python3.7/site-packages/spot`. This step usually takes around 20 mins.
3. To train the agent in ZoneEnv, you will need Mujoco installed, as well as an [active license](http://www.mujoco.org/index.html). 
    - `pip install mujoco-py==2.0.2.9`
    - `pip install -e src/envs/safety/safety-gym/`

## Examples
### To train the agent in ZoneEnv with Transformer
`python train_agent.py`

## Office Gridworld
<p align="center">
    <img width="700" src="https://github.com/Charlie0257/T2TL/blob/main/README_file/office_env.png">
</p>

## MiniCraft-like gridworld
<p align="center">
    <img width="700" src="https://github.com/Charlie0257/T2TL/blob/main/README_file/Mini-craft_env_all.png">
</p>

## ZoneEnv
<p align="center">
    <img width="700" src="https://github.com/Charlie0257/T2TL/blob/main/README_file/Zones_env_all.png">
</p>

## Dimension Comparison
<p align="center">
    <img width="700" src="https://github.com/Charlie0257/T2TL/blob/main/README_file/dim_comparison_all.png">
</p>

## Comments
Our codebase in ZoneEnv builds heavily on LTL2Action and https://github.com/LTL2Action/LTL2Action. Thanks for open-sourcing!

## Citation

```
@article{zhang2022exploiting,
  title={Exploiting Transformer in Reinforcement Learning for Interpretable Temporal Logic Motion Planning},
  author={Zhang, Hao and Wang, Hao and Kan, Zhen},
  journal={arXiv preprint arXiv:2209.13220},
  year={2022}
}
``
