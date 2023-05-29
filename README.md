# T2TL
Exploiting Transformer in Reinforcement Learning for Interpretable Temporal Logic Motion Planning (**Under-review**)

If any questions, feel free to contact: zcharlie0257@gmail.com.

## Installation instructions

We recommend using Python 3.7 or Python 3.8 to run this code.

1. `pip install -r requirements.txt`
2. Install [Spot-2.9](https://spot.lrde.epita.fr/install.html), there are many ways:
    - Follow the installation instructions at the link. Spot should be installed in `/usr/local/lib/python3.7/site-packages/spot`. This step usually takes around 20 mins.
    - `conda install -c conda-forge spot` (Note that this command needs Python is 3.8+)
3. To train the agent in ZoneEnv, you will need Mujoco installed, as well as an [active license](http://www.mujoco.org/index.html). 
    - `pip install mujoco-py==2.0.2.9`
    - `pip install -e src/envs/safety/safety-gym/`

## Examples
1. `cd dis_src/`
2. To train the agent with GNN method, run:
    - `python T1TL.py  --gnn RGCN_8x32_ROOT_SHARED`
3. To train the agent with  pretrained GNN method, run:
    - `python train_PreGNNAgent.py`
4. To train the agent with T1TL method, run:
    - `python T1TL.py`
5. To train the agent with pretrained T1TL method, run:
    - `python T1TL_pretrain.py`
6. To train the agent with T2TL method, run:
    - `python T2TL.py`
7. To train the agent with pretrained T2TL method, run:
    - `python T2TL_pretrain..py`

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


## Citation

```
@article{zhang2022exploiting,
  title={Exploiting Transformer in Reinforcement Learning for Interpretable Temporal Logic Motion Planning},
  author={Zhang, Hao and Wang, Hao and Kan, Zhen},
  journal={arXiv preprint arXiv:2209.13220},
  year={2022}
}
``
