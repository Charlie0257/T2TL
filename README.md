# T2TL
[**Exploiting Transformer Sparse Reward in Reinforcement Learning for Interpretable Temporal Logic Motion Planning**](https://ieeexplore.ieee.org/document/10167731)

If any questions, feel free to contact: zcharlie0257@gmail.com.

## Installation instructions

We recommend using Python 3.7 or Python 3.8 to run this code.

1. `pip install -r requirements.txt` or `conda env create -f conda_env.yml` (Note that .yml is only effective for Python 3.7)
2. Install [Spot-2.9](https://spot.lrde.epita.fr/install.html):
    - For Python 3.7, follow the installation instructions at the link. Spot should be installed in `/usr/local/lib/python3.7/site-packages/spot`. This step usually takes around 20 mins.
    - For Python 3.8+, `conda install -c conda-forge spot` (Note that this command needs Python is 3.8+)
3. To train the agent in ZoneEnv, you will need Mujoco installed, as well as an [active license](http://www.mujoco.org/index.html). 
    - `pip install mujoco-py==2.0.2.9`
    - `pip install -e src/envs/safety/safety-gym/`
    - `pip install numpy==1.21.5`, if Python 3.7 or `pip install numpy==1.23`, if Python 3.8
4. Install dgl:
    - If you only reproduce the results from Transformer-encoded method, just `pip install dgl-cu113 -f https://data.dgl.ai/wheels/repo.html`
    - If you also want to reproduce the results from GNN-encoded method, we found the dgl team has not maintained dgl_cu111==0.7a210520. You can install it as follows:
        - download the dgl file from [google drive](https://drive.google.com/drive/folders/1Mg7wXciMgWWbeL93DG1npUD4HhI9mQHk?usp=drive_link)
        - `unzip dgl.zip` and `unzip dgl_cu111-0.7a210520.dist-info.zip`
        - `mv -f dgl/ /home/{your_workspace_name}/anaconda3/envs/T2TL/lib/python3.7/site-packages/`
        - `mv -f dgl_cu111-0.7a210520.dist-info/ /home/{your_workspace_name}/anaconda3/envs/T2TL/lib/python3.7/site-packages/`
        

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

## WaterWorld
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
@ARTICLE{10167731,
  author={Zhang, Hao and Wang, Hao and Kan, Zhen},
  journal={IEEE Robotics and Automation Letters}, 
  title={Exploiting Transformer in Sparse Reward Reinforcement Learning for Interpretable Temporal Logic Motion Planning}, 
  year={2023},
  volume={8},
  number={8},
  pages={4831-4838},
  doi={10.1109/LRA.2023.3290511}}
``
