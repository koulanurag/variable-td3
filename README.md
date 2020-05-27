# pytorch-variable-td3
<a target="_blank" href="https://docs.google.com/presentation/d/1TKRy9va3qgIlia7pjdZkcLV9ht9bjQkUUcND4byrARg/edit?usp=sharing"><img src="https://legismusic.com/wp-content/uploads/2019/04/Google-Slides.png" width="100px" height="100px"/></a>
<a target="_blank" href="https://www.overleaf.com/project/5eccded67908040001d77a7e"><img src="https://ftp.riken.jp/tex-archive/info/guide-latex-fr/Images/overleaf.png" width="80px" height="80px"/></a>


## Installation
1. For classic tasks and gym mujcoco , do following:
    ```bash
    conda env create -f  env.yml # creates env with name "pytorch-variable-td3"
    ```
2. For dm_control, create a seperate conda env.:
    ```bash
    conda env create -f env_dmcontrol.yml # creates env with name "td3_dmcontrol"
    ```

## Usage
- ```$ conda activate <env_name>```
- Train: ```$ python main.py --case classic_control --env Pendulum-v0 --opr train```
- Test: ```$ python main.py --case classic_control --env Pendulum-v0 --opr test```

    |Required Arguments | Description|  
    |:-------------|:-------------|  
    | `--case {classic_control,box2d,mujoco}` |It's used for switching between different domains(and configs)|  
    | `--env` |Name of the environment <br><br> Environments corresponding to ease case: <br> `classic_control` : {Pendulum-v0, MountainCarContinuous-v0} <br> `box2d` : _{LunarLanderContinuous-v2, BipedalWalker-v3, BipedalWalkerHardcore-v3}_ <br>`mujoco`: _[(refer here)](https://gym.openai.com/envs/#mujoco)_|  
    | `--opr {train,test}` |select the operation to be performed|

- Visualize Results: ```tensorboard --logdir=./results```
- Summarize plots in Plotly:
    ```bash
    $ cd scripts
    $ python summary_graphs.py --logdir=../results/classic_control --opr extract_summary 
    $ python summary_graphs.py --logdir=../results/classic_control --opr plot
    ```
