# variable-td3

|<a target="_blank" href="https://docs.google.com/presentation/d/1TKRy9va3qgIlia7pjdZkcLV9ht9bjQkUUcND4byrARg/edit?usp=sharing"><img src="https://lh3.ggpht.com/9rwhkrvgiLhXVBeKtScn1jlenYk-4k3Wyqt1PsbUr9jhGew0Gt1w9xbwO4oePPd5yOM=w300" width="65px" height="65px"/></a> | <a target="_blank" href="https://www.overleaf.com/project/5eccded67908040001d77a7e"><img src="https://images.ctfassets.net/nrgyaltdicpt/6qSXAo1CYEeBn5RkKLOR64/19c74bfb9a32772e353ff25c6f0070f5/ologo_square_colour_light_bg.png" width="70px" height="70px"/></a>| <a target="_blank" href="https://drive.google.com/drive/folders/135kdBROapppjGWIXl2cQLRJYu_trHi0k?usp=sharing"><img src="https://services.google.com/fh/files/misc/logo_drive_color_2x_web_96dp.png" width="70px" height="70px"/></a>| 
|:-------------:|:-------------:|:---------:|
|Slide| Overleaf| Google Drive|

## Installation
1. Install [conda](https://docs.conda.io/en/latest/miniconda.html)
2. For classic tasks, do following:
    ```bash
    conda env create -f  env.yml # creates env with name "vtd3"
    ```
3. `Optional`: For gym mujoco env.(requires license) do following:

    - Requires [mjpro 150 and mujoco license](https://www.roboti.us/index.html)
    - ```conda activate vtd3 # activates conda-env. in (1)```
    - ```pip install 'gym[mujoco]'```
4. `Optional`: For dm_control & cassie, create a separate conda env. having mujoco 2.0 :
    - Requires [mujoco200 and mujoco license](https://www.roboti.us/index.html)
    ```bash
    conda env create -f env_mj2.yml # creates env with name "vtd3_mj2"
    ```

## Usage
- ```$ conda activate <env_name>```
- Train: ```$ python main.py --case classic_control --env Pendulum-v0 --opr train```
- Test: ```$ python main.py --case classic_control --env Pendulum-v0 --opr test```

    |Required Arguments | Description|  
    |:-------------|:-------------|  
    | `--case {classic_control,box2d,mujoco,dm_control,cassie}` |It's used for switching between different domains(and configs)|  
    | `--env` |Name of the environment <br><br> Environments corresponding to ease case: <br> `classic_control` : {Pendulum-v0, MountainCarContinuous-v0} <br> `box2d` : _{LunarLanderContinuous-v2, BipedalWalker-v3, BipedalWalkerHardcore-v3}_ <br>`mujoco`: _[(refer here)](https://gym.openai.com/envs/#mujoco)_ <br> `dm_control`: _[(refer here)](https://github.com/zuoxingdong/dm2gym)_ <br> `cassie`: {Cassie-v0} |  
    | `--opr {train,test}` |select the operation to be performed|

- Visualize Results: ```tensorboard --logdir=./results```
- Summarize plots in Plotly:
    ```bash
    $ cd scripts
    $ python summary_graphs.py --logdir=../results/classic_control --opr extract_summary 
    $ python summary_graphs.py --logdir=../results/classic_control --opr plot
    ```
