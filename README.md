# variable-td3

Traditionally, we learn a policy, and action is determined for every time-step. However, in many cases, it is also viable to simply repeat an action for multiple time-steps rather than determining a new action every time.  This repeat factor is usually manually tuned and is kept constant. We hypothesize that keeping it constant may not be an ideal-policy as there could be scenarios in an environment where we need fine-step control as well as there could be scenarios where a larger-step control is feasible. 
For example, if we think of Lunar-Lander, we may need fine-step control as we are closer to the ground and attempting to land as compared to moments when we are high up in the space and large-repeat action may be feasible.

In this work, we learn a policy that learns an action as well as the time-step for which this action should be repeated. This gives the policy the ability to have large as well as fine-step control. We also hypothesize that learning to repeat an action may also lead to better sample efficiency.  


|<a target="_blank" href="https://docs.google.com/presentation/d/1TKRy9va3qgIlia7pjdZkcLV9ht9bjQkUUcND4byrARg/edit?usp=sharing"><img src="https://lh3.ggpht.com/9rwhkrvgiLhXVBeKtScn1jlenYk-4k3Wyqt1PsbUr9jhGew0Gt1w9xbwO4oePPd5yOM=w300" width="65px" height="65px"/></a> |
|:-------------:|
|Slide|


`Status:`
- Code provided as it is and no major updates expected.
- This work is similar to the paper ["Learning to repeat: fine grained action repetition for deep reinforcement learning"](https://arxiv.org/pdf/1702.06054v1.pdf)


## Installation
1. Install [conda](https://docs.conda.io/en/latest/miniconda.html)
2. For classic tasks, do following:
    ```bash
    conda env create -f  env.yml # creates env with name "vtd3"
    ```
3. `Optional`: For gym mujoco env. do following:

    - Requires [mjpro 150 and mujoco license](https://www.roboti.us/index.html)
    - ```conda activate vtd3 # activates conda-env. in (2)```
    - ```pip install 'gym[mujoco]'```
4. `Optional`: For dm_control, create a separate conda env. having mujoco 2.0 :
    - Requires [mujoco200 and mujoco license](https://www.roboti.us/index.html)
    - ```conda env create -f env_mj2.yml # creates env with name "vtd3_mj2"```

_Having trouble during installation ?, please refer [here](#installation-troubleshooting)_

## Usage
- ```$ conda activate <env_name>```
- Train: ```$ python main.py --case classic_control --env Pendulum-v0 --opr train```
- Test: ```$ python main.py --case classic_control --env Pendulum-v0 --opr test```

    |Required Arguments | Description|  
    |:-------------|:-------------|  
    | `--case {classic_control,box2d,mujoco,dm_control}` |It's used for switching between different domains(and configs)|  
    | `--env` |Name of the environment <br><br> Environments corresponding to ease case: <br> `classic_control` : {Pendulum-v0, MountainCarContinuous-v0} <br> `box2d` : _{LunarLanderContinuous-v2, BipedalWalker-v3, BipedalWalkerHardcore-v3}_ <br>`mujoco`: _[(refer here)](https://gym.openai.com/envs/#mujoco)_ <br> `dm_control`: _[(refer here)](https://github.com/zuoxingdong/dm2gym)_ |  
    | `--opr {train,test}` |select the operation to be performed|

- Visualize Results: ```tensorboard --logdir=./results```
- Summarize plots in Plotly:
    ```bash
    $ cd scripts
    $ python summary_graphs.py --logdir=../results/classic_control --opr extract_summary 
    $ python summary_graphs.py --logdir=../results/classic_control --opr plot
    ```

## Installation Troubleshooting
### Windows : 
- 
  - `Error`: ` error: command 'swig.exe' failed ...`  
  - `Fix`: install swigy from [here](http://www.swig.org/download.html) and add it in your path using this [reference](https://www.youtube.com/watch?v=HDD9QqLtAws).
- 
  - `Error`: ` error: Microsoft Visual C++ 14.0 is required. ...`
  - `Fix`: by installing build tools from [here.](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
