# pytorch-variable-td3

## Installation
```conda env create -f environment.yml```

## Usage
- Train: ```$ python main.py --case classic_control --env Pendulum-v0 --opr train```
- Test: ```$ python main.py --case classic_control --env Pendulum-v0 --opr test```

    |Required Arguments | Description|  
    |:-------------|:-------------|  
    | `--env`                          |Name of the environment|  
    | `--case {classic_control,box2d,mujoco}` |It's used for switching between different domains(and configs) <br><br> Environments corresponding to ease case: <br> `classic_control` : {Pendulum-v0, MountainCarContinuous-v0} <br> `box2d` : _{LunarLanderContinuous-v2, BipedalWalker-v3, BipedalWalkerHardcore-v3}_ <br>`mujoco`: _[(refer here)](https://gym.openai.com/envs/#mujoco)_|  
    | `--opr {train,test}` |select the operation to be performed|

- Visualize Results: ```tensorboard --logdir=./results```
- Summarize plots in Plotly:
    ```bash
    $ cd scripts
    $ python summary_graphs.py --logdir=../results/classic_control --opr extract_summary 
    $ python summary_graphs.py --logdir=../results/classic_control --opr plot
    ```
  
