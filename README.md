# pytorch-variable-td3

## Installation
```conda env create -f environment.yml```

## Usage
- Train: ```$ python main.py --case classic_control --env Pendulum-v0 --opr train```
- Test: ```$ python main.py --case classic_control --env Pendulum-v0 --opr test```
- Visualize Results: ```tensorboard --logdir=./results```
- Summarize plots in Plotly:
    ```bash
    $ cd scripts
    $ python summary_graphs.py --logdir=../results/classic_control --opr extract_summary 
    $ python summary_graphs.py --logdir=../results/classic_control --opr plot
    ```

<table>
<thead>
<th >Required Arguments</th>
<th >Description</th>
</thead>
<tbody>
<tr>
	<td> <i>--env</i> </td>
	<td>Name of the environment</td>
</tr>
<tr>
	<td rowspan="4"> <i>--case {atari,classic_control,box2d}</i> </td>
	<td>It's used for switching between different domains(and configs).<br> <br>Environments corresponding to each case:</td>
</tr>
<tr>
	<td><i>classic_control : {Pendulum-v0, MountainCarContinuous-v0}</td>
</tr>
<tr>
	<td><i>box2d : {LunarLanderContinuous-v2, BipedalWalker-v3, BipedalWalkerHardcore-v3} </i></td>
</tr>
<tr>
	<td><i><a href="https://gym.openai.com/envs/#mujoco(https://gym.openai.com/envs/#mujoco)">mujoco</a></i></td>
</tr>
<tr>
	<td> <i>--opr {train,test}</i> </td>
	<td>operation to be performed</td>
</tr>
</tbody>
</table>
