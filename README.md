# pytorch-variable-td3

## Installation
```conda env create -f environment.yml```

## Usage

```$ python main.py --case classic_control --env Pendulum-v0 --opr train```

## Plotting:
```bash
$ cd scripts
$ python summary_graphs.py --logdir=../results/classic_control --opr extract_summary 
$ python summary_graphs.py --logdir=../results/classic_control --opr plot
```