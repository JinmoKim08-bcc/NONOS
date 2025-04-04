# NONOS: Neural Oscillation and Non-Oscillation Separator for Securing Temporal Resolution

This repository is the official implementation of [NONOS: Neural Oscillation and Non-Oscillation Separator for Securing Temporal Resolution]. 

![Image](https://github.com/user-attachments/assets/1bf9fdc9-344d-447e-a66c-4d226c90be70)

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Release note
The convenience update coming soon!

## Generate simulation data

To generate the simulation data in the paper, run this command:

```train
python generate_sim.py --exp_num 4_1_1
```
- The spectral parameterization process with SpecParam takes considerable time.

## Training

To train the model(s) in the paper, run this command:

```train
python NONOS_train.py --save_path <path_to_save> --fname_osc <path_to_osc> --fname_osc <path_to_nosc> --gpus 4 --mode <simple or SpecParam> --specpara-result <path_to_result> --epochs 200
```

- We provide two versions of models: NONOS-simple and NONOS-SpecParam.
- For NONOS-SpecParam, a pre-calculated spectral parameterization result is required for smooth computation.


## Evaluation

To evaluate my model on a specific data, run:

```eval
python NONOS_eval.py --model_file mymodel.checkpoint --fname_osc <path_to_osc> --fname_osc <path_to_nosc> --gpus 4
```

## Results

Our model achieves the following performance on :

![image](https://github.com/jkwrbcc/NONOS/assets/170528215/32012f6c-864a-476e-b2ae-d4bbc0d3a995)

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 
