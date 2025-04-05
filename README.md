# NONOS: Neural Oscillation and Non-Oscillation Separator for Securing Temporal Resolution

This repository is the official implementation of [NONOS: Neural Oscillation and Non-Oscillation Separator for Securing Temporal Resolution]. 

![Image](https://github.com/user-attachments/assets/1bf9fdc9-344d-447e-a66c-4d226c90be70)

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
- We recommend using Linux when working with multiple GPUs.

## Procedure

### 1. Create an environment
To create a conda environment, run this command:

```train
conda create -n nonos python=3.9
```

### 2. Activate the environment
```train
conda activate nonos
```

### 3. Move to the cloned directory
```train
cd /your_path/NONOS/
```

### 4. Install required libraries
```train
pip install -r requirements.txt
```

### 5. Generate a simulation data
```train
python generate_sim_for_Figure1.py --save_path <path_to_save> --num_data <number_of_data>
```

### 6. Extract spectral parameters of oscillation & non-oscillation
```train
python extract_aperiodic_guides.py --load_fname <fname_to_load> --fs <sampling_rate> --t_len <time_length> --save_path <path_to_save>
```
- Here, the data to be loaded must be in NumPy format and have a shape of (N, T), where N is the number of data samples and T is the number of data points.
  
### 7. Train the model
```train
python NONOS.py --load_path <path_to_load> --gpus <number_of_gpus> --epochs <number_of_epochs> --mode 0
```
- The trained model will be saved in the same directory as the loaded data.

### 8. Test the model
To examine the results, run this command first:
```train
python NONOS.py --load_path <path_to_load> --gpus <number_of_gpus> --epochs <number_of_epochs> --mode 0
```

Finally, use the Jupyter notebook to see figures!:
```train
Plot_results.ipynb
```
