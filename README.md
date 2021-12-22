# ML-OTF
This repository contains the code that can be used to reproduce the 2D results from the master thesis: [DeepONet Generated Optimal Test Functions for the Finite Element Method](https://repository.tudelft.nl/islandora/object/uuid:0051e66b-2a12-4b74-bf5d-2f1b486651cd). The 2D results will be expanded and published in the near future.

Additional links:
* Introductory video on DeepONets: [link](https://www.youtube.com/watch?v=1bS0q0RkoH0)
* Deep Operator Network (DeepONet) paper: [link](https://arxiv.org/abs/1910.03193)
* Optimal test functions paper: [link](http://web.pdx.edu/~gjay/pub/dpg2.pdf)

## Implementing DeepONet Generated Optimal Test Functions
### Generate a training dataset
1. Define the weak formulation and the inner product in `twoD/gen_solutions/weak_forms.py` that are used to build the optimal test functions.
2. Run `twoD/gen_solutions/gen_matrices.py` to build the matrices corresponding to the inner product that you defined in `twoD/gen_solutions/weak_forms.py` and corresponding to a particular set of discretisations. The range of discretisations can be set by changing `n_min` and `n_max`. Define where you want the matrices to be saved before running.
3. Run `twoD/gen_solutions/build_training_dataset.py` to build the training dataset. Define where you want the results to be stored before running by setting `matrix_files_locations`, `training_dataset_location`, `testing_dataset_location`. The function `gen_data_points` can be changed into `gen_data_points_var` to generate data for a variable Péclet number. 

### Train a DeepONet
Once a training dataset has been generated this comes down to running `twoD/train_deeponet.py`. The batch size per replica can be set in this script, as the number of epochs. Make sure to specify the storage location, model name, training dataset and testing dataset before runnning. Also make sure that the `num_vars` `is set correctly depending on whether variables other than the spatial ones are used, so that the TFRecords can be read correctly.

### Implement the DeepONet
Once the DeepONet is trained it can be implemented into the `twoD/fem_deeponet.py` code by specifying `model_path`. The results can be compared to the Galerkin implementation by running `fem_galerkin.py`.
