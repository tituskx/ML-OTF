# ML-OTF
This repository contains the code that can be used to reproduce the 2D results from the master thesis: [DeepONet Generated Optimal Test Functions for the Finite Element Method](https://repository.tudelft.nl/islandora/object/uuid:0051e66b-2a12-4b74-bf5d-2f1b486651cd). The 2D results will be expanded and published in the near future.

Additional links:
* Introductory video on DeepONets: [link](https://www.youtube.com/watch?v=1bS0q0RkoH0)
* Deep Operator Network (DeepONet) paper: [link](https://arxiv.org/abs/1910.03193)
* Optimal test functions paper: [link](http://web.pdx.edu/~gjay/pub/dpg2.pdf)

## Implementing DeepONet Generated Optimal Test Functions
### Generate a training dataset
1. Run `twoD/gen_solutions/gen_matrices.py` to build a matrix corresponding to a particular inner product (defined in `twoD/gen_solutions/weak_forms.py` and corresponding to a particular set of discretisations. The range of discretisations can be set by changing `n_min` and `n_max`.
 
3. Train a DeepONet
4. Implement the network
