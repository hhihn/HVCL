# Hierarchically Structured Task-Agnostic Continual Learning
Code basis for Hierarchically Structured Task-Agnostic Continual Learning, which you can find as pre-print on arxiv under [this](https://arxiv.org/pdf/2110.12667.pdf) link. You can also check out our [project page](https://sites.google.com/view/hvcl/home).

## Minimum Requirements: 

- Tensorflow 2.5.0
- Tensorflow Probability 0.11.0
- NumPy 1.19.5
- gym 0.15.7
- scipy 1.4.1
- pybullet 3.0.8
- cloudpickle 1.2.2
- pybullet-gym from https://github.com/benelot/pybullet-gym

## Implementation for Dense MoVE Layers
Dense layers can be found in the file `DenseMoVE.py` under the python class `DenseMoVE`, which inherts from `keras.layers`. Thus, it can be used as part of a model or in a feedforward chain.
See the following example on how to instantiate a dense MoVE layer:
```
import tensorflow_probability as tfp
ds = tfp.distributions

kl_divergence_function = (lambda q, p: ds.kl_divergence(q, p))
entropy_function = (lambda p: p.entropy())

layer_out = DenseMoE(units=64, 
                     expert_activation=tf.nn.relu,
                     gating_activation=tf.nn.softmax,
                     n_experts=2, gating_beta=0.5,
                     expert_beta=1.0, name="layer_0",
                     diversity_bonus=0.05,
                     entropy_fun=entropy_function,
                     kl_div_fun=kl_divergence_function)(layer_in)
```
The lambda functions 
## Implementation for Conv MoVE Layers
Conv layers can be found in the file `ConvMoVE.py` under the python class `ConvMoE`, which inherts from `keras.layers`. Thus, it can be used as part of a model or in a feedforward chain. We provide 1D, 2D, and 3D convolutional layers.
See the following example on how to instantiate a 2D conv MoVE layer:
```
layer_out = Conv2DMoVE(n_filters=64,
                        kernel_size=(3, 3),
                        expert_activation=tf.nn.relu,
                        strides=1,
                        padding='same',
                        gating_activation=tf.nn.softmax,
                        n_experts=2,
                        gating_beta=0.5,
                        expert_beta=1.0, name="conv0",
                        diversity_bonus=0.01,
                        entropy_fun=entropy_function,
                        kl_div_fun=kl_divergence_function)(class_input)
```
## Implementation for the Continual RL Experiments
CRL experiments can be run using the following command:
```
python3 SAC_main.py --hvcl --n_exp 4 --hidden_u 64 --e_beta 1.0 --g_beta 0.5
```
or you can call 
```
python3 SAC_main.py --help
```
to see a full list of all avalaible hyper-parameters.

## Baseline EWC and UCL CRL Experiments
We provide the implementation we used for EWC and UCL for CRL in the accordingly named folder.

## Repo Structure
-`ExperienceBuffer.py` and `sum_tree.py` contain the implementation of prioritized replay buffer
- `SparseDispatcher.py` contains the implementation of the sparse top-k sampling technique
- `initializer.py` implementsn itializers that initiatial each expert weight as  Xavier and He initialization
- `parallel_gym.py` contains necessary functiosn to run CRL experiments in parallel by executing several envs in parallel
- `HVCL_rl_*.py` contain the HVCL implementation of SAC
-
## Acknowledgements
The SAC implementation is based on [this](https://github.com/RickyMexx/SAC-tf2) repo and the UCL implementation for CRL can be found in [this](https://github.com/csm9493/UCL) repo. `SparseDispatcher.py` is based on [this](https://github.com/tensorflow/tensor2tensor) repo.

## Citation
If you use any parts of this repo, please use the following citation:

```
@article{hihn2021mixture,
  title={Mixture-of-Variational-Experts for Continual Learning},
  author={Hihn, Heinke and Braun, Daniel A},
  journal={arXiv preprint arXiv:2110.12667},
  year={2021}
}
```
