Code basis for Hierarchically Structured Task-Agnostic Continual Learning, which you can find as pre-print on arxiv under [this](https://arxiv.org/pdf/2110.12667.pdf) link. You can also check out our [projet page](https://sites.google.com/view/hvcl/home).

# Hierarchically Structured Task-Agnostic Continual Learning
Code for HVCL Paper contains:

## Implementation for Dense MoVE Layers
Dense layers can be found in the file `BayesianDenseMoe.py` under the python class `BayesianDenseMoE`, which inherts from `keras.layers`. Thus, it can be used as part of a model or in a feedforward chain.
See the following example on how to instantiate such a layer:
```
layer_out = BayesianDenseMoE(units=64, expert_activation=tf.nn.relu,
                                             gating_activation=tf.nn.softmax,
                                             n_experts=2, gating_beta=0.5,
                                             expert_beta=1.0, name="layer_0",
                                             diversity_bonus=0.05,
                                             entropy_fun=entropy_function,
                                             kl_div_fun=kl_divergence_function)(layer_in)
```

## Implementation for Conv MoVE Layers
Conv layers can be found in the file `BayesianConvMoe.py` under the python class `BayesianConvMoE`, which inherts from `keras.layers`. Thus, it can be used as part of a model or in a feedforward chain. We provide 1D, 2D, and 3D convolutional layers.

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

# Minimum Requirements: 

- Tensorflow 2.5.0
- Tensorflow Probability 0.11.0
- NumPy 1.19.5
- gym 0.15.7
- scipy 1.4.1
- pybullet 3.0.8
- pybullet-gym from https://github.com/benelot/pybullet-gym
