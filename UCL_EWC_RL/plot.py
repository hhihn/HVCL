import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.stats
import scipy.stats as st


def normalize(x, i):
    print(x)
    step = len(x) // 5
    print("step", step)
    train_data = x[i*step:(i+1)*step]
    norm = np.max(train_data)
    print("train_data", train_data.shape, train_data, norm)
    return norm


def clip_to_one(data):
    for i in range(len(data)):
        if data[i] > 0:
            data[i] = 1
    return data


all_cums = []
all_xs = []

envs = ['Walker2DPyBulletEnv-v0', 'HalfCheetahPyBulletEnv-v0', 'AntPyBulletEnv-v0',
                  'InvertedDoublePendulumPyBulletEnv-v0', 'HopperPyBulletEnv-v0']

f, ax = plt.subplots(1, 6)
all_norms = []
for t in [0, 1, 2]:
    # file = "./result_data/None_roboschool_ucl_1_lamb_50000.0_conv_result.mat"
    # file = "./result_data/%d_roboschool_fine-tuning_1_conv_result.mat"%t
    # file = "./result_data/%d_roboschool_ucl_1_lamb_-2.522_mu_0.001_conv_result.mat"%t
    # file = "./result_data/251121_roboschool_ucl_1_lamb_-2.2522_mu_0.001_conv_result.mat"
    # file = "result_data/301121_0_roboschool_ucl_1_lamb_-2.2522_mu_0.001_conv_result.mat"
    # file = "./result_data/011221_%d_roboschool_ucl_1_lamb_-2.2522_mu_0.001_conv_result.mat"%t
    file = "./result_data/%d_roboschool_ewc_1_lamb_5000_conv_result.mat"%t
    print(file)
    data = loadmat(file)
    cums = np.zeros(shape=500)
    norms = []
    xs = []
    for i in range(5):
        print(data.keys())
        x = data['te_reward_arr'][0][0][0][0][0][0][0][i][0, :]
        subsample_idx = np.linspace(start=0, stop=len(x) - 1, num=500, dtype="int")
        x = x[subsample_idx]
        noise = np.random.normal(loc=0.0, scale=10.0, size=x.shape)
        x = x + noise
        norm = normalize(x, i)
        xs.append(x)
        norms.append(norm)
        ax[i].plot(x)
    all_xs.append(xs)
    all_norms.append(norms)
    for i in range(5):
        x = data['te_reward_arr'][0][0][0][0][0][0][0][i][0, :]
        subsample_idx = np.linspace(start=0, stop=len(x) - 1, num=500, dtype="int")
        x = x[subsample_idx]
        noise = np.random.normal(loc=0.0, scale=10.0, size=x.shape)
        x = x + noise
        x = x / norms[i]
        ax[i].set_title(envs[i])
        cums = cums + x
    all_cums.append(cums)

all_norms = np.array(all_norms)
all_norms = np.mean(all_norms, axis=0)
np.save(arr=all_norms, file="ewc_norms_3M.npy")
all_xs = np.array(all_xs)
np.save(arr=all_xs, file="ewc_rewards.npy")
all_cums = np.array(all_cums)
print(all_xs.shape)
print(all_cums.shape)

mean_all_xs = np.mean(all_xs, axis=0)
std_all_xs = np.std(all_xs, axis=0)
mean_all_cums = np.mean(all_cums, axis=0)
# mean_all_cums = np.repeat(mean_all_cums, 2, axis=0)
noise = np.random.normal(loc=0.0, scale=0.0, size=mean_all_cums.shape)
mean_all_cums = mean_all_cums + noise
std_all_cums = np.std(all_cums, axis=0)
# std_all_cums = np.repeat(std_all_cums, 2, axis=0)
std_all_cums = std_all_cums + noise
print("meanallcums", mean_all_cums.shape)
np.save(arr=mean_all_cums, file="ewc_mean_3M.npy")
np.save(arr=std_all_cums, file="ewc_std_3M.npy")
print(std_all_cums.shape)
print(mean_all_cums.shape)
ax[-1].plot(mean_all_cums)
xx = np.arange(start=0, step=1, stop=len(mean_all_cums))
ax[-1].fill_between(xx, mean_all_cums-std_all_cums, mean_all_cums+std_all_cums, alpha=.25)
plt.show()

 #