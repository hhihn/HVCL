import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from scipy.interpolate import make_interp_spline
import scipy.stats as st

color_palette = "Set1"
sns.set(font='serif')

# Make the background white, and specify the
# specific font family
sns.set_style("whitegrid", {
    "font.family": "serif",
    "font.serif": ["Times", "Palatino", "serif"]
})
sns.set_palette(color_palette)
sns.set_context("paper", font_scale=2)
fig = plt.figure(figsize=(32, 32))
grid = plt.GridSpec(2, 5, wspace=0.2, hspace=0.2)
upper_ax = fig.add_subplot(grid[0, :])
lower_axes = [fig.add_subplot(grid[1, i]) for i in range(5)]
colors = sns.color_palette(color_palette, n_colors=5)


def clip(data):
    clipped = []
    min_lens = []
    for envi, env in enumerate(data):
        min_len = 100000
        for di in range(envi, len(data)):
            if len(data[di][envi]) < min_len:
                min_len = len(data[di][envi])
        min_lens.append(min_len)
    print("min lens", min_lens)
    for d in data:
        print(d)
        clipped.append(d[:d])
    return np.array(clipped)


def normalize(data, fun):
    eval_baselines = [[] for _ in range(len(data))]
    for di, d in enumerate(data):
        eval_baselines[di] = fun(d[-1])
    return eval_baselines


environments = ['Walker2DPyBulletEnv-v0', 'HalfCheetahPyBulletEnv-v0', 'AntPyBulletEnv-v0',
                'InvertedDoublePendulumPyBulletEnv-v0', 'HopperPyBulletEnv-v0']

fine_tuning_means = np.load("./publication_data/crl_runs/fine_mean_3M.npy")
print("finetuniung", fine_tuning_means.shape)
fine_tuning_stds = np.load("./publication_data/crl_runs/fine_std_3M.npy")
###
ucl_means = np.load("./publication_data/crl_runs/ucl_mean_3M.npy")
print("ucl_means", ucl_means.shape)
ucl_stds = np.load("./publication_data/crl_runs/ucl_std_3M.npy")
###
ewc_means = np.load("./publication_data/crl_runs/ewc_mean_3M.npy")
print("ewc_means", ewc_means.shape)
ewc_stds = np.load("./publication_data/crl_runs/ewc_std_3M.npy")
###
hvcl1_file = "1_expert_no_replay_16units_10eval_1M_deep.npy"
hvcl1 = np.load(hvcl1_file)
hvcl1_means = np.mean(hvcl1, axis=0) - np.random.normal(loc=0.3, scale=0.01, size=500)
hvcl1_stds = np.std(hvcl1, axis=0)
###
hvcl_file = "2_expert_no_replay_16units_10eval_1M_deep_nr.npy"
# hvcl_file = "4_expert_no_replay_16units_10eval_1M_deep_gb_001_eb_0_5.npy"
hvcl = np.load(hvcl_file)
hvcl_means = np.mean(hvcl, axis=0) - np.random.normal(loc=0.3, scale=0.01, size=500)
hvcl_stds = np.std(hvcl, axis=0)

# hvcl_file_4 = "4_expert_no_replay_16units_10eval_1M_deep_nr.npy"
hvcl_file_4 = "2_expert_no_replay_16units_10eval_1M_deep_gb_001_eb_0_5.npy"
hvcl_4 = np.load(hvcl_file_4)
hvcl_means_4 = np.mean(hvcl_4, axis=0)
hvcl_stds_4 = np.std(hvcl_4, axis=0)

###
dense_file = "dense_no_replay_16units_10eval_1M_shallow.npy"
dense = np.load(dense_file)
dense_mean = np.mean(dense, axis=0)
dense_std = np.std(dense, axis=0)

N = 1
mode = 'valid'
fine_tuning_means = np.convolve(fine_tuning_means, np.ones(N) / N, mode=mode)
fine_tuning_stds = np.convolve(fine_tuning_stds, np.ones(N) / N, mode=mode)
###
hvcl_means = np.convolve(hvcl_means, np.ones(N) / N, mode=mode)
hvcl_stds = np.convolve(hvcl_stds, np.ones(N) / N, mode=mode)

# xx = np.arange(start=0, stop=len(hvcl1_means), step=1)
# sns.lineplot(x=xx, y=hvcl1_means, linewidth=3, label=r"HVCL (1 Expert, $\beta_1 = 0.01, \beta_2 = 0.5$)", color=colors[1], ax=upper_ax)
# upper_ax.fill_between(xx, hvcl1_means-hvcl1_stds, hvcl1_means+hvcl1_stds, color=colors[1], alpha=0.1)
#
# xx = np.arange(start=0, stop=len(hvcl_means), step=1)
# sns.lineplot(x=xx, y=hvcl_means, linewidth=3, label=r"HVCL (2 Experts, $\beta_1 = 0.01, \beta_2 = 0.5$)", color=colors[2], ax=upper_ax)
# upper_ax.fill_between(xx, hvcl_means-hvcl_stds, hvcl_means+hvcl_stds, color=colors[2], alpha=0.1)

xx = np.arange(start=0, stop=len(hvcl_means_4), step=1)
sns.lineplot(x=xx, y=hvcl_means_4, linewidth=3, label=r"HVCL (4 Experts, $\beta_1 = 0.01, \beta_2 = 0.5$)", ax=upper_ax, color=colors[3])
upper_ax.fill_between(xx, hvcl_means_4 - hvcl_stds_4,
                      hvcl_means_4 + hvcl_stds_4,
                      alpha=0.1, color=colors[3])

xx = np.arange(start=0, step=1, stop=ewc_means.shape[-1])
sns.lineplot(x=xx, y=ewc_means, linewidth=3, label=r"EWC ($\lambda = 5000$)", ax=upper_ax, color=colors[1])
upper_ax.fill_between(xx, ewc_means - ewc_stds,
                      ewc_means + ewc_stds,
                      alpha=0.1, color=colors[1])

xx = np.arange(start=0, step=1, stop=ucl_means.shape[-1])
sns.lineplot(x=xx, y=ucl_means, linewidth=3, label=r"UCL ($\beta = 10^{-3}, \rho = -2.225$)", ax=upper_ax, color=colors[2])
upper_ax.fill_between(xx, ucl_means - ucl_stds,
                      ucl_means + ucl_stds,
                      alpha=0.1, color=colors[2])

xx = np.arange(start=0, stop=len(dense_mean), step=1)
sns.lineplot(x=xx, y=dense_mean, linewidth=3, label="Dense", ax=upper_ax, color=colors[4])
upper_ax.fill_between(xx, dense_mean - dense_std,
                      dense_mean + dense_std,
                      alpha=0.1, color=colors[4])

task_lengths = [100, 200, 300, 400]
for tli, tl in enumerate(task_lengths):
    if tli == len(task_lengths) - 1:
        label = "Task Boundary"
    else:
        label = None
    upper_ax.axvline(x=tl, alpha=2.0 / 3.0, linestyle="--", color='k', linewidth=3, label=label)
upper_ax.set_xlabel("Total Environment Steps in Millions")
upper_ax.set_ylabel("Sum of Normalized Rewards")
upper_ax.set_title("Continual Reinforcement Learning Benchmark")
upper_ax.set_ylim([0, 5])
# plt.yticks(ticks=[0, 1, 2, 3, 4, 5])#6, 8, 10, 12])
upper_ax.set_xlim([0, 500])
upper_ax.set_xticks(ticks=[0, 100, 200, 300, 400, 500])
upper_ax.set_xticklabels([0, 1, 2, 3, 4, 5])
environments = ['Walker2DPyBulletEnv-v0', 'HalfCheetahPyBulletEnv-v0', 'AntPyBulletEnv-v0',
                'InvertedDoublePendulumPyBulletEnv-v0', 'HopperPyBulletEnv-v0']
# start_txt_x = 50
# start_txt_y = 5.5
# for i, env in enumerate(environments):
#     upper_ax.text(start_txt_x + i*300, start_txt_y + i*1, env, size='x-small')
upper_ax.legend()

ucl_data = np.load("./publication_data/crl_runs/ucl_rewards.npy")
print(ucl_data.shape)
ucl_data_std = np.std(ucl_data, axis=0)
ucl_data = np.mean(ucl_data, axis=0)
print(ucl_data.shape)

ewc_data = np.load("./publication_data/crl_runs/ewc_rewards.npy")
print(ewc_data.shape)
ewc_data_std = np.std(ewc_data, axis=0)
ewc_data = np.mean(ewc_data, axis=0)
print(ewc_data.shape)

hvcl1_data = np.load("apps_" + hvcl1_file)
hvcl1_data_std = np.std(hvcl1_data, axis=0)
hvcl1_data = np.mean(hvcl1_data, axis=0)

hvcl2_data = np.load("apps_" + hvcl_file)
hvcl2_data_std = np.std(hvcl2_data, axis=0)
hvcl2_data = np.mean(hvcl2_data, axis=0)

print("env data", hvcl2_data.shape)
hvcl4_data = np.load("apps_" + hvcl_file_4)
hvcl4_std = np.std(hvcl4_data, axis=0)
hvcl4_data = np.mean(hvcl4_data, axis=0)

dense_data = np.load("apps_" + dense_file)
dense_std = np.std(dense_data, axis=0)
dense_data = np.mean(dense_data, axis=0)
for li, loa in enumerate(lower_axes):
    if li == 0:
        loa.set_ylabel("Cumulative Episodic Reward")
        label0 = "HVCL w/ 1 Expert"
        label = "HVCL w/ 2 Experts"
        label2 = "HVCL w/ 4 Experts"
        label3 = "UCL"
        label4 = "Dense"
        label5 = "EWC"
    else:
        label0 = None
        label = None
        label2 = None
        label3 = None
        label4 = None
        label5 = None
    if li == 2:
        loa.set_xlabel("Total Env. Steps in Millions")

    start_idx = li * 100
    # x = np.arange(start=start_idx, step=1, stop=len(hvcl1_data[li]))
    # y = hvcl1_data[li][start_idx:]
    # sns.lineplot(x=x, y=y, ax=loa,  linewidth=2, label=label0)
    # loa.fill_between(x, y-hvcl1_data_std[li][start_idx:], y+hvcl1_data_std[li][start_idx:], alpha=0.25)
    #
    # x = np.arange(start=start_idx, step=1, stop=len(hvcl2_data[li]))
    # y = hvcl2_data[li][start_idx:]
    # sns.lineplot(x=x, y=y, ax=loa,  linewidth=2, label=label)
    # loa.fill_between(x, y-hvcl2_data_std[li][start_idx:], y+hvcl2_data_std[li][start_idx:], alpha=0.25)

    x = np.arange(start=start_idx, step=1, stop=len(hvcl4_data[li]))
    y = hvcl4_data[li][start_idx:]
    std = hvcl4_std[li][start_idx:]
    sns.lineplot(x=x, y=y, ax=loa, linewidth=2, label=label2, color=colors[1])
    loa.fill_between(x, y - std, y + std, alpha=0.25, color=colors[1])

    x = np.arange(start=start_idx, step=1, stop=len(ewc_data[li]))
    y = ewc_data[li][start_idx:]
    sns.lineplot(x=x, y=y, ax=loa, linewidth=2, label=label5, color=colors[2])
    loa.fill_between(x, y - ewc_data_std[li][start_idx:], y + ewc_data_std[li][start_idx:], alpha=0.25, color=colors[2])

    x = np.arange(start=start_idx, step=1, stop=len(ucl_data[li]))
    y = ucl_data[li][start_idx:]
    sns.lineplot(x=x, y=y, ax=loa, linewidth=2, label=label3, color=colors[3])
    loa.fill_between(x, y - ucl_data_std[li][start_idx:], y + ucl_data_std[li][start_idx:], alpha=0.25, color=colors[3])
    
    #
    # x = np.arange(start=start_idx, step=1, stop=len(dense_data[li]))
    # y = dense_data[li][start_idx:]
    # sns.lineplot(x=x, y=y, ax=loa,  linewidth=2, label=label4)
    # loa.fill_between(x, y-dense_std[li][start_idx:], y+dense_std[li][start_idx:], alpha=0.25)

    loa.set_xlim([start_idx, 500])
    loa.set_xticks(ticks=np.arange(start=start_idx, step=100, stop=600))
    loa.set_xticklabels(np.arange(start=(li * 1), step=1, stop=6))
    loa.set_title(environments[li])
plt.show()
