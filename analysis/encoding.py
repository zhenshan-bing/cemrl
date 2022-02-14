import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import colorsys

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

number2name_ml10 = {
    0: 'reach-v1',
    1: 'push-v1',
    2: 'pick-place-v1',
    3: 'door-open-v1',
    4: 'drawer-close-v1',
    5: 'button-press-topdown-v1',
    6: 'peg-insert-side-v1',
    7: 'window-open-v1',
    8: 'sweep-v1',
    9: 'basketball-v1',
    10: 'drawer-open-v1',
    11: 'door-close-v1',
    12: 'shelf-place-v1',
    13: 'sweep-into-v1',
    14: 'lever-pull-v1'}

number2name_cheetah_multi_task = {
    1: 'velocity',
    2: 'goal direction',
    3: 'goal',
    4: 'rollover',
    5: 'stand-up'}

number2name = number2name_cheetah_multi_task
cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']


def plot_encodings_split_with_rewards(epoch, exp_directory, save=False, normalize=False, legend=False):
    encoding_storage = pickle.load(open(os.path.join(exp_directory, "encoding_" + str(epoch) + ".p"), "rb"))
    base_tasks = list(encoding_storage.keys())
    #rewards_per_base_task = [sum([encoding_storage[base][key]['reward_mean'] / len(list(encoding_storage[base].keys())) for key in encoding_storage[base].keys()]) for base in base_tasks]
    if len(base_tasks) == 15:
        figsize = (20, 5)
    elif len(base_tasks) == 10:
        figsize = (15, 5)
    elif len(base_tasks) == 1:
        figsize = (7, 5)
    elif len(base_tasks) == 3:
        figsize = (7, 5)
    else:
        figsize = None
    fig, axes_tuple = plt.subplots(nrows=3, ncols=len(base_tasks), sharex='col', sharey='row', gridspec_kw={'height_ratios': [3, 1, 1]}, figsize=figsize)
    if len(axes_tuple.shape) == 1:
        axes_tuple = np.expand_dims(axes_tuple, 1)
    latent_dim = encoding_storage[base_tasks[0]][next(iter(encoding_storage[base_tasks[0]]))]['mean'].shape[0]

    # Normalization over base tasks of dim
    if normalize:
        normalizer = []
        mean_std = ['mean', 'std']
        for dim in range(latent_dim):
            temp_dict = {}
            for element in mean_std:
                values = np.array([a[element][dim] for base in base_tasks for a in list(encoding_storage[base].values())])
                temp_dict[element] = dict(mean=values.mean(), std=values.std())
            normalizer.append(temp_dict)


    for i, base in enumerate(base_tasks):
        # encodings
        #target_values = np.array([encoding_storage[base][key]['target'][2] for key in encoding_storage[base].keys()])
        #sort_indices = np.argsort(target_values)
        for dim in range(latent_dim):
            x_values = np.array([a['mean'][dim] for a in list(encoding_storage[base].values())])#[sort_indices]
            y_values = np.array([a['std'][dim] for a in list(encoding_storage[base].values())])#[sort_indices]
            #Normalize
            if normalize:
                x_values = (x_values - normalizer[dim]['mean']['mean']) / (normalizer[dim]['mean']['std'] + 1e-9)
                y_values = (y_values - normalizer[dim]['std']['mean']) / (normalizer[dim]['std']['std'] + 1e-9)
            label_string = "Encoding $z_" + str(dim) + "$"
            #axes_tuple[0][i].errorbar(target_values[sort_indices], x_values, yerr=y_values, fmt=".", label=label_string)
            axes_tuple[0][i].errorbar(np.array(list(encoding_storage[base].keys())), x_values, yerr=y_values, fmt=".", label=label_string)#, capsize=2
            if axes_tuple.shape[1] > 1:
                #axes_tuple[0][i].set_title("Base Task " + str(i))
                nameWithoutVersion = '-'.join(number2name[base].split('-')[:-1])
                if len(nameWithoutVersion.split('-')) > 2:
                    split_name = '-'.join(nameWithoutVersion.split('-')[:2]) + " \n " + '-'.join(nameWithoutVersion.split('-')[2:])
                else:
                    split_name = nameWithoutVersion
                axes_tuple[0][i].set_title(split_name)
            else:
                axes_tuple[0][i].set_title("Epoch " + str(epoch), fontsize=14)
        # rewards
        #axes_tuple[2][i].plot(np.array(list(encoding_storage[base].keys())), [encoding_storage[base][i]['reward_mean'] for i in encoding_storage[base].keys()], 'x')
        axes_tuple[2][i].bar(np.array(list(encoding_storage[base].keys())), [encoding_storage[base][i]['reward_mean'] for i in encoding_storage[base].keys()], width=0.01, align='center')

        # base task encodings
        #axes_tuple[1][i].plot(target_values[sort_indices], [np.argmax(a['base']) for a in list(encoding_storage[base].values())], 'x', label="Base encoding $\mathbf{y}$")
        axes_tuple[1][i].plot(list(encoding_storage[base].keys()), [np.argmax(a['base']) for a in list(encoding_storage[base].values())], 'x', label="Base encoding $\mathbf{y}$")
        axes_tuple[1][i].set_xlabel("Specification", fontsize=12)
        axes_tuple[1][i].set_yticks(np.arange(-1, len(base_tasks), 1), minor=True)


        axes_tuple[1][0].set_ylim(-1, 10)  #len(base_tasks)
        axes_tuple[0][i].grid()
        axes_tuple[1][i].grid(which='minor')
        axes_tuple[1][i].grid(which='major')
        axes_tuple[2][i].grid()
        axes_tuple[0][0].set_ylabel('Encoding $\mathbf{z}$', fontsize=12)
        axes_tuple[1][0].set_ylabel('Base task \n encoding $\mathbf{y}$', fontsize=12)
        axes_tuple[2][0].set_ylabel('Average \n reward $R$', fontsize=12)
    if legend:
        axes_tuple[0][-1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
        axes_tuple[1][-1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
    if save:
        plt.tight_layout()
        fig.savefig(exp_directory + "/encoding_epoch_" + str(epoch) + ("_normalized" if normalize else "") + "_with_rewards" + ".pdf", format="pdf")
    fig.show()
    # print("Here to create plot 1")
    print("Created plot")


def plot_encodings_split_with_rewards_cheetah(epoch, exp_directory, save=False, normalize=False, legend=False):
    encoding_storage = pickle.load(open(os.path.join(exp_directory, "encoding_" + str(epoch) + ".p"), "rb"))
    base_tasks = list(encoding_storage.keys())
    #rewards_per_base_task = [sum([encoding_storage[base][key]['reward_mean'] / len(list(encoding_storage[base].keys())) for key in encoding_storage[base].keys()]) for base in base_tasks]
    if len(base_tasks) == 15:
        figsize = (20, 5)
    elif len(base_tasks) == 10:
        figsize = (15, 5)
    elif len(base_tasks) == 1:
        figsize = (7, 5)
    elif len(base_tasks) == 3:
        figsize = (7, 5)
    else:
        figsize = None
    fig, axes_tuple = plt.subplots(nrows=3, ncols=len(base_tasks), sharex='col', sharey='row', gridspec_kw={'height_ratios': [3, 1, 1]}, figsize=figsize)
    if len(axes_tuple.shape) == 1:
        axes_tuple = np.expand_dims(axes_tuple, 1)
    latent_dim = encoding_storage[base_tasks[0]][next(iter(encoding_storage[base_tasks[0]]))]['mean'].shape[0]

    # Normalization over base tasks of dim
    if normalize:
        normalizer = []
        mean_std = ['mean', 'std']
        for dim in range(latent_dim):
            temp_dict = {}
            for element in mean_std:
                values = np.array([a[element][dim] for base in base_tasks for a in list(encoding_storage[base].values())])
                temp_dict[element] = dict(mean=values.mean(), std=values.std())
            normalizer.append(temp_dict)


    for i, base in enumerate(base_tasks):
        # encodings
        #target_values = np.array([encoding_storage[base][key]['target'][2] for key in encoding_storage[base].keys()])
        #sort_indices = np.argsort(target_values)
        for dim in range(latent_dim):
            x_values = np.array([a['mean'][dim] for a in list(encoding_storage[base].values())])#[sort_indices]
            y_values = np.array([a['std'][dim] for a in list(encoding_storage[base].values())])#[sort_indices]
            #Normalize
            if normalize:
                x_values = (x_values - normalizer[dim]['mean']['mean']) / (normalizer[dim]['mean']['std'] + 1e-9)
                y_values = (y_values - normalizer[dim]['std']['mean']) / (normalizer[dim]['std']['std'] + 1e-9)
            label_string = "Encoding $z_" + str(dim) + "$"
            #axes_tuple[0][i].errorbar(target_values[sort_indices], x_values, yerr=y_values, fmt=".", label=label_string)
            axes_tuple[0][i].errorbar(np.array(list(encoding_storage[base].keys())), x_values, yerr=y_values, fmt=".", label=label_string)#, capsize=2
            if axes_tuple.shape[1] > 1:
                #axes_tuple[0][i].set_title("Base Task " + str(i))
                nameWithoutVersion = '-'.join(number2name[base].split('-')[:-1])
                if len(nameWithoutVersion.split('-')) > 2:
                    split_name = '-'.join(nameWithoutVersion.split('-')[:2]) + " \n " + '-'.join(nameWithoutVersion.split('-')[2:])
                else:
                    split_name = nameWithoutVersion
                split_name = number2name[base]
                axes_tuple[0][i].set_title(split_name)
            else:
                axes_tuple[0][i].set_title("Epoch " + str(epoch), fontsize=14)
        # rewards
        #axes_tuple[2][i].plot(np.array(list(encoding_storage[base].keys())), [encoding_storage[base][i]['reward_mean'] for i in encoding_storage[base].keys()], 'x')
        axes_tuple[2][i].bar(np.array(list(encoding_storage[base].keys())), [encoding_storage[base][i]['reward_mean'] for i in encoding_storage[base].keys()], width=0.01, align='center')

        # base task encodings
        #axes_tuple[1][i].plot(target_values[sort_indices], [np.argmax(a['base']) for a in list(encoding_storage[base].values())], 'x', label="Base encoding $\mathbf{y}$")
        axes_tuple[1][i].plot(list(encoding_storage[base].keys()), [np.argmax(a['base']) for a in list(encoding_storage[base].values())], 'x', label="Base encoding $\mathbf{y}$")
        axes_tuple[1][i].set_xlabel("Specification", fontsize=12)
        axes_tuple[1][i].set_yticks(np.arange(-1, len(base_tasks), 1), minor=True)


        axes_tuple[1][0].set_ylim(-1, 10)  #len(base_tasks)
        axes_tuple[0][i].grid()
        axes_tuple[1][i].grid(which='minor')
        axes_tuple[1][i].grid(which='major')
        axes_tuple[2][i].grid()
        axes_tuple[0][0].set_ylabel('Encoding $\mathbf{z}$', fontsize=12)
        axes_tuple[1][0].set_ylabel('Base task \n encoding $\mathbf{y}$', fontsize=12)
        axes_tuple[2][0].set_ylabel('Average \n reward $R$', fontsize=12)
    if legend:
        axes_tuple[0][-1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
        axes_tuple[1][-1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
    if save:
        plt.tight_layout()
        fig.savefig(exp_directory + "/encoding_epoch_" + str(epoch) + ("_normalized" if normalize else "") + "_with_rewards" + ".pdf", format="pdf")
    fig.show()
    print("Created plot")


def plot_encodings_split(epoch, exp_directory, save=False, normalize=False, legend=False):
    encoding_storage = pickle.load(open(os.path.join(exp_directory, "encoding_" + str(epoch) + ".p"), "rb"))
    base_tasks = list(encoding_storage.keys())
    if len(base_tasks) == 10:
        figsize = (15, 5)
    elif len(base_tasks) == 1:
        figsize = (7, 5)
    elif len(base_tasks) == 3:
        figsize = (7, 5)
    else:
        figsize = None
    fig, axes_tuple = plt.subplots(nrows=2, ncols=len(base_tasks), sharex='col', sharey='row', gridspec_kw={'height_ratios': [3, 1]}, figsize=figsize)
    if len(axes_tuple.shape) == 1:
        axes_tuple = np.expand_dims(axes_tuple, 1)
    latent_dim = encoding_storage[base_tasks[0]][next(iter(encoding_storage[base_tasks[0]]))]['mean'].shape[0]
    base_task_encodings = [np.argmax(a['base']) for base in base_tasks for a in list(encoding_storage[base].values())]

    # Normalization over base tasks of dim
    if normalize:
        normalizer = []
        mean_std = ['mean', 'std']
        for dim in range(latent_dim):
            temp_dict = {}
            for element in mean_std:
                values = np.array([a[element][dim] for base in base_tasks for a in list(encoding_storage[base].values())])
                temp_dict[element] = dict(mean=values.mean(), std=values.std())
            normalizer.append(temp_dict)


    for i, base in enumerate(base_tasks):
        fontsize=26
        # encodings
        #target_values = np.array([encoding_storage[base][key]['target'][2] for key in encoding_storage[base].keys()])
        #sort_indices = np.argsort(target_values)
        for dim in range(latent_dim):
            x_values = np.array([a['mean'][dim] for a in list(encoding_storage[base].values())])#[sort_indices]
            y_values = np.array([a['std'][dim] for a in list(encoding_storage[base].values())])#[sort_indices]
            #Normalize
            if normalize:
                x_values = (x_values - normalizer[dim]['mean']['mean']) / (normalizer[dim]['mean']['std'] + 1e-9)
                y_values = (y_values - normalizer[dim]['std']['mean']) / (normalizer[dim]['std']['std'] + 1e-9)
            label_string = "Encoding $z_" + str(dim) + "$"

            # 2 classes: capsize=3, elinewidth=3, capthick=3, markersize=9
            # more classes: capsize=2, elinewidth=2, capthick=2, markersize=7
            axes_tuple[0][i].errorbar(np.array(list(encoding_storage[base].keys())), x_values, yerr=y_values,
                                      fmt="d", color='tab:green', label=label_string, capsize=2, elinewidth=2, capthick=2, markersize=7,
                                      markerfacecolor='yellow', markeredgecolor='black')

            if axes_tuple.shape[1] > 1:
                #axes_tuple[0][i].set_title("Base Task " + str(i))
                nameWithoutVersion = '-'.join(number2name[base].split('-')[:-1])
                if len(nameWithoutVersion.split('-')) > 2:
                    split_name = '-'.join(nameWithoutVersion.split('-')[:2]) + " \n " + '-'.join(nameWithoutVersion.split('-')[2:])
                else:
                    split_name = nameWithoutVersion
                split_name = number2name[base]
                axes_tuple[0][i].set_title(split_name, fontsize=fontsize)
            else:
                axes_tuple[0][i].set_title("Epoch " + str(epoch), fontsize=fontsize)

        # base task encodings
        axes_tuple[1][i].plot(list(encoding_storage[base].keys()), [np.argmax(task['base']) for task in list(encoding_storage[base].values())], 'd', color='yellow', markersize=7, markerfacecolor='yellow', markeredgecolor='black')  # markersize=7 for multiple tasks, 9 for two
        axes_tuple[1][i].set_xlabel("Specification", fontsize=fontsize)
        axes_tuple[1][i].set_ylim(-1, np.max(base_task_encodings) + 1)
        axes_tuple[0][i].tick_params(axis="x", labelsize=fontsize)
        axes_tuple[0][i].tick_params(axis="y", labelsize=fontsize)
        axes_tuple[1][i].tick_params(axis="x", labelsize=fontsize)
        axes_tuple[1][i].tick_params(axis="y", labelsize=fontsize)
        axes_tuple[1][i].set_yticks(np.arange(-1, np.max(base_task_encodings) + 2, 1))
        axes_tuple[0][i].grid(b=True, which='major', alpha=1)
        axes_tuple[1][i].grid(which='minor')
        axes_tuple[1][i].grid(which='major')
        axes_tuple[0][0].set_ylabel('Encoding $\mathbf{z}$', fontsize=fontsize)
        axes_tuple[1][0].set_ylabel('Encoding $\mathbf{y}$', fontsize=fontsize)
        plt.tight_layout()
    if legend:
        axes_tuple[0][-1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True, fontsize=fontsize)
        axes_tuple[1][-1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True, fontsize=fontsize)
    plt.subplots_adjust(wspace=0.15, hspace=0.15)
    plt.grid(b=True, which='major', alpha=1)
    if save:
        fig.savefig(exp_directory + "/encoding_epoch_" + str(epoch) + ("_normalized" if normalize else "") + ".pdf", format="pdf", bbox_inches = "tight")
    plt.show()
    print(exp_directory)
    print("Created plot")


def plot_encodings(epoch, exp_directory, save=False, normalize=False):
    encoding_storage = pickle.load(open(os.path.join(exp_directory, "encoding_" + str(epoch) + ".p"), "rb"))
    base_tasks = list(encoding_storage.keys())
    fig, axes_tuple = plt.subplots(ncols=len(base_tasks), sharey='row')
    #fig, axes_tuple = plt.subplots(ncols=len(base_tasks), sharey='row')
    fig, axes_tuple = plt.subplots(ncols=len(base_tasks), sharey='row', figsize=(15, 3))
    #fig.suptitle("Epoch " + str(epoch), fontsize="x-large")
    if len(base_tasks) == 1: axes_tuple = [axes_tuple]
    latent_dim = encoding_storage[base_tasks[0]][next(iter(encoding_storage[base_tasks[0]]))]['mean'].shape[0]
    for i, base in enumerate(base_tasks):
        for dim in range(latent_dim):
            x_values = np.array([a['mean'][dim] for a in list(encoding_storage[base].values())])
            y_values = np.array([a['std'][dim] for a in list(encoding_storage[base].values())])
            #Normalize
            if normalize:
                mean = x_values.mean()
                std = x_values.std()
                x_values = (x_values - mean) / (std + 1e-9)
                mean = y_values.mean()
                std = y_values.std()
                y_values = (y_values - mean) / (std + 1e-9)
            axes_tuple[i].errorbar(list(encoding_storage[base].keys()), x_values, yerr=y_values, fmt=".", label="Encoding $\mathbf{z}$")
        axes_tuple[i].plot(list(encoding_storage[base].keys()), [np.argmax(a['base']) for a in list(encoding_storage[base].values())], 'x', label="Class encoding $\mathbf{y}$")
        #axes_tuple[i].set_title("Base Task " + str(i) + ", Epoch " + str(epoch))
        axes_tuple[i].set_title("Base Task " + str(i))
        #axes_tuple[i].set_title("Epoch " + str(epoch))
        #axes_tuple[i].set_xlabel("Specification") #, fontsize=10
        axes_tuple[i].grid()
        #axes_tuple[i].set_ylim(-0.1, 0.1)
        #axes_tuple[i].legend()
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
    #fig.text(0.0, 0.5, 'Encoding $\mathbf{z}$', va='center', rotation='vertical')
    #plt.subplots_adjust(wspace=0, hspace=0)
    if save:
        fig.savefig(exp_directory + "/encoding_epoch" + str(epoch) + ("_normalized" if normalize else "") +".pdf", dpi=300, format="pdf")
    plt.show()
    print("Created plot")


def plot_encodings_2D(epoch, exp_directory):
    encoding_storage = pickle.load(open(os.path.join(exp_directory, "encoding_" + str(epoch) + ".p"), "rb"))
    base_tasks = list(encoding_storage.keys())
    fig, ax = plt.subplots()
    for i, base in enumerate(base_tasks):
        specification = np.array(list(encoding_storage[base].keys()))
        spec_max = specification.max()
        means1 = [a['mean'][0] for a in list(encoding_storage[base].values())]
        means2 = [a['mean'][1] for a in list(encoding_storage[base].values())]
        vars1 = [a['mean'][0] for a in list(encoding_storage[base].values())]
        vars2 = [a['mean'][1] for a in list(encoding_storage[base].values())]
        points = ax.scatter(means1, means2, c=specification, cmap='autumn', zorder=0)
        ax.errorbar(means1, means2, xerr=np.array(vars1) / 2, yerr=np.array(vars2) / 2, alpha=0.2, fmt="o", color="black", zorder=-2)
        for j in range(len(encoding_storage[base])):
            #color = np.expand_dims(np.array(colorsys.hsv_to_rgb(hue[j], 1, 1)), 0)
            e = Ellipse((means1[j], means2[j]), vars1[j], vars2[j], fill=False, zorder=-1)
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(0.2)
            #e.set_color(color[j])

        fig.colorbar(points)
        plt.show()


if __name__ == "__main__":
    #plot_encodings_split(0, "/path/to/exp", save=False, normalize=False)
    pass