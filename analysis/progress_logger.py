import csv
from datetime import datetime
import matplotlib.pyplot as plt
import json
import os
import numpy as np
import argparse

# plt.style.use('seaborn')


base_dir = os.path.dirname(os.path.abspath(__file__))
progress_file = base_dir + '/' + 'progress.csv'
variant_file = base_dir + '/' + 'variant.json'
database_file = base_dir + '/' + 'experiment_database.json'


def load_json(json_string):
    with open(json_string, 'r') as f:
        return json.load(f)


def dump_json(json_string, dict_to_dump):
    with open(json_string, 'w') as f:
        json.dump(dict_to_dump, f, indent=3)


def delete_from_database(id):
    database = load_json(database_file)
    del database[str(id)]
    dump_json(database_file, database)


def comment_experiment(id):
    database = load_json(database_file)
    comment = input("\nWrite your comment: ")
    database[str(id)]["description"] = comment
    dump_json(database_file, database)


def show_entries(entries='all', things=["name", "algo", "env", "date"]):
    database = load_json(database_file)
    if entries == "all":
        keys = database.keys()
    else:
        keys = [str(x) for x in entries]

    res = dict((key, dict((a, database[key][a]) for a in things)) for key in keys)
    for key in keys:
        print(key, json.dumps(res[key]))


def plot_multiple_runs(experiments, benchmark=None, log=True, name=None, save=False):
    database = load_json(database_file)
    experiments = [str(x) for x in experiments]
    x_axis = "n_timesteps"
    y_axis = "testAverageReturn"

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    def plot_median_interquartile(data, name, color):
        x = database[data[0]][x_axis]
        y = np.zeros((len(data), len(x)))
        for i, exp in enumerate(data):
            assert database[exp][x_axis] == x
            y[i] = np.array(database[exp][y_axis])

        median = np.nanmedian(y, axis=0)
        percentile_25 = np.nanpercentile(y, 25, axis=0)
        percentile_75 = np.nanpercentile(y, 75, axis=0)
        plt.plot(x, median, label=name, color=cycle[color])
        plt.fill_between(x, percentile_25, percentile_75, alpha=0.25, color=cycle[color], linewidth=0)

    fig = plt.figure()
    plot_median_interquartile(experiments, "CEMRL", 0)
    if benchmark is not None:
        benchmark = [str(x) for x in benchmark]
        plot_median_interquartile(benchmark, "PEARL", 3)

    plt.grid(b=True, which='major', alpha=1)
    fontsize=26
    plt.legend(fontsize=20, loc='upper left')
    plt.xlabel("Training transition $n$", fontsize=fontsize)
    plt.ylabel("Average Return $R$", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title(name, fontsize=26)
    #fig.axes[0].set_xticks(np.arange(20000, 250000, 50000)) # for cheetah
    plt.xscale("log")
    plt.tight_layout()
    if save:
        now = datetime.now()
        datename = now.strftime("%Y_%m_%d_%H_%M_%S")
        plt.savefig(os.path.dirname(os.path.realpath(__file__)) + "/figures/" + datename + '_' + name + ".pdf", dpi=600, format="pdf")
    plt.show()


def manage_logging(exp_dir, save=False):
    # Check if file is already logged
    database = load_json(database_file)
    exp_name = exp_dir.split('/')[-1]
    env_name = exp_dir.split('/')[-2]

    exp_id = -1
    for key in database.keys():
        if database[key]["name"] == exp_name:
            exp_id = key
            break

    if exp_id != -1:
        plot_multiple_runs([exp_id], benchmark=None, name=env_name, save=save)
    else:
        new_id = add_experiment_to_database(exp_name, "cemrl", env_name, '_'.join(exp_name.split('_')[:3]), "no comment", "0.0", "0.0",
                                   progress_name=exp_dir + "/" + "progress.csv", variant_name=exp_dir + "/" + "variant.json")
        plot_multiple_runs([new_id], benchmark=None, name=env_name, save=False)


def add_experiment_to_database(name, algo, env, date, description, algo_version, env_version, progress_name=progress_file, variant_name=variant_file):
    '''
    Add experiment data from different algorithms to database
    Following attributes are saved:
    name
    algo
    env
    date
    description
    param_log
    n_timesteps
    time
    trainAverageReturn
    testAverageReturn
    '''
    database = load_json(database_file)

    # read from progress file
    columns = []
    with open(progress_name, 'rU') as f:
        reader = csv.reader(f)
        for row in reader:
            if columns:
                for i, value in enumerate(row):
                    columns[i].append(value)
            else:
                # first row
                columns = [[value] for value in row]
    # you now have a column-major 2D array of your file.
    progress_dict = {c[0]: c[1:] for c in columns}

    # find out next id
    if list(database.keys()) == []:
        new_id = 0
    else:
        new_id = str(max([int(x) for x in list(database.keys())]) + 1)

    database[new_id] = {}
    database[new_id]["name"] = name
    database[new_id]["algo"] = algo
    database[new_id]["env"] = env
    database[new_id]["env_version"] = env_version
    database[new_id]["date"] = date
    database[new_id]["description"] = description

    if algo == "r2l":
        database[new_id]["algo_version"] = "0"
        database[new_id]["param_log"] = load_json("params.json")
        database[new_id]["n_timesteps"] = [float(i) for i in progress_dict["n_timesteps"]]
        database[new_id]["time"] = [float(i) for i in progress_dict["Time"]]
        database[new_id]["trainAverageReturn"] = [float(i) for i in progress_dict["train-AverageReturn"]]
        database[new_id]["testAverageReturn"] = [float(i) for i in progress_dict["train-AverageReturn"]]

    elif algo == "pearl":
        database[new_id]["algo_version"] = algo_version
        database[new_id]["param_log"] = load_json(variant_name)
        database[new_id]["n_timesteps"] = [float(i) for i in progress_dict["Number of env steps total"]]
        database[new_id]["time"] = [float(i) for i in progress_dict["Total Train Time (s)"]]
        database[new_id]["trainAverageReturn"] = [float(i) for i in progress_dict["AverageReturn_all_train_tasks"]]
        database[new_id]["testAverageReturn"] = [float(i) for i in progress_dict["AverageReturn_all_test_tasks"]]

    elif algo == "cemrl":
        database[new_id]["algo_version"] = algo_version
        database[new_id]["param_log"] = load_json(variant_name)
        database[new_id]["n_timesteps"] = [float(i) for i in progress_dict["n_env_steps_total"]]
        database[new_id]["time"] = [float(i) for i in progress_dict["time_total"]]
        database[new_id]["trainAverageReturn"] = [float(i) for i in progress_dict["train_eval_avg_reward_deterministic"]]
        database[new_id]["testAverageReturn"] = [float(i) for i in progress_dict["test_eval_avg_reward_deterministic"]]
        if "train_eval_success_rate" in progress_dict:
            database[new_id]["trainSuccessRate"] = [float(i) for i in progress_dict["train_eval_success_rate"]]
        if "test_eval_success_rate" in progress_dict:
            database[new_id]["testSuccessRate"] = [float(i) for i in progress_dict["test_eval_success_rate"]]

    dump_json(database_file, database)
    return new_id


def get_all_from_env(env_name):
    database = load_json(database_file)

    own_exp_ids = []
    benchmark_exp_ids = []
    for key in database.keys():
        if database[key]["env"] == env_name and database[key]["algo"] == "cemrl":
            own_exp_ids.append(int(key))
        elif database[key]["env"] == env_name and database[key]["algo"] == "pearl":
            benchmark_exp_ids.append(int(key))
    return own_exp_ids, benchmark_exp_ids


def add_experiement_manually():
    name = "2020_12_20_11_14_49"
    algo = "pearl"
    env = "cheetah-stationary-vel"
    date = "2020_12_20"
    description = "pearl reproduction"
    algo_version = "0.0"
    env_version = "0.0"
    add_experiment_to_database(name, algo, env, date, description, algo_version, env_version)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default=None)
    parser.add_argument('--exp_id', type=int, default=-1)
    parser.add_argument('--save', dest='save', action='store_true')
    parser.set_defaults(save=False)
    parser.add_argument('--list', dest='list', action='store_true')
    parser.set_defaults(list=False)
    args = parser.parse_args()

    if args.env_name is not None:
        own_exp_ids, benchmark_exp_ids = get_all_from_env(args.env_name)
        if benchmark_exp_ids == []:
            benchmark_exp_ids = None
        plot_multiple_runs(own_exp_ids, benchmark=benchmark_exp_ids, name=args.env_name, save=args.save)

    if args.exp_id >= 0:
        plot_multiple_runs([args.exp_id], benchmark=None, save=args.save)

    if args.list is True:
        show_entries('all')


    #delete_from_database(0)
    #show_entries('all')
    #show_entries([49], ["name", "algo", "algo_version", "env_version", "env", "date", "description", "param_log"])
    #plot_multiple_runs([0], benchmark=None, name="some experiment", save=False)

if __name__ == '__main__':
    main()
