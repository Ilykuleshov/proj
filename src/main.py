
import os
import yaml
import argparse

from experiment import Experiment


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path', help="Path to experiment_config.yaml")

    args = parser.parse_args()

    with open(args.yaml_path) as f:
        params = yaml.safe_load(f)

    save_folder = params["logging"]["save_folder"]

    try:
        tau = params["method"]["tau"]
    except KeyError:
        tau = None

    try:
        gamma = params["method"]["gamma"]
    except KeyError:
        gamma = None

    try:
        batch_size = params["method"]["batch_size"]
    except KeyError:
        batch_size = None

    if save_folder is not None and not os.path.exists(save_folder):
        os.makedirs(save_folder)

    with open(os.path.join(params["logging"]["save_folder"], "config.yml"), 'w') as f:
        yaml.dump(params, f)

    experiment = Experiment(
        n=params["task"]["matrix"]["n"],
        kappa=params["task"]["matrix"]["kappa"],
        scale=params["task"]["scale"],
        alpha=params["method"]["alpha"],
        beta=params["method"]["beta"],
        tau=tau,
        gamma=gamma,
        batch_size=batch_size,
        save_folder=save_folder,
    )

    print(experiment.__dict__)

    experiment.run(num_iters=params["method"]["num_iters"])
    print("Training is completed.")

