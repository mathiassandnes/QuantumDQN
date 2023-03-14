from datetime import datetime

from TrainingHandler import TrainingHandler
from source.utils import Hyperparameters
import os
import multiprocessing

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def run_thread(thread, trials, hparams):
    for trial in range(trials):

        hyperparameters = hparams.select_hyperparameters()

        created_at = datetime.now().strftime(f'%Y-%m-%d_%H-%M-%S_thread_{thread}_trial_{trial}')

        print(f'Running {created_at} \n'
              f'Hyperparameters: {hyperparameters}')

        for run_num in range(5):
            h = TrainingHandler(run_num, created_at, hyperparameters)
            h.run()


if __name__ == '__main__':
    n_procs = 2
    n_trials = 2
    processes = []

    hparams = Hyperparameters()

    for i in range(n_procs):
        p = multiprocessing.Process(target=run_thread, args=(i, n_trials, hparams))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
