from datetime import datetime

from TrainingHandler import TrainingHandler, select_hyperparameters
import os
import multiprocessing

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def run_thread(thread, trials):
    for trial in range(trials):
        print(f'Running thread: {thread + 1}, trial {trial + 1}')

        hyperparameters = select_hyperparameters()

        created_at = datetime.now().strftime(f'%Y-%m-%d_%H-%M-%S_thread_{thread}_trial_{trial}')
        for run_num in range(3):
            h = TrainingHandler(run_num, created_at, hyperparameters)
            h.run()


if __name__ == '__main__':
    n_procs = 2
    n_trials = 2
    processes = []

    for i in range(n_procs):
        p = multiprocessing.Process(target=run_thread, args=(i, n_trials))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
