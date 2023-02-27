from TrainingHandler import TrainingHandler
import os
import threading

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cProfile
import pstats


def run_thread(thread, trials):
    for trial in range(trials):
        print(f'Running thread: {thread + 1}, trial {trial + 1}')

        with cProfile.Profile() as pr:
            h = TrainingHandler(thread, trial)
            h.run()

        stats = pstats.Stats(pr)
        stats.sort_stats(pstats.SortKey.TIME)
        stats.print_stats()


if __name__ == '__main__':
    n_threads = 1
    n_trials = 1
    threads = []

    # run every configuration thrice

    for thread in range(n_threads):
        t = threading.Thread(target=run_thread, args=(thread, n_trials))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
