import subprocess
import re
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

def compile_and_run(nb_threads, load_balance, num_runs=10):
    run_times = []


    compile_command = f"make MEASURE=1 NB_THREADS={nb_threads} LOADBALANCE={load_balance}"
    subprocess.run(compile_command, shell=True, check=True)

    for _ in range(num_runs):
        run_command = f"./mandelbrot-256-500-375--2-0.6--1-1-{nb_threads}-{load_balance}"
        process = subprocess.run(run_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = process.stdout.decode('utf-8')

        match = re.search(r'Time: (\d+\.\d+)', output)
        if match:
            run_time = float(match.group(1))
            run_times.append(run_time)
        else:
            print(f"Error: Couldn't extract time information for {nb_threads} threads and {load_balance} load balance.")
            return None

    average_time = sum(run_times) / num_runs
    return average_time

def generate_and_save_plot(load_balance):
    num_threads_list = [1, 2, 3,4,5,6,7, 8,16,32] 
    timings = []

    for num_threads in num_threads_list:
        timing = compile_and_run(num_threads, load_balance)
        if timing is not None:
            timings.append((num_threads, timing))

    thread_numbers, run_times = zip(*timings)
    plt.plot(thread_numbers, run_times, marker='o')
    plt.title(f'Average Runtime vs Number of Threads (Load Balance: {load_balance})')
    plt.xlabel('Number of Threads')
    plt.ylabel('Average Runtime (seconds)')

    plt.xticks(thread_numbers)

    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

    plt.grid(True)
    
    plt.savefig(f'load_balance{load_balance}.png')
    plt.close()

def main():
    load_balance_list = [0, 1, 2] 
    for load_balance in load_balance_list:
        generate_and_save_plot(load_balance)

if __name__ == "__main__":
    main()
