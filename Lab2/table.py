import subprocess
import re
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

def compile_and_run(nb_threads, non_blocking, measure,num_runs=20):
    
    max_times = []

    compile_command = f"make NON_BLOCKING={non_blocking} MEASURE={measure} NB_THREADS={nb_threads}"
    subprocess.run(compile_command, shell=True, check=True)

    for _ in range(num_runs):
        run_times = []
        run_command = f"./stack"
        process = subprocess.run(run_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = process.stdout.decode('utf-8')
        matches = re.compile(r'Thread \d+ time: \d+\.\d+').finditer(output)
        if matches:
            for match in matches:
                run_time = float(re.search(r'\d+\.\d+', match.group()).group())
                run_times.append(run_time)
            max_times.append(max(run_times))
        else:
  
            print(f"Error: Couldn't extract time information for {nb_threads} threads and {non_blocking} non_blocking.")
            print(f"run command: {run_command} output: {output}")
            return None


    max_time = sum(max_times)/num_runs
    return max_time

def generate_and_save_plot(non_blocking, measure):
    num_threads_list = [4,5,6,7,8,9,10,11,12,13,14,15,16] 
    timings = []

    for num_threads in num_threads_list:
        timing = compile_and_run(num_threads, non_blocking, measure)
        if timing is not None:
            timings.append((num_threads, timing))

    thread_numbers, run_times = zip(*timings)
    plt.plot(thread_numbers, run_times, marker='o')
    plt.title(f'Maximum Runtime vs Number of Threads (non_blocking: {non_blocking})')
    plt.xlabel('Number of Threads')
    plt.ylabel('Maximum Runtime (seconds)')

    plt.xticks(thread_numbers)

    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.6f'))

    plt.grid(True)
    
    plt.savefig(f'non_blocking={non_blocking} measure={measure}.png')
    plt.close()

def main():
    non_blocking_list = [0, 1] 
    measure_list = [1, 2]
    for non_blocking in non_blocking_list:
        for measure in measure_list:
            generate_and_save_plot(non_blocking, measure)

if __name__ == "__main__":
    main()
