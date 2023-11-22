import subprocess
import re
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

def compile_and_run(nb_threads, non_blocking, num_runs=10):
    run_times = []


    compile_command = f"make NON_BLOCKING={non_blocking} MEASURE=1 NB_THREADS={nb_threads}"
    subprocess.run(compile_command, shell=True, check=True)

    for _ in range(num_runs):
        run_command = f"./stack"
        process = subprocess.run(run_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = process.stdout.decode('utf-8')

        matches = re.compile(r'Thread \d+ time: ').finditer(output)
        if matches:
            for match in matches:
                run_time = float(re.search(r'\d+\.\d+', match.group()).group())
                run_times.append(run_time)
        else:
  
            print(f"Error: Couldn't extract time information for {nb_threads} threads and {non_blocking} non_blocking.")
            print(f"run command: {run_command} output: {output}")
            return None


    average_time = sum(run_times) / num_runs
    return average_time

def generate_and_save_plot(non_blocking):
    num_threads_list = [1, 2, 3,4,5,6,7, 8,16,32] 
    timings = []

    for num_threads in num_threads_list:
        timing = compile_and_run(num_threads, non_blocking)
        if timing is not None:
            timings.append((num_threads, timing))

    thread_numbers, run_times = zip(*timings)
    plt.plot(thread_numbers, run_times, marker='o')
    plt.title(f'Average Runtime vs Number of Threads (non_blocking: {non_blocking})')
    plt.xlabel('Number of Threads')
    plt.ylabel('Average Runtime (seconds)')

    plt.xticks(thread_numbers)

    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

    plt.grid(True)
    
    plt.savefig(f'non_blocking{non_blocking}.png')
    plt.close()

def main():
    non_blocking_list = [0, 1] 
    for non_blocking in non_blocking_list:
        generate_and_save_plot(non_blocking)

if __name__ == "__main__":
    main()
