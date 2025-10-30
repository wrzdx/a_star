from concurrent.futures import ProcessPoolExecutor
from statistics import mean, median, stdev
import time
from interactor import Interactor
import multiprocessing as mp
from typing import Dict, Any

A_STAR = "a_star.py"
BACKTRACKING = "backtracking.py"


def print_statistics(stats: Dict[str, Any]):
    print()
    print("ALGORITHM COMPARISON STATISTICS")
    print("=" * 100)

    print(
        f"\n{'Algorithm':>12} {'Radius':>8} {'Wins':>6} {'Losses':>8} {'Win Rate':>10}  {'Total':>8} {'Mean Time':>12}  {'Median Time':>12}  {'Std Time':>12}"
    )
    print("-" * 100)

    for radius_name, radius_stats in stats.items():
        radius_num = radius_name.split("_")[1]

        for algo_name, algo_stats in radius_stats.items():
            wins = algo_stats["wins"]
            losses = algo_stats["losses"]
            total = wins + losses
            win_rate = (wins / total * 100) if total > 0 else 0

            algo_display = "A*" if "a_star" in algo_name else "Backtracking"

            if algo_stats["times"]:
                times = algo_stats["times"]
                mean_time = mean(times)
                median_time = median(times)
                std_time = stdev(times) if len(times) > 1 else 0

                print(
                    f"{algo_display:>12} {radius_num:>8} {wins:>6} {losses:>8} {win_rate:>9.1f}% {total:>8} {mean_time:>12.3f}s {median_time:>12.3f}s {std_time:>12.3f}s"
                )
            else:
                print(
                    f"{algo_display:>12} {radius_num:>8} {wins:>6} {losses:>8} {win_rate:>9.1f}% {total:>8} {mean_time:>12.3f}s {median_time:>12.3f}s {std_time:>12.3f}s"
                )

            print


A_STAR = "a_star.py"
BACKTRACKING = "backtracking.py"


def run_single_iteration(iteration_id):
    """Запускает одну итерацию сравнения"""
    try:
        stats = {
            "radius_1": {
                "a_star": {"wins": 0, "losses": 0, "times": []},
                "backtracking": {"wins": 0, "losses": 0, "times": []},
            },
            "radius_2": {
                "a_star": {"wins": 0, "losses": 0, "times": []},
                "backtracking": {"wins": 0, "losses": 0, "times": []},
            },
        }

        interactor = Interactor(1)
        interactor.set_random_tokens()
        x, y = interactor.gollum

        # Radius 1 - A*
        start_time = time.time()
        answer_a_star_1, history_a_star_1 = interactor.start(A_STAR)
        time_a_star_1 = time.time() - start_time
        path_a_star_1 = history_a_star_1.splitlines()[-1]

        # Radius 1 - Backtracking
        interactor.update_map(False)
        interactor.set_token(x, y, "G")
        start_time = time.time()
        answer_backtracking_1, history_backtracking_1 = interactor.start(BACKTRACKING)
        time_backtracking_1 = time.time() - start_time
        path_backtracking_1 = history_backtracking_1.splitlines()[-1]

        # Radius 2 - A*
        interactor.radius = 2
        interactor.update_map(False)
        interactor.set_token(x, y, "G")
        start_time = time.time()
        answer_a_star_2, history_a_star_2 = interactor.start(A_STAR)
        time_a_star_2 = time.time() - start_time
        path_a_star_2 = history_a_star_2.splitlines()[-1]

        # Radius 2 - Backtracking
        interactor.update_map(False)
        interactor.set_token(x, y, "G")
        start_time = time.time()
        answer_backtracking_2, history_backtracking_2 = interactor.start(BACKTRACKING)
        time_backtracking_2 = time.time() - start_time
        path_backtracking_2 = history_backtracking_2.splitlines()[-1]

        # Reference answer
        interactor.radius = 5
        interactor.update_map(False)
        interactor.set_token(x, y, "G")
        reference_answer, _ = interactor.start(A_STAR)

        # Update stats for radius 1
        if path_a_star_1 != "Failed":
            stats["radius_1"]["a_star"]["wins"] += (answer_a_star_1 == reference_answer)
            stats["radius_1"]["a_star"]["losses"] += (answer_a_star_1 != reference_answer)
            stats["radius_1"]["a_star"]["times"].append(time_a_star_1)

        if path_backtracking_1 != "Failed":
            stats["radius_1"]["backtracking"]["wins"] += (answer_backtracking_1 == reference_answer)
            stats["radius_1"]["backtracking"]["losses"] += (answer_backtracking_1 != reference_answer)
            stats["radius_1"]["backtracking"]["times"].append(time_backtracking_1)

        # Update stats for radius 2
        if path_a_star_2 != "Failed":
            stats["radius_2"]["a_star"]["wins"] += (answer_a_star_2 == reference_answer)
            stats["radius_2"]["a_star"]["losses"] += (answer_a_star_2 != reference_answer)
            stats["radius_2"]["a_star"]["times"].append(time_a_star_2)

        if path_backtracking_2 != "Failed":
            stats["radius_2"]["backtracking"]["wins"] += (answer_backtracking_2 == reference_answer)
            stats["radius_2"]["backtracking"]["losses"] += (answer_backtracking_2 != reference_answer)
            stats["radius_2"]["backtracking"]["times"].append(time_backtracking_2)

        return stats

    except Exception as e:
        print(f"Error in iteration {iteration_id}: {e}")
        return None


def run_comparison_parallel(number_of_runs: int = 1000):
    """Запускает сравнение в нескольких процессах"""
    
    # Определяем количество процессов (можно настроить)
    num_processes = min(mp.cpu_count(), 8)  # Не более 8 процессов
    
    print(f"Running {number_of_runs} iterations using {num_processes} processes...")
    
    aggregated_stats = {
        "radius_1": {
            "a_star": {"wins": 0, "losses": 0, "times": []},
            "backtracking": {"wins": 0, "losses": 0, "times": []},
        },
        "radius_2": {
            "a_star": {"wins": 0, "losses": 0, "times": []},
            "backtracking": {"wins": 0, "losses": 0, "times": []},
        },
    }

    completed = 0
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(run_single_iteration, i) for i in range(number_of_runs)]
        
        for future in futures:
            try:
                result = future.result(timeout=300)  # 5 минут таймаут на итерацию
                if result:
                    for radius in ["radius_1", "radius_2"]:
                        for algo in ["a_star", "backtracking"]:
                            aggregated_stats[radius][algo]["wins"] += result[radius][algo]["wins"]
                            aggregated_stats[radius][algo]["losses"] += result[radius][algo]["losses"]
                            aggregated_stats[radius][algo]["times"].extend(result[radius][algo]["times"])
                
                completed += 1
                elapsed = time.time() - start_time
                eta = (elapsed / completed) * (number_of_runs - completed) if completed > 0 else 0
                
                print(f"\rProgress: {completed}/{number_of_runs} ({completed/number_of_runs*100:.1f}%) "
                      f"Elapsed: {elapsed:.1f}s ETA: {eta:.1f}s", end="", flush=True)
                      
            except Exception as e:
                print(f"\nError processing future: {e}")
                continue

    print(f"\nCompleted {completed} iterations in {time.time() - start_time:.1f} seconds")
    return aggregated_stats


if __name__ == "__main__":
    print("Starting algorithm comparison...")
    stats = run_comparison_parallel(1000)
    print_statistics(stats)