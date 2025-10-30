from statistics import mean, median, stdev
import time
from interactor import Interactor
from typing import Dict, Any

A_STAR = "a_star.py"
BACKTRACKING = "backtracking.py"


def run_comparison(number_of_runs: int = 100):
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

    for i in range(number_of_runs):
        print(
            f"\rProgress: {100 * i / number_of_runs:.1f}%", end="", flush=True
        )

        interactor = Interactor(1)
        interactor.set_random_tokens()
        x, y = interactor.gollum

        start_time = time.time()
        answer_a_star_1, history_a_star_1 = interactor.start(A_STAR)
        time_a_star_1 = time.time() - start_time
        path_a_star_1 = history_a_star_1.splitlines()[-1]

        interactor.update_map(False)
        interactor.set_token(x, y, "G")
        start_time = time.time()
        answer_backtracking_1, history_backtracking_1 = interactor.start(
            BACKTRACKING
        )
        time_backtracking_1 = time.time() - start_time
        path_backtracking_1 = history_backtracking_1.splitlines()[-1]

        interactor.radius = 2
        interactor.update_map(False)
        interactor.set_token(x, y, "G")

        start_time = time.time()
        answer_a_star_2, history_a_star_2 = interactor.start(A_STAR)
        time_a_star_2 = time.time() - start_time
        path_a_star_2 = history_a_star_2.splitlines()[-1]

        interactor.update_map(False)
        interactor.set_token(x, y, "G")
        start_time = time.time()
        answer_backtracking_2, history_backtracking_2 = interactor.start(
            BACKTRACKING
        )
        time_backtracking_2 = time.time() - start_time
        path_backtracking_2 = history_backtracking_2.splitlines()[-1]

        interactor.radius = 5
        interactor.update_map(False)
        interactor.set_token(x, y, "G")
        reference_answer, _ = interactor.start(A_STAR)

        if path_a_star_1 != "Failed":
            stats["radius_1"]["a_star"]["wins"] += (
                answer_a_star_1 == reference_answer
            )
            stats["radius_1"]["a_star"]["losses"] += (
                answer_a_star_1 != reference_answer
            )
            stats["radius_1"]["a_star"]["times"].append(time_a_star_1)

        if path_backtracking_1 != "Failed":
            stats["radius_1"]["backtracking"]["wins"] += (
                answer_backtracking_1 == reference_answer
            )
            stats["radius_1"]["backtracking"]["losses"] += (
                answer_backtracking_1 != reference_answer
            )
            stats["radius_1"]["backtracking"]["times"].append(
                time_backtracking_1
            )

        if path_a_star_2 != "Failed":
            stats["radius_2"]["a_star"]["wins"] += (
                answer_a_star_2 == reference_answer
            )
            stats["radius_2"]["a_star"]["losses"] += (
                answer_a_star_2 != reference_answer
            )
            stats["radius_2"]["a_star"]["times"].append(time_a_star_2)

        if path_backtracking_2 != "Failed":
            stats["radius_2"]["backtracking"]["wins"] += (
                answer_backtracking_2 == reference_answer
            )
            stats["radius_2"]["backtracking"]["losses"] += (
                answer_backtracking_2 != reference_answer
            )
            stats["radius_2"]["backtracking"]["times"].append(
                time_backtracking_2
            )

    return stats


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


# Run the comparison and print statistics
if __name__ == "__main__":
    print("Starting algorithm comparison...")
    stats = run_comparison(10)
    print_statistics(stats)
