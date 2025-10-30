import random
from statistics import mean, median, mode, stdev
import time
from interactor import Interactor
from utility import print_map_colored


A_STAR = "a_star.py"
BACKTRACKING = "backtracking.py"


def run_comparison(number_of_runs: int = 100):
    stats = {
        "a_star": {"wins": 0, "losses": 0, "times": []},
        "backtracking": {"wins": 0, "losses": 0, "times": []},
    }

    for i in range(1, number_of_runs + 1):
        # Setup interactor with random tokens
        interactor = Interactor(random.randint(1, 2))
        interactor.set_random_tokens()
        x, y = interactor.gollum

        # A*
        start_time = time.time()
        answer0, history0 = interactor.start(A_STAR)
        time0 = time.time() - start_time
        path0 = history0.splitlines()[-1]

        # Clear map for next run
        interactor.update_map(False)
        interactor.set_token(x, y, "G")

        # Backtracking
        start_time = time.time()
        answer1, history1 = interactor.start(BACKTRACKING)
        time1 = time.time() - start_time
        path1 = history1.splitlines()[-1]

        # Clear map for next run
        interactor.update_map(False)
        interactor.set_token(x, y, "G")
        # Answer

        interactor.radius = 5
        answer2, history2 = interactor.start(A_STAR)

        # Check for failures
        if path0 == "Failed" or path1 == "Failed":
            interactor.update_map(False)
            interactor.set_token(x, y, "G")
            print_map_colored(interactor.world_map)
            break

        # Update statistics
        stats["a_star"]["wins"] += answer0 == answer2
        stats["a_star"]["losses"] += answer0 != answer2
        stats["a_star"]["times"].append(time0)

        stats["backtracking"]["wins"] += answer1 == answer2
        stats["backtracking"]["losses"] += answer1 != answer2
        stats["backtracking"]["times"].append(time1)

        print(f"\rProgress: {100 * i / number_of_runs:.1f}%", end="", flush=True)

    return stats


def print_statistics(stats):
    print("\n" + "=" * 60)
    print("Statistics:")
    print("=" * 60)

    for algo in ["a_star", "backtracking"]:
        data = stats[algo]
        total = data["wins"] + data["losses"]
        win_rate = (data["wins"] / total * 100) if total > 0 else 0

        print(f"\n{algo.upper()}:")
        print(f"  Wins: {data['wins']}")
        print(f"  Losses: {data['losses']}")
        print(f"  Win Rate: {win_rate:.1f}%")

        if data["times"]:
            print("  Time Statistics (seconds):")
            print(f"    mean = {mean(data['times']):.3f}")
            print(f"    median = {median(data['times']):.3f}")
            print(f"    mode = {mode(map(lambda x: round(x, 3), data['times']))}")
            print(f"    std = {stdev(data['times']):.3f}")


# Run the comparison and print statistics
stats = run_comparison(100)
print_statistics(stats)
