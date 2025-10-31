from interactor import Interactor
from utility import print_map_colored

while True:
    interactor = Interactor(1)
    interactor.set_random_tokens()
    x, y = interactor.gollum

    answer_a_star_1, history_a_star_1 = interactor.start("a_star.py")
    path_a_star_1 = history_a_star_1.splitlines()[-1]

    interactor.radius = 5
    interactor.update_map(False)
    interactor.set_token(x, y, "G")
    reference_answer, history_a_star_2 = interactor.start("a_star.py")
    path_a_star_2 = history_a_star_2.splitlines()[-1]

    if answer_a_star_1 != reference_answer and answer_a_star_1 == -1 and path_a_star_1:
        # interactor.update_map(False)
        interactor.set_token(x, y, "G")
        print("A* with raduis 1:")
        print_map_colored(interactor.world_map, path_a_star_1)
        print()
        print("A* with raduis 5:")
        print_map_colored(interactor.world_map, path_a_star_2)
        print()

        break