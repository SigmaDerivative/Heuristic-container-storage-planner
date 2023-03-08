from Container import Container
from Solver import Solver
from Vessel import Vessel

# Set data file
filename = "data/instance2.txt"

if __name__ == "__main__":
    # Read from data file
    with open(filename, "r", encoding="UTF-8") as datafile:
        data = datafile.readlines()

        vessel_dimensions = list(map(int, data[:3]))

        weights = list(map(int, data[3:]))

    vessel = Vessel(*vessel_dimensions)

    containers = []
    for idx, weight in enumerate(weights):
        containers.append(Container(idx, weight))

    # Create a Solver object
    solver = Solver(vessel.n_bays, vessel.n_stacks, vessel.n_tiers)

    # Construct an initial solution and evaluate it
    print("Default construct")
    solver.construct()
    solver.calculate_objective(containers)
    print(solver.objective)

    print("TASK 1")
    solver.construction_improved(containers)
    solver.calculate_objective(containers)
    print(solver.objective)

    # Improvement phase
    print("TASK 2A")
    solver.local_search_two_swap(containers, n_iterations=100, greedy=True)
    solver.calculate_objective(containers)
    print(solver.objective)

    print("TASK 2B")
    solver.local_search_three_swap(containers, n_iterations=100, greedy=True)
    solver.calculate_objective(containers)
    print(solver.objective)

    print("TASK 3")
    solver.tabu_search_heuristic(containers, n_iterations=100)
    solver.calculate_objective(containers)
    print(solver.objective)

    # Print the solution
    # solver.print_solution()
