import copy
from itertools import combinations

import numpy as np

from Container import Container


class Solver:
    def __init__(self, n_bays, n_stacks, n_tiers):
        self.n_bays = n_bays
        self.n_stacks = n_stacks
        self.n_tiers = n_tiers

        self.flow_x = [
            [[0 for _ in range(n_tiers)] for _ in range(n_stacks)]
            for _ in range(n_bays)
        ]
        self.objective = float("inf")
        # Center-of-gravity
        self.cog = [0, 0]
        self.total_weight_containers = 0

    def copy(self):
        new_solution = Solver(self.n_bays, self.n_stacks, self.n_tiers)

        for bay in range(self.n_bays):
            for stack in range(self.n_stacks):
                for tier in range(self.n_tiers):
                    new_solution.flow_x[bay][stack][tier] = self.flow_x[bay][stack][
                        tier
                    ]

        new_solution.objective = self.objective
        new_solution.cog = self.cog
        new_solution.total_weight_containers = self.total_weight_containers

        return new_solution

    def construct(self):
        """
        Simple construction heuristic.
        Takes the first container in the list, and places it in the
        first location. The next is placed in the second location and
        so on.
        """
        i = 0

        for bay in range(self.n_bays):
            for stack in range(self.n_stacks):
                for tier in range(self.n_tiers):
                    self.flow_x[bay][stack][tier] = i
                    i += 1

    def calculate_objective(self, containers):
        """
        Calculate the objective function for the current solution. Updates self.objective.
        """

        gravity_goal = [self.n_bays / 2.0, self.n_stacks / 2.0]
        gravity_this = [0.0, 0.0]

        sum_container_weight = 0

        for bay in range(self.n_bays):
            for stack in range(self.n_stacks):

                sum_tier = 0

                for tier in range(self.n_tiers):
                    # get container by container_id
                    container = Container.find_container(
                        containers, self.flow_x[bay][stack][tier]
                    )
                    sum_tier += container.weight
                    sum_container_weight += container.weight

                gravity_this[0] += (bay + 0.5) * sum_tier
                gravity_this[1] += (stack + 0.5) * sum_tier

        gravity_this[0] /= sum_container_weight
        gravity_this[1] /= sum_container_weight

        evaluation = (gravity_goal[0] - gravity_this[0]) ** 2 + (
            gravity_goal[1] - gravity_this[1]
        ) ** 2

        self.objective = evaluation
        self.cog = gravity_this
        self.total_weight_containers = sum_container_weight

        return evaluation

    def print_solution(self):
        print("Current solution:")

        for bay in range(self.n_bays):
            for stack in range(self.n_stacks):
                for tier in range(self.n_tiers):
                    print(
                        f"Bay: {bay}, stack: {stack}, tier: {tier}, container: {self.flow_x[bay][stack][tier]}"
                    )

    def construction_improved(self, containers):
        """Places the heaviest container nearest the middle of the ship."""
        # Calculate the desired center of gravity
        cog = np.array([self.n_bays, self.n_stacks]) / 2

        # get euclidean distance from center of gravity to each container
        distances = []
        # start with tier to distribute more evenly
        for tier in range(self.n_tiers):
            # then stacks
            for stack in range(self.n_stacks):
                for bay in range(self.n_bays):
                    distance = np.linalg.norm(cog - np.array([bay + 0.5, stack + 0.5]))
                    # distance =  math.dist([bay + 0.5, stack + 0.5], cog)
                    distances.append(
                        {"bay": bay, "stack": stack, "tier": tier, "dist": distance}
                    )

        # Sort the containers by weight
        sorted_containers = Container.sort_array_weight_descending(containers)

        # Sort the distances by distance
        sorted_distances = sorted(distances, key=lambda x: x["dist"])

        # Place the containers in sorted order
        for idx, container in enumerate(sorted_containers):
            bay = sorted_distances[idx]["bay"]
            stack = sorted_distances[idx]["stack"]
            tier = sorted_distances[idx]["tier"]
            self.flow_x[bay][stack][tier] = container.container_id
            # print(
            #     f"Bay: {bay}, stack: {stack}, tier: {tier}, "
            #     f"container: {container.container_id}, "
            #     f"weight: {container.weight}, "
            #     f"distance: {sorted_distances[idx]['dist']}"
            # )
        # print(self.flow_x)

    def num_to_placement(self, num):
        bay = num % self.n_bays
        stack = (num // self.n_bays) % self.n_stacks
        tier = num // (self.n_stacks * self.n_bays)
        return bay, stack, tier

    def two_swap(self, a, b):
        from_bay, from_stack, from_tier = self.num_to_placement(a)
        to_bay, to_stack, to_tier = self.num_to_placement(b)
        (
            self.flow_x[from_bay][from_stack][from_tier],
            self.flow_x[to_bay][to_stack][to_tier],
        ) = (
            self.flow_x[to_bay][to_stack][to_tier],
            self.flow_x[from_bay][from_stack][from_tier],
        )

    def local_search_two_swap(self, containers):
        improvement = True

        indices = list(range(len(containers)))
        swap_combos = list(combinations(indices, 2))

        while improvement:
            original_objective = copy.deepcopy(self.calculate_objective(containers))

            improvement = False
            improvement_amount = 0
            best_from = None
            best_to = None

            # swap containers and get improvement value
            for swap_combo in swap_combos:
                self.two_swap(swap_combo[0], swap_combo[1])

                cur_obj = copy.deepcopy(self.calculate_objective(containers))

                current_improvement_amount = original_objective - cur_obj

                if current_improvement_amount > improvement_amount:
                    improvement_amount = current_improvement_amount
                    # print(improvement_amount)
                    best_from = swap_combo[0]
                    best_to = swap_combo[1]

                # swap back
                self.two_swap(swap_combo[1], swap_combo[0])

            # if improvement, swap containers
            if improvement_amount > 0:
                improvement = True
                self.two_swap(best_from, best_to)

    def three_swap(self, a, b, c):
        a_bay, a_stack, a_tier = self.num_to_placement(a)
        b_bay, b_stack, b_tier = self.num_to_placement(b)
        c_bay, c_stack, c_tier = self.num_to_placement(c)

        (
            self.flow_x[a_bay][a_stack][a_tier],
            self.flow_x[b_bay][b_stack][b_tier],
            self.flow_x[c_bay][c_stack][c_tier],
        ) = (
            self.flow_x[b_bay][b_stack][b_tier],
            self.flow_x[c_bay][c_stack][c_tier],
            self.flow_x[a_bay][a_stack][a_tier],
        )

    def local_search_three_swap(self, containers):
        improvement = True

        indices = list(range(len(containers)))
        swap_combos = list(combinations(indices, 3))

        while improvement:

            original_objective = copy.deepcopy(self.calculate_objective(containers))

            improvement = False
            improvement_amount = 0
            best_a = None
            best_b = None
            best_c = None

            # swap containers and get improvement value
            for swap_combo in swap_combos:
                self.three_swap(swap_combo[0], swap_combo[1], swap_combo[2])

                cur_obj = copy.deepcopy(self.calculate_objective(containers))
                current_improvement_amount = original_objective - cur_obj

                if current_improvement_amount > improvement_amount:
                    improvement_amount = current_improvement_amount
                    best_a = swap_combo[0]
                    best_b = swap_combo[1]
                    best_c = swap_combo[2]

                # swap back
                self.three_swap(swap_combo[0], swap_combo[2], swap_combo[1])

            # if improvement, swap containers
            if improvement_amount > 0:
                improvement = True
                self.three_swap(best_a, best_b, best_c)

    def tabu_search_heuristic(self, containers, n_iterations):
        pass
