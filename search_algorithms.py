import random
import math
import data
from solution_checker import isFeasable
from solution_checker import cost
from generators import generate_random_solution
from statistics import mean

def blind_random_search(s0):
    best_sol = s0
    cost_best_sol = cost(best_sol)
    for _ in range(10000):
        cur_sol = generate_random_solution()
        if isFeasable(cur_sol) and cost(cur_sol)<cost_best_sol:
            best_sol = cur_sol
            cost_best_sol = cost(best_sol)
    return best_sol

def local_search(s0,operator):
    best_sol = s0
    cost_best_sol = cost(best_sol)
    for _ in range(10000):
        cur_sol = operator(best_sol)
        if isFeasable(cur_sol) and cost(cur_sol)<cost_best_sol:
            best_sol = cur_sol
            cost_best_sol = cost(best_sol)
            print(best_sol,cost_best_sol)
    return best_sol

def simulated_annealing(s0,operator):
    T_final = 5
    best_sol = s0
    incumbent = s0
    deltas = []

    # find temps
    for _ in range(0,100):
        new_sol = operator(incumbent)
        cost_incumbent = cost(incumbent)
        delta = cost(new_sol) - cost_incumbent
        if delta < 0 and isFeasable(new_sol):
            incumbent = new_sol
            if cost_incumbent < cost(best_sol):
                best_sol = incumbent
        elif isFeasable(new_sol):
            if random.random() < 0.8:
                incumbent = new_sol
            deltas.append(delta)
    if deltas:
        delta_avg = mean(deltas)
    else:
        print("Got no deltas")
        delta_avg = 10000
    T_start = -delta_avg/math.log(0.8)
    alfa = (T_final/T_start)**(1/9990)

    temp = T_start
    cost_incumbent = cost(incumbent)
    for _ in range(9990):
        new_sol = operator(incumbent)
        delta = cost(new_sol) - cost_incumbent
        Feasable = isFeasable(new_sol)
        if delta < 0 and Feasable:
            incumbent = new_sol
            cost_incumbent = cost(incumbent)
            if cost_incumbent < cost(best_sol):
                best_sol = incumbent
        elif Feasable and random.random() < math.exp(-delta/temp):
            incumbent = new_sol
            cost_incumbent = cost(incumbent)
        temp *= alfa
    return best_sol


def simulated_annealing_multiple_ops(s0,operators,weights):
    T_final = 5
    best_sol = s0
    incumbent = s0
    deltas = []

    # find temps
    for _ in range(0,100):
        operator = random.choices(operators,weights)[0]
        new_sol = operator(incumbent)
        cost_incumbent = cost(incumbent)
        delta = cost(new_sol) - cost_incumbent
        if delta < 0 and isFeasable(new_sol):
            incumbent = new_sol
            if cost_incumbent < cost(best_sol):
                best_sol = incumbent
        elif isFeasable(new_sol):
            if random.random() < 0.8:
                incumbent = new_sol
            deltas.append(delta)
    if deltas:
        delta_avg = mean(deltas)
    else:
        print("got no deltas")
        delta_avg = 10000

    T_start = -delta_avg/math.log(0.8)
    alfa = (T_final/T_start)**(1/9990)

    temp = T_start
    cost_incumbent = cost(incumbent)
    for _ in range(9990):
        operator = random.choices(operators,weights)[0]
        new_sol = operator(incumbent)
        delta = cost(new_sol) - cost_incumbent
        Feasable = isFeasable(new_sol)
        if delta < 0 and Feasable:
            incumbent = new_sol
            cost_incumbent = cost(incumbent)
            if cost_incumbent < cost(best_sol):
                best_sol = incumbent
                print(incumbent, cost_incumbent, operator)
        elif Feasable and random.random() < math.exp(-delta/temp):
            incumbent = new_sol
            cost_incumbent = cost(incumbent)
        temp *= alfa
    return best_sol
