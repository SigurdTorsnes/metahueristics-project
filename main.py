import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
from statistics import mean
import data_formating
import data
from generators import gen_dummy_sol
import search_algorithms as algs
import operators as ops
from solution_checker import cost
from solution_checker import isFeasable
from solution_checker import cost_outsource
from solution_checker import cost_of_vehicle
import time
import helpers as help

case1 = 'Call_7_Vehicle_3'
case2 = 'Call_18_Vehicle_5'
case3 = 'Call_35_Vehicle_7'
case4 = 'Call_80_Vehicle_20'
case5 = 'Call_130_Vehicle_40'
case6 = 'Call_300_Vehicle_90'


def test_algorithm():
    costs = []
    best_sol = gen_dummy_sol()
    best_cost = cost(best_sol)
    for i in range(1):
        sol = gen_dummy_sol()
        sol = algs.simulated_annealing_multiple_ops(sol,[ops.reinsert_1,ops.swap2_similar,ops.improved_reinsert])
        # sol = algs.local_search(sol,ops.improved_reinsert)
        # sol = algs.local_search(sol, ops.reinsert_1)
        c = cost(sol)
        if c < best_cost:
            best_sol = sol
            best_cost = c
        print(sol,c)
        costs.append(c)
    average_cost = mean(costs)
    return best_sol, best_cost, costs, average_cost

def test_operator(operator):
    # sol = [4, 4, 0, 2, 2, 0, 5, 5, 3, 3, 0, 1, 1, 6, 6, 7, 7]
    for i in range(15):
        sol = gen_dummy_sol()
        sol = operator(sol)
        print(sol, cost(sol), isFeasable(sol))
    return sol

def add_instance(search_algorithm,name):
    start = time.time()
    start_sol = gen_dummy_sol()
    total_cost = 0
    start_cost = cost(start_sol)
    solutions = []
    amount = 10
    best_sol = start_sol
    
    for i in range(amount):
        cur_sol = search_algorithm(start_sol,ops.swap2)
        if cost(cur_sol) < cost(best_sol):
            best_sol = cur_sol
        solutions.append(cur_sol)
        total_cost += cost(cur_sol)
    end = time.time()
    avg_time = (end - start)/10

    avg = total_cost/amount
    b_c = cost(best_sol)

    improvement = (start_cost-b_c)/start_cost
    improvement *= 100
    print(best_sol, cost(best_sol),start_cost)
    # print(start_cost,cost(best_sol))
    instance = np.array([[name,round(avg),cost(best_sol),round(improvement,1),round(avg_time,1)]])
    return instance

cur_case = case1
data.init() # initialize global data variables
data_formating.read_data(cur_case) # fill global data variables with data
data_formating.generate_data()

######
# instance = add_instance(algs.simulated_annealing,"SA")
# header = [['',cur_case,cur_case,cur_case],
#           ['Average objective','Best objective', 'Improvement (%)','Running time (seconds)']]
# df = pd.DataFrame(instance,columns=['Algorithm','Average objective','Best objective', 'Improvement (%)','Running time'])
# df.set_index('Algorithm',inplace=True)
# df.columns = header
# print(df)
######


# sol =  [4, 4, 2, 2, 0, 7, 7, 0, 1, 3, 1, 3, 0, 6, 6, 5, 5]

# v = help.pick_good_vehicle_by_call(sol,2)
# s,e = help.find_vehicle_indices(sol,v)

# print(v,s,e,sol)
# test_operator(ops.improved_reinsert)
sol = gen_dummy_sol()
sol = algs.simulated_annealing_multiple_ops(sol,[ops.improved_reinsert,ops.improved_reinsert,ops.improved_reinsert])

print((sol,cost(sol)))