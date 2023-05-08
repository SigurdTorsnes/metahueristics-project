import random
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
from statistics import mean
import data_preperation
import data
from generators import gen_dummy_sol
import search_algorithms as algs
import operators as ops
from solution_checker import cost
from solution_checker import isFeasable
from solution_checker import cost_outsource
from solution_checker import cost_of_vehicle
import time
import sys
import helpers
import final_operators as fops

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
        sol = algs.simulated_annealing_multiple_ops(sol,[ops.reinsert_1,ops.swap2_similar,ops.smart_one_insert])
        c = cost(sol)
        if c < best_cost:
            best_sol = sol
            best_cost = c
        print(sol,c)
        costs.append(c)
    average_cost = mean(costs)
    return best_sol, best_cost, costs, average_cost

def test_operator(operator):
    sol = gen_dummy_sol()
    temp = sol
    
    print(temp)
    for i in range(15):
        temp = operator(sol)
        if isFeasable(temp):
            sol = temp
        print(cost(temp), isFeasable(temp))
    return sol
def test_k_operator(operator,sol,k):
    temp = sol[:]
    print(temp)
    for i in range(15):
        temp = operator(sol,k)
        if isFeasable(temp):
            sol = temp
        print(temp,cost(temp), isFeasable(temp))
    return sol

def add_instance(search_algorithm,name,r_ops,i_ops):
    start = time.time()
    start_sol = gen_dummy_sol()
    total_cost = 0
    start_cost = cost(start_sol)
    solutions = []
    amount = 10
    best_sol = start_sol
    
    for i in range(amount):
        cur_sol = search_algorithm(start_sol,r_ops,i_ops)
        if cost(cur_sol) < cost(best_sol):
            best_sol = cur_sol
        solutions.append(cur_sol)
        total_cost += cost(cur_sol)
    end = time.time()
    avg_time = (end - start)/amount

    avg = total_cost/amount
    b_c = cost(best_sol)

    improvement = (start_cost-b_c)/start_cost
    improvement *= 100
    # print('\n',best_sol, cost(best_sol),start_cost)
    instance = np.array([[name,round(avg),cost(best_sol),round(improvement,1),round(avg_time,1)]])
    return instance

cur_case = case3
data.init() # initialize global data variables
data_preperation.read_data(cur_case) # fill global data variables with data
data_preperation.generate_data()
sys.stdout = open('VisualizeRuns/testing.txt','wt')


# operators = [ops.smart_one_insert,ops.swap_similar_vehicles,ops.swap2_similar]#,ops.reinsert_1]

###############################
# time:
# case 1  11 s
# case 2  78 s / random r best insert 30 s / 
# case 3  550 s / 400s / 270 w k=1 / 123s w random + 1insert / 
# case 4: 2300 s
# case 5: :(
# case 6: :(
# new time:
# case 1: 15 s
# case 2: 56 s
# case 3: 173 s
# case 4: 511 s
# case 5: 954 s
# case 6:

# new new time:
# case 1: 10s
# case 2: 38 
# case 3: 138 ss
# case 4: 280s :)
# case 5: 481s
# 63 costly random similar vehicle
# 47 coslty random similar
# 48 random similar

# escape algorithm takes about 20-30 seconds of the search
# removal with random insert: 
#   k_similar   : 6s
#   k_costly    : 20s
#   k_random    : 5s

# insertion with random remove:
# k_quick           : 34s
# first_feasible    : 54s / 35s if break at infeasible
# k_costly          : 138s
removal_ops = [fops.remove_k_random,fops.remove_k_costly,fops.remove_k_similar]
insertion_ops = [fops.insert_k_best_pos, fops.insert_first_feasible,fops.insert_k_quick]

instance = add_instance(algs.ALNS,"ALNS",removal_ops,insertion_ops)
header = [['',cur_case,cur_case,cur_case],
          ['Average objective','Best objective', 'Improvement (%)','Running time (seconds)']]
df = pd.DataFrame(instance,columns=['Algorithm','Average objective','Best objective', 'Improvement (%)','Running time'])
df.set_index('Algorithm',inplace=True)
df.columns = header
print(df)

###############################

# sol = [4, 4, 7, 7, 0, 2, 2, 0, 5, 5, 3, 3, 0, 6, 6]
# helpers.find_k_expensive_calls(sol,k=5)

# sol = algs.ALNS(sol,operators)
# print(sol,cost(sol),isFeasable(sol))
# print(cost(gen_dummy_sol()))

# print(helpers.pick_random_vehicles_by_call(2))
# s = fops.k_reinsert_random(s,1)
# print(s,cost(s))
# for i in range(5):
