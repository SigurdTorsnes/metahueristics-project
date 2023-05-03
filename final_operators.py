import helpers
import data
import numpy as np
import random
from solution_checker import isFeasable
from solution_checker import cost_of_vehicle
from solution_checker import cost


### removal operators
def remove_k_outsourced(sol,q):
    solution = sol[:]
    dummy_index = helpers.find_dummy_index(solution)
    dummy_calls = list(set(solution[dummy_index:]))
    calls = np.random.choice(dummy_calls,size=min(q,len(dummy_calls)),replace=False)
    for call in calls:
        solution.remove(call)
        solution.remove(call)
    return solution, calls

def remove_k_costly(sol,k):
    solution = sol[:]
    calls = helpers.find_k_expensive_calls(sol,k)
    for call in calls:
        solution.remove(call)
        solution.remove(call)
    return solution, calls

def remove_k_similar(sol,k):
    solution = sol[:]
    call1 = random.randint(1,data.num_calls)
    similarites = []
    for cur_call in range(1,data.num_calls+1): # improve
        if call1 != cur_call:
            cur_similarity = helpers.get_similarity(call1,cur_call)
            similarites.append([cur_call, cur_similarity])

    similarites = np.array(similarites)
    population = similarites[:,0].astype(int)
    weights = similarites[:,1]
    weights = 1/weights
    weights = (weights/max(weights)) # **x to intensify 
    weights = (weights/sum(weights))
    calls = np.random.choice(population,p=weights,size=min(len(population),k),replace=False)
    for call in calls:
        solution.remove(call)
        solution.remove(call)
    return solution, calls

def remove_k_random(sol,k):
    pop = np.arange(1,data.num_calls+1)
    calls = np.random.choice(pop,size=k,replace=False)
    solution = sol[:]
    for call in calls:
        solution.remove(call)
        solution.remove(call)
    return solution, calls

def remove_vehicle(sol,k):
    # ratio = data.num_vehicles//data.num_calls
    solution = sol[:]
    v = random.randrange(data.num_vehicles)
    start, end = helpers.find_vehicle_indices(solution,v)
    calls = list(set(solution[start:end]))
    solution[start:end] = []
    return solution,calls

def remove_costly_and_similar_vehicles(sol,k):
    # ratio = data.num_vehicles//data.num_calls
    # k = int(ratio*k)+1
    solution = sol[:]
    # weights = []
    # vehicles = np.arange(data.num_vehicles)
    # for v in vehicles:
    #     cv = cost_of_vehicle(solution,v)
    #     weights.append(cv)
    # weights = np.array(weights)+1
    # v1 = random.choices(vehicles,weights)[0] # costly
    v1 = random.randrange(data.num_vehicles)
    v2 = helpers.find_similar_vehicle(v1) # similar to v1
    vs = [v1,v2]
    calls = []
    # print("start")
    for v in vs:
        # print(solution)
        v = random.randrange(data.num_vehicles)
        start, end = helpers.find_vehicle_indices(solution,v)
        for i in set(solution[start:end]):
            calls.append(i)
        solution[start:end] = []
    # print(solution,calls)
    return solution,calls,vs

### insert_operators
def random_insert(sol,calls):
    solution = sol[:]
    for call in calls:
        index = random.choice(np.arange(len(solution)))
        solution.insert(index,call)
        solution.insert(index,call)
    return solution

def insert_k_quick(sol,calls):
    random.shuffle(calls)
    base = sol[:]
    for call in calls:
        base.append(call)
        base.append(call)
    best_sol = sol[:]
    to_change = sol[:]
    for call in calls:
        improved = False
        best_diff = data.call_info[call-1][4]
        v = helpers.pick_most_empty_vehicle_insert(to_change,call)
        temp, temp_diff = helpers.insert_at_optimal_position(to_change,call,v)
        if temp_diff < best_diff:
            best_sol = temp
            best_diff = temp_diff
            improved = True
        if not improved:
            best_sol.append(call)
            best_sol.append(call)
        to_change = best_sol
    return best_sol


def insert_k_best_pos(sol,calls, vs=[]):
    random.shuffle(calls)
    best_sol = sol[:]
    to_change = sol[:]
    
    for call in calls: # k calls
        improved = False
        best_cost_diff = data.call_info[call-1][4]
        if len(vs)!=0:
            vehicles = vs[:]
        else:
            # print("emp")
            vehicles = helpers.get_valid_vehicles(call)
        # random.shuffle(vehicles)
        # vehicles = [vehicles[0]]
        for v in vehicles: # v vehicles * k calls
            temp, temp_cost_diff = helpers.insert_at_optimal_position(to_change,call,v)
            # temp,temp_cost_diff = sol, 200
            # temp_cost = cost(temp) # Make this use the difference from the insert method above to improve time
            if temp_cost_diff < best_cost_diff:
                best_sol = temp
                best_cost_diff = temp_cost_diff
                improved = True
        if not improved:
            best_sol.append(call)
            best_sol.append(call)
        to_change = best_sol
    return best_sol
















#### swaps
def k_exchange_similar_calls(sol, q):
    solution = sol[:]
    # find similar call
    call1 = random.randint(1,data.num_calls)
    # best_similarity = 1000000
    similarites = []
    for cur_call in range(1,data.num_calls+1): # improve
        if call1 != cur_call:
            cur_similarity = helpers.get_similarity(call1,cur_call)
            similarites.append([cur_call, cur_similarity])

    similarites = np.array(similarites)
    population = similarites[:,0].astype(int)
    weights = similarites[:,1]
    weights = 1/weights
    weights = (weights/max(weights)) # **x to intensify 
    call2 = random.choices(population,weights)[0]
    solution = np.array(solution)
    np.place(solution,solution==call1,-1)
    np.place(solution,solution==call2,call1)
    np.place(solution,solution==-1,call2)
    return list(solution)


def k_exchange_similar_vehicles(sol,q):
    solution = sol[:]
    vehicle_1 = random.randrange(0,data.num_vehicles)
    vehicle_2 = helpers.find_similar_vehicle(vehicle_1)
    vehicle_1,vehicle_2 = min(vehicle_1,vehicle_2), max(vehicle_1,vehicle_2)

    start_1,end_1 = helpers.find_vehicle_indices(solution,vehicle_1)
    start_2,end_2 = helpers.find_vehicle_indices(solution,vehicle_2)

    calls_1 = solution[start_1:end_1] 
    calls_2 = solution[start_2:end_2]
    return solution
