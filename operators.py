import random
import data
import numpy as np
import math
from solution_checker import isFeasable
from solution_checker import cost
from solution_checker import cost_of_vehicle
from solution_checker import cost_outsource
import helpers

##############################
### ASSIGNMENT 4 OPERATORS ###
##############################

def swap_similar_vehicles(sol):
    solution = sol[:]
    vehicle_1 = random.randrange(0,data.num_vehicles)
    vehicle_2 = helpers.find_similar_vehicle(vehicle_1)
    vehicle_1,vehicle_2 = min(vehicle_1,vehicle_2), max(vehicle_1,vehicle_2)

    start_1,end_1 = helpers.find_vehicle_indices(solution,vehicle_1)
    start_2,end_2 = helpers.find_vehicle_indices(solution,vehicle_2)

    calls_1 = solution[start_1:end_1] 
    calls_2 = solution[start_2:end_2]
    solution[start_2:end_2], solution[start_1:end_1] = calls_1, calls_2
    return solution

def smart_one_insert(sol):
    solution = sol[:]

    dummy_index = helpers.find_dummy_index(solution)
    dummy_calls = list(set(solution[dummy_index:]))
    dummy_amount = len(dummy_calls)
    p_dummy_removal = math.log(dummy_amount+1)/math.log(data.num_calls) # keeps high probability for long

    if random.random() <= p_dummy_removal:
        call = random.choice(dummy_calls)
    else:
        start,end,vehicle_id_remove = helpers.pick_good_vehicle_removal(solution) # removal_vehicle
        call = helpers.pick_expensive_calls_in_vehicle(solution,vehicle_id_remove,start,end)[0]

    vehicle_id = helpers.pick_good_vehicle_by_call(call) # insertion_vehicle
    vehicle_start_index,vehicle_end_index = helpers.find_vehicle_indices(solution,vehicle_id)
    vehicle_calls = solution[vehicle_start_index:vehicle_end_index]

    if vehicle_start_index==vehicle_end_index or not vehicle_calls:
        # print(call,solution)
        solution = helpers.reinsert_at_index(solution,call,vehicle_end_index)
        # print("end")
        return solution
    
    call_pickuptime = data.call_info[call-1][6]
    pickuptime_diffs = []

    visited = []
    w = 1
    for c in vehicle_calls:
        w = 1
        cur_pickuptime = data.call_info[c-1][6]
        if c in visited:
            w = 2
        pickuptime_diffs.append((cur_pickuptime-call_pickuptime)*w)
        visited.append(c)

    if pickuptime_diffs[-1] < 0:
        pickuptime_diffs.append(pickuptime_diffs[-1]/4)
    pickuptime_diffs = np.array(pickuptime_diffs)
    weights = abs(pickuptime_diffs)
    weights[np.where(weights<0.1)] = 0.1
    with np.errstate(divide='ignore'):
        weights = (1/weights)**2
    weights[np.where(weights>100)] = 1
    
    indices = np.arange(len(weights))
    i = random.choices(indices,weights)[0] # used for insertion later
    c_index = solution.index(call)
    solution.remove(call)
    solution.remove(call)

    if c_index < vehicle_start_index:
        vehicle_start_index -= 2
        vehicle_end_index -= 2
    if vehicle_start_index <= c_index < vehicle_end_index:
        vehicle_end_index -= 2
    if vehicle_start_index == vehicle_end_index:
        solution.insert(vehicle_start_index,call)
        solution.insert(vehicle_start_index,call)
        return solution
    pickup_insertion_index = vehicle_start_index+i
    # reinsert pickup:
    solution.insert(pickup_insertion_index,call)

    # reinsert delivery (base case):
    temp = solution[:]
    temp.insert(pickup_insertion_index+1,call)
    best_sol = temp

    for index in range(pickup_insertion_index+2,vehicle_end_index+1):
        # reinsert delivery:
        temp = solution[:]
        temp.insert(index,call)
        if isFeasable(temp):
            if cost_of_vehicle(temp,vehicle_id) < cost_of_vehicle(best_sol,vehicle_id):
                best_sol = temp 
                continue
        else:
            break
    return best_sol

def swap2_similar(sol):
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

##########################
### PREVIOUS OPERATORS ###
##########################

def reinsert_1(sol):
    solution = sol[:]
    dummy_index = helpers.find_dummy_index(solution)

    c1 = random.randint(1,data.num_calls)
    solution = list(filter(lambda x: x != c1, solution))

    new_dummy_index = helpers.find_dummy_index(solution)

    separator_indices = [i for i, x in enumerate(solution) if x == 0]
    separator_indices

    if new_dummy_index == dummy_index:
        insert_index = random.randrange(0,dummy_index)
    else:
        insert_index = random.randrange(0,len(solution))
    if insert_index >= new_dummy_index:
        solution.append(c1)
        solution.append(c1)
        return solution

    vehicles_before_call = [x for x in separator_indices if x<insert_index]
    vehicles_after_call = [x for x in separator_indices if x>=insert_index]
    solution.insert(insert_index,c1) # Insert first call at random before dummy

    vehicles_before_call.insert(0,-1)

    start = vehicles_before_call[-1]+1
    end = vehicles_after_call[0]
    if (start!=end):
        new_indx = random.randrange(start,end)
    else:
        new_indx=start
    
    solution.insert(new_indx,c1) # Insert second call in same vehicle
    return solution

def swap3(sol):
    solution = sol[:]
    call = -1
    dummy_indx = 0
    while call != 0:
        dummy_indx -= 1
        call = solution[dummy_indx]
    dummy_indx+=1
    dummy_indx = len(solution) + dummy_indx
    
    i1 = random.randrange(0,dummy_indx)
    c1 = solution[i1]

    while c1 == 0 and dummy_indx==len(solution):
        i1 = random.randrange(0,dummy_indx)
        c1 = solution[i1]

    if c1 == 0:
        i2 = random.randrange(dummy_indx,len(solution))
        c2 = solution[i2]
        i3 = random.randrange(0,len(solution))
        c3 = solution[i3]
        while c3 == c1 or c3 == c2:
            i3 = random.randrange(0,len(solution))
            c3 = solution[i3]
        
        solution[i2] = c1
        solution = list(filter(lambda x: x != c2, solution))
        del solution[i1]
        solution.insert(i1,c2)
        solution.insert(i1,c2)
        
        solution = np.array(solution)
        np.place(solution,solution==c2,-1)
        np.place(solution,solution==c3,c2)
        np.place(solution,solution==-1,c3)
        return solution

    i2 = random.randrange(0,len(solution))
    c2 = solution[i2]
    while c2 == c1 or c2 == 0:
        i2 = random.randrange(0,len(solution))
        c2 = solution[i2]
    i3 = random.randrange(0,len(solution))
    c3 = solution[i3]
    while c3 == c2 or c3 == c1 or c3 == 0:
        i3 = random.randrange(0,len(solution))
        c3 = solution[i3]

    solution = np.array(solution)
    np.place(solution,solution==c1,-3)
    np.place(solution,solution==c2,-1)
    np.place(solution,solution==c3,-2)
    np.place(solution,solution==-1,c1)
    np.place(solution,solution==-2,c2)
    np.place(solution,solution==-3,c3)
    return list(solution)

def swap2(sol):
    solution = sol[:]
    call = -1
    dummy_indx = 0
    while call != 0:
        dummy_indx -= 1
        call = solution[dummy_indx]
    dummy_indx+=1
    dummy_indx = len(solution) + dummy_indx
    
    i1 = random.randrange(0,dummy_indx)
    c1 = solution[i1]
    if dummy_indx==len(solution):
        while c1 == 0:
            i1 = random.randrange(0,dummy_indx)
            c1 = solution[i1]

    if c1 == 0:
        i2 = random.randrange(dummy_indx,len(solution))
        c2 = solution[i2]
        solution[i2] = c1
        solution = list(filter(lambda x: x != c2, solution))
        del solution[i1]
        solution.insert(i1,c2)
        solution.insert(i1,c2)
        return solution

    i2 = random.randrange(0,len(solution))
    c2 = solution[i2]
    while c2 == c1 or c2 == 0:
    # while c1 == c2 and c2 != 0:
        i2 = random.randrange(0,len(solution))
        c2 = solution[i2]

    solution = np.array(solution)
    np.place(solution,solution==c1,-1)
    np.place(solution,solution==c2,c1)
    np.place(solution,solution==-1,c2)

    return list(solution)

###################################
### FAILED/UNFINISHED OPERATORS ###
###################################

def quick_insert(sol):
    solution = sol[:]
    dummy_index = helpers.find_dummy_index(solution)
    dummy_calls = list(set(solution[dummy_index:]))
    _,vehicle_end_index,vehicle_id = helpers.pick_good_vehicle_insertion(solution)
    valid_calls = data.valid_calls[vehicle_id][1:]
    good_calls = list(set(dummy_calls) & set(valid_calls))
    if good_calls:
        call = random.choice(good_calls)
    else:
        return solution
    solution = helpers.reinsert_at_index(solution,call,vehicle_end_index)
    return solution

def swap3_similar(sol):
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
    ### change this to alter probability of selection
    weights = np.array(weights)
    weights = weights/sum(weights)
    calls = np.random.choice(population,2,replace=False,p=weights)
    call2 = calls[0]
    call3 = calls[1]
    # print(call1,call2,call3)
    solution = np.array(solution)
    np.place(solution,solution==call1,-3)
    np.place(solution,solution==call2,-1)
    np.place(solution,solution==call3,-2)
    np.place(solution,solution==-1,call1)
    np.place(solution,solution==-2,call2)
    np.place(solution,solution==-3,call3)
    return list(solution)

def smart_k_insert(sol):
    k = 2 # currently just using k=2

    solution = sol[:]
    for i in range(k):
        call = helpers.find_k_expensive_calls(solution,k=1)[0]
        vehicle_id = helpers.pick_good_vehicle_by_call(call)
        vehicle_start_index,vehicle_end_index = helpers.find_vehicle_indices(solution,vehicle_id)
        vehicle_calls = solution[vehicle_start_index:vehicle_end_index]
        if vehicle_start_index==vehicle_end_index:
            solution = helpers.reinsert_at_index(solution,call,vehicle_end_index)
            continue
        if not vehicle_calls:
            solution = helpers.reinsert_at_index(solution,call,vehicle_end_index)
            continue

        # Insertion weights:
        cp = data.call_info[call-1][6]
        cps = []

        visited = []
        w = 1
        for c in vehicle_calls:
            w = 1
            cur_cp = data.call_info[c-1][6]
            if c in visited:
                w = 2
            cps.append((cur_cp-cp)*w)
            visited.append(c)

        if cps[-1] < 0:
            cps.append(cps[-1]/4)
        cps = np.array(cps)
        cps = abs(cps)
        cps[np.where(cps<0.001)] = 0.001
        weights =(1/cps)**2
        weights[np.where(weights>100)] = 1

        indices = np.arange(len(cps))
        i = random.choices(indices,weights)[0]

        c_index = solution.index(call)
        solution.remove(call)
        solution.remove(call)
        # reinsert pickup call
        if c_index < vehicle_start_index:
            vehicle_start_index -= 2
            vehicle_end_index -= 2
        if vehicle_start_index <= c_index < vehicle_end_index:
            vehicle_end_index -= 2
        if vehicle_start_index == vehicle_end_index:
            solution.insert(vehicle_start_index,call)
            solution.insert(vehicle_start_index,call)
            continue
        solution.insert(vehicle_start_index+i,call)

        temp = solution[:]
        temp.insert(vehicle_start_index+i+1,call)
        best_sol = temp
        for index in range(vehicle_start_index+i+2,vehicle_end_index+1):
            temp = solution[:]
            temp.insert(index,call)
            if isFeasable(temp):
                if cost_of_vehicle(temp,vehicle_id) < cost_of_vehicle(best_sol,vehicle_id):
                    best_sol = temp 
                    # print(best_sol, cost(best_sol),"By reinsert+")
                    continue
            else:
                break
        solution = best_sol
    return solution
