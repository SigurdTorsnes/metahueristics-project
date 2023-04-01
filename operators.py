import random
import data
import numpy as np
import math
from solution_checker import isFeasable
from solution_checker import cost
from solution_checker import cost_of_vehicle
from solution_checker import cost_outsource
import helpers as help

##############################
### ASSIGNMENT 4 OPERATORS ###
##############################

def change_order(sol):
    solution = sol[:]
    i1 = random.randrange(len(solution))
    i2 = i1+1
    while solution[i1] != 0 and i2 !=0:
        i1 = random.randrange(len(solution))
        i2 = i1+1
    solution[i1], solution[i2] = solution[i2], solution[i1]
    return solution

def improved_reinsert(sol):
    solution = sol[:]

    dummy_index = help.find_dummy_index(solution) # dummy_index = separator_indices[-1] + 1 (beware edge case with len(solution))
    calls_in_dummy = solution[dummy_index:]
    if len(calls_in_dummy) == 0:
        return solution
    
    ### TODO: Maybe change the order here to: 
    # remove expensive call; pick vehicle to place call in; pick best position in vehicle;

    call = find_expensive_call(solution)


    # find vehicle to put in
    vehicle_id = help.pick_good_vehicle_by_call(call)
    vehicle_start_index,vehicle_end_index = help.find_vehicle_indices(solution,vehicle_id)
    # pick call available for vehicle
    # valid_calls_for_vehicle = data.valid_calls[vehicle_id][1:]
    vehicle_calls = solution[vehicle_start_index:vehicle_end_index]
    # good_choices = list(set(valid_calls_for_vehicle).symmetric_difference(set(vehicle_calls)))
    # call = random.choice(good_choices)

    if vehicle_start_index==vehicle_end_index:
        solution = help.reinsert_at_index(solution,call,vehicle_end_index)
        # print("end")
        return solution



    ###### Finding index to place in #######
    # Excact option:
    # i1 = data.pickup_sorted_by_time.index(call)
    # i2 = data.pickup_sorted_by_time.index(vehicle_calls[0])
    # i = 0
    # for c in vehicle_calls:
    #     i2 = data.pickup_sorted_by_time.index(c)
    #     if i2 < i1:
    #         i+= 1
    #     else:
    #         break

    # Weighted option:
    i1 = data.pickup_sorted_by_time.index(call)
    weights = []
    for c in vehicle_calls:
        i2 = data.pickup_sorted_by_time.index(c)
        diff = abs(i2-i1)
        if diff == 0:
            diff = 2
        weights.append(1/diff)

    if not vehicle_calls:
        solution = help.reinsert_at_index(solution,call,vehicle_end_index)
        return solution
    indices = np.arange(len(vehicle_calls))
    i = random.choices(indices,weights)[0]
    # print(indices,weights, i)

    ############################################

    # remove call     
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
        return solution
    solution.insert(vehicle_start_index+i,call)

    temp = solution[:]
    temp.insert(vehicle_start_index+i+1,call)
    best_sol = temp
    # reinsert delivery call

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
    return best_sol


def remove_expensive(sol): 
    solution = sol [:]
    dummy_index = help.find_dummy_index(solution) # dummy_index = separator_indices[-1] + 1 (beware edge case with len(solution))
    calls_in_dummy = solution[dummy_index:]
    calls_in_vehicles = list(set(solution).symmetric_difference(set(calls_in_dummy)))
    if len(calls_in_dummy)/data.num_calls < 0.2:
        expensive_call_1 = find_expensive_call(solution)
        expensive_call_2 = find_expensive_call(solution)
        while expensive_call_1 == expensive_call_2:
            expensive_call_2 = find_expensive_call(solution)
        
        help.put_in_dummy(solution,expensive_call_1)
        help.put_in_dummy(solution,expensive_call_2)
        return solution

    if len(calls_in_dummy)/2 < data.num_calls/2:
        expensive_call = find_expensive_call(solution)
    else:
        expensive_call = random.choice(calls_in_dummy)
    help.put_in_dummy(solution,expensive_call)

    return solution, expensive_call

def swap2_similar(sol):
    solution = sol[:]
    # find similar call
    call1 = random.randint(1,data.num_calls)
    # best_similarity = 1000000
    similarites = []
    for cur_call in range(1,data.num_calls+1): # improve
        if call1 != cur_call:
            cur_similarity = help.get_similarity(call1,cur_call)
            similarites.append([cur_call, cur_similarity])

    similarites = np.array(similarites)
    population = similarites[:,0].astype(int)
    weights = similarites[:,1]
    ### change this to alter probability of selection
    weights = (weights/max(weights))

    call2 = random.choices(population,weights)[0]
    solution = np.array(solution)
    np.place(solution,solution==call1,-1)
    np.place(solution,solution==call2,call1)
    np.place(solution,solution==-1,call2)
    return list(solution)


def reinsert_with_similar(sol):
    solution = sol[:]
    dummy_index = help.find_dummy_index(solution) # dummy_index = separator_indices[-1] + 1 (beware edge case with len(solution))
    calls_in_dummy = solution[dummy_index:]
    calls_in_vehicles = list(set(solution).symmetric_difference(set(calls_in_dummy)))
    if len(calls_in_vehicles) <= 1:
        return sol # no calls to remove
    good_choices = calls_in_vehicles[1:]

    if not good_choices:
        print("no good choises for similar_insert")
        return solution

    if len(calls_in_dummy)/2 < data.num_calls/2:
        expensive_call = find_expensive_call(solution)
    else:
        expensive_call = random.choice(calls_in_dummy)
    vehicle_start_index,vehicle_end_index,vehicle_id = help.pick_good_vehicle(solution)
    # random.choice
    # pick call available for vehicle
    # valid_calls_for_vehicle = data.valid_calls[vehicle_id][1:]
    # vehicle_calls = solution[vehicle_start_index:vehicle_end_index]
    
    # good_choices = list(set(solution).symmetric_difference(set(calls_in_dummy)))

    similarites = []
    for cur_call in good_choices: # improve
        if expensive_call != cur_call:
            cur_similarity = help.get_similarity(expensive_call,cur_call)
            similarites.append([cur_call, cur_similarity])

    similarites = np.array(similarites)
    population = similarites[:,0].astype(int)
    weights = similarites[:,1]
    ### change this to alter probability of selection
    weights = (weights/max(weights))

    similar_call = random.choices(population,weights)[0]
    # solution.remove(best_call)
    # solution.remove(best_call)
    
    solution.remove(expensive_call)
    solution.remove(expensive_call)
    # call_to_remove = find_expensive_call_in_vehicle(solution,vehicle_id)
    # solution = help.put_in_dummy(solution,call_to_remove)
    similar_call_indices = [i for i, x in enumerate(solution) if x == similar_call]
    # print(similar_call_indices)
    solution.insert(min(similar_call_indices[1]+1,len(solution)-1),expensive_call)
    solution.insert(similar_call_indices[0],expensive_call)
    # print(vehicle_start_index,vehicle_end_index,vehicle_id, expensive_call, most_similar_call)
    return solution

# returns most expensive call in solution
def find_expensive_call1(solution):
    dummy_index = help.find_dummy_index(solution)
    calls_in_dummy = solution[dummy_index:]
    calls_in_vehicles = list(set(solution).symmetric_difference(set(calls_in_dummy)))
    calls_in_vehicles = calls_in_vehicles[1:] # removes 0, is sorted by set

    call_costs = []
    for call in calls_in_vehicles:
        vehicle_id = help.find_vehicle_of_call(solution,call)
        total_cost = cost_of_vehicle(solution,vehicle_id)
        temp = solution[:]
        temp.remove(call)
        temp.remove(call)
        cost_without_call = cost_of_vehicle(temp,vehicle_id)
        diff = total_cost-cost_without_call
        call_costs.append([call,diff])
    if not call_costs:
        print("no expensive calls")
        return random.choice(calls_in_dummy)
    call_costs = np.array(call_costs)
    costs_as_float = call_costs[:,1].astype(float)

    ### change this to alter probability of selection
    weights = (costs_as_float/max(costs_as_float))
    if sum(weights) < 0.05:
        print("low weights")
        return random.choice(calls_in_dummy)
    call = random.choices(call_costs[:,0],weights)[0]
    return call

def find_expensive_call(solution):
    dummy_index = help.find_dummy_index(solution)
    calls_in_dummy = solution[dummy_index:]
    calls_in_vehicles = list(set(solution).symmetric_difference(set(calls_in_dummy)))
    calls_in_vehicles = calls_in_vehicles[1:] # removes 0, is sorted by set

    call_costs = []
    for call in calls_in_vehicles:
        vehicle_id = help.find_vehicle_of_call(solution,call)
        total_cost = cost_of_vehicle(solution,vehicle_id)
        temp = solution[:]
        temp.remove(call)
        temp.remove(call)
        cost_without_call = cost_of_vehicle(temp,vehicle_id)
        diff = total_cost-cost_without_call
        call_costs.append([call,diff])

    dummy_cost = cost_outsource(solution)
    for call in calls_in_dummy:
        temp = solution[:]
        temp.remove(call)
        temp.remove(call)
        cost_without_call = cost_outsource(temp)
        diff = dummy_cost-cost_without_call
        call_costs.append([call,diff])
    call_costs = np.array(call_costs)
    costs_as_float = call_costs[:,1].astype(float)

    ### change this to alter probability of selection
    weights = (costs_as_float/max(costs_as_float))
    if sum(weights) < 0.05:
        print("low weights")
        return random.choice(calls_in_dummy)
    call = random.choices(call_costs[:,0],weights)[0]
    return call

# returns most expensive call found in given vehicle
def find_expensive_call_in_vehicle(solution, vehicle_id):
    separator_indices = [i for i, x in enumerate(solution) if x == 0]
    if vehicle_id == 0:
        vehicle_start_index = 0
    else:
        vehicle_start_index = separator_indices[vehicle_id-1]+1
    vehicle_end_index = separator_indices[vehicle_id]
    if vehicle_start_index == vehicle_end_index:
        return 0
    calls_in_vehicle = [x for x in solution[vehicle_start_index:vehicle_end_index]]
    calls_in_vehicle = list(set(calls_in_vehicle))
    largest_diff = 0
    most_expensive_call = None
    total_cost = cost_of_vehicle(solution,vehicle_id)

    for call in calls_in_vehicle:
        temp = solution[:]
        temp.remove(call)
        temp.remove(call)
        cost_without_call = cost_of_vehicle(temp,vehicle_id)
        diff = total_cost-cost_without_call
        if diff > largest_diff:
            largest_diff = diff
            most_expensive_call = call
            # costly_calls.append(most_expensive_call)
    return most_expensive_call#, largest_diff

def put_last_call_in_dummy(solution,vehicle):
    separator_indecies = [i for i, x in enumerate(solution) if x == 0]
    index = separator_indecies[vehicle]-1
    call = 0
    if index >= 0:
        call = solution[index]
    if call != 0:
        solution = help.put_in_dummy(solution,call)
    return solution


#######################
### BASIC OPERATORS ###
#######################

def reinsert_1(sol):
    solution = sol[:]


    # find call
    # find valid vehicles for given call
    # find range of indexes that can be chosen 

    # maybe create function for this
    
    dummy_index = help.find_dummy_index(solution)

    c1 = random.randint(1,data.num_calls)
    solution = list(filter(lambda x: x != c1, solution))

    new_dummy_index = help.find_dummy_index(solution)

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