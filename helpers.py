import random
import data
import math
import numpy as np
import solution_checker as check

def find_dummy_index(solution):
    dummy_call = -1
    dummy_index = 0
    while dummy_call != 0:
        dummy_index -= 1
        dummy_call = solution[dummy_index]
    dummy_index+=1
    dummy_index = len(solution) + dummy_index
    return dummy_index

def find_vehicle_of_call(solution,call):
    call_index = solution.index(call)
    separator_indices = [i for i, x in enumerate(solution) if x == 0]
    vehicle_id= 0
    for i in separator_indices:
        if i > call_index:
            break
        vehicle_id += 1
    return vehicle_id

def find_vehicle_indices(solution,vehicle_id):
    separator_indices = [i for i, x in enumerate(solution) if x == 0]
    separator_indices.append(0)
    if vehicle_id == 0:
        start = 0
    else:
        start = separator_indices[vehicle_id-1]+1
    end = separator_indices[vehicle_id]
    return start,end

def pick_random_vehicle_by_call(call):
    population = []
    for i in range(data.num_vehicles):
        if call in data.valid_calls[i+1]:
            population.append(i)
    vehicle = random.choice(population)
    return vehicle

def pick_random_vehicles_by_call(call):
    population = []
    for i in range(data.num_vehicles): # this can be done in preprocessing 
        if call in data.valid_calls[i+1]:
            population.append(i)
    population = np.array(population)
    vehicles = np.random.choice(population,size=min(len(population),1+data.num_vehicles//5),replace=False)
    return vehicles

def get_valid_vehicles(call):
    population = []
    for i in range(data.num_vehicles): # this can be done in preprocessing 
        if call in data.valid_calls[i+1]:
            population.append(i)
    population = np.array(population)
    return population

def pick_most_empty_vehicle_insert(solution,call):
    vehicles = get_valid_vehicles(call)
    sizes = []
    for vehicle in vehicles:
        start,end = find_vehicle_indices(solution,vehicle)
        size = end - start
        sizes.append(size)
    weights = 1/(np.array(sizes)+1)
    v = random.choices(vehicles,weights)[0]
    return v
    

### finds vehicle based on amount of calls
def pick_good_vehicle_insertion(solution,call):
    separator_indices = [i for i, x in enumerate(solution) if x == 0]
    vehicle_sizes = [x-separator_indices[i-1]-1 for i,x in enumerate(separator_indices)]
    vehicles = get_valid_vehicles(call)
    vehicle_sizes[0] = separator_indices[0]
    vehicle_weights = [1/math.exp(x+1) for x in vehicle_sizes]

    vehicle_end_index = random.choices(separator_indices,vehicle_weights, k=1)[0]
    vehicle_id = separator_indices.index(vehicle_end_index)
    if vehicle_id == 0:
        vehicle_start_index = 0
    else:
        vehicle_start_index = separator_indices[vehicle_id-1]+1
    return vehicle_start_index,vehicle_end_index,vehicle_id

def pick_expensive_calls_in_vehicle(solution, vehicle_id,start,end):
    call_costs = []
    calls_in_vehicle = calls_between(solution,start,end)
    total_cost = check.cost_of_vehicle(solution,vehicle_id)
    if len(calls_in_vehicle) == 1:
        return calls_in_vehicle
    for call in calls_in_vehicle: 
        temp = solution[:]
        temp.remove(call)
        temp.remove(call)
        cost_without_call = check.cost_of_vehicle(temp,vehicle_id)
        diff = total_cost-cost_without_call
        call_costs.append([call,diff])
    call_costs = np.array(call_costs)
    costs_as_float = call_costs[:,1].astype(float)
    weights = costs_as_float/costs_as_float.sum()
    picked = np.random.choice(call_costs[:,0], 1, replace=False, p=weights)
    return picked

def pick_good_vehicle_removal(solution):
    separator_indices = [i for i, x in enumerate(solution) if x == 0]
    vehicle_sizes = [x-separator_indices[i-1]-1 for i,x in enumerate(separator_indices)]
    vehicle_sizes[0] = separator_indices[0]
    tot_len = sum(vehicle_sizes)
    vehicle_weights = [math.pow(x,1.3) for x in vehicle_sizes]
    vehicle_end_index = random.choices(separator_indices,vehicle_weights, k=1)[0]
    vehicle_id = separator_indices.index(vehicle_end_index)
    if vehicle_id == 0:
        vehicle_start_index = 0
    else:
        vehicle_start_index = separator_indices[vehicle_id-1]+1
    return vehicle_start_index,vehicle_end_index,vehicle_id

def calls_between(solution,start_index,end_index):
    calls = list(set(solution[start_index:end_index]))
    return calls
    
def reinsert_at_index(solution,call,index):
    solution.remove(call)
    solution.remove(call)
    solution.insert(index,call)
    solution.insert(index,call)
    return solution

def put_in_dummy(solution,call):
    solution.remove(call)
    solution.remove(call)
    solution.append(call)
    solution.append(call)
    return solution

def get_nodes(call1,call2):
    c1_pickup_node = data.call_info[call1][0]
    c2_pickup_node = data.call_info[call2][0]
    c1_delivery_node = data.call_info[call1][1]
    c2_delivery_node = data.call_info[call2][1]
    return c1_pickup_node, c2_pickup_node, c1_delivery_node, c2_delivery_node

def get_travel_cost(node1,node2,vehicle_id):
    index = data.num_vehicles*((node1-1)*data.num_nodes+node2)-data.num_vehicles+vehicle_id
    # travel_cost = data.travel_times_and_cost[index][4]
    travel_cost = data.travel_cost[index]
    return travel_cost

def get_travel_time(node1,node2,vehicle_id):
    index = data.num_vehicles*((node1-1)*data.num_nodes+node2)-data.num_vehicles+vehicle_id
    # travel_time = data.travel_times_and_cost[index][3]
    travel_time = data.travel_time[index]
    return travel_time

def cost_relativity(call1,call2):
    c1_p,c2_p,c1_d,c2_d =get_nodes(call1,call2)
    pickup_travel_cost = get_travel_cost(c1_p,c2_p,1)
    delivery_travel_cost = get_travel_cost(c1_d,c2_d,1)
    total_travel_cost = pickup_travel_cost+delivery_travel_cost
    return total_travel_cost

def time_relativity(call1,call2):
    c1_p,c2_p,c1_d,c2_d =get_nodes(call1,call2)
    pickup_travel_time = get_travel_time(c1_p,c2_p,1)
    delivery_travel_time = get_travel_time(c1_d,c2_d,1)
    return (pickup_travel_time+delivery_travel_time),(pickup_travel_time+delivery_travel_time)/data.largest_travel_time

def size_relativity(call1,call2):
    s1 = data.call_info[call1-1,3]
    s2 = data.call_info[call2-1,3]
    size_r = abs(s1-s2)
    return size_r

def get_similarity(call1,call2):
    if call1 == call2:
        raise ValueError("Cannot find similarity of the same call")
    c1, c2 = min(call1,call2), max(call1,call2)
    index =(data.num_calls*(c1-1)+c2-c1-1)
    return data.call_relativity[index][-1]

def find_similar_vehicle(v1):
    row_indexes = []
    for v2 in range(data.num_vehicles):
        if v1 == v2:
            continue
        index = v1*data.num_vehicles+v2
        row_indexes.append(index)
    selection = data.vehicle_similarities.loc[row_indexes,:]
    weights = selection['total'].values
    weights = weights+0.001 # this is to avoid 100% chance of picking vehicles that may have exactly equal features
    weights = 1/pow(weights,1)
    weights = weights/sum(weights)
    vehicles = selection['v2'].values
    vehicle = np.random.choice(vehicles,size=None,replace=False,p=weights)
    return vehicle

def find_k_expensive_calls(solution,k=1):
    dummy_index = find_dummy_index(solution)
    calls_in_dummy = solution[dummy_index:]
    calls_in_vehicles = list(set(solution).symmetric_difference(set(calls_in_dummy)))
    calls_in_vehicles = calls_in_vehicles[1:] # removes 0, is sorted by set
    calls_in_dummy = list(set(calls_in_dummy))

    call_costs = []
    for call in calls_in_vehicles:
        vehicle_id = find_vehicle_of_call(solution,call)
        temp = solution[:]
        i1 = temp.index(call)
        temp.remove(call)
        i2 = temp.index(call)
        temp.remove(call)
        cost1 = check.cost_insert(temp,call,i1,vehicle_id)
        cost2 = check.cost_insert(temp,call,i2,vehicle_id)
        cost = cost1+cost2
        call_costs.append([call,cost])

    for call in calls_in_dummy:
        cost = data.call_info[call][3]
        call_costs.append([call,cost])
    call_costs = np.array(call_costs)
    costs_as_float = call_costs[:,1].astype(float)
    # print(costs_as_float)
    weights = costs_as_float**5/sum(costs_as_float**5)
    # print(weights**5/sum(weights**5.))
    picked = np.random.choice(call_costs[:,0],k,replace=False,p= weights)
    return picked


 # this is a helper function for reinsert operators
 # First make the picking of vehicle bettter (?)
 # make the placement of pickup call optimal
 # if runtime is ok, discard the selection of vehicles, and check all (or subset of) instead.
def reinsert_at_optimal_position2(solution,call):
    vehicle_id = pick_random_vehicle_by_call(call) # insertion_vehicle
    vehicle_start_index,vehicle_end_index = find_vehicle_indices(solution,vehicle_id)
    vehicle_calls = solution[vehicle_start_index:vehicle_end_index]

    if vehicle_start_index==vehicle_end_index or not vehicle_calls:
        # print(call,solution)
        solution = reinsert_at_index(solution,call,vehicle_end_index)
        # print("end")
        return solution
    
    # find pickup_index:
    call_pickuptime = data.call_info[call-1][6]
    pickuptime_diffs = []
    visited = []
    w = 1
    for c in vehicle_calls:
        w = 1
        cur_pickuptime = data.call_info[c-1][6]
        if c in visited:
            w = 2
        pickuptime_diffs.append((cur_pickuptime-call_pickuptime)*w**2)
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
    found_feasible = False
    if check.isFeasable(temp):
        found_feasible = True

    for index in range(pickup_insertion_index+2,vehicle_end_index+1):
        # reinsert delivery:
        temp = solution[:]
        temp.insert(index,call)
        if check.isFeasable(temp):
            found_feasible = True
            if check.cost_of_vehicle(temp,vehicle_id) < check.cost_of_vehicle(best_sol,vehicle_id):
                best_sol = temp 
                continue
        else:
            break
    if found_feasible:
        return best_sol
    else:
        best_sol = put_in_dummy(best_sol,call)
        return best_sol

def reinsert_at_optimal_position(sol,call, vehicle_id):
    solution = sol[:]
    # vehicle_id = pick_random_vehicle_by_call(call) # insertion_vehicle
    vehicle_start_index,vehicle_end_index = find_vehicle_indices(solution,vehicle_id)
    vehicle_calls = solution[vehicle_start_index:vehicle_end_index]

    if vehicle_start_index==vehicle_end_index or not vehicle_calls:
        # print(call,solution)
        solution = reinsert_at_index(solution,call,vehicle_end_index)
        # print("end")
        return solution
    
    # find pickup_index:

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
    # reinsert pickup:
    found_feasible = False
    best_sol = solution[:]
    best_sol.insert(vehicle_start_index,call)
    best_sol.insert(vehicle_start_index,call)
    # reinsert delivery (base case):
    for i in range(vehicle_start_index,vehicle_end_index+1):
        base = solution[:]
        base.insert(i,call)
        temp = base[:]
        # reinsert delivery (base case):
        temp.insert(i+1,call)
        # best_sol = temp
        # print("     ",temp)
        if check.isFeasable(temp):
            best_sol = temp
            found_feasible = True
        else:
            continue
        for index in range(i+2,vehicle_end_index+2):
            # reinsert delivery:
            temp = base[:]
            temp.insert(index,call)
            # print("         ",temp)
            if check.isFeasable(temp):
                found_feasible = True
                if check.cost_of_vehicle(temp,vehicle_id) < check.cost_of_vehicle(best_sol,vehicle_id):
                    # print("             ",temp)
                    best_sol = temp 
                    continue
            else:
                break
    if found_feasible:
        return best_sol
    else:
        solution.append(call)
        solution.append(call)
        return solution
    
def find_pickup_start_index(vehicle_calls,call):
    # min_time_p = data.call_info[:,5][call-1]
    min_time_p = data.call_info[call][4]
    last_c = 0
    for c in vehicle_calls:
        max_time = data.call_info[call][5]
        # max_time = data.call_info[:,6][c-1]
        if max_time < min_time_p:
            last_c  = c
    if last_c == 0:
        start = 0 # not quite
        pass
    else:
        start = vehicle_calls.index(last_c)+1

    # print(vehicle_calls,last_c,start,call)
    return start
    
def can_be_infront(call1,call2):
    delivery_max_1 =data.call_info[call1-1,8]
    previous_node = data.call_info[call2-1][2]
    pickup_max_2 =data.call_info[call2-1,6]
    node = data.call_info[call2-1][1]
    index = data.num_vehicles*((previous_node-1)*data.num_nodes+node)-data.num_vehicles-1 #uses just first vehicle
    # travel_time = data.travel_times_and_cost[index][3]
    travel_time = data.travel_time[index]
    if delivery_max_1+travel_time<pickup_max_2: # + time between nodes of the calls
        return False
    return True

def find_elapsed_time(vehicle_calls,v_id): # undone
    index = v_id*data.num_calls+call-1
    node_times = data.node_times_and_cost[index]
    visited = []
    for call in vehicle_calls:
        if call in visited:
            # delivery
            time += node_times
        else:
            # pickup
            pass

def insert_first_feasible(sol,call):
    solution = sol[:]
    vs = get_valid_vehicles(call)
    random.shuffle(vs)
    for vehicle_id in vs:
        vehicle_start_index,vehicle_end_index = find_vehicle_indices(solution,vehicle_id) 
        vehicle_calls = solution[vehicle_start_index:vehicle_end_index] 
        if vehicle_start_index==vehicle_end_index or not vehicle_calls:
            solution.insert(vehicle_start_index,call)
            solution.insert(vehicle_start_index,call)
            return solution
        
        best_sol = solution[:]
        feasible_offset = find_pickup_start_index(vehicle_calls,call) 
        start = vehicle_start_index+feasible_offset 
        best_sol.insert(start,call) 
        best_sol.insert(start,call)
        j = 0
        
        for i in range(start,vehicle_end_index+1): 
            j += 1
            base = solution[:]
            base.insert(i,call)
            feasible, failed_call, reason = check.isFeasibleVehicle(base,vehicle_id)
            if reason == 'pickup' or reason == 'delivery':
                continue

            temp = base[:]
            temp.insert(i+1,call)   
            feasible, failed_call, reason = check.isFeasibleVehicle(temp,vehicle_id)
            if reason == 'delivery' and failed_call == call:
                break
            if feasible: 
                best_sol = temp
                return best_sol

            # for index in range(i+2,vehicle_end_index+2):
            #     temp = base[:]
            #     temp.insert(index,call)
            #     feasible, failed_call, reason = check.isFeasibleVehicle(temp,vehicle_id)
            #     if reason == 'delivery' and failed_call == call:
            #         break
            #     if feasible:
            #         best_sol = temp
            #         return best_sol
    solution.append(call)
    solution.append(call)
    return solution


def insert_at_optimal_position(sol,call, vehicle_id): # v total vehicles * k calls [1,qmax] 10000*data.vnum*kavg
    solution = sol[:] # 0s
    vehicle_start_index,vehicle_end_index = find_vehicle_indices(solution,vehicle_id) # 1 sec
    vehicle_calls = solution[vehicle_start_index:vehicle_end_index] # 0s

    if vehicle_start_index==vehicle_end_index or not vehicle_calls: # 0s
        diff_pickup = check.cost_insert(solution,call,vehicle_start_index,vehicle_id)
        solution.insert(vehicle_start_index,call)
        diff_delivery = check.cost_insert(solution,call,vehicle_start_index+1,vehicle_id)
        solution.insert(vehicle_start_index,call)
        diff = diff_pickup+diff_delivery
        return solution, diff

    found_feasible = False
    best_sol = solution[:]
    feasible_offset = find_pickup_start_index(vehicle_calls,call) # 1 s
    #0s:
    start = vehicle_start_index+feasible_offset 
    # print(solution[start:vehicle_end_index])
    best_diff = data.call_info[call][3]
    best_sol.insert(start,call) 
    best_sol.insert(start,call)
    j = 0

    for i in range(start,vehicle_end_index+1): 
        j += 1
        base = solution[:]
        base_diff = check.cost_insert(solution,call,i,vehicle_id) # 1.5s * i (about 5)
        base.insert(i,call)
        feasible, failed_call, reason = check.isFeasibleVehicle(base,vehicle_id)
        if reason == 'pickup' or reason == 'delivery':
            continue

        temp = base[:]
        temp.insert(i+1,call)   
        cost_diff = check.cost_insert(base,call,i+1,vehicle_id) + base_diff
        feasible, failed_call, reason = check.isFeasibleVehicle(temp,vehicle_id)
        if reason == 'delivery' and failed_call == call:
            break
        if cost_diff < best_diff and feasible: # 2.7s * i
            found_feasible = True
            best_sol = temp
            best_diff = cost_diff 


        for index in range(i+2,vehicle_end_index+2):
            temp = base[:]
            cost_diff = check.cost_insert(temp,call,index,vehicle_id) + base_diff
            temp.insert(index,call)

            feasible, failed_call, reason = check.isFeasibleVehicle(temp,vehicle_id)
            if reason == 'delivery' and failed_call == call:
                break
            if cost_diff < best_diff and feasible:
                found_feasible = True
                best_diff = cost_diff
                best_sol = temp

    if found_feasible:
        return best_sol, best_diff
    else:
        solution.append(call)
        solution.append(call)
        return solution, best_diff
    

def insert_at_almost_optimal_position(sol,call, vehicle_id): # v total vehicles * k calls [1,qmax] 10000*data.vnum*kavg
    solution = sol[:] # 0s
    vehicle_start_index,vehicle_end_index = find_vehicle_indices(solution,vehicle_id) # 1 sec
    vehicle_calls = solution[vehicle_start_index:vehicle_end_index] # 0s

    if vehicle_start_index==vehicle_end_index or not vehicle_calls: # 0s
        diff_pickup = check.cost_insert(solution,call,vehicle_start_index,vehicle_id)
        solution.insert(vehicle_start_index,call)
        diff_delivery = check.cost_insert(solution,call,vehicle_start_index+1,vehicle_id)
        solution.insert(vehicle_start_index,call)
        diff = diff_pickup+diff_delivery
        return solution, diff

    found_feasible = False
    best_sol = solution[:]
    # feasible_offset = find_pickup_start_index(vehicle_calls,call) # 1 s
    #0s:
    start = vehicle_start_index#+feasible_offset 
    # print(solution[start:vehicle_end_index])
    best_diff = data.call_info[call][3]
    best_sol.insert(start,call) 
    best_sol.insert(start,call)
    j = 0

    for i in range(start,vehicle_end_index+1): 
        j += 1
        base = solution[:]
        base_diff = check.cost_insert(solution,call,i,vehicle_id) # 1.5 * i (about 5)
        base.insert(i,call)
        temp = base[:]
        temp.insert(i+1,call)
        cost_diff = check.cost_insert(base,call,i+1,vehicle_id) + base_diff
        if check.isFeasibleVehicle(temp,vehicle_id):
            found_feasible = True
            if cost_diff < best_diff:# 2.7 * i
                best_sol = temp
                best_diff = cost_diff
        else:
            continue
        for index in range(i+2,vehicle_end_index+2):
            temp = base[:]
            cost_diff = check.cost_insert(temp,call,index,vehicle_id) + base_diff
            temp.insert(index,call)
            if check.isFeasibleVehicle(temp,vehicle_id):
                found_feasible = True
                if cost_diff < best_diff:
                    best_diff = cost_diff
                    best_sol = temp
            else:
                break
    if found_feasible:
        return best_sol, best_diff
    else:
        solution.append(call)
        solution.append(call)
        return solution, best_diff
    

    