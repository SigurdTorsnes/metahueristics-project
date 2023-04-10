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

def pick_good_vehicle_by_call(call):
    population = []
    for i in range(data.num_vehicles):
        if call in data.valid_calls[i][1:]:
            population.append(i)
    vehicle = random.choice(population)
    return vehicle


### finds vehicle based on amount of calls
def pick_good_vehicle_insertion(solution):
    separator_indices = [i for i, x in enumerate(solution) if x == 0]
    vehicle_sizes = [x-separator_indices[i-1]-1 for i,x in enumerate(separator_indices)]
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
    c1_pickup_node = data.call_info[call1-1][1]
    c2_pickup_node = data.call_info[call2-1][1]
    c1_delivery_node = data.call_info[call1-1][2]
    c2_delivery_node = data.call_info[call2-1][2]
    return c1_pickup_node, c2_pickup_node, c1_delivery_node, c2_delivery_node

def get_travel_cost(node1,node2,vehicle_id):
    index = data.num_vehicles*((node1-1)*data.num_nodes+node2)-data.num_vehicles+vehicle_id
    travel_cost = data.travel_times_and_cost[index][4]
    return travel_cost

def get_travel_time(node1,node2,vehicle_id):
    index = data.num_vehicles*((node1-1)*data.num_nodes+node2)-data.num_vehicles+vehicle_id
    travel_cost = data.travel_times_and_cost[index][3]
    return travel_cost

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
        total_cost = check.cost_of_vehicle(solution,vehicle_id)
        temp = solution[:]
        temp.remove(call)
        temp.remove(call)
        cost_without_call = check.cost_of_vehicle(temp,vehicle_id)
        diff = total_cost-cost_without_call
        call_costs.append([call,diff])

    dummy_cost = check.cost_outsource(solution)
    for call in calls_in_dummy:
        temp = solution[:]
        temp.remove(call)
        temp.remove(call)
        cost_without_call = check.cost_outsource(temp)
        diff = dummy_cost-cost_without_call
        call_costs.append([call,diff])
    call_costs = np.array(call_costs)
    costs_as_float = call_costs[:,1].astype(float)

    weights = costs_as_float/costs_as_float.sum()
    picked = np.random.choice(call_costs[:,0],k,replace=False,p= weights)
    return picked