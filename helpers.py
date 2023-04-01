import random
import data

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
    separator_indices.insert(0,0)
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

def pick_good_vehicle(solution):
    separator_indices = [i for i, x in enumerate(solution) if x == 0]
    vehicle_sizes = [x-separator_indices[i-1]-1 for i,x in enumerate(separator_indices)]
    vehicle_sizes[0] = separator_indices[0]
    tot_len = sum(vehicle_sizes)


    vehicle_weights = [tot_len+1/(x+1) for x in vehicle_sizes]
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