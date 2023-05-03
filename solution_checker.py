import data
import helpers as helpers

def cost(solution):
    cost = 0
    vehicle_index = 1
    previous_node = data.vehicle_info[vehicle_index-1][1]
    picked_up = []
    node = data.vehicle_info[vehicle_index-1][1]
    dummy = []
    for call in solution:
        if vehicle_index-1 >= data.num_vehicles:
            if call not in dummy:
                cost += data.call_info[call-1][4]
                dummy.append(call)
            continue

        if call == 0:
            vehicle_index += 1
            if vehicle_index-1 >= data.num_vehicles:
                continue
            else: 
                previous_node = data.vehicle_info[vehicle_index-1][1]
        else:
            if call not in picked_up:
                node = data.call_info[call-1][1]
                index = (vehicle_index-1)*data.num_calls+call-1
                transfer_cost = data.node_times_and_cost[index][3]
                picked_up.append(call)
            else:
                node = data.call_info[call-1][2]
                index = (vehicle_index-1)*data.num_calls+call-1
                transfer_cost = data.node_times_and_cost[index][5]
                
            index = data.num_vehicles*((previous_node-1)*data.num_nodes+node)-data.num_vehicles-1+vehicle_index
            travel_cost = data.travel_times_and_cost[index][4]
            cost += travel_cost + transfer_cost
            previous_node = node
    return cost

def isFeasable(solution):
    vehicle_index = 0
    previous_node = data.vehicle_info[vehicle_index][1]
    time = data.vehicle_info[vehicle_index][2]
    node_traversal = []
    picked_up = []
    vehicle_capacity = data.vehicle_info[vehicle_index][3]
    node = data.vehicle_info[vehicle_index][1]
    node_traversal.append(node)
    for call in solution:
        if call == 0:
            vehicle_index += 1
            if vehicle_index == data.num_vehicles:
                return True
            else:
                previous_node = data.vehicle_info[vehicle_index][1]
                time = data.vehicle_info[vehicle_index][2]
                vehicle_capacity = data.vehicle_info[vehicle_index][3]
        else:
            index = vehicle_index*data.num_calls+call-1
            node_times = data.node_times_and_cost[index]
            package_size = data.call_info[call-1][3]
            valid_vehicle_calls = data.valid_calls[vehicle_index][1:]
            if call not in valid_vehicle_calls:
                return False # call not valid for vehicle
            if call not in picked_up:
                node = data.call_info[call-1][1]
                index = data.num_vehicles*((previous_node-1)*data.num_nodes+node)-data.num_vehicles+vehicle_index
                travel_time = data.travel_times_and_cost[index][3]
                time += travel_time
                min_pickuptime = data.call_info[call-1][5]
                max_pickuptime = data.call_info[call-1][6]
                node_time = node_times[2]
                time = max(time,min_pickuptime)
                if min_pickuptime <= time <= max_pickuptime:
                    time += node_time
                    vehicle_capacity -= package_size
                    if vehicle_capacity < 0:
                        return False # not enough capacity
                    picked_up.append(call)
                else: 
                    return False # Not picked up in time
            else:
                min_deliverytime = data.call_info[call-1][7]
                max_deliverytime = data.call_info[call-1][8]
                node_time = node_times[4]
                node = data.call_info[call-1][2]
                index = data.num_vehicles*((previous_node-1)*data.num_nodes+node)-data.num_vehicles+vehicle_index
                travel_time = data.travel_times_and_cost[index][3]
                time += travel_time
                time = max(time,min_deliverytime)
                if min_deliverytime <= time <= max_deliverytime:
                    time += node_time
                    vehicle_capacity += package_size
                else:
                    return False # Not delivered in time
            previous_node = node  
    return True

def isFeasibleVehicle(solution,vehicle_id):
    start,end = helpers.find_vehicle_indices(solution,vehicle_id)
    vehicle_calls = solution[start:end]
    previous_node = data.vehicle_info[vehicle_id][1]
    time = data.vehicle_info[vehicle_id][2]
    node_traversal = []
    picked_up = []
    vehicle_capacity = data.vehicle_info[vehicle_id][3]
    node = data.vehicle_info[vehicle_id][1]
    node_traversal.append(node)
    for call in vehicle_calls:
        index = vehicle_id*data.num_calls+call-1
        node_times = data.node_times_and_cost[index]
        package_size = data.call_info[call-1][3]
        valid_vehicle_calls = data.valid_calls[vehicle_id][1:]
        if call not in valid_vehicle_calls:
            return False, call,"vehicle"
        if call not in picked_up:
            node = data.call_info[call-1][1]
            index = data.num_vehicles*((previous_node-1)*data.num_nodes+node)-data.num_vehicles+vehicle_id
            travel_time = data.travel_times_and_cost[index][3]
            time += travel_time
            min_pickuptime = data.call_info[call-1][5]
            max_pickuptime = data.call_info[call-1][6]
            node_time = node_times[2]
            time = max(time,min_pickuptime)
            if min_pickuptime <= time <= max_pickuptime:
                time += node_time
                vehicle_capacity -= package_size
                if vehicle_capacity < 0:
                    return False, call,"capacity"
                picked_up.append(call)
            else: 
                return False, call,"pickup"
        else:
            min_deliverytime = data.call_info[call-1][7]
            max_deliverytime = data.call_info[call-1][8]
            node_time = node_times[4]
            node = data.call_info[call-1][2]
            index = data.num_vehicles*((previous_node-1)*data.num_nodes+node)-data.num_vehicles+vehicle_id
            travel_time = data.travel_times_and_cost[index][3]
            time += travel_time
            time = max(time,min_deliverytime)
            if min_deliverytime <= time <= max_deliverytime:
                time += node_time
                vehicle_capacity += package_size
            else:
                return False,call, "delivery"
        previous_node = node
    return True, None, None

def cost_of_vehicle(solution,vehicle_id):   
    separator_indices = [i+1 for i, x in enumerate(solution) if x == 0]
    separator_indices.insert(0,0)

    start_index = separator_indices[vehicle_id]
    end_index = separator_indices[vehicle_id+1]-1
    cost = 0
    previous_node = data.vehicle_info[vehicle_id][1]
    picked_up = []
    node = data.vehicle_info[vehicle_id][1]
    if start_index == end_index:
        return 0
    for i in range(start_index,end_index):
        call = solution[i]
        if call not in picked_up:
            node = data.call_info[call-1][1]
            index = vehicle_id*data.num_calls+call-1
            transfer_cost = data.node_times_and_cost[index][3]
            picked_up.append(call)
        else:
            node = data.call_info[call-1][2]
            index = vehicle_id*data.num_calls+call-1
            transfer_cost = data.node_times_and_cost[index][5]
            
        index = data.num_vehicles*((previous_node-1)*data.num_nodes+node)-data.num_vehicles+vehicle_id
        travel_cost = data.travel_times_and_cost[index][4]
        cost += travel_cost + transfer_cost
        previous_node = node
    
    return cost

def cost_outsource(solution):
    dummy_index = helpers.find_dummy_index(solution)
    dummy_calls = set(solution[dummy_index:])
    cost = 0
    for call in list(dummy_calls):
        cost += data.call_info[call-1][4]
    return cost


def cost_insert(sol,call,index,vehicle_index):
    # print("-----------")
    solution = sol[:]
    solution.insert(index,call)
    call_before = sol[index-1]
    total_cost = 0
    # print("before",call_before)
    i = solution.index(call_before)
    if call in sol:
        cur_node = data.call_info[call-1][2]
        ti = (vehicle_index)*data.num_calls+call-1    
        transfer_cost = data.node_times_and_cost[ti][5] 
        # print("CisDelivery")
    else:
        cur_node = data.call_info[call-1][1]
        ti = (vehicle_index)*data.num_calls+call-1    
        transfer_cost = data.node_times_and_cost[ti][3] 
        # print("CisPickup")

    if call_before == 0:
        # print("isZero")
        node_before = data.vehicle_info[vehicle_index][1]
    elif i != max(0,index-1):
        node_before = data.call_info[call_before-1][2]
        # print("1isDelivery")
    else:
        node_before = data.call_info[call_before-1][1]
        # print("1isPickup")

    ti = data.num_vehicles*((node_before-1)*data.num_nodes+cur_node)-data.num_vehicles+vehicle_index
    travel_cost = data.travel_times_and_cost[ti][4]
    total_cost += travel_cost + transfer_cost

    call_after = sol[index]
    # print("after",call_after)
    i = solution.index(sol[index])
    if call_after == 0:
        # print("isZero")
        # print(total_cost, "return")
        # print(00,call_before,call,call_after)
        # print(sol)
        return total_cost
    elif i != min(len(solution),index+1):
        # print(i, index+1)
        node_after = data.call_info[call_after-1][2]
    else:
        # print(i, index+1)
        node_after = data.call_info[call_after-1][1]

    ti = data.num_vehicles*((cur_node-1)*data.num_nodes+node_after)-data.num_vehicles+vehicle_index
    travel_cost = data.travel_times_and_cost[ti][4]
    total_cost += travel_cost

    previous_cost = 0
    ti = data.num_vehicles*((node_before-1)*data.num_nodes+node_after)-data.num_vehicles+vehicle_index
    travel_cost = data.travel_times_and_cost[ti][4]
    previous_cost += travel_cost

    diff = total_cost-previous_cost
    # print(call_before,call,call_after)
    # print(sol)
    return diff