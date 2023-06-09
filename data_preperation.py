import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import helpers
import data

def read_data(case):
    f = open("Data/"+case+".txt", "r")
    percent_count = 0
    ntc_key = 0
    ttc_key = 0
    for line in f:
        line = line.strip()
        if line[0] == "%":
            percent_count += 1
            continue
        match percent_count:
            case 1: # number of nodes
                data.num_nodes = int(line)
            case 2: # number of vehicles
                data.num_vehicles = int(line)
            case 3: # for each vehicle: vehicle index, home node, starting time, capacity
                line = line.split(',')
                line = [eval(i) for i in line]
                data.vehicle_info.append(line)
            case 4: # number of calls
                data.num_calls = int(line)
            case 5: # for each vehicle, vehicle index, and then a list of calls that can be transported using that vehicle
                line = line.split(',')
                line = [eval(i) for i in line]
                key = line[0]
                vals = line[1:]
                pair = (key,vals)
                data.valid_calls.append(pair)
            case 6: # for each call: call index, origin node, destination node, size, cost of not transporting, lowerbound timewindow for pickup, upper_timewindow for pickup, lowerbound timewindow for delivery, upper_timewindow for delivery
                line = line.split(',')
                line = [eval(i) for i in line]
                key = line[0]
                vals = line[1:]
                pair = (key,vals)
                data.call_info.append(pair)
            case 7: # travel times and costs: vehicle, origin node, destination node, travel time (in hours), travel cost (in Euro)
                line = line.split(',')
                line = [eval(i) for i in line]
                time = line[3]
                cost = line[4]
                time_pair = (ttc_key,time)
                cost_pair = (ttc_key,cost)
                data.travel_cost.append(cost_pair)
                data.travel_time.append(time_pair)
                ttc_key += 1
            case 8: # node times and costs: vehicle, call, origin node time (in hours), origin node costs (in Euro), destination node time (in hours), destination node costs (in Euro)
                line = line.split(',')
                line = [eval(i) for i in line]
                # print(line)
                pair = (ntc_key,line)
                data.node_times_and_cost.append(pair)
                ntc_key+=1
    data.vehicle_info = np.array(data.vehicle_info)
    data.call_info = dict(data.call_info)
    data.valid_calls = dict(data.valid_calls)
    data.travel_cost = dict(data.travel_cost)
    data.travel_time = dict(data.travel_time)
    data.node_times_and_cost = dict(data.node_times_and_cost)

def generate_data():
    call_info = pd.DataFrame.from_dict(data.call_info,orient="index")
    call_info.columns = ["origin_node", "destination_node", "size", "cost_not_transporting", "lowerbound_timewindow_pickup", "upper_timewindow_pickup", "lowerbound_timewindow_delivery", "upper_timewindow_delivery"]
    data.pickup_sorted_by_time = list(call_info.sort_values(by="upper_timewindow_pickup").index.values)
    # data.largest_travel_cost = data.travel_times_and_cost[:,3].max()
    # data.largest_travel_time = data.travel_times_and_cost[:,2].max()
    # print(call_info)
    # data.latest_pickup_time = data.call_info[:,5].max()
    # data.latest_delivery_time = data.call_info[:,7].max()
    generate_call_relativity()
    generate_vehicle_similarity()

def generate_call_relativity():
    call_relativity = []
    for i in range(1,data.num_calls+1):
        for j in range(1,data.num_calls+1):
            cost_r = helpers.cost_relativity(i,j)
            time_pickup_r = time_pickup_relativity(i,j)
            time_delivery_r = time_delivery_relativity(i,j)
            size_r = size_relativity(i,j)
            vehicle_compatibility_r = vehicle_compatibility(i,j)
            call_pair = [i,j,cost_r,time_pickup_r,time_delivery_r,vehicle_compatibility_r,size_r]
            call_relativity.append(call_pair)
    call_r_df = pd.DataFrame(call_relativity)
    columns = ['call1','call2','cost','time_pickup','time_delivery','vehicle_compatibility','call_size']
    call_r_df.columns = columns
    call_r_df = call_r_df.convert_dtypes()
    call_r_df = call_r_df.query("call1 != call2")
    #[call_r_df.columns[2:]]
    for column in columns[2:]:
        call_r_df[column] = call_r_df[column] /call_r_df[column].abs().max()
    # call_r_df.iloc[:,2:] = (call_r_df.iloc[:,2:]-call_r_df.iloc[:,2:].min())/(call_r_df.iloc[:,2:].max()-call_r_df.iloc[:,2:].min())
    call_r_df['total'] = call_r_df.cost + call_r_df.time_pickup + call_r_df.time_delivery + call_r_df.vehicle_compatibility + call_r_df.call_size
    # call_r_df = call_r_df.round(4)
    call_r_df = call_r_df.sort_values(by=['call1','call2'])
    data.call_relativity = call_r_df.to_numpy()

def time_pickup_relativity(call1,call2):
    pickup_max_1 =data.call_info[call1][5]
    pickup_max_2 =data.call_info[call2][5]
    diff = (pickup_max_1-pickup_max_2)**2

    return diff

def time_delivery_relativity(call1,call2):
    delivery_max_1 =data.call_info[call1][7]
    delivery_max_2 =data.call_info[call2][7]
    diff = (delivery_max_1-delivery_max_2)**2
    return diff

def size_relativity(call1,call2):
    s1 = data.call_info[call1][2]
    s2 = data.call_info[call2][2]
    size_r = abs(s1-s2)/max(s1,s2)
    return size_r

def vehicle_compatibility(call1,call2):
    call1_compatible_vehicles = []
    call2_compatible_vehicles = []
    for vehicle_id in range(data.num_vehicles):
        valid = data.valid_calls[vehicle_id+1]
        if call1 in valid:
            call1_compatible_vehicles.append(vehicle_id)
        if call2 in valid:
            call2_compatible_vehicles.append(vehicle_id)
    total = max(len(call1_compatible_vehicles),len(call2_compatible_vehicles))
    compatibility_r = len(set(call2_compatible_vehicles).symmetric_difference(set(call1_compatible_vehicles)))/total
    return compatibility_r

def normalize(column):
    return column/max(column)

def generate_vehicle_similarity():
    for v1 in range(0,data.num_vehicles):
        for v2 in range(0,data.num_vehicles):
            starting_node_distance = helpers.get_travel_cost(data.vehicle_info[v1][1],data.vehicle_info[v2][1],0)
            starting_node_distance = starting_node_distance # not taking abs value results in negative numbers
            starting_time_diff = (data.vehicle_info[v1][2]-data.vehicle_info[v2][2])
            capacity_diff = (data.vehicle_info[v1][3]-data.vehicle_info[v2][3])
            info = np.array([v1,v2,starting_node_distance,starting_time_diff,capacity_diff])
            # info = info/max(info)
            index = v1*data.num_vehicles+v2
            row = data.vehicle_info
            # vehicle_similarity.update({f'{index}':f"{info}"})
            data.vehicle_similarities.append(info)
    columns = ['v1','v2','start_node_dist','start_time_diff','capacity_diff']
    data.vehicle_similarities = pd.DataFrame(data.vehicle_similarities,index=np.arange(data.num_vehicles**2),columns=columns)
    data.vehicle_similarities.iloc[:,2:] = pd.DataFrame.apply(data.vehicle_similarities.iloc[:,2:],normalize,axis=0)
    data.vehicle_similarities.iloc[:,2:] = data.vehicle_similarities.iloc[:,2:]**2#(math.log(data.num_calls)/2)
    data.vehicle_similarities = data.vehicle_similarities.query(" v1 != v2")
    data.vehicle_similarities['total'] = data.vehicle_similarities.iloc[:,2:].sum(axis=1)
    data.vehicle_similarities = data.vehicle_similarities.sort_values(by=['total','v1'])

    data.vehicle_similarities = data.vehicle_similarities.fillna(0)
    data.vehicle_similarities = data.vehicle_similarities