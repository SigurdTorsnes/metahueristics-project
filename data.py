
def init():
    global num_calls
    global num_nodes
    global num_vehicles
    global vehicle_info
    global valid_calls
    global call_info
    global travel_times_and_cost
    global node_times_and_cost
    global pickup_sorted_by_time
    global largest_travel_cost
    global largest_travel_time
    global latest_pickup_time
    global latest_delivery_time
    global call_relativity

    num_calls = 0
    num_nodes = 0
    num_vehicles = 0
    pickup_sorted_by_time = []
    vehicle_info = []
    valid_calls = []
    call_info = []
    travel_times_and_cost = []
    node_times_and_cost = []
    largest_travel_cost = 0
    largest_travel_time = 0
    latest_pickup_time = 0
    latest_delivery_time = 0
    call_relativity = []