import random
import data

def generate_random_solution():
    solution = []
    calls = random.sample(range(1,data.num_calls+1),data.num_calls)
    for _ in range(data.num_vehicles):
        calls.insert(random.randint(0, len(calls)), 0) 

    temp = []
    for i in calls:
        if i == 0:
            temp.extend(temp)
            random.shuffle(temp)
            solution.extend(temp)
            solution.append(0)
            temp = []
            continue
        temp.append(i)
    temp.extend(temp)
    random.shuffle(temp)
    solution.extend(temp)

    return solution

def gen_dummy_sol():
    sol = []
    for i in range(data.num_vehicles):
        sol.append(0)
    for j in range(data.num_calls):
        sol.append(j+1)
        sol.append(j+1)
    return sol