import random
import math
import data
import numpy as np
from solution_checker import isFeasable
from solution_checker import cost
from generators import generate_random_solution
from statistics import mean
import helpers
import final_operators as fops

def blind_random_search(s0):
    best_sol = s0
    cost_best_sol = cost(best_sol)
    for _ in range(10000):
        cur_sol = generate_random_solution()
        if isFeasable(cur_sol) and cost(cur_sol)<cost_best_sol:
            best_sol = cur_sol
            cost_best_sol = cost(best_sol)
    return best_sol

def local_search(s0,operator):
    best_sol = s0
    cost_best_sol = cost(best_sol)
    for _ in range(10000):
        cur_sol = operator(best_sol)
        if isFeasable(cur_sol) and cost(cur_sol)<cost_best_sol:
            best_sol = cur_sol
            cost_best_sol = cost(best_sol)
            print(best_sol,cost_best_sol)
    return best_sol

def simulated_annealing(s0,operator):
    T_final = 5
    best_sol = s0
    incumbent = s0
    deltas = []

    # find temps
    for _ in range(0,100):
        new_sol = operator(incumbent)
        cost_incumbent = cost(incumbent)
        delta = cost(new_sol) - cost_incumbent
        if delta < 0 and isFeasable(new_sol):
            incumbent = new_sol
            if cost_incumbent < cost(best_sol):
                best_sol = incumbent
        elif isFeasable(new_sol):
            if random.random() < 0.8:
                incumbent = new_sol
            deltas.append(delta)
    if deltas:
        delta_avg = mean(deltas)
    else:
        print("Got no deltas")
        delta_avg = 10000
    T_start = -delta_avg/math.log(0.8)
    alfa = (T_final/T_start)**(1/9990)

    temp = T_start
    cost_incumbent = cost(incumbent)
    for _ in range(9990):
        new_sol = operator(incumbent)
        delta = cost(new_sol) - cost_incumbent
        Feasable = isFeasable(new_sol)
        if delta < 0 and Feasable:
            incumbent = new_sol
            cost_incumbent = cost(incumbent)
            if cost_incumbent < cost(best_sol):
                best_sol = incumbent
        elif Feasable and random.random() < math.exp(-delta/temp):
            incumbent = new_sol
            cost_incumbent = cost(incumbent)
        temp *= alfa
    return best_sol

def alt_escape(sol,operators,weights):
    incumbent = sol[:]
    best_sol = incumbent[:]

    for i in range(5):
        incumbent = helpers.put_in_dummy(incumbent,random.randint(1,data.num_calls))
        for k in range(500):
            operator = random.choices(operators,[1,1,1])[0]
            new_sol = operator(incumbent)

            if isFeasable(new_sol):
                incumbent = new_sol
                if (cost(incumbent) < cost(best_sol)):
                    best_sol = incumbent
                    return best_sol
    return best_sol

        

def escape_algorithm(incumbent,operators,weights):
    # At local minima is given. We need to escape.
    # Options:
    #   reinsert random call in dummy and run operators on the new solution, if no new best_sol,
    #   remove more and try again
    best_sol = incumbent[:]
    print(best_sol)

    for i in range(30):
        for j in range(10):
            operator = random.choices(operators,weights)[0]
            new_sol = operator(incumbent)
            if isFeasable(new_sol) and cost(new_sol) != cost(incumbent):
                incumbent = new_sol
                print("Feasable")
                print(incumbent,cost(incumbent), cost(best_sol))
                if cost(incumbent) < cost(best_sol):
                    print(i,j)
                    best_sol = incumbent
                break

        for k in range(50):
            operator = random.choices(operators,[1,1,1])[0]
            new_sol = operator(incumbent)
            cost_incumbent = cost(incumbent)
            delta = cost(new_sol) - cost_incumbent

            if isFeasable(new_sol) and (cost(incumbent) < cost(best_sol)):
                print(cost(best_sol))
                best_sol = incumbent
                return best_sol
    
    return best_sol

def adaptive_search(s0,operators):
    total_runs = 10000
    start_runs =  100
    main_runs = total_runs-start_runs
    operator_indexes = np.arange(len(operators))
    periods = 3
    runs_in_period = main_runs//periods
    best_sol = s0
    incumbent = s0
    T_final = 5
    deltas = []
    weights = np.ones(len(operators))
    scores = np.ones(len(operators))

    # find temps
    for _ in range(0,start_runs):
        op_index = random.choices(operator_indexes,weights)[0]
        operator = operators[op_index]
        new_sol = operator(incumbent)
        cost_incumbent = cost(incumbent)
        delta = cost(new_sol) - cost_incumbent
        if delta < 0 and isFeasable(new_sol):
            incumbent = new_sol
            scores[op_index] += 1
            if cost_incumbent < cost(best_sol):
                best_sol = incumbent
                scores[op_index] += 2
        elif isFeasable(new_sol):
            if random.random() < 0.8:
                incumbent = new_sol
            deltas.append(delta)
    if deltas:
        delta_avg = mean(deltas)
    else:
        print("got no deltas")
        delta_avg = 10000

    T_start = -delta_avg/math.log(0.9)
    alfa = (T_final/T_start)**(1/9990)

    temp = T_start
    cost_incumbent = cost(incumbent)
    weights = scores/sum(scores)
    for _ in range(periods):
        scores = np.ones(len(operators))
        print(weights, cost(best_sol))
        
        for _ in range(runs_in_period):
            op_index = random.choices(operator_indexes,weights)[0]
            operator = operators[op_index]
            new_sol = operator(incumbent)
            cost_incumbent = cost(incumbent)
            delta = cost(new_sol) - cost_incumbent
            # log score
            feasable = isFeasable(new_sol)
            if feasable:
                scores[op_index] += 1
            if delta < 0 and feasable:
                scores[op_index] += 50
                incumbent = new_sol
                if cost_incumbent < cost(best_sol):
                    scores[op_index] += 100
                    best_sol = incumbent
            elif feasable and random.random() < math.exp(-delta/temp):
                incumbent = new_sol
                cost_incumbent = cost(incumbent)
            temp *= alfa
        scores = scores/sum(scores)
        weights = scores/weights
        weights = weights/sum(weights)
        
    print(weights, cost(best_sol))
    return best_sol
            

def simulated_annealing_multiple_ops(s0,operators,weights):
    T_final = 5
    best_sol = s0
    incumbent = s0
    deltas = []

    # find temps
    for _ in range(0,100):
        operator = random.choices(operators,weights)[0]
        new_sol = operator(incumbent)
        cost_incumbent = cost(incumbent)
        delta = cost(new_sol) - cost_incumbent
        if delta < 0 and isFeasable(new_sol):
            incumbent = new_sol
            if cost_incumbent < cost(best_sol):
                best_sol = incumbent
        elif isFeasable(new_sol):
            if random.random() < 0.8:
                incumbent = new_sol
            deltas.append(delta)
    if deltas:
        delta_avg = mean(deltas)
    else:
        print("got no deltas")
        delta_avg = 10000

    T_start = -delta_avg/math.log(0.8)
    alfa = (T_final/T_start)**(1/9990)

    temp = T_start
    cost_incumbent = cost(incumbent)
    for _ in range(9990):
        operator = random.choices(operators,weights)[0]
        new_sol = operator(incumbent)
        delta = cost(new_sol) - cost_incumbent
        Feasable = isFeasable(new_sol)
        if delta < 0 and Feasable:
            incumbent = new_sol
            cost_incumbent = cost(incumbent)
            if cost_incumbent < cost(best_sol):
                best_sol = incumbent
                # print(incumbent, cost_incumbent, operator)
        elif Feasable and random.random() < math.exp(-delta/temp):
            incumbent = new_sol
            cost_incumbent = cost(incumbent)
        temp *= alfa
    return best_sol

def escape(sol,best_cost):
    incumbent = sol[:]
    count = 0
    for i in range(30):
        temp, calls = fops.remove_k_random(incumbent,int(2+data.num_calls*0.1))
        temp = fops.insert_k_quick(temp,calls)
        # print(temp, cost(temp),isFeasable(temp))
        if temp != incumbent and isFeasable(temp):
            count += 1
            incumbent = temp
            if cost(temp) < best_cost:
                return incumbent, True
            if count == 3:
                break
    return incumbent, False

def intensify(sol,best_cost):
    incumbent = sol[:]
    # print("--------------------")
    # print(sol, cost(sol))
    for i in range(10):
        temp, calls, vehicles = fops.remove_costly_and_similar_vehicles(incumbent,k=2)
        temp = fops.insert_k_best_pos(temp,calls,vehicles)
        if temp != incumbent and isFeasable(temp):
            # print(temp, cost(temp))
            if cost(temp) < best_cost:
                return incumbent, True
    return incumbent, False

def ALNS(sol,removal_ops,insertion_ops):
    print("Cost,Iteration")
    solution = sol[:]
    best_sol = solution
    best_cost = cost(best_sol)
    incumbent = solution
    cost_incumbent = cost(incumbent)

    total_runs = 10000
    r_ops_indexes = np.arange(len(removal_ops))
    i_ops_indexes = np.arange(len(insertion_ops))
    periods = 20
    runs_in_period = total_runs//periods

    r_weights = np.ones(len(removal_ops))
    i_weights = np.ones(len(insertion_ops))
    D = 0.2
    q_expected = 1
    q_max = int(2+math.log(data.num_calls))
    qs = np.arange(1,q_max+1)
    q_weights = np.empty(q_max)
    q_weights.fill(q_expected)
    q_weights = np.subtract(q_weights,qs)
    q_weights = 1/((q_weights**2)+1)
    r = 0.7
    e = 0
    a = 0
    thresh = 0.005
    itr_of_last_improvement = -1
    itr_since_last_improvement = -1
    r_tot_scores = np.ones(len(removal_ops))
    i_tot_scores = np.ones(len(insertion_ops))
    for i in range(periods):
        q_selected = []
        i_op_count = np.ones(len(insertion_ops))
        i_scores = np.ones(len(insertion_ops))
        r_op_count = np.ones(len(removal_ops))
        r_scores = np.ones(len(removal_ops))
        for j in range(runs_in_period):
            # if a > 500:
            #     a = 0
            #     # print(best_sol, "before intensify")
            #     incumbent, improved = intensify(best_sol,best_cost)
            #     cost_incumbent = cost(incumbent)
            #     # print(incumbent, "after intensify")
            #     if improved:
            #         q_expected = 1
            #         thresh = 0.005
            #         best_sol = incumbent
            #         best_cost = cost(best_sol)
            if e > 400:
                e = 0
                # print(best_sol, "before escape")
                incumbent, improved = escape(best_sol,best_cost)
                cost_incumbent = cost(incumbent)
                # print(incumbent, "after escape")
                if improved:
                    q_expected = 1
                    thresh = 0.005
                    best_sol = incumbent
                    best_cost = cost(best_sol)

            # print(q,q_weights)
            q = random.choices(qs,q_weights)[0]
            q_selected.append(q)
            r_op_index = random.choices(r_ops_indexes,r_weights)[0]
            r_operator = removal_ops[r_op_index]
            r_op_count[r_op_index] +=1
            i_op_index = random.choices(i_ops_indexes,i_weights)[0]
            i_operator = insertion_ops[i_op_index]
            i_op_count[i_op_index] +=1
            temp, calls = r_operator(incumbent,q)
            new_sol = i_operator(temp,calls)

            cost_new_sol = cost(new_sol)
            delta = cost_new_sol-cost_incumbent
            print(f'{cost_incumbent},{((i*runs_in_period)+j)}')

            if delta < 0 and isFeasable(new_sol):
                incumbent = new_sol
                cost_incumbent = cost(incumbent)
                itr_of_last_improvement = (i*runs_in_period)+j
                i_scores[i_op_index] += 2
                r_scores[r_op_index] += 2

                if cost_incumbent < best_cost:
                    e = 0
                    a = 0
                    # print(incumbent,cost_incumbent, r_operator,i_operator, q,q_expected,(i*runs_in_period)+j)
                    q_expected = 1
                    thresh = 0.005
                    i_scores[i_op_index] += 2
                    r_scores[r_op_index] += 2
                    best_sol = incumbent
                    best_cost = cost_incumbent
            elif cost_new_sol < best_cost + D and isFeasable(new_sol):
                # itr_of_last_accepted = (i*runs_in_period)+j
                i_scores[i_op_index] += 1
                r_scores[r_op_index] += 1
                incumbent = new_sol
                cost_incumbent = cost_new_sol

            e += 1
            a += 1
            D = 0.2*((total_runs-((i*runs_in_period)+j))/total_runs)*best_cost
            itr_since_last_improvement = (i*runs_in_period)+j-itr_of_last_improvement

            if itr_since_last_improvement/total_runs > thresh:
                itr_of_last_improvement = (i*runs_in_period)+j
                q_expected = min(q_expected+1,q_max)
                # print("increased to: ",q_expected)
                thresh = thresh*2
                q_weights = np.empty(q_max)
                q_weights.fill(q_expected)
                q_weights = np.subtract(q_weights,qs)
                q_weights = 1/((q_weights**2)+1)
                # print(q_weights)
        # print(scores)
        r_tot_scores += r_scores
        i_tot_scores += i_scores
        # print(r_tot_scores,r_op_count)
        # print(i_tot_scores,i_op_count)
        # r_scores = r_scores/r_op_count
        # i_scores = i_scores/i_op_count

        r_scores = r_scores/sum(r_scores)
        r_weights = r_weights/sum(r_weights)
        r_weights = r_weights*(1-r)+r*r_scores
        i_scores = i_scores/sum(i_scores) 
        i_weights = i_weights/sum(i_weights)
        i_weights = i_weights*(1-r)+r*i_scores
        # print(r_weights,(i*runs_in_period))
        
    # print(q_cs)
    return best_sol


def alt_annealing(s0,operators,weights):
    best_sol = s0
    incumbent = s0
    cost_incumbent = cost(incumbent)
    best_cost = cost(best_sol)
    D = 0.2
    iterations = 10000

    for i in range(iterations):
        operator = random.choices(operators,weights)[0]
        new_sol = operator(incumbent)
        new_cost = cost(new_sol)
        delta = new_cost - cost_incumbent
        Feasable = isFeasable(new_sol)
        if delta < 0 and Feasable:
            incumbent = new_sol
            cost_incumbent = cost(incumbent)
            if cost_incumbent < best_cost:
                best_sol = incumbent
                best_cost = cost(best_sol)
                print(incumbent, cost_incumbent, operator, i)
        elif Feasable and new_cost < best_cost + D:
            incumbent = new_sol
            cost_incumbent = cost(incumbent)
        D = 0.2*((iterations-i)/iterations)*best_cost
    return best_sol