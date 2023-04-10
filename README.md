# Metahueristics-project
## Running
To run the code you need python with numpy and pandas installed. I run it with pandas version 1.5.0, numpy version 1.23.3 and python version 3.10.7. When installed run main.py to run the algorithm, make sure the working directory is the same as this file. The default settings run case 1 (7 calls 3 vehicles) using simulated annealing with equal weights and the operators made for this assignment. You can change the current case, operators and weights easily in the main.py file. The result is written into a text file located in the /outputs folder.

The output file is named Output_"casename".txt and contains every improvement made with the algorithm, including which operator was used. At the very end of the file is the table containing the average cost, best cost, % improvement and running time. The best found solution is located right above this table. 

Note that the preprocessing might take a minute for larger cases. Before the algorithm runs some features based on the input data are created and stored in global variables. The intention here is to have as much as possible available in numpy arrays with (hopefully) fast lookup time instead of computing things over and over.

## Operators
### smart_one_insert
This operator does most of the heavy work. It works in 5 steps: (details in parentheses)
1. pick a vehicle to remove a call from /or remove from dummy (weighted selection, the more calls in a vehicle/dummy, the more likely to remove a call from it)
2. pick a call within the vehicle to remove (weighted selection, the more expensive the call is, the more likely to remove the call)
3. find vehicle to insert the call in (random selection, but can only choose vehicles that the removed call is valid for)
4. find position to insert the pickup of the call (weighted selection, weights are determined by the difference in max pickup time for the removed call and all other calls in the vehicle. A small difference means the calls have similar pickup times and they are therefore placed close to eachother)
5. find position to insert the delivery of the call (exact approach, checks every option in front of the pickup node and calculetes the cost of the vehicle, the position with the lowest cost is selected)

### swap2_similar
This is a fairly simple operator using some data from preprocessing. Works as follows:
1. Select a call that is not outsourced (random selection)
2. Select a second call (weighted selection, calls are more likely to be picked the more similar they are to the call picked in 1.)
3. Swap the two calls 

The similarity of calls are determined by a total score of the following features (more similar the lower the score):

* cost difference (cost to travel from pickup to pickup and delivery to delivery)
* max pickup time difference
* max delivery time difference
* size difference
* compatibility of vehicles

I have not really played around with the weighting of each feature, except for normalizing all of them to [0,1]

### swap_similar_vehicles
This is pretty much the same as swap2-similar but for vehicles. The similarity measure of vehicles are a bit different however, it is a total of the following features:
* cost between starting nodes
* starting time difference
* capacity difference

As with the swap2_similar, I've not really done any tuning of the weights of features, but they are normalized to [0,1].

### The cooperation of the operators
The 1-insert operator is the most important operator. It greedely reinserts expensive calls into good positions. 
The swapping operators works nicely with this operator, because the good positioning we get from the reinsert are likely to be good positions when swapping with other
calls or even entire vehicles. By comparing the results to the scoreboard of the course, it seems the operators work well for the smaller cases, but they are not as
impressive for the larger ones. I honestly do not have a good explanation for why this is, maybe you have some ideas or suggestions for improvement? 

### Future operator plans

For the next assignment I plan to make all operators k-swap and k-reinsert instead of 2-swap and 1-reinsert. Other than that I want to try 
tweaking weighting parameters and play with the selection processes, other suggestions are more than welcome. 

