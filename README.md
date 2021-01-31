# Project: Knapsack problem solved with Gate-based VQE and Annealing-based DQM.
# Team: qa_fold (Advanced Hybrid Division)

Ziwei Qiu, Ilan Mitnikov, Yusheng Zhao, Nakul Aggarwal , Victor Onofre 

## Annealing: Implement the Knapsack problem with the DQM solver
To represent the effect of any decision by numerical values, the outcome of the decision can be measured by a single value representing gain, profit, loss, cost, or some other data category. The comparison of these values gives a total order on the set of all options for a decision. Finding the option with the highest or lowest value can be difficult because the set of available options may be extensive. The knapsack problem is one of this decision process.

The knapsack problem can be formally defined as follows: We are given an item set N, consisting of n items j with profit Pj and weight Wj, and the capacity value c. The objective is to select a subset of N such that the total profit of the selected items is maximized and the total weight does not exceed c. In this work, we demonstrated solving the Bounded Knapsack Problem with the D-wave Ocean Discrete Quadratic Model (DQM) solver, where we are allowed to take multiple pieces for each item so the variable can take discrete values 0,1,2,... up to b. This extended Knapsack problem has a direct application in stock portofolio optimization where we show a proof-of-concept demonstration in the notebook 'Knapsack_DQM.ipynb'. 

Knapsack problems appear in many real-world decision-making processes, including home energy management, cognitive radio networks, resource management in software, power allocation management, relay selection in secure cooperative wireless communication, etc. 

Future works on this project include:
1. Study in more detail and quantitatively the actual quantum advantage of solving this NP-hard problem on large dataset over classical methods by running on real quantum computers. 
2. Implement different variants of the Knapsack problem, e.g. by adding more constraints, adding m knapsacks with different capacities or optimizing the Unbounded Knapsack Problem where an unlimited amount of each item is availabl.
3. Use Knapsack problem as a subroutine and combine it with other NP-hard problem to solve complicated tasks challenging to classical computers.
4. In the stock selection application, we can better quantify the profits instead of just using earnings as the metric, and have more realistic assumptions.



## Gate-based: VQE Game

Check out some info in the [event's repository](https://github.com/iQuHACK/2021) to get started.

Having a README in your team's repository facilitates judging. A good README contains:
* a clear title for your project,
* a short abstract,
* the motivation/goals for your project,
* a description of the work you did, and
* proposals for future work.

You can find a potential README template in [one of last year's projects](https://github.com/iQuHACK/QuhacMan).

Feel free to contact the staff with questions over our [event's slack](https://iquhack.slack.com), or via iquhack@mit.edu.

Good luck!
