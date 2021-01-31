# Project: Knapsack problem solved with Gate-based VQE and Annealing-based DQM.

Team: Hybrid Demons (Advanced Hybrid Division)
Ziwei Qiu, Ilan Mitnikov, Yusheng Zhao, Nakul Aggarwal , Victor Onofre 

The knapsack problem can be formally defined as follows: We are given an item set N, consisting of n items j with profit Pj and weight Wj, and the capacity value c. The objective is to select a subset of N such that the total profit of the selected items is maximized and the total weight does not exceed c. (see more details here: https://en.wikipedia.org/wiki/Knapsack_problem)

In this project, we work on solving the Knapsack problem with both gate-based VQE methods running on IonQ hardware and annealing-based DQM/BQM methods running on D-Wave hardware, and to compare between the two methods. We further show using the DQM solver to implement the bounded Knapsack problem.


## Annealing: Implement the bounded Knapsack problem with the DQM solver
We demonstrated solving the Bounded Knapsack Problem with the D-wave Ocean Discrete Quadratic Model (DQM) solver, where we are allowed to take multiple pieces for each item so the variable can take discrete values 0,1,2,... up to b. This extended Knapsack problem has a direct application in stock portofolio optimization where we show a proof-of-concept demonstration in the notebook 'Knapsack_DQM.ipynb'. 

Knapsack problems appear in many real-world decision-making processes, including home energy management, cognitive radio networks, resource management in software, power allocation management, relay selection in secure cooperative wireless communication, etc. 

Future works on this project include:
(1) Study in more detail and quantitatively the actual quantum advantage of solving this NP-hard problem on large dataset over classical methods by running on real quantum computers. 
(2) Implement different variants of the Knapsack problem, e.g. by adding more constraints, adding m knapsacks with different capacities or optimizing the Unbounded Knapsack Problem where an unlimited amount of each item is availabl.
(3) Use Knapsack problem as a subroutine and combine it with other NP-hard problem to solve complicated tasks challenging to classical computers.
(4) In the stock selection application, we can better quantify the profits instead of just using earnings as the metric, and have more realistic assumptions.



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
