import pdb
from copy import deepcopy
from math import isclose

import matplotlib.pyplot as plt
import numpy as np
import qiskit.optimization.applications.ising.knapsack as ks

from qiskit import Aer, QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.ignis.verification import marginal_counts
from qiskit.optimization.applications.ising import knapsack
from qiskit.providers.jobstatus import JobStatus
from qiskit.visualization import plot_histogram
from qiskit_ionq_provider import IonQProvider
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from qiskit import transpile, execute
from qiskit.result import Result
from qiskit.providers.aer import QasmSimulator
from qiskit.test.mock import FakeMelbourne


def replace_0_w_m1(state):
    # helper func no longer used
    # keep it there just in case
    return [int(i) if int(i) == 1 else -1 for i in state]


def update_params(qc, vals):
    # assign numerical value to parameters in a
    # parametrized circuit

    # list parameters to later put into dicts
    params = qc.ordered_parameters

    pd = {}

    for i, p in enumerate(params):
        pd[p] = float(vals[i])

    qc = qc.bind_parameters(pd)
    return qc


class PlayGround:

    def __init__(self, vals, weights, max_weight, shots=8192):

        self.provider = IonQProvider(token='laldTW63uAIPEfNaJwfljDU6OT3p7bKr')

        # Get an IonQ simulator backend to run circuits on:
        self.backend = self.provider.get_backend("ionq_simulator")

        # for testing purposes only
        # self.backend = QasmSimulator.from_backend(FakeMelbourne())

        # get the weighted Pauli Operators
        self.ham = ks.get_operator(vals, weights, max_weight)[0]
        # get the shift for Maximum values
        self.shift = ks.get_operator(vals, weights, max_weight)[1]

        self.shots = shots

        # number of qubits used
        self.n_qubits = self.ham.num_qubits

        # build the ansatz from this module
        self.ansatz = RealAmplitudes(self.n_qubits, reps=2)

        # number of parameters for this ansatz
        self.num_params = self.ansatz.num_parameters

        # default an initial energy to be large value
        self.prev_energy = 100000000000
        self.ans = 100000000

    def energy_state_solver(self, vals, weights, max_weight):
        # guys, might want to fill this in?
        # or just put in a dummy value here for the sake
        # of presentation

        # calls dwave to solve for the state and energy

        ans = 0
        self.ans = ans

        return

    def energy(self, result):
        # this function is used to calculate the energy for Ising Model
        # Since we now use Weighted Pauli Operators, it is no longer used
        # but I keep it here just in case we need to use to last minute to cover our asses

        e = 0
        for state, count in result.get_counts().items():
            state = replace_0_w_m1(state)
            curr_e = 0
            for pair, weight in self.ham.items():
                curr_e += weight * state[self.n_qubits - 1 -
                                         pair[0]] * state[self.n_qubits - 1 -
                                                          pair[1]]
                e += curr_e * count
                return e / self.shots

    def get_result(self, qcs, shots=1000):

        # Then run the circuit:

        # for obtaining the result dictionary format
        # had to do this otherwise don't know how to make
        # a result object containing many quantumcircuit results

        # job = execute(qcs,self.backend)
        # result = job.result()
        # rd = result.to_dict()
        # for k,v in rd.items():
        #     print(k)
        #     print(v)
        #     if k == "results":
        #         for kk,vv in
        # pdb.set_trace()

        full_rd = None  # full result dictionary
        for aqc in qcs:
            # had to transpile otherwise will have illegal gates
            t_aqc = transpile(aqc, self.backend)
            job = self.backend.run(t_aqc, shots=shots)

            # save job_id
            # job_id_bell = job.job_id()

            # Fetch the result:
            result = job.result()
            rd = result.to_dict()
            if full_rd is None:
                full_rd = rd
            else:
                full_rd['results'].append(rd['results'])

        fin_result = Result.from_dict(full_rd)
        return fin_result

    def compute_energy(self, cur_param):

        # compute the energy correspond to our WeightedPauliOperators
        # with ansatz having parameters: cur_param

        qc_binded = update_params(self.ansatz, cur_param)

        # had to do this for provider's limitation
        qc_cur = QuantumCircuit(self.n_qubits)
        qc_cur.compose(qc_binded, inplace=True)

        # append to evaluation circuit for state initialized by the ansatz,
        # this is in qc_cur
        # append the appropriate rotation to compute
        # expectation value for WeightedPaulioperators
        wpauli_circuits = self.ham.construct_evaluation_circuit(qc_cur, False)

        result = self.get_result(wpauli_circuits, shots=self.shots)

        e = self.ham.evaluate_with_result(result, False)

        return e

    def plot_energy_landscape(self,
                              prev_param,
                              which_params,
                              p_rng0,
                              p_rng1,
                              step=10):

        # plot the energies landscapes around the prev_param
        # due to limitations we only vary two parameters at a time

        energies = np.ones((step, step))

        theta0i = prev_param[which_params[0]]
        theta1i = prev_param[which_params[1]]

        theta0r = np.linspace(theta0i + p_rng0[0], theta0i + p_rng0[1], step)
        theta1r = np.linspace(theta1i + p_rng1[0], theta1i + p_rng1[1], step)

        for i, theta0 in enumerate(theta0r):
            for j, theta1 in enumerate(theta1r):

                cur_param = prev_param
                cur_param[which_params[0]] = theta0
                cur_param[which_params[1]] = theta1

                # e = self.compute_energy(cur_param)
                e = 1

                # e = self.energy(result)
                # print(e)
                # pdb.set_trace()

                energies[i][j] = e

        X, Y = np.meshgrid(theta0r, theta1r)
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # in reality, it will take too much time to do this step
        # for demonstration, just use random number generator
        energies = np.random.random((step, step))

        ax.plot_surface(X,
                        Y,
                        energies,
                        cmap=cm.coolwarm,
                        linewidth=0,
                        antialiased=False)

        ax.set_ylabel('theta{}'.format(which_params[0]))
        ax.set_ylabel('theta{}'.format(which_params[1]))

        plt.show()

    def ask_move(self):

        # asks which direction and magnitude you want to move
        print("Anon bid me how thee wanteth to moveth in the phase space.")
        print("Chooseth thy grise wisely!")
        print("Lacking valor moves shall beest did punish")

        changes = np.zeros((self.num_params,))

        delta = np.pi / 100
        for i in range(self.num_params):
            j = input('How many steps for parameter {}?'.format(i))
            changes[i] = j * delta

        return changes

    def ask_show_param(self):

        # asks which parameter to show the plot
        print(
            "Which two parameters would you like to see for the next energy landscape plot"
        )

        var1, var2 = input("Enter two numbers between 0 and {1} here: ".format(
            self.num_params)).split()

        return int(var1), int(var2)

    def punish_player(self, cur_energy):

        # if you get to a higher energy, give random kick to parameters

        if self.prev_energy < cur_energy:
            print(
                "Bad move, a strong wind blows you to somewhere else in phase space"
            )
            punish = np.random.random((self.num_params,)) * np.pi / 100
            return punish
        else:
            print("Well done, your current energy is {}".format(cur_energy))
            print("You maybe one step closer to the ground state energy {}".
                  format(self.gs_energy))
            self.prev_energy = cur_energy
            return np.zeros((self.num_params,))

    def do_move(self, new_params):

        # actually change the update parameters

        e = self.compute_energy(new_params)

        punish_step = self.punish_player(e)
        return punish_step + new_params  # update the prev_param

    def play(self):
        prev_param = np.pi * np.random.random((self.num_params,)) - np.pi / 2
        next_param = None
        counter = 0
        while not isclose(self.prev_energy, self.ans, rel_tol=0.05):
            vary_param = self.ask_show_param()

            self.plot_energy_landscape(prev_param, vary_param,
                                       [-np.pi / 10, np.pi / 10],
                                       [-np.pi / 10, np.pi / 10], 10)
            moves = self.ask_move()
            next_param = prev_param + moves
            prev_param = self.do_move(next_param)
            if counter >= 5:
                print("You mad bro?")

            counter += 1
        print("Brave Soul, congratulations!")
        print("Now you know the hard works a VQE has to do right?")

### loop in the main function until player either rage quits or reach the place


def main():

    # declare the knapsack problem here
    vals = [1, 2, 3]
    weights = [2, 4, 5]
    max_weight = 8

    # previously used these for easier hamiltonian game
    # hamiltonain = {(0, 1): 0.1, (1, 2): 0.3, (0, 2): -.5}
    # n_qubits = 4
    # shots = 8192

    # initialize the PlayGround for all the work
    pg = PlayGround(vals, weights, max_weight)

    pg.play()


if __name__ == "__main__":

    main()
