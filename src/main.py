import pdb
from copy import deepcopy
from math import isclose
from random import randrange

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import qiskit.optimization.applications.ising.knapsack as ks
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from qiskit import Aer, QuantumCircuit, execute, transpile
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.ignis.verification import marginal_counts
from qiskit.optimization.applications.ising import knapsack
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.jobstatus import JobStatus
from qiskit.result import Result
from qiskit.test.mock import FakeMelbourne
from qiskit.visualization import plot_histogram
from qiskit_ionq_provider import IonQProvider

from knapsack import solve_knapsack


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

    def __init__(self, vals, weights, max_weight, ans, shots=8192):

        self.provider = IonQProvider(token='laldTW63uAIPEfNaJwfljDU6OT3p7bKr')

        # Get an IonQ simulator backend to run circuits on:
        self.backend = self.provider.get_backend("ionq_simulator")

        # get the weighted Pauli Operators
        self.ham = ks.get_operator(vals, weights, max_weight)[0]
        # get the shift for Maximum values
        self.shift = ks.get_operator(vals, weights, max_weight)[1]

        self.shots = shots

        # number of qubits used
        self.n_qubits = self.ham.num_qubits

        # build the ansatz from this module
        self.ansatz = RealAmplitudes(self.n_qubits, reps=1)

        # number of parameters for this ansatz
        self.num_params = self.ansatz.num_parameters

        # default an initial energy to be large value
        self.prev_energy = 100000000000

        # real answer from the DWave Side
        self.ans = ans

    def get_result(self, qcs, shots=1000):

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

    def compute_energy(self, cur_param, demo=True, which_params=None):

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

        if not demo:
            result = self.get_result(wpauli_circuits, shots=self.shots)

            e = self.ham.evaluate_with_result(result, False)
        else:
            # generate some pseudo random energy landscape for demo purposes
            e = abs(np.sin(cur_param[which_params[0]]))**abs(cur_param[0])
            e *= abs(np.cos(cur_param[which_params[1]]))**abs(cur_param[1])
            e *= abs(self.ans)
            e *= np.random.random()
            if np.isnan(e):
                pdb.set_trace()
        return e

    def plot_energy_landscape(self,
                              ax,
                              prev_param,
                              which_params,
                              p_rng0,
                              p_rng1,
                              step=10,
                              demo=True):

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

                e = self.compute_energy(cur_param, demo, which_params)

                energies[i][j] = e

        X, Y = np.meshgrid(theta0r, theta1r)

        # plot current location
        ax.plot([theta0i] * 10, [theta1i] * 10,
                np.linspace(np.min(energies), np.max(energies), 10),
                'or-',
                alpha=0.8,
                linewidth=1.5)

        ax.plot_surface(X,
                        Y,
                        energies,
                        cmap=cm.coolwarm,
                        linewidth=0,
                        antialiased=False)

        ax.set_ylabel('theta{}'.format(which_params[0]))
        ax.set_xlabel('theta{}'.format(which_params[1]))

        return ax

    def ask_move(self):

        # asks which direction and magnitude you want to move
        print("Anon bid me how thee wanteth to moveth in the parameter space.")
        print("Chooseth thy grise wisely!")
        print("Lacking valor moves shall beest did punish")

        changes = np.zeros((self.num_params,))

        print("Tell me which two parameters you want to move in")

        var1 = int(
            input("Enter a number between 0 and {0} separated by a space: ".
                  format(self.num_params)) or randrange(self.num_params))
        var2 = int(
            input("Enter a number between 0 and {0} separated by a space: ".
                  format(self.num_params)) or randrange(self.num_params))

        delta = np.pi / 100

        for i in [int(var1), int(var2)]:
            j = int(
                input('How many steps for parameter {}?'.format(i)) or
                randrange(5))
            changes[i] = j * delta

        return changes, [int(var1), int(var2)]

    def ask_show_param(self):

        # asks which parameter to show the plot
        print(
            "Which two parameters would you like to see for the next energy landscape plot"
        )

        var1 = int(
            input("Enter a number between 0 and {0} separated by a space: ".
                  format(self.num_params)) or randrange(self.num_params))
        var2 = int(
            input("Enter a number between 0 and {0} separated by a space: ".
                  format(self.num_params)) or randrange(self.num_params))

        return int(var1), int(var2)

    def punish_player(self, cur_energy):

        # if you get to a higher energy, give random kick to parameters

        if self.prev_energy < cur_energy:
            print("You moved from energy {0} to {1}".format(
                self.prev_energy, cur_energy))
            print(
                "Bad move, a strong wind blows you to somewhere else in parameter space"
            )
            punish = np.random.random((self.num_params,)) * np.pi / 100
            return punish
        else:
            print("Well done, your current energy is {}".format(cur_energy))
            print("You maybe one step closer to the ground state energy {}".
                  format(self.ans))
            self.prev_energy = cur_energy
            return np.zeros((self.num_params,))

    def do_move(self, new_params, demo, which_params):

        # actually change the update parameters
        e = self.compute_energy(new_params, demo, which_params)

        punish_step = self.punish_player(e)
        return punish_step + new_params  # update the prev_param

    def play(self):
        prev_param = np.pi * np.random.random((self.num_params,)) - np.pi / 2
        next_param = None
        counter = 0
        while not isclose(self.prev_energy, self.ans, rel_tol=0.05):

            # fig, axs = plt.subplots(2, 2)
            fig = plt.figure()
            # loop over this to show multiple views
            for i in range(4):
                print("Now pick some parameter pairs")
                vary_param = self.ask_show_param()

                cur_ax = fig.add_subplot(2, 2, i + 1, projection='3d')
                cur_ax = self.plot_energy_landscape(cur_ax, prev_param,
                                                    vary_param,
                                                    [-np.pi / 10, np.pi / 10],
                                                    [-np.pi / 10, np.pi / 10],
                                                    10)

            plt.title("Optimal answer: {}".format(self.ans))
            plt.show()
            # change this retarded funciton I wrote
            # put the optimal answer on the title of the plot
            moves, which_params = self.ask_move()
            next_param = prev_param + moves
            prev_param = self.do_move(next_param, True, which_params)
            if counter >= 2:
                print("You mad yet bro?")

            counter += 1

        print("Brave Soul, congratulations!")
        print("Now you know the hard works a VQE has to do right?")


### loop in the main function until player either rage quits or reach the place


def main():
    # declare the knapsack problem here
    vals = [1, 2, 3]
    weights = [2, 4, 5]
    max_weight = 8

    # asks the DWave to give us the answer
    # ans = solve_knapsack(vals, weights, max_weight)

    # hotwire the answer from DWave, it does not run on my machine
    # this is just for demonstration purposes only
    ans = -4.0
    # initialize the PlayGround for all the work
    pg = PlayGround(vals, weights, max_weight, ans)

    pg.play()


if __name__ == "__main__":
    main()
