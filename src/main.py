import pdb
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import qiskit.optimization.applications.ising.knapsack as ks
# import Aer here, before calling qiskit_ionq_provider
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

def replace_0_w_m1(state):
    return [int(i) if int(i) == 1 else -1 for i in state]


def update_params(qc, vals):
    params = qc.ordered_parameters
    pd = {}
    for i, p in enumerate(params):
        pd[p] = float(vals[i])
    qc = qc.bind_parameters(pd)
    return qc


def energy_state_solver():
    # calls dwave to solve for the state and energy
    pass


def ask_show_param():

    # asks which parameter to show the plot

    pass


def ask_move(num_params):
    # asks which direction and magnitude you want to move

    pass


def punish_player():

    # if you get to a higher energy, give random kick to parameters

    pass


def do_move():

    # actually change the update parameters
    pass


class PlayGround:

    def __init__(self, n_qubits, shots, hamiltonian):

        self.provider = IonQProvider(token='laldTW63uAIPEfNaJwfljDU6OT3p7bKr')
        # Get an IonQ simulator backend to run circuits on:
        self.backend = self.provider.get_backend("ionq_simulator")

        self.ham = hamiltonian
        self.shots = shots
        self.n_qubits = n_qubits

    def energy(self, result):
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

    def get_result(self, qc, shots=1000):

        # Then run the circuit:
        # pdb.set_trace()
        job = self.backend.run(qc, shots=shots)
        # save job_id
        # job_id_bell = job.job_id()

        # Fetch the result:
        result = job.result()
        return result

    def plot_energy_landscape(self,
                              qc,
                              prev_param,
                              which_params,
                              p_rng0,
                              p_rng1,
                              step=10):

        energies = np.ones((step, step))

        theta0i = prev_param[which_params[0]]
        theta1i = prev_param[which_params[1]]

        theta0r = np.linspace(theta0i + p_rng0[0], theta0i + p_rng0[1], step)
        theta1r = np.linspace(theta1i + p_rng1[0], theta1i + p_rng1[1], step)
        # for i, theta0 in enumerate(theta0r):
        #     for j, theta1 in enumerate(theta1r):

        #         cur_param = prev_param
        #         cur_param[which_params[0]] = theta0
        #         cur_param[which_params[1]] = theta1

        #         qc_cur = update_params(qc, cur_param)
        #         result = self.get_result(qc_cur, shots=self.shots)
        #         e = self.energy(result)
        #         print(e)
        #         # pdb.set_trace()

        #         energies[i][j] = e
        X, Y = np.meshgrid(theta0r, theta1r)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        energies = np.random.random((step,step))
        ax.plot_surface(X,
                        Y,
                        energies,
                        cmap=cm.coolwarm,
                        linewidth=0,
                        antialiased=False)
        ax.set_ylabel('theta{}'.format(which_params[0]))
        ax.set_ylabel('theta{}'.format(which_params[1]))
        plt.show()


def comp_answer():

    # compare the L2 distance to the actual answer

    pass


def apply_ham():

    pass


def make_ansatz_circuit():
    pass


### loop in the main function until player either rage quits or reach the place


def main():

    hamiltonain = {(0, 1): 0.1, (1, 2): 0.3, (0, 2): -.5}
    n_qubits = 4
    shots = 1
    pg = PlayGround(n_qubits, shots, hamiltonain)
    # Create a bell state circuit.
    qc = RealAmplitudes(n_qubits, reps=2)
    qc.measure_all()
    num_params = len(qc.ordered_parameters)
    init_param = np.pi * np.random.random((num_params,)) - np.pi / 2
    vary_param = [0, 1]
    pg.plot_energy_landscape(qc, init_param, vary_param, [-np.pi / 4, np.pi / 4],
                             [-np.pi / 4, np.pi / 4],10)


if __name__ == "__main__":

    main()
