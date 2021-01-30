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
from qiskit import transpile,execute
from qiskit.result import Result
from qiskit.providers.aer import QasmSimulator
from qiskit.test.mock import FakeMelbourne

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




class PlayGround:

    def __init__(self, vals, weights, max_weight, shots=8192):

        self.provider = IonQProvider(token='laldTW63uAIPEfNaJwfljDU6OT3p7bKr')
        # Get an IonQ simulator backend to run circuits on:
        self.backend = self.provider.get_backend("ionq_simulator")
        # self.backend = QasmSimulator.from_backend(FakeMelbourne())
        self.ham = ks.get_operator(vals, weights, max_weight)[0]
        self.shift = ks.get_operator(vals, weights, max_weight)[1]

        self.shots = shots
        self.n_qubits = self.ham.num_qubits

        self.ansatz = RealAmplitudes(self.n_qubits, reps=2)
        self.num_params = self.ansatz.num_parameters

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

    def get_result(self, qcs, shots=1000):

        # Then run the circuit:
        # pdb.set_trace()

        # job = execute(qcs,self.backend)
        # result = job.result()
        # rd = result.to_dict()
        # for k,v in rd.items():
        #     print(k)
        #     print(v)
        #     if k == "results":
        #         for kk,vv in 
        # pdb.set_trace()
        
        full_rd = None # full result dictionary
        for aqc in qcs:
            t_aqc = transpile(aqc,self.backend)
            job = self.backend.run(t_aqc,shots=shots)
            # save job_id
            # job_id_bell = job.job_id()

            # Fetch the result:
            result = job.result()
            rd = result.to_dict()
            if full_rd is None:
                full_rd = rd
            else:
                full_rd['results'].append(rd['results'])

        pdb.set_trace()
        fin_result = Result().from_dict(full_rd)
        return result

    def compute_energy(self, cur_param):
        
        qc_binded = update_params(self.ansatz, cur_param)

        qc_cur = QuantumCircuit(self.n_qubits)
        qc_cur.compose(qc_binded, inplace=True)

        wpauli_circuits = self.ham.construct_evaluation_circuit(
            qc_cur, False)

        result = self.get_result(wpauli_circuits, shots=self.shots)

        e = self.ham.evaluate_with_result(result, False)

        return e

    def plot_energy_landscape(self,
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

        for i, theta0 in enumerate(theta0r):
            for j, theta1 in enumerate(theta1r):

                cur_param = prev_param
                cur_param[which_params[0]] = theta0
                cur_param[which_params[1]] = theta1

                e = self.compute_energy(cur_param)
                
                pdb.set_trace()
                # e = self.energy(result)
                # print(e)
                # pdb.set_trace()

                energies[i][j] = e
                
        X, Y = np.meshgrid(theta0r, theta1r)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        
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


    def ask_move(self, num_params):
        # asks which direction and magnitude you want to move
        print("Anon bid me how thee wanteth to moveth in the phase space.")
        print("Chooseth thy grise wisely!")
        print("Lacking valor moves shall beest did punish")

        changes = np.zeros((num_params,))

        delta = np.pi/100
        for i in range(num_params):
            j = input('How many steps for parameter {}?'.format(i))
            changes[i] = j * delta

            return changes




    def ask_show_param(self, num_params):

        # asks which parameter to show the plot
        print("Which two parameters would you like to see for the next energy landscape plot")

        var1, var2 = input("Enter two numbers between 0 and {1} here: ".format(num_params)).split()

        return int(var1), int(var2)




    def punish_player(self, cur_energy):

        # if you get to a higher energy, give random kick to parameters

        if self.prev_energy < cur_energy:
            print("Bad move, a strong wind blow you to somewhere else in phase space")
            punish = np.random.random((self.num_params,)) * np.pi/100
            return punish
        else:
            print("Well done, your current energy is {}".format(cur_energy))
            print("You maybe one step closer to the ground state energy {}".format(self.gs_energy))
            self.prev_energy = cur_energy
            return np.zeros((self.num_params,))
        
    def do_move(self, new_params, punish=False):

        # actually change the update parameters
        
        e = self.compute_energy(new_params)

        if punish:
            self.punish_player(e)
        else:
            self.prev_energy = e


### loop in the main function until player either rage quits or reach the place


def main():

    vals = [1, 2, 3]
    weights = [2, 4, 5]
    max_weight = 8
    # hamiltonain = {(0, 1): 0.1, (1, 2): 0.3, (0, 2): -.5}
    # n_qubits = 4
    # shots = 1
    pg = PlayGround(vals, weights, max_weight)

    init_param = np.pi * np.random.random((pg.num_params,)) - np.pi / 2
    vary_param = [0, 1]
    pg.plot_energy_landscape(init_param, vary_param, [-np.pi / 10, np.pi / 10],
                             [-np.pi / 10, np.pi / 10], 2)


if __name__ == "__main__":

    main()
