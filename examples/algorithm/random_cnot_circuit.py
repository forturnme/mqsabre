from mindquantum import Circuit, X
from joblib import dump, load
import numpy as np


class RandomCNOTCircuitsGenerator():
    """
    Generate random CNOT circuits.
    with at most max_gates CNOT gates and at least min_gates CNOT gates.
    and with at most max_qubits qubits and at least min_qubits qubits.
    it can dump the circuits to file and load it from file.
    """
    def __init__(self, min_gates, max_gates, min_qubits, max_qubits, n_circuits):
        self.min_gates = min_gates
        self.max_gates = max_gates
        self.min_qubits = min_qubits
        self.max_qubits = max_qubits
        self.n_circuits = n_circuits
        self.circuits = None

    def generate(self):
        """
        generate num random CNOT circuits.
        """
        num = self.n_circuits
        self.circuits = None
        circuits = []
        for i in range(num):
            num_gates = np.random.randint(self.min_gates, self.max_gates + 1)
            num_qubits = np.random.randint(self.min_qubits, self.max_qubits + 1)
            circuit = Circuit()
            for j in range(num_gates):
                control = np.random.randint(0, num_qubits)
                target = np.random.randint(0, num_qubits)
                while control == target:
                    target = np.random.randint(0, num_qubits)
                circuit += X.on(control, target)
            circuits.append(circuit)
        self.circuits = circuits
        return circuits

    def dump(self, filename):
        """
        dump circuits to file.
        """
        dump(self.circuits, "circuits/"+filename)

    def load(self, filename):
        """
        load circuits from file.
        """
        circuits = load("circuits/"+filename)
        self.circuits = circuits
        return circuits
    
