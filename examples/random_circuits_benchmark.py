import numpy as np
import matplotlib.pyplot as plt
from fake_devices import Grid5x5, FakeGuadalupe
from algorithm.random_cnot_circuit import RandomCNOTCircuitsGenerator
from algorithm.mapping_sampler import DifferentialMappingSearch
from mindquantum.device import SABRE
from tqdm import tqdm
from sys import stdout
from IPython.display import display_svg
from mindquantum.io.display import draw_topology


circuits_dir = "circuits/"
circuits_file_name = "circuits_20_g80_150_q8_16.joblib"

# device = Grid5x5()
device = FakeGuadalupe()

# test parameters
repeat = 10
no_extra = False

# seed everything
np.random.seed(42)

# first check if circuits are dumped to file.
circuits_dataset = RandomCNOTCircuitsGenerator(80, 150, 8, 16, 20)
try:
    circuits = circuits_dataset.load(circuits_file_name)
except:
    circuits = circuits_dataset.generate()
    circuits_dataset.dump(circuits_file_name)
# circuits[0].svg()

# then run Differential Mapping Search on all test circuits.
m_name = 'Differential Mapping Search'
n_added_gates = np.zeros((len(circuits), repeat), dtype=int)
n_qubits = 25 if device.name == 'grid5x5' else 16
for circ in tqdm(circuits):
    n_lobits = circ.n_qubits
    solver = DifferentialMappingSearch(circ, device.topology, n_qubits, n_lobits, 
                                       no_extra, 0.5, 0.3, 0.2)
    for i in range(repeat):
        new_circ, _, _ = solver.solve(n_sample=4, n_iter=20, sabre_iter=1, lr=0.1)
        n_added_gates[circuits.index(circ), i] = len(new_circ) - len(circ)
n_added_gates_mean = np.mean(n_added_gates, axis=1)
stdout.flush()

print(f"{m_name} mean added gates: {n_added_gates_mean}")
print(f"{m_name} total mean added gates: {np.mean(n_added_gates_mean)}")
stdout.flush()

