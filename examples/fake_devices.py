"""
define fake devices for testing
by mq.device.GridQubits
"""
from mindquantum.device import GridQubits, QubitNode, LinearQubits, QubitsTopology
from mindquantum.io.display import draw_topology


class FakeDevice():
    def __init__(self, n_qubits, name, edges:list) -> None:
        self.n_qubits = n_qubits
        self.name = name
        self.edges = edges
        self.topology = None

    def _build_topology(self):
        """
        build topology for fake device.
        """
        self.topology = QubitsTopology([QubitNode(i, poi_x=i) for i in range(self.n_qubits)])
        for e in self.edges:
            self.topology[e[0]] << self.topology[e[1]]
            self.topology[e[1]] << self.topology[e[0]]
    
    def draw_topology(self):
        draw_topology(self.topology)
        

class Grid5x5(FakeDevice):
    """
    5*5 grid qubits.
    """
    def __init__(self) -> None:
        n_qubits = 25
        name = "grid5x5"
        edges = []
        super().__init__(n_qubits, name, edges)
        self.topology = GridQubits(5, 5)
    

class Linear20(FakeDevice):
    """
    20 linear qubits.
    """
    def __init__(self) -> None:
        n_qubits = 20
        name = "linear20"
        edges = []
        super().__init__(n_qubits, name, edges)
        self.topology = LinearQubits(20)


class FakeGuadalupe(FakeDevice):
    """
    fake guadalupe grid qubits.
    """
    def __init__(self):
        n_qubits = 16
        name = "guadalupe"
        edges = [[0,1],[1,2],[1,4],[2,3],[3,5],[4,7],[5,8],[7,6],
                      [7,10],[8,9],[8,11],[10,12],[11,14],[12,13],
                      [12,15],[14,13]]
        super().__init__(n_qubits, name, edges)
        self._build_topology()
        
