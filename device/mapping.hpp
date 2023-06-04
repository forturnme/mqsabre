//   Copyright 2023 <Huawei Technologies Co., Ltd> and 2023 Hanlin Long
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.

#ifndef MINDQUANTUM_DEVICE_MAPPING_HPP_
#define MINDQUANTUM_DEVICE_MAPPING_HPP_

#include <list>
#include <memory>
#include <set>
#include <string>
#include <utility>

#include "device/mapping.hpp"
#include "device/topology.hpp"
#include "ops/basic_gate.hpp"
#include "ops/gate_id.hpp"
namespace mindquantum::mapping {
struct Gate {
    std::string type;  // gate's type, such as CNOT or X
    int q1, q2;        // logical qubit number, if
                       // gate is single gate, q2=q1
    std::string tag;   // user defined tag
    Gate(const std::string& type, int q1, int q2, const std::string& tag) : type(type), q1(q1), q2(q2), tag(tag) {
    }
};

std::pair<qbit_t, VT<Gate>> GateToAbstractGate(const VT<std::shared_ptr<BasicGate>>& gates);

/**
 * @brief Get the circuit DAG object
 *
 * @param n number of logical qubits
 * @param gates logical circuit from left to right
 * @return vector<vector<int>> DAG of logical circuit
 */
VT<VT<int>> GetCircuitDAG(int n, const VT<Gate>& gates);

class SABRE {
 private:
    int num_logical;
    int num_physical;
    VT<Gate> gates;  // logical circuit
    VT<VT<int>> G;   // physical coupling graph

    VT<VT<int>> DAG;   // DAG of logical circuit
    VT<VT<int>> RDAG;  // reverse graph of DAG

    VT<VT<int>> D;  // nearest neighbor cost

    VT<int> decay;  // decay of each logical qubit

    double W;       // parameter between F and E
    double delta1;  // decay of a single gate
    double delta2;  // decay of a CNOT gate

    /**
     * @brief judge whether gate[g] can be executable under mapping pi
     *
     * @param pi current mapping from logical qubit to physical qubit
     * @param g id of gate
     * @return true when pi[g.q1] and pi[g.q2] is neighbor in G
     */
    inline bool IsExecutable(const VT<int>& pi, int g) const;

    /**
     * @brief Get the mapping from physical qubit to logical qubit
     *
     * @param pi mapping from logical to physical
     * @return vector<int> mapping from physical to logical
     */
    inline VT<int> GetReversePi(const VT<int>& pi) const;

    /**
     * @brief get the candidate SWAP list when there is no executable gate.
     *   If edge (x,z) in G and gate (x,y) in F, then SWAP (x,z) is possible.
     * @param F current front layer
     * @param pi current mapping from logical to physical
     * @return set<pair<int, int>> set of candidate SWAP list, containing physical id.
     */
    std::set<std::pair<int, int>> ObtainSWAPs(const std::list<int>& F, const VT<int>& pi) const;

    /**
     * @brief Get the next layer of F in DAG, only considering CNOT gates.
     *   Single gates can always be executed, so there is no need to consider.
     * @param F current front layer
     * @param DAG
     * @param indeg current in-degree of DAG
     * @return list<int> the next layer of F, ignoring single gates.
     */
    std::list<int> GetNextLayer(const std::list<int>& F, const VT<VT<int>>& DAG, const VT<int>& indeg) const;

    /**
     * @brief Get the extended set E
     *   There are many ways to generate E. Here we just use the next layer of F.
     * @param F current front layer
     * @param DAG
     * @param indeg current in-degree of DAG
     * @return list<int> extended set E
     */
    std::list<int> GetExtendedSet(const std::list<int>& F, const VT<VT<int>>& DAG, const VT<int>& indeg) const;

    /**
     * @brief basic heuristic function
     *      H = \sum_{g\in F} D[pi[g.q1]][pi[g.q2]]
     * @param F set of gates' id
     * @param pi mapping from logical to physical
     * @return double
     */
    double HBasic(const std::list<int>& F, const VT<int>& pi) const;

    /**
     * @brief heuristic function considering extended set E
     *   H = 1 / |F| * HBasic(F, pi) + W / |E| * HBasic(E, pi)
     * @param F current front layer
     * @param E extended set
     * @param pi mapping from logical to physical
     * @return double
     */
    double HLookAhead(const std::list<int>& F, const std::list<int>& E, const VT<int>& pi) const;

    /**
     * @brief heuristic function considering trade-off between circuit depth and gates number.
     *
     * @param F current front layer
     * @param E extended set
     * @param pi mapping from logical to physical
     * @param SWAP physical SWAP, using logical id
     * @param decay decay of logical qubits
     * @return double
     */
    double H(const std::list<int>& F, const std::list<int>& E, const VT<int>& pi, const std::pair<int, int>& SWAP,
             const VT<double>& decay) const;

 public:
    /**
     * @brief Construct a new SABRE object
     *
     * @param num_logical number of logical qubits
     * @param gates logical circuit from left to right
     * @param num_physical number of physical qubits
     * @param G physical coupling graph
     */
    SABRE(const VT<std::shared_ptr<BasicGate>>& circ, const std::shared_ptr<QubitsTopology>& coupling_graph);

    /**
     * @brief heuristic search algorithm to generate physical circuit
     *
     * @param pi Initial mapping from logical to physical
     * @param DAG
     * @return vector<Gate> physical circuit that can be executed on hardware,
     *      and modified pi that maps logical qubits to physical qubits.
     */
    VT<Gate> HeuristicSearch(VT<int>& pi, const VT<VT<int>>& DAG);

    /**
     * @brief one-turn iterate to update Initial Mapping.
     *
     * @param pi Input initial mapping
     */
    void IterOneTurn(VT<int>& pi);

    /**
     * @brief solve qubit mapping problem
     *
     * @param iter_num iterate times to update random initial mapping
     * @param W parameter to look-ahead
     * @param delta1 decay of single gate
     * @param delta2 decay of CNOT gate, decay of SWAP will be 3*delta2
     * @return pair<vector<Gate>, pair<vector<int>, vector<int>>>
     *      (gs, (pi0, pi1)), gs is generated physical circuit,
     *                        pi0 is initial mapping from logical to physical
     *                        pi1 is final mapping from logical to physical
     */
    std::pair<VT<VT<int>>, std::pair<VT<int>, VT<int>>> Solve(int iter_num, double W, double delta1, double delta2);

    inline void SetParameters(double W, double delta1, double delta2);
};
}  // namespace mindquantum::mapping
#endif
