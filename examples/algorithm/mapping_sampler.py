import numpy as np
from itertools import product
from mindquantum.device import LinearQubits, GridQubits
from mindquantum.core.circuit import Circuit
from mindquantum.device import SABRE


def softmax(x: np.array):
    # return softmaxed x
    xr = x.copy().flatten()
    xrsum = np.sum(np.exp(xr))
    for j in range(len(xr)):
        xr[j] = np.exp(xr[j])/xrsum
    return np.resize(xr, x.shape)


class BaysianPermuteSampler():
    def __init__(self, n_sel, n_total, p_array) -> None:
        assert p_array.shape == (n_sel, n_total)
        self.p_array = p_array
        self.p_array /= np.sum(p_array)
        self.n_sel = n_sel
        self.n_total = n_total
        self.sample_list = list(range(n_total))

    def set_prob_array(self, p_array):
        self.p_array = p_array
        self.p_array /= np.sum(p_array)

    def plain_sample(self, size=1):
        new_sample = np.zeros(self.n_sel)
        # ALWAYS FROM THE FIRST PHY QUBIT
        sample_sequence = list(range(self.n_sel))
        # new_probvec /= np.sum(new_probvec)
        # new_probvec = softmax(new_probvec)
        for i in range(len(self.p_array)):
            self.p_array[i] = softmax(self.p_array[i])
        # sample first digit within the first place
        new_probvec = self.p_array[sample_sequence[0]]
        new_sample[sample_sequence[0]] = np.random.choice(a=self.sample_list, p=new_probvec)
        next = 1
        while next < self.n_sel:
            avaliable_indices = []
            for a in self.sample_list:
                if a not in new_sample[sample_sequence[:next]]:
                    avaliable_indices.append(a)
            new_probvec = self.p_array[sample_sequence[next], avaliable_indices]
            new_probvec /= np.sum(new_probvec)
            new_sample[sample_sequence[next]] = np.random.choice(a=avaliable_indices, p=new_probvec)
            next += 1
        return new_sample

    def softmax_plain_sample(self, size=1):
        new_sample = np.zeros(self.n_sel)
        # ALWAYS FROM THE FIRST PHY QUBIT
        sample_sequence = list(range(self.n_sel))
        # sample first digit within the first place
        new_probvec = self.p_array[sample_sequence[0]]
        # new_probvec /= np.sum(new_probvec)
        new_probvec = softmax(new_probvec)
        # print(new_probvec)
        new_sample[sample_sequence[0]] = np.random.choice(a=self.sample_list, p=new_probvec)
        next = 1
        while next < self.n_sel:
            avaliable_indices = []
            for a in self.sample_list:
                if a not in new_sample[sample_sequence[:next]]:
                    avaliable_indices.append(a)
            new_probvec = self.p_array[sample_sequence[next], avaliable_indices]
            # new_probvec /= np.sum(new_probvec)
            new_probvec = softmax(new_probvec)
            # print(new_probvec)
            new_sample[sample_sequence[next]] = np.random.choice(a=avaliable_indices, p=new_probvec)
            next += 1
        return new_sample

    def sample(self, size=1):
        new_sample = np.zeros(self.n_sel)
        # RANDOMLY DECIDE WHERE TO SAMPLE FIRST
        sample_sequence = np.random.permutation(list(range(self.n_sel)))
        # sample_sequence = list(range(self.n_sel))
        # sample first digit within the first place
        new_probvec = self.p_array[sample_sequence[0]]
        new_probvec /= np.sum(new_probvec)
        new_sample[sample_sequence[0]] = np.random.choice(a=self.sample_list, p=new_probvec)
        next = 1
        while next < self.n_sel:
            avaliable_indices = []
            for a in self.sample_list:
                if a not in new_sample[sample_sequence[:next]]:
                    avaliable_indices.append(a)
            new_probvec = self.p_array[sample_sequence[next], avaliable_indices]
            new_probvec /= np.sum(new_probvec)
            new_sample[sample_sequence[next]] = np.random.choice(a=avaliable_indices, p=new_probvec)
            next += 1
        return new_sample

    def softmax_uniform_sample(self, size=1):
        samples = np.zeros((size, self.n_sel), dtype=int)
        for ii in range(size):
            new_sample = np.zeros(self.n_sel)
            sample_space = list(product(range(self.n_sel), range(self.n_total)))
            new_probvec = self.p_array.flatten()
            new_probvec = softmax(new_probvec)
            smp = np.random.choice(a=range(len(sample_space)), p=new_probvec)
            smp = sample_space[smp]
            new_sample[smp[0]] = smp[1]
            sel_row = [smp[0]]
            sel_col = [smp[1]]
            while len(sel_row) < self.n_sel:
                avaliable_rows = []
                avaliable_cols = []
                for i in range(self.n_sel):
                    if i not in sel_row:
                        avaliable_rows.append(i)
                for i in range(self.n_total):
                    if i not in sel_col:
                        avaliable_cols.append(i)
                new_probvec = self.p_array.copy()
                new_probvec = np.delete(new_probvec, sel_row, axis=0)
                new_probvec = np.delete(new_probvec, sel_col, axis=1)
                new_probvec = new_probvec.flatten()
                # new_probvec /= np.sum(new_probvec)
                new_probvec = softmax(new_probvec)
                sample_space = list(product(avaliable_rows, avaliable_cols))
                smp = np.random.choice(a=range(len(sample_space)), p=new_probvec)
                smp = sample_space[smp]
                new_sample[smp[0]] = smp[1]
                sel_row.append(smp[0])
                sel_col.append(smp[1])
            samples[ii, :] = new_sample
        return samples

    def uniform_sample(self, size=1):
        samples = np.zeros((size, self.n_sel), dtype=int)
        for ii in range(size):
            new_sample = np.zeros(self.n_sel)
            sample_space = list(product(range(self.n_sel), range(self.n_total)))
            new_probvec = self.p_array.flatten()
            new_probvec /= np.sum(new_probvec)
            smp = np.random.choice(a=range(len(sample_space)), p=new_probvec)
            smp = sample_space[smp]
            new_sample[smp[0]] = smp[1]
            sel_row = [smp[0]]
            sel_col = [smp[1]]
            while len(sel_row) < self.n_sel:
                avaliable_rows = []
                avaliable_cols = []
                for i in range(self.n_sel):
                    if i not in sel_row:
                        avaliable_rows.append(i)
                for i in range(self.n_total):
                    if i not in sel_col:
                        avaliable_cols.append(i)
                new_probvec = self.p_array.copy()
                new_probvec = np.delete(new_probvec, sel_row, axis=0)
                new_probvec = np.delete(new_probvec, sel_col, axis=1)
                new_probvec = new_probvec.flatten()
                new_probvec /= np.sum(new_probvec)
                sample_space = list(product(avaliable_rows, avaliable_cols))
                smp = np.random.choice(a=range(len(sample_space)), p=new_probvec)
                smp = sample_space[smp]
                new_sample[smp[0]] = smp[1]
                sel_row.append(smp[0])
                sel_col.append(smp[1])
            samples[ii, :] = new_sample
        return samples

    def derive_sample(self):
        """
        derive a sample from the current permutation array.
        first select the [i,j] with biggest p_array[i,j],
        and then remove row i and column j from the permutation array.
        then select the [i,j] with biggest p_array[i,j] among the remaining.
        """
        new_sample = np.zeros(self.n_sel, dtype=int)
        sample_space = list(product(range(self.n_sel), range(self.n_total)))
        new_probvec = self.p_array.flatten()
        smp = np.argmax(new_probvec)
        smp = sample_space[smp]
        new_sample[smp[0]] = int(smp[1])
        sel_row = [smp[0]]
        sel_col = [smp[1]]
        while len(sel_row) < self.n_sel:
            avaliable_rows = []
            avaliable_cols = []
            for i in range(self.n_sel):
                if i not in sel_row:
                    avaliable_rows.append(i)
            for i in range(self.n_total):
                if i not in sel_col:
                    avaliable_cols.append(i)
            new_probvec = self.p_array.copy()
            new_probvec = np.delete(new_probvec, sel_row, axis=0)
            new_probvec = np.delete(new_probvec, sel_col, axis=1)
            new_probvec = new_probvec.flatten()
            sample_space = list(product(avaliable_rows, avaliable_cols))
            smp = np.argmax(a=range(len(sample_space)))
            smp = sample_space[smp]
            new_sample[smp[0]] = int(smp[1])
            sel_row.append(smp[0])
            sel_col.append(smp[1])
        return new_sample

    def hist(self, samples, normalize=False):
        histogram = np.zeros_like(self.p_array)
        for smp in samples:
            for i, d in enumerate(smp):
                histogram[i, int(d)] += 1
        if normalize:
            histogram /= np.sum(histogram)
        return histogram


def score_perm_mat(p_array, scores, samples=100):
    '''
    for reference only.
    '''
    sampler = BaysianPermuteSampler(5, 7, p_array)
    grad = np.ones((5,7)) * 1e-5
    score = 0
    for _ in range(samples):
        cscore = 0
        perm = sampler.uniform_sample(1)
        # kgrad = 0
        for i, j in enumerate(perm):
            score -= scores[i,int(j)]
            cscore -= scores[i,int(j)]
        for i, j in enumerate(perm):
            grad[i,int(j)] += cscore*(-p_array[i,int(j)])
            
        # for i in range(len(grad)):
        #     for j in range(len(grad[0])):
        #         grad[i,j] += score*(-p_array[i,j] + 1e-5)
    return score / samples, grad / samples


class DifferentialMappingSearch():
    """
    search for initial layout by gradient descent.
    maintains an hash table of visited layouts.
    it should be noted that the gradient is calculated by sampling,
    and the sampling process is not deterministic.
    
    should be initialized with a circuit and a GridQubits object.
    """
    def __init__(self, circ: Circuit, grid_qubit: GridQubits,
                 n_qubits: int, n_lobits: int, no_extra=False,
                 sabre_w=0.5, sabre_delta1=0.3, sabre_delta2=0.2) -> None:
        self.circ = circ
        self.grid_qubit = grid_qubit
        self.sabre = SABRE(self.circ, self.grid_qubit)
        self.hash_table = {}
        self.n_qubits = n_qubits
        self.n_lobits = n_lobits
        self.p_array = None
        self.sabre_w = sabre_w
        self.no_extra = no_extra
        self.n_gates = len(circ)
        self.sabre_delta1 = sabre_delta1
        self.sabre_delta2 = sabre_delta2
    
    def _hash_layout(self, layout):
        """
        hash for layout.
        it should be unique because the layout is a permutation.
        hash = i[0]+i[1]*self.n_qubits+i[2]*self.n_qubits**2+...
        """
        layout_hash = 0
        for i, j in enumerate(layout):
            layout_hash += j * self.n_lobits**i
        return layout_hash

    def score_layout(self, layout, sabre_iter=1):
        """
        score a layout.
        run sabre on the layout and return the added gate number.
        """
        # print(layout)
        new_circ, _, _ = self.sabre.solve_init(layout, sabre_iter, w=self.sabre_w, 
                                         delta1=self.sabre_delta1, 
                                         delta2=self.sabre_delta2, 
                                         no_extra=self.no_extra)
        score = len(new_circ) - self.n_gates
        return score
    
    def sample_batch(self, n_sample, sabre_iter=1):
        """
        sample a batch of layouts and score them.
        if layout is in hash table, find its score in hash table.
        if layout is not in hash table, add it to hash table.
        """
        sampler = BaysianPermuteSampler(self.n_lobits, self.n_qubits, self.p_array)
        samples = sampler.uniform_sample(n_sample)
        cscore = 0
        cgrad = np.ones((self.n_lobits, self.n_qubits)) * 1e-5
        for sample in samples:
            hash = self._hash_layout(sample)
            if hash in self.hash_table:
                score = self.hash_table[hash]
            else:
                score = self.score_layout(sample, sabre_iter)
                self.hash_table[hash] = score
            # now calculate gradient
            for i, j in enumerate(sample):
                cgrad[i,j] -= score*(self.p_array[i,j])
            cscore += score
        return cscore / n_sample, cgrad / n_sample

    def solve(self, n_sample: int, n_iter=100, sabre_iter=1, lr=1e-2,
              solver_sabre_iter=1):
        """
        the solver.
        """
        # initialize the permutation array
        self.p_array = np.ones((self.n_lobits, self.n_qubits))
        self.p_array /= np.sum(self.p_array)
        ref_delta = np.ones((self.n_lobits, self.n_qubits))*1e-5
        # start iteration
        for s in range(n_iter):
            # sample a batch of layouts
            score, grad = self.sample_batch(n_sample, sabre_iter)
            # update permutation array
            self.p_array += lr * grad
            self.p_array = np.maximum(self.p_array, ref_delta)
        # return the best layout
        deriver = BaysianPermuteSampler(self.n_lobits, self.n_qubits, self.p_array)
        pi = deriver.derive_sample()
        return self.sabre.solve_init(pi, solver_sabre_iter, w=self.sabre_w, 
                                         delta1=self.sabre_delta1, 
                                         delta2=self.sabre_delta2, 
                                         no_extra=self.no_extra)

