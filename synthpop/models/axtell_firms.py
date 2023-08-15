from numba import njit
import numpy as np
from ..abstract import AbstractModel

@njit
def compute_output(a2f, firm_number, es, a, b, beta):

    e = es[a2f == firm_number].sum()
    return a * e + b * (e**beta)

@njit
def compute_utility(a2f, firm_number, es, a, b, ei, thi, beta):

    output = compute_output(a2f, firm_number, es, a, b, beta)
    n = (a2f == firm_number).sum()
    utility = (output / n)**(thi) * (1 - ei)**(thi)
    return utility

@njit
def compute_optimal_effort(a2f, firm_number, agent, es, thi, a, b):

    e_i = es[a2f == firm_number].sum() - es[agent]
    if b == 0:
        optimal_effort = thi - (1 - thi) * e_i
    else:
        numerator = - a - 2*b*(e_i - thi) + np.sqrt( a**2 + 4 * a * b * thi**2 * (1 + e_i) + 4 * (b * thi)**2 * (1 + e_i)**2 )
        denom = 2 * b * (1 + thi)
        optimal_effort = numerator / denom
    if optimal_effort > 0:
        return optimal_effort
    else:
        return 0.

#@njit
def _simulate(a2f, es, ths, phs, T, a, b, beta, V, seed=None, verbose=False):

    if not (seed is None):
        np.random.seed(seed)

    ts = [0.]
    ass = [np.inf]
    a2fs = [a2f.copy()]
    ess = [es.copy()]

    # Forward simulate
    t = 0.
    N = a2f.size
    cumul_firms_existed = N
    normed_phs = phs / phs.sum()
    break_point = 0.1
    while t < T:

        # Next event time
        delta_t = np.random.exponential(1. / phs.sum())
        agent = np.argmax(np.random.multinomial(1, normed_phs))
        if t + delta_t > T:
            ts.append(T)
            ass.append(agent)
            a2fs.append(a2f[agent])
            ess.append(es[agent])
            break
        # Corresponding agent
        thi = ths[agent]

        # Agent chooses action
        # Compute utilities from all actions
        stay_effort = compute_optimal_effort(a2f, a2f[agent], agent, es, thi, a, b)
        best_effort = stay_effort
        _es = es
        _es[agent] = stay_effort
        # Compute hypothetical utility of staying at firm at new effort level
        stay_util = compute_utility(a2f, a2f[agent], _es, a, b, stay_effort, thi, beta)
        best_util = stay_util
        # Start new firm
        _a2f = a2f
        _a2f[agent] = cumul_firms_existed + 1
        # Compute hypothetical optimal effort if starts new firm
        new_effort = compute_optimal_effort(_a2f, _a2f[agent], agent, es, thi, a, b)
        _es[agent] = new_effort
        # Compute corresponding utility of optimal effort at new firm
        new_util = compute_utility(_a2f, _a2f[agent], _es, a, b, new_effort, thi, beta)
        new_firm = False
        friend_firm = False
        if new_util > stay_util:
            new_firm = True
            best_effort = new_effort
            best_util = new_util
        # Now check for all friends' firms. Assume agents lie on ring
        for v in range(-V, V+1):
            if v == 0:
                continue
            friend_label = (agent + v) % N
            _a2f[agent] = a2f[friend_label]
            friend_effort = compute_optimal_effort(_a2f, _a2f[agent], agent, es, thi, a, b)
            _es[agent] = friend_effort
            friend_util = compute_utility(_a2f, _a2f[agent], _es, a, b, _es[agent], thi, beta)
            if friend_util > best_util:
                friend_firm = True
                new_firm = False
                best_util = friend_util
                best_effort = friend_effort
        # Choose the one that maximises utility
        if friend_firm:
            a2f[agent] = a2f[friend_label]
        elif new_firm:
            cumul_firms_existed += 1
            a2f[agent] = cumul_firms_existed
        es[agent] = best_effort
        if verbose:
            if (t < break_point) and (t + delta_t > break_point):
                print(t)
                break_point += 0.1
        t += delta_t

        ts.append(t)
        ass.append(agent)
        a2fs.append(a2f[agent])
        ess.append(es[agent])
    return ts, ass, a2fs, ess


class AxtellModel(AbstractModel):

    def __init__(self, max_time=1., v=1, N=1000, ):

        self._max_time = max_time
        self._N = N
        self._v = min([v, int(N / 2)])

    def initialize(self):
        pass
    
    def step(self, *args, **kwargs):
        pass

    def observe(self, x):
        return x

    def run(self, generator):
        es, ths, phs, beta, a, b = generator(self._N)
        a2f = np.arange(self._N)
        return _simulate(a2f, es, ths, phs, self._max_time, a, b, beta, self._v)
        
    def _reconstruct(self, last_a2prop, agent, entry):

        last_a2prop[agent] = entry
        return last_a2prop

    def reconstruct_firms(self, last_a2f, agent, firm_number):

        return self._reconstruct(last_a2f, agent, firm_number)

    def reconstruct_effort(self, last_a2eff, agent, effort):

        return self._reconstruct(last_a2eff, agent, effort)

