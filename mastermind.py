"""
Mastermind solver

author: Pierre Thibault
date: March 2015
"""

import numpy as np
import time
import sets

def log0(x):
    return np.log2(x) if x != 0 else 1


class MMplayer(object):

    def __init__(self, Np, clist):
        """
        Mastermind player.
        Np: number of pegs
        clist: list of "colors" (characters)
        """
        self.clist = clist
        self.Nc = len(clist)
        self.Np = Np

        self.initialize()

    def initialize(self, seed=None):
        """
        Get ready for a new game. Pick a random sequence of colors. 
        """
        self.Nmoves = 0
        self.solution = np.random.randint(self.Nc, size=(self.Np,))

    def new_move(self, move):
        """
        Receive a new move and return the results.
        """
        move = np.asarray(move)
        m = 0
        for i,j in zip(self.solution, move):
            m += (i==j)

        S = 0
        for c in range(self.Nc):
            ss = sum(self.solution == c)
            sm = sum(move == c)
            S += abs(ss - sm)
        c = self.Np - S/2 - m

        return (m,c)


class MMsolver(object):

    def __init__(self, Np, clist, max_templates=None, max_time=60):
        """
        Initialize with Np pegs and the list of colors clist.
        """
        self.clist = clist
        self.Nc = len(clist)
        self.Np = Np
        self.N = self.Nc**Np

        self.max_templates = max_templates
        self.max_time = max_time

        self.initialize()

    def initialize(self):
        """
        Preparare everything for a new game.
        """

        if self.Nc*self.Np*self.N > 1e9:
            raise RuntimeError('System too large. Sorry.')

        # Create all states
        allstates = np.array(zip(*np.unravel_index(range(self.N), self.Np*(self.Nc,))))

        # Initialise possible solutions
        self.solutions = allstates.copy()
        self.allstates = allstates

        # List of moves
        self.moves = []

    def choose_move(self):
        """
        Compute statistics and return a move.
        """

        if not self.moves:
            move = range(self.Np)
            self.moves.append(move)
            return move

        # Randomly pick templates from all available states
        solidx = np.random.permutation(self.N)

        # Stopping criterion
        if (self.max_templates is not None) and (self.max_templates < N):
            solidx = solidx[:self.max_templates]

        best_entropy = 0.
        t0 = time.time()
        for i in solidx:
            template = self.allstates[i]
            entropy = self._entropy(template)
            if entropy > best_entropy:
                # Better candidate
                best_template = template
                best_entropy = entropy 
                print '%s (%f)' % (str(template), entropy)

            if self.max_time is not None:
                if time.time() - t0 > self.max_time:
                    print 'time out'
                    break

        self.moves.append(best_template)
        return best_template

    def new_result(self, mc, move=None):
        """
        Receive result, eliminate possible solutions.
        """

        if move is None:
            move = self.moves[-1]

        (allm, allc) = self._mc(self.solutions, move)
        pmc = allm*self.Np + allc
        self.solutions = self.solutions[pmc == mc[0]*self.Np + mc[1]]

    @property
    def solved(self):
        return len(self.solutions) == 1

    def _mc(self, states, template):

        template = np.asarray(template)
        m = (states == template).sum(axis=-1)

        cstates = np.array([(states == c).sum(axis=-1) for c in range(self.Nc) ]).T
        ctemplate = np.array([ (template == c).sum() for c in range(self.Nc) ])
        S = abs(cstates - ctemplate).sum(axis=-1)
        c = self.Np - S/2 - m

        return (m,c)

    def _entropy(self, template, ensemble=None):
        """
        Given an ensemble of possible configuration,
        compute the entropy given the template.
        """

        if ensemble is None:
            ensemble = self.solutions

        N = len(ensemble)

        # Initialize the outcome dict
        possible_outcomes = [(mi,ci) for mi in range(self.Np+1) for ci in range(self.Np-mi+1)]
        outcomes = dict((p,0) for p in possible_outcomes)

        (allm, allc) = self._mc(ensemble, template)
        for (m,c) in zip(allm, allc):
           outcomes[(m,c)] += 1

        # Compute entropy
        entropy = sum(ni * log0(ni) for ni in outcomes.values()) / N - log0(N)

        return -entropy


if __name__ == "__main__":
    import sys
    #Np = int(sys.argv[1])
    #clist = [x for x in sys.argv[2]]
    Np = 4
    clist = 'abcdef'
    mmplayer = MMplayer(Np, clist)
    mmsolver = MMsolver(Np, clist)

    print 'Solution is %s' % str(mmplayer.solution)

    t0 = time.time()
    while not mmsolver.solved:
        move = mmsolver.choose_move()
        result = mmplayer.new_move(move)
        mmsolver.new_result(result)
        print result, len(mmsolver.solutions)
    print 'Solution is %s' % str(mmsolver.solutions[0])
    print 'Solved in %d turns and %f seconds' % (len(mmsolver.moves), time.time() - t0)

