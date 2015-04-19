"""
Mastermind solver

author: Pierre Thibault
date: March 2015
"""

import numpy as np
import time
import multiprocessing as mp
import itertools

class Dummy(object):
    pass
myglobals = Dummy()  # Container for shared arrays


# Useful function
def log0(x):
    return np.log2(x) if x != 0 else 1


class EntropyWorker(mp.Process):
    def __init__(self, rank, size, queue, max_templates, max_time):
        """
        Process worker that computes entropies
        """
        super(EntropyWorker, self).__init__()
        self.rank = rank
        self.size = size
        self.q = queue
        self.max_templates = max_templates
        self.max_time = max_time

    def run(self):
        """
        Compute as many entropies as possible in the given time.
        """
        N = myglobals.N
        t0 = time.time()

        # Loop through states that belong to this process
        prog0 = 0.
        Ntot = myglobals.Nc**myglobals.Np
        for k, i in enumerate(myglobals.solidx[self.rank::self.size]):
            prog = np.floor((100.*k)/Ntot)
            if prog != prog0:
                print '(%d) %3d %%' % (self.rank, (prog))
            prog0 = prog

            # Get template
            template = myglobals.allstates[i]

            # Compute matches
            m = (myglobals.solutions == template).sum(axis=-1)

            # Compute color matches
            ctemplate = np.array([(template == c).sum() for c in range(myglobals.Nc)])
            S = abs(myglobals.csol - ctemplate).sum(axis=-1)
            c = myglobals.Np - S/2 - m

            # Combine them for binning
            outcomes = np.bincount(m*myglobals.Np + c)

            # Compute entropy
            entropy = sum(ni * log0(ni) for ni in outcomes) / N - log0(N)

            # Put the result in the queue
            self.q.put((-entropy, i))

            # Exit if some conditions are met.
            if (self.max_templates is not None) and (k>= self.max_templates):
                MMsolver._announce('Max number of templates (%d) reached' % self.max_templates)
                break
            if ((self.max_time is not None) and (time.time() - t0 > self.max_time)):
                MMsolver._announce('Time out after %d templates' % k)
                break
        return


class MMplayer(object):

    def __init__(self, Np, Nc):
        """
        Mastermind player.
        Np: number of pegs
        Nc: number of colours
        """
        self.Nc = Nc
        self.Np = Np

        self.initialize()

    def initialize(self, solution=None):
        """
        Get ready for a new game. Pick a random sequence of colors. 
        """
        self.Nmoves = 0
        if solution is None:
            self.solution = np.random.randint(self.Nc, size=(self.Np,))
        else:
            self.solution = np.array(solution)

    def new_move(self, move):
        """
        Receive a new move and return the results.
        """
        move = np.asarray(move)
        m = 0
        for i, j in zip(self.solution, move):
            m += (i == j)

        S = 0
        for c in range(self.Nc):
            ss = sum(self.solution == c)
            sm = sum(move == c)
            S += abs(ss - sm)
        c = self.Np - S/2 - m

        self._announce('Result: %d good, %d wrong place' % (m,c))
        return m, c

    @staticmethod
    def _announce(message):
        print ('Player >>>>> %40s' % message)


class MMsolver(object):

    def __init__(self, Np, Nc, mp=True, max_templates=None, max_time=60):
        """
        Initialize with Np pegs and Nc colours.
        """
        self.Nc = Nc
        self.Np = Np
        self.N = self.Nc**Np

        self.max_templates = max_templates
        self.max_time = max_time

        self.mp = mp

        self.initialize()

    def initialize(self, start=None, compute_entropy=False):
        """
        Prepare everything for a new game.
        """

        if self.Nc*self.Np*self.N > 1e9:
            raise RuntimeError('System too large. Sorry.')

        # Create all states
        allstates = np.array(zip(*np.unravel_index(range(self.N), self.Np*(self.Nc,))))

        # Initialise possible solutions
        self.solutions = allstates.copy()
        self.allstates = allstates

        self.start = start
        self.compute_entropy = compute_entropy

        # List of moves
        self.moves = []

    def choose_move(self):
        """
        Compute statistics and return a move.
        """

        # Update the color info
        self.csol = np.array([(self.solutions == c).sum(axis=-1) for c in range(self.Nc)]).T

        # Special case for the first move.
        if not self.moves:
            if self.start is not None:
                move = np.array(self.start)
            else:
                move = np.arange(self.Np)
            self.moves.append(move)
            if self.compute_entropy:
                entropy = self.entropy(move)
            else:
                entropy = np.nan
            self._announce('Move : %s (entropy: %f bits)' % (str(move), entropy))
            return move, entropy

        # Special case for the solution
        if len(self.solutions) == 1:
            move = self.solutions[0]
            self.moves.append(move)
            entropy = 0. # self._entropy(move)
            self._announce('Solution : %s (entropy: %f bits)' % (str(move), entropy))
            return move, entropy


        # Randomly pick templates from all available states
        solidx = np.random.permutation(self.N)

        # Compute maximum possible entropy.
        max_ent = min(log0(len(self.solutions)), log0(self.Nc+1)+log0(self.Nc+2)-1)

        if self.mp:

            # Put all useful arrays in the globals dict before forking
            myglobals.allstates = self.allstates
            myglobals.solutions = self.solutions
            myglobals.csol = self.csol
            myglobals.solidx = solidx
            myglobals.N = len(self.solutions)
            myglobals.Nc = self.Nc
            myglobals.Np = self.Np

            # create queues
            q = mp.Queue()

            # create processes
            numworkers = 4
            processes = [EntropyWorker(rank=i, size=numworkers, queue=q, max_templates=self.max_templates, max_time=self.max_time) for i in range(numworkers)]

            # Run the processes and join
            for p in processes:
                p.start()

            best_entropy = 0.
            # Get the results from the queue.
            go = True
            while go:
                if not q.empty():
                    r = q.get()
                    if r[0] > best_entropy:
                        # Better candidate
                        best_template = self.allstates[r[1]]
                        best_entropy = r[0]
                        self._announce('%s (%f bits)' % (str(best_template), best_entropy))
                else:
                    go = False
                    for p in processes:
                        go |= p.is_alive()

            for p in processes:
                p.join()

        else:
            best_entropy = 0.
            t0 = time.time()
            can_exit = False
            prog0 = 0.
            for k, i in enumerate(solidx):
                template = self.allstates[i]
                entropy = self.entropy(template)
                prog = np.floor((1000.*k)/self.N)
                if prog != prog0:
                    print '%3.2f %%' % (prog/10.)
                prog0 = prog

                if entropy > best_entropy:
                    # Better candidate
                    best_template = template
                    best_entropy = entropy
                    can_exit = True

                if can_exit:
                    if (self.max_templates is not None) and (k >= self.max_templates):
                        self._announce('Max number of templates (%d) reached' % self.max_templates)
                        break
                    if ((self.max_time is not None) and (time.time() - t0 > self.max_time)):
                        self._announce('Time out after %d templates' % k)
                        break
                    if abs(best_entropy - max_ent) < 1e-4:
                        self._announce('Reached max entropy.')
                        break

        self.moves.append(best_template)
        self._announce('Move : %s (entropy: %f bits)' % (str(best_template), best_entropy))
        return best_template, best_entropy

    def new_result(self, mc, move=None):
        """
        Receive result, reevaluate possible solutions.
        """
        if move is None:
            move = self.moves[-1]
        else:
            self.moves[-1] = move
            self.csol = np.array([(self.solutions == c).sum(axis=-1) for c in range(self.Nc) ]).T
            self._announce('Entropy : %f' % self.entropy(move))

        (allm, allc) = self._mc(move)
        pmc = allm*self.Np + allc
        Nbefore = len(self.solutions)
        self.solutions = self.solutions[pmc == mc[0]*self.Np + mc[1]].copy()
        self._announce('Just learned %f bits!' % (log0(Nbefore)-log0(len(self.solutions))))
        self._announce('%d remaining possible solutions' % len(self.solutions))

    @property
    def solved(self):
        return len(self.solutions) == 1

    @staticmethod
    def _announce(message):
        print 'Solver >>>>> ' + ('%40s' % message)

    def _mc(self, template):

        template = np.asarray(template)
        m = (self.solutions == template).sum(axis=-1)
        ctemplate = np.array([(template == c).sum() for c in range(self.Nc)])
        S = abs(self.csol - ctemplate).sum(axis=-1)
        c = self.Np - S/2 - m

        return m, c

    def entropy(self, template):
        """
        Compute the entropy of the possible solutions conditional to the template.
        """

        N = len(self.solutions)

        (allm, allc) = self._mc(template)
        outcomes = np.bincount(allm*self.Np + allc)

        # Compute entropy
        entropy = sum(ni * log0(ni) for ni in outcomes) / N - log0(N)

        return -entropy


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        Ntries = 1
    else:
        Ntries = int(sys.argv[1])
    Np, Nc = 6, 10
    mmplayer = MMplayer(Np, Nc)
    mmsolver = MMsolver(Np, Nc, max_time=1)

    tlist = []
    nlist = []
    for i in range(Ntries):
        print 'New problem'
        mmplayer.initialize()
        mmsolver.initialize()

        t0 = time.time()
        while not mmsolver.solved:
            move, entropy = mmsolver.choose_move()
            result = mmplayer.new_move(move)
            mmsolver.new_result(result)
        tlist.append(time.time() - t0)
        nlist.append(len(mmsolver.moves))
        print 'Solution is %s (check: %s)' % (str(mmsolver.solutions[0]), str(mmplayer.solution))
        print 'Solved in %d turns and %f seconds' % (nlist[-1], tlist[-1])

    print 'Avg time: %f' % np.mean(tlist)
    print 'Avg number of moves: %f' % np.mean(nlist)
