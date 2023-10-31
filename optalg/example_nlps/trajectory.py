# Problem
# with trajectory
# SOS: last point, regularization
# N obstacles


# Linear Least Squares
## Ax = b

# Quadratic Function
##


import numpy as np
import math

try:
    from ..optalg.interface.nlp import NLP
    from ..optalg.interface.objective_type import OT

except BaseException:
    from optalg.interface.nlp import NLP
    from optalg.interface.objective_type import OT

# lets just use the  some waypoints at N/2 and N/3


class PointTrajOpt(NLP):
    def __init__(self, N):
        """
        Arguments:
        ---
            N: integer

        N is num of points, dimension is 2.
        x = [ x(1) , y(1) , x(2), y(2) , ... ]
        sos cost on velocity:

        we use a waypoint formulation.
        """
        self.N = N
        self.obs = [np.array([1.3, 1.3])]
        self.goal = np.array([2, 2])

        self.waypoints = [np.array([1., 0.]), np.array([2, 1])]
        self.refs = [int(1 * self.N / 3), int(2 * self.N / 3)]

        self.pg = 2
        self.po = 1

    def evaluate(self, x):

        n = self.getDimension()

        Jgoal = np.zeros((2, n))
        ygoal = x[-2:] - self.goal
        Jgoal[:, -2:] = np.identity(2)

        yvel = np.zeros(n)
        Jvel = np.zeros((n, n))

        for i in range(0, self.N):
            # x
            if i == 0:
                # x
                yvel[2 * i] = x[2 * i]
                Jvel[2 * i, 2 * i] = 1
                # y
                yvel[2 * i + 1] = x[2 * i + 1]
                Jvel[2 * i + 1, 2 * i + 1] = 1
            else:
                yvel[2 * i] = x[2 * i] - x[2 * (i - 1)]
                Jvel[2 * i, 2 * i] = 1
                Jvel[2 * i, 2 * (i - 1)] = -1
                # y
                yvel[2 * i + 1] = x[2 * i + 1] - x[2 * (i - 1) + 1]
                Jvel[2 * i + 1, 2 * i + 1] = 1
                Jvel[2 * i + 1, 2 * (i - 1) + 1] = -1

        num_obs = len(self.obs)
        c = 0
        Jc = np.zeros(n)
        for i in range(self.N):
            p = x[2 * i:2 * i + 2]
            for j in range(num_obs):
                dd = p - self.obs[j]
                d2 = dd @ dd
                dnorm = np.sqrt(d2)
                c += 1.0 / dnorm
                Jc[2 * i:2 * i + 2] += - dd / (dnorm**3)

        cway = 0
        gway = np.zeros(n)

        for ref, waypoint in zip(self.refs, self.waypoints):
            # for i in range(self.N):
            sl = slice(2 * ref, 2 * ref + 2)
            xx = x[sl]
            cway += math.exp((xx - waypoint) @ (xx - waypoint))
            gway[sl] += 1 * math.exp((xx - waypoint) @
                                     (xx - waypoint)) * 2 * (xx - waypoint)

        Jout = np.vstack((self.po * gway, Jvel, self.pg * Jgoal))
        yout = np.concatenate(
            (np.array([self.po * cway]), yvel, self.pg * ygoal))
        return yout, Jout

    def getFeatureTypes(self):
        """
        See Also
        ------
        MathematicalProgram.getFeatureTypes
        """
        return [OT.f] + (self.getDimension() + 2) * [OT.r]

    def getDimension(self):
        return 2 * self.N

    def getFHessian(self, x):

        n = self.getDimension()
        H = np.zeros((n, n))

        for ref, waypoint in zip(self.refs, self.waypoints):
            sl = slice(2 * ref, 2 * ref + 2)

            xx = x[sl]

            ee = math.exp((xx - waypoint) @ (xx - waypoint))
            H[sl, sl] += ee * 2 * np.identity(2)
            H[sl, sl] += ee * 4 * \
                (xx - waypoint).reshape(2, -1) @ (xx - waypoint).reshape(-2, 2)
        return self.po * H

    def getInitializationSample(self):

        xx = np.linspace(0, 2, self.N)
        yy = np.linspace(0, 2, self.N)
        out = np.zeros(2 * self.N)
        for i in range(self.N):
            out[2 * i] = xx[i]
            out[2 * i + 1] = yy[i]

        return out

    def report(self, x):
        """
        displays semantic information on the last query

        Parameters
        ----

        Returns
        ----
        output: string
        """
        # draw with matplotlib
        assert(len(x) % 2 == 0)
        n = int(len(x) / 2)
        xs = [x[2 * i] for i in range(n)]
        ys = [x[2 * i + 1] for i in range(n)]

        import matplotlib.pyplot as plt
        import numpy as np

        # plt.style.use('_mpl-gallery')
        # make data

        # plot
        fig, ax = plt.subplots()

        ax.plot(xs, ys, '--bo')
        way_x = [i[0] for i in self.waypoints]
        way_y = [i[1] for i in self.waypoints]
        ax.plot(way_x, way_y, 'r*')
        ax.plot([self.goal[0]], [self.goal[1]], 'k*')

        # ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
        #        ylim=(0, 8), yticks=np.arange(1, 8))

        plt.show()
