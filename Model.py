from math import comb
import numpy as np
from scipy.optimize import minimize


# Generic single parameter model. Provide p_2, p_1, p_0 as functions of a parameter x. Use the fit method to predict x
# from experimental data
class Model:
    def __init__(self, multiplexing=1, detection_probability=1, ploidy=1):
        """ Instantiate Model object
        :param multiplexing: The number of NPs per sequenced 'tube'
        :param detection_probability: The probability that a sectioned bead is successfully detected (or 'sequenced')
        :param ploidy: The number of copies of each indistinguishable homologous locus
        """
        self.ploidy = ploidy
        self.multiplexing = multiplexing
        if detection_probability > 1 or detection_probability < 0:
            raise ValueError("detection_probability must be between zero and one")
        self.detection_probability = detection_probability

    def fit(self, m, p, initial_guess, cost=None, tol=1e-100, method=None):
        """
        :param m: Experimentally measured values for m_0, m_1, m_2
        :param p: A function returning a tuple of values for p_0, p_1, p_2.

        :return: The optimal parameters for the functions p such that the cost function is minimized.
        """

        if cost is None:
            cost = Model.default_cost

        return minimize(lambda params: cost(self.predict(p(*params)), m),
                        initial_guess, tol=tol, method=method)

    def predict(self, p):
        """ Given a set of p's, predict the m's. That is, turn the probabilities associated with each individual
        slice into numbers that reflect the experimental data. I will write an actual description at some point"""
        self.p_0 = p[0]
        self.p_1 = p[1]
        self.p_2 = p[2]

        return self.multiplex()

    def multiplex(self):
        m_0 = self.detect(0, 0) ** self.multiplexing
        m_1 = 2 * ((sum([self.detect(i, 0) for i in range(0, self.ploidy + 1)]) ** self.multiplexing) - m_0)
        m_2 = 1 - m_0 - m_1

        return m_0, m_1, m_2

    def detect(self, alpha, beta):
        def A(i, j):
            if self.ploidy == 1:
                return 1
            if self.ploidy == 2:
                return ((alpha == 1) * (i == 2) + 1) * ((beta == 1) * (j == 2) + 1)

        return sum([
            sum([
                self.collapse_homologs(i, j) * ((1 - self.detection_probability) ** (i + j - alpha - beta))
                * A(i, j)
                for j in range(beta, self.ploidy + 1)
            ])
            for i in range(alpha, self.ploidy + 1)
        ]) * (self.detection_probability ** (alpha + beta))

    def collapse_homologs(self, alpha, beta):
        def A(i, j):
            if (i + j <= self.ploidy) and (2 * i + j == alpha + beta) and (beta >= i) and (alpha >= i):
                return ((self.p_2 ** i) * (self.p_1 ** j) * (self.p_0 ** (self.ploidy - i - j)) * comb(self.ploidy, i) *
                        comb(self.ploidy - i, j))
            return 0

        return sum([
            sum([
                A(i, j)
                for j in range(0, self.ploidy + 1)
            ])
            for i in range(0, self.ploidy + 1)
        ])

    def __str__(self):
        return ("multiplexing: " + str(self.multiplexing) + "\ndetection_probability: " + str(
            self.detection_probability)
                + "\nploidy: " + str(self.ploidy))

    @staticmethod
    def default_cost(predicted_m, m):
        return np.sum(np.power(np.array([*m]) - np.array([*predicted_m]), 2))

    @staticmethod
    def co_cost(predicted_m, m):
        return ((predicted_m[2] / (predicted_m[1] + predicted_m[2])) - (m[2] / (m[1] + m[2]))) ** 2


class StaticModel(Model):
    def __init__(self, slice_range, slice_width, multiplexing=1, detection_probability=1, ploidy=1):
        super(StaticModel, self).__init__(multiplexing=multiplexing, detection_probability=detection_probability,
                                          ploidy=ploidy)
        self.slice_range = slice_range
        self.slice_width = slice_width

    def p(self, distance):
        return (StaticModel.p_0(distance, self.slice_range, self.slice_width),
                StaticModel.p_1(distance, self.slice_range, self.slice_width),
                StaticModel.p_2(distance, self.slice_range, self.slice_width))

    def fit(self, m, initial_guess, cost=None, tol=1e-100, method=None):
        return super().fit(m, self.p, initial_guess, cost, tol, method)

    def __str__(self):
        return ("multiplexing: " + str(self.multiplexing) + "\ndetection_probability: " + str(
            self.detection_probability)
                + "\nploidy: " + str(self.ploidy) + "\nslice_range: " + str(self.slice_range) + "\nslice_width: "
                + str(self.slice_width))

    @staticmethod
    def p_0(D, l, h):
        if D < h:
            return 1 - (2 * h + D) / (2 * l)
        else:
            return 1 + ((h ** 2) / (2 * l * D)) - 2 * h / l

    @staticmethod
    def p_1(D, l, h):
        if D < h:
            return D / (2 * l)
        else:
            return h / l - (h ** 2) / (2 * D * l)

    @staticmethod
    def p_2(D, l, h):
        return (2 * h * min(D, h) - min(D, h) ** 2) / (2 * D * l)


class SLICE(Model):
    def __init__(self, u, t, multiplexing=1, detection_probability=1, ploidy=1):
        super(SLICE, self).__init__(multiplexing=multiplexing, detection_probability=detection_probability, ploidy=ploidy)
        self.u = u
        self.t = t

    # def fit(self):
    #     

    def predict(self, pi):
        self.pi = pi
        return self.multiplex()

    def collapse_homologs(self, alpha, beta):
        # c is a three tuple representing the probability that 0, 1 or 2 loci of an AB couple are segregated in a NP
        c = tuple([self.pi * self.t[i] + (1 - self.pi) * self.u[i]
                   for i in range(len(self.u))])
        if self.ploidy == 1:
            return c[alpha + beta]
        # Formulas for the diploid case taken from SI section page 10 (eqs 10)
        elif self.ploidy == 2:
            v_0 = c[0] + c[1]
            if alpha == 0 and beta == 0:
                return c[0] ** 2
            elif alpha == 1 and beta == 1:
                return 2 * ((1 - 2 * v_0 + c[0]) * c[0] + (v_0 - c[0]) ** 2)
            elif alpha == 2 and beta == 2:
                return (1 - 2 * v_0 + c[0]) ** 2
            elif (alpha == 1 and beta == 0) or (alpha == 0 and beta == 1):
                return 2 * (v_0 - c[0]) * c[0]
            elif (alpha == 2 and beta == 0) or (alpha == 0 and beta == 2):
                return (v_0 - c[0]) ** 2
            elif (alpha == 2 and beta == 1) or (alpha == 1 and beta == 2):
                return 2 * (1 - 2 * v_0 + c[0]) * (v_0 - c[0])
            else:
                raise ValueError("Invalid values of alpha and beta for this ploidy")
        else:
            raise ValueError("SLICE model not configured to handle this ploidy yet")
