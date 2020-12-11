from random import random, randint
from math import sqrt, exp, log, tan, pi, e
import matplotlib.pyplot as plt
import numpy
from scipy.special import gamma, gammainc
import scipy.stats as STATS

numbers = [5, 10, 100, 1000, 100000]
PARAM_I = 1
PARAM_A = 9
SAMPLE_SIZE = 5


def gammaGC(k, alpha):
    while True:
        b = k - 1
        A = k + b
        s = sqrt(A)
        x = -1
        t = 0
        while x < 0:
            u = random()
            t = s * tan(pi * (u - 0.5))
            x = b + t
        u = random()
        check = exp(b * log(x / b) - t + log(1 + t * t / A))
        if u <= check:
            return x * 1 / alpha


def gammaGS(k, alpha):
    m1 = 0.36787944117144232159
    b = 1.0 + m1 * k
    while True:
        u = random()
        p = b * u
        if p > 1.0:
            x = -log((b - p) / k)
            u = random()
            if u <= x ** (k - 1):
                return x * 1 / alpha
        else:
            x = p ** (1 / k)
            u = random()
            if u <= e ** (-x):
                return x * 1 / alpha


def isInt(n):
    return int(n) == float(n)


def getRandom():
    k = random()
    if k == 0:
        return 1
    else:
        return k


def getGammaPDF(x, alpha, rate):
    return ((rate ** alpha) / gamma(alpha)) * (x ** (alpha - 1)) * (e ** (-(rate * x)))


def simulateExperiment(k, alpha):
    # return rgamma(k,alpha)
    return gammaGS(k, alpha) if k <= 1.0 else gammaGC(k, alpha)


def getSample(n):
    sample = []
    for i in range(n):
        sample.append(simulateExperiment(PARAM_A, PARAM_I))
    return sample


def getEDF(sample):
    values, left_bounds, right_bounds = [], [], []

    firstString = "%3.5f :          x <= %3.5f"
    otherString = "%3.5f : %3.5f < x <= %3.5f"
    lastString = "%3.5f : %3.5f > x"

    sample.sort()

    left_bounds.append(0)
    right_bounds.append(sample[0])
    values.append(0)
    print(firstString % (0, sample[0]))

    l, r = 0, 0
    count = 0

    while r + 1 <= len(sample):
        if sample[l] == sample[r]:
            r = r + 1
            count = count + 1
        else:
            left_bounds.append(sample[l])
            right_bounds.append(sample[r])
            values.append(count / len(sample))
            print(otherString % (count / len(sample), sample[l], sample[r]))
            l = r

    left_bounds.append(sample[l])
    right_bounds.append(sample[len(sample) - 1] + 3)
    values.append(1)
    print(lastString % (1, sample[len(sample) - 1]))

    return values, left_bounds, right_bounds


def getCDF(rangeT):
    x, y = [], []
    for t in rangeT:
        x.append(t)
        y.append(gammainc(PARAM_A, t))
    return x, y


def getRandomColour():
    return "#%06x" % randint(0, 0xFFFFFF)


def drawEDFgraph(n, quantity):
    max_elem = 0
    for i in range(1, quantity + 1):
        sample = getSample(n)
        v, l, r = getEDF(sample)
        max_elem = max(max_elem, sample[len(sample) - 1])
        plt.hlines(v, l, r, color=(getRandomColour()), label="EDF of Sample " + str(i), alpha=0.6)

    cdf_x, cdf_y = getCDF(range(0, 20))
    plt.plot(cdf_x, cdf_y, color=(getRandomColour()), label="CDF", alpha=0.6)

    max_elem += 3

    plt.legend()
    plt.grid()
    plt.savefig("Gamma EDF and CDF (N=%d)" % n)
    plt.clf()


def upper_limit_difference(N, count):
    max_diff = []
    n = []
    for i in range(count):
        n.append(getSample(N))

    for i in range(count):
        for j in range(i + 1, count):
            diff = 0
            for elem in range(N):
                diff = max(diff, abs(n[i][elem] - n[j][elem]))
            max_diff.append(diff)
    return max_diff


def getQuantile(sample, a):
    sample.sort()
    k = int(a * (len(sample) - 1))
    if (k + 1) < (a * len(sample)):
        return sample[k + 1]
    elif (k + 1) == (a * len(sample)):
        return (sample[k] + sample[k + 1]) / 2
    elif (k + 1) > (a * len(sample)):
        return sample[k]


def getTheoreticalQuantile(level):
    return STATS.gamma.ppf(level, PARAM_A)


def drawPolygonGraph(n, quantity):
    for i in range(1, quantity + 1):
        sample = getSample(n)
        sample.sort()
        width = sample[len(sample) - 1] - sample[0]
        width /= len(sample) - 1
        bounds = numpy.arange(0, int(sample[len(sample) - 1] + 2), 1)
        values = numpy.zeros(len(bounds))

        cx = 0
        rf = 0
        while cx < len(sample):
            if sample[cx] < bounds[rf]:
                values[rf] += 1
                cx += 1
            else:
                rf += 1

        frequency = []
        for t in values:
            frequency.append(t / len(sample))

        plt.plot(bounds, frequency, label="Sample " + str(i), alpha=0.6)

    plt.legend()
    plt.grid()

    ProbabilityDensityFunction = []
    for x in range(100):
        ProbabilityDensityFunction.append(getGammaPDF(x, PARAM_A, PARAM_I))

    plt.plot(range(100), ProbabilityDensityFunction)
    plt.savefig("Gamma Frequency polygons (N=%d)" % n)
    plt.clf()


def drawHistogramGraph(n, quantity):
    for i in range(1, quantity + 1):
        plt.hist(getSample(n), int(100 / 5), label="Sample " + str(i), alpha=0.6)

    plt.legend()
    plt.grid()
    plt.savefig("Gamma Frequency Histograms (N=%d)" % n)
    plt.clf()


# Вывод данных для отчёта

def getReport():
    for N in numbers:

        # Negative binomial distribution
        # 2.1
        for i in range(SAMPLE_SIZE):
            print(getSample(N))

        # 2.2
        sample = getSample(N)
        print("Sample: ", sample)
        print("EDF: ")
        drawEDFgraph(N, 5)

        print(upper_limit_difference(N, 5))
        # 2.3
        sample = getSample(N)
        print("Sample:          ", sample)
        sample.sort()
        print("Variation row: ", sample)

        sample = getSample(N)
        levels = [0.1, 0.5, 0.7]
        for level in levels:
            sample = getSample(N)
            print("Квантиль уровня %f:" % level, getQuantile(sample, level),
                  "Теоретический квантиль уровня %f:" % level, getTheoreticalQuantile(level))
        # 2.4
        drawPolygonGraph(N, SAMPLE_SIZE)
        drawHistogramGraph(N, SAMPLE_SIZE)


if __name__ == '__main__':
    getReport()