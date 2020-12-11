from random import random, randint
import matplotlib.pyplot as plt
from scipy.special import binom, betainc
from collections import Counter
import scipy.stats as STATS


def isInt(n):
    return int(n) == float(n)


def getNegativeBinomPDF(z, m, p):
    if not (isInt(z) and z >= 0):
        raise Exception("Error. Incorrect value for Z argument")
    if not (isInt(m) and m > 0):
        raise Exception("Error. Incorrect value for M argument")
    if not (0 < p < 1):
        raise Exception("Error. Incorrect value for P argument")
    return binom((z + m - 1), z) * (p ** m) * ((1 - p) ** z)


def getNegativeBinomCDF(z, m, p):
    return betainc(m, z + 1, p)


def simulateExperiment(m, p):
    success, failure = 0, 0
    while (success < m):
        if (random() < p):
            success += 1
        else:
            failure += 1
    return failure


def getSample(n):
    sample = []
    for i in range(n):
        sample.append(simulateExperiment(PARAM_M, PARAM_P))
    return sample


def getEDF(sample):
    values, left_bounds, right_bounds = [], [], []

    firstString = "%2.2f :      x <= %2.0f"
    otherString = "%2.2f : %2.0f < x <= %2.0f"
    lastString = "%2.2f : %2.0f > x"

    sample.sort()

    left_bounds.append(0)
    right_bounds.append(sample[0])
    values.append(0)
    print(firstString % (0, sample[0]))

    l, r = 0, 0
    count = 0

    while r + 1 <= len(sample):
        if (sample[l] == sample[r]):
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


def getCDF(rangeX):
    values, left, right = [], [], []

    for x in rangeX:
        left.append(x)
        right.append(x + 1)
        values.append(getNegativeBinomCDF(x, PARAM_M, PARAM_P))
    return values, left, right


# Квантили
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
    return STATS.nbinom.ppf(level, PARAM_M, PARAM_P)


def getRandomColour():
    return "#%06x" % randint(0, 0xFFFFFF)


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


def drawEDFgraph(n, quantity):
    max_elem = 0
    for i in range(1, quantity + 1):
        sample = getSample(n)
        v, l, r = getEDF(sample)
        max_elem = max(max_elem, sample[len(sample) - 1])
        plt.hlines(v, l, r, color=(getRandomColour()), label="Sample " + str(i), alpha=0.6)
    max_elem += 3

    cdf_v, cdf_l, cdf_r = getCDF(range(0, max_elem))
    plt.hlines(cdf_v, cdf_l, cdf_r, color=(0, 0, 0), label="CDF")
    plt.xlim(0, min(80, max_elem))

    plt.legend()
    plt.grid()
    plt.savefig("NegativeBinom EDF and CDF (N=%d)" % n)
    plt.clf()


def drawPolygonGraph(n, quantity):
    for i in range(1, quantity + 1):
        sample = getSample(n)
        sample.sort()
        counted_sample = Counter(sample)
        set_sample = list(set(sample))
        set_sample.sort()

        frequency = []
        for x in set_sample:
            fr = counted_sample[x] / len(sample)
            frequency.append(fr)
        plt.plot(set_sample, frequency, label="Sample " + str(i), alpha=0.6)
    plt.legend()
    plt.grid()

    ProbabilityDensityFunction = []
    for x in range(100):
        ProbabilityDensityFunction.append(getNegativeBinomPDF(x, PARAM_M, PARAM_P))
    plt.plot(range(100), ProbabilityDensityFunction)
    plt.savefig("NegativeBinom Frequency polygons (N=%d)" % n)
    plt.clf()


def drawHistogrGraph(n, quantity):
    for i in range(1, quantity + 1):
        plt.hist(getSample(n), int(100 / 5), label="Sample " + str(i), alpha=0.6)
    plt.legend()
    plt.grid()
    plt.savefig("NegativeBinom Frequency Histograms (N=%d)" % n)
    plt.clf()


# Вывод данных для отчёта
numbers = [5, 10, 100]

PARAM_M = 4
PARAM_P = 0.2
SAMPLE_SIZE = 5

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

    print("Разность: ", upper_limit_difference(N, 5))

    # 2.3
    sample = getSample(N)
    print("Sample:          ", sample)
    sample.sort()
    print("Variation row: ", sample)

    levels = [0.1, 0.5, 0.7]
    for level in levels:
        sample = getSample(N)
        print("Квантиль уровня %f:" % level, getQuantile(sample, level),
              "Теоретический квантиль уровня %f:" % level, getTheoreticalQuantile(level))

    # 2.4
    drawPolygonGraph(N, SAMPLE_SIZE)
    drawHistogrGraph(N, SAMPLE_SIZE)
