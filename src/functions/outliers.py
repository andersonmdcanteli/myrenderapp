### ------ IMPORTS ------ ###

# --- Third part --- #
import numpy as np



### ------ Constants ------ ###

DIXON_TABLE = {
    "n_rep" : (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30),
    0.10 : (0.941, 0.765, 0.642, 0.560, 0.507, 0.554, 0.512, 0.477, 0.576, 0.546, 0.521, 0.546, 0.525, 0.507, 0.490, 0.475, 0.462, 0.450, 0.440, 0.430, 0.421, 0.413, 0.406, 0.399, 0.393, 0.387, 0.381, 0.376),
    0.05 : (0.970, 0.829, 0.710, 0.625, 0.568, 0.615, 0.570, 0.534, 0.625, 0.592, 0.565, 0.590, 0.568, 0.548, 0.531, 0.516, 0.503, 0.491, 0.480, 0.470, 0.461, 0.452, 0.445, 0.438, 0.432, 0.426, 0.419, 0.414),
    0.01 : (0.994, 0.926, 0.821, 0.740, 0.680, 0.725, 0.677, 0.639, 0.713, 0.675, 0.649, 0.674, 0.647, 0.624, 0.605, 0.589, 0.575, 0.562, 0.551, 0.541, 0.532, 0.524, 0.516, 0.508, 0.501, 0.495, 0.489, 0.483)
}


GRUBBS_ONE_TABLE = {
    "n_rep" : (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30),
    0.10 : (1.153, 1.463, 1.672, 1.822, 1.938, 2.032, 2.110, 2.176, 2.234, 2.285, 2.331, 2.371, 2.409, 2.443, 2.475, 2.504, 2.532, 2.557, 2.580, 2.603, 2.624, 2.644, 2.663, 2.681, 2.698, 2.714, 2.730, 2.745),
    0.05 : (1.155, 1.481, 1.715, 1.887, 2.020, 2.126, 2.215, 2.290, 2.355, 2.412, 2.462, 2.507, 2.549, 2.585, 2.620, 2.651, 2.681, 2.709, 2.733, 2.758, 2.781, 2.802, 2.822, 2.841, 2.859, 2.876, 2.893, 2.908),
    0.01 : (1.155, 1.496, 1.764, 1.973, 2.139, 2.274, 2.387, 2.482, 2.564, 2.636, 2.699, 2.755, 2.806, 2.852, 2.894, 2.932, 2.968, 3.001, 3.031, 3.060, 3.087, 3.112, 3.135, 3.157, 3.178, 3.199, 3.218, 3.236),
}



### ------ Functions ------ ###

def _check_outermost_observation(x):
    lower = np.mean(x) - np.min(x)
    upper = np.max(x) - np.mean(x)
    if upper >= lower:
        return "max"
    else:
        return "min"

# --- For Dixon test --- #

def _check_dixon_division_by_zero(x_exp, position_1, position_2):
    if x_exp[position_1] == x_exp[position_2]:
        return True
    else:
        return False

def _r10(x_exp, which):
    if which == "min":
        statistic = (x_exp[1] - x_exp[0])/(x_exp[-1]-x_exp[0])
    else:
        statistic = (x_exp[-1] - x_exp[-2])/(x_exp[-1]-x_exp[0])
    return statistic

def _r11(x_exp, which):
    if which == "min":
        statistic = (x_exp[1] - x_exp[0])/(x_exp[-2]-x_exp[0])
    else:
        statistic = (x_exp[-1] - x_exp[-2])/(x_exp[-1]-x_exp[1])
    return statistic

def _r21(x_exp, which):
    if which == "min":
        statistic = (x_exp[2] - x_exp[0])/(x_exp[-2]-x_exp[0])
    else:
        statistic = (x_exp[-1] - x_exp[-3])/(x_exp[-1]-x_exp[1])
    return statistic

def _r22(x_exp, which):
    if which == "min":
        statistic = (x_exp[2] - x_exp[0])/(x_exp[-3]-x_exp[0])
    else:
        statistic = (x_exp[-1] - x_exp[-3])/(x_exp[-1]-x_exp[2])
    return statistic

def dixon(x_exp, alfa):
    which = _check_outermost_observation(x_exp)
    n_rep = x_exp.size
    critical = DIXON_TABLE[alfa][n_rep-3]

    x_exp = np.sort(x_exp, kind='quicksort')

    if which == "min":
        outlier = x_exp[0]
        if 3 <= n_rep <= 7:
            erro = _check_dixon_division_by_zero(x_exp, -1, 0)
            if erro:
                statistic = None
            else:
                statistic = _r10(x_exp, which="min")
        elif 8 <= n_rep <= 10:
            erro = _check_dixon_division_by_zero(x_exp, -2, 0)
            if erro:
                statistic = None
            else:
                statistic = _r11(x_exp, which="min")
        elif 11 <= n_rep <= 13:
            erro = _check_dixon_division_by_zero(x_exp, -2, 0)
            if erro:
                statistic = None
            else:
                statistic = _r21(x_exp, which="min")
        else:
            erro = _check_dixon_division_by_zero(x_exp, -3, 0)
            if erro:
                statistic = None
            else:
                statistic = _r22(x_exp, which="min")

    else:
        outlier = x_exp[-1]
        if 3 <= n_rep <= 7:
            erro = _check_dixon_division_by_zero(x_exp, -1, 0)
            if erro:
                statistic = None
            else:
                statistic = _r10(x_exp, which="max")
        elif 8 <= n_rep <= 10:
            erro = _check_dixon_division_by_zero(x_exp, -1, 1)
            if erro:
                statistic = None
            else:
                statistic = _r11(x_exp, which="max")
        elif 11 <= n_rep <= 13:
            erro = _check_dixon_division_by_zero(x_exp, -1, 1)
            if erro:
                statistic = None
            else:
                statistic = _r21(x_exp, which="max")

        else:
            erro = _check_dixon_division_by_zero(x_exp, -1, 2)
            if erro:
                statistic = None
            else:
                statistic = _r22(x_exp, which="max")

    return statistic, critical, outlier, erro



# --- For Grubbs test --- #


def _one(x_exp, which):
    if which == "min":
        statistic = (np.mean(x_exp) - x_exp[0])/np.std(x_exp, ddof=1)
    else:
        statistic = (x_exp[-1] - np.mean(x_exp))/np.std(x_exp, ddof=1)
    return statistic

def grubbs(x_exp, alfa):
    n_rep = x_exp.size
    critical = GRUBBS_ONE_TABLE[alfa][n_rep-3]

    which = _check_outermost_observation(x_exp)
    x_exp = np.sort(x_exp, kind='quicksort')

    if which == "min":
        outlier = x_exp[0]
    else:
        outlier = x_exp[-1]


    if np.std(x_exp, ddof=1) < 10e-9:
        return None, critical, outlier, True
    else:
        statistic = _one(x_exp, which)
        return statistic, critical, outlier, False




















#
