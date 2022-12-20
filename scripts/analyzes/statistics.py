from scipy import stats
import scikit_posthocs as sp
import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter


def compute_critical_difference(avg_ranks, N, alpha="0.05", type="nemenyi"):
    """ From https://gist.github.com/pavlin-policar/c071d09baef45b7d310778d2414255fc.
    Reference: DEMŠAR, Janez. Statistical comparisons of classifiers over multiple data sets.
    The Journal of Machine learning research, 2006, vol. 7, p. 1-30.

    Returns critical difference for Nemenyi or Bonferroni-Dunn test
    according to given alpha (either alpha="0.05" or alpha="0.1") for average
    ranks and number of tested data sets N. Type can be either "nemenyi" for
    for Nemenyi two tailed test or "bonferroni-dunn" for Bonferroni-Dunn test.

    Parameters
    ----------
    avg_ranks: np.ndarray
        Should contain the average rank of each method. So if we have methods
        1, 2, 3 and 4, there should be 4 entries in avranks.
    N: int
        The number of tested data sets.
    """

    k = len(avg_ranks)

    d = {
        ("nemenyi", "0.05"): [0, 0, 1.959964, 2.343701, 2.569032, 2.727774,
                              2.849705, 2.94832, 3.030879, 3.101730, 3.163684,
                              3.218654, 3.268004, 3.312739, 3.353618, 3.39123,
                              3.426041, 3.458425, 3.488685, 3.517073, 3.543799],
        ("nemenyi", "0.1"): [0, 0, 1.644854, 2.052293, 2.291341, 2.459516,
                             2.588521, 2.692732, 2.779884, 2.854606, 2.919889,
                             2.977768, 3.029694, 3.076733, 3.119693, 3.159199,
                             3.195743, 3.229723, 3.261461, 3.291224, 3.319233],
        ("bonferroni-dunn", "0.05"): [0, 0, 1.960, 2.241, 2.394, 2.498, 2.576,
                                      2.638, 2.690, 2.724, 2.773],
        ("bonferroni-dunn", "0.1"): [0, 0, 1.645, 1.960, 2.128, 2.241, 2.326,
                                     2.394, 2.450, 2.498, 2.539],
    }

    q = d[(type, alpha)]

    cd = q[k] * (k * (k + 1) / (6.0 * N)) ** 0.5

    return cd


def r_stat(df, m, n, pvalue, title):
    # Defining the R script and loading the instance in Python
    r = ro.r
    r['source']('Nemenyi.R')
    # Loading the function we have defined in R.
    r_function = ro.globalenv['nemenyi']

    output_directory = "../../results/statistics"

    importr('base')
    with localconverter(ro.default_converter + pandas2ri.converter):
        #Invoking the R function and getting the result
        df_result_r = r_function(df, m, pvalue, output_directory, title)

        #Converting it back to a pandas dataframe.
        pd_from_r_df = ro.conversion.rpy2py(df_result_r)
        print("Result: ", pd_from_r_df)

        cd = compute_critical_difference(pd_from_r_df, n, alpha="0.05", type="nemenyi")
        print('cd = ', cd)


def r_launcher():

    # Reading and processing data
    colNames = ["learners", "tasks", "data"]
    #tasks = ["res1", "res2", "res3", "res4", "res5", "res6", "res7", "res8", "res9", "res10"]

    #vals1 = [44, 56, 53, 46, 53, 46, 42, 47, 46, 45]
    #vals2 = [35, 46, 38, 47, 37, 38, 44, 46, 44, 35]
    #vals3 = [32, 42, 54, 43, 32, 32, 43, 36, 8, 29]

    tasks = ["res1", "res2", "res3", "res4"]
    n = 4
    vals1 = [2, 2, 2, 2]
    vals2 = [3, 3, 3, 3]
    vals3 = [1, 1, 1, 1]

    a1 = np.array([np.repeat("data_group1", n), tasks, vals1]).T
    a2 = np.array([np.repeat("data_group2", n), tasks, vals2]).T
    a3 = np.array([np.repeat("data_group3", n), tasks, vals3]).T

    joined = np.concatenate((a1, a2, a3))
    df = pd.DataFrame(joined, columns=colNames)

    title = "test"

    r_stat(df, 'data', n, 0.05, title)


def python_stat():
    # Data groups
    data_group1 = [1, 1, 0, 1]
    data_group2 = [0, 0, 0, 0]
    data_group3 = [0, 0, 0, 0]


    # Conduct the Friedman Test
    friedman, pvalue = stats.friedmanchisquare(data_group1, data_group2, data_group3)

    print("friedman:")
    print(friedman)

    print("pvalue:")
    print(pvalue)

    # Combine three groups into one array
    data = np.array([data_group1, data_group2, data_group3])

    print(data)

    # Conduct the Nemenyi post-hoc test
    nemenyi = sp.posthoc_nemenyi_friedman(data.T)

    print("nemenyi:")
    print(nemenyi)

if __name__ == '__main__':
    r_launcher()