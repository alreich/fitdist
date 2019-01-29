#!/Users/areich/anaconda/bin/python

"""fitdist: Find the continuous probability distribution that best fits a dataset

.. moduleauthor:: Al Reich (al.reich@gmail.com)

"""

import sys
import os
import logging
import csv

import numpy as np
import pandas as pd
import scipy.stats
#import statsmodels.api as sm

# Names of all continuous distributions in scipy.stats
cont_dist_names = (
    'alpha',            # Alpha continuous random variable
    'anglit',           # Anglit continuous random variable
    'arcsine',          # Arcsine continuous random variable
    'beta',             # Beta continuous random variable
    'betaprime',        # Beta prime continuous random variable
    'bradford',         # Bradford continuous random variable
    'burr',             # Burr continuous random variable
    'cauchy',           # Cauchy continuous random variable
    'chi',              # Chi continuous random variable
    'chi2',             # Chi-squared continuous random variable
    'cosine',           # Cosine continuous random variable
    'dgamma',           # Double gamma continuous random variable
    'dweibull',         # Double Weibull continuous random variable
    #'erlang',           # Erlang continuous random variable -------------- Often causes problems (so removed)
    'expon',            # Exponential continuous random variable
    'exponweib',        # Exponentiated Weibull continuous random variable
    'exponpow',         # Exponential power continuous random variable
    'f',                # F continuous random variable
    'fatiguelife',      # Fatigue-life (Birnbaum-Sanders) continuous random variable
    'fisk',             # Fisk continuous random variable
    'foldcauchy',       # Folded Cauchy continuous random variable
    'foldnorm',         # Folded normal continuous random variable
    'frechet_r',        # Frechet right (or Weibull minimum) continuous random variable
    'frechet_l',        # Frechet left (or Weibull maximum) continuous random variable
    'genlogistic',      # Generalized logistic continuous random variable
    'genpareto',        # Generalized Pareto continuous random variable
    'genexpon',         # Generalized exponential continuous random variable
    'genextreme',       # Generalized extreme value continuous random variable
    'gausshyper',       # Gauss hypergeometric continuous random variable
    'gamma',            # Gamma continuous random variable
    'gengamma',         # Generalized gamma continuous random variable
    'genhalflogistic',  # Generalized half-logistic continuous random variable
    'gilbrat',          # Gilbrat continuous random variable
    'gompertz',         # Gompertz (or truncated Gumbel) continuous random variable
    'gumbel_r',         # Right-skewed Gumbel continuous random variable
    'gumbel_l',         # Left-skewed Gumbel continuous random variable
    'halfcauchy',       # Half-Cauchy continuous random variable
    'halflogistic',     # Half-logistic continuous random variable
    'halfnorm',         # Half-normal continuous random variable
    'hypsecant',        # Hyperbolic secant continuous random variable
    'invgamma',         # Inverted gamma continuous random variable
    'invgauss',         # Inverse Gaussian continuous random variable
    'invweibull',       # Inverted Weibull continuous random variable
    'johnsonsb',        # Johnson SB continuous random variable
    'johnsonsu',        # Johnson SU continuous random variable
    'ksone',            # General Kolmogorov-Smirnov one-sided test
    'kstwobign',        # Kolmogorov-Smirnov two-sided test for large N
    'laplace',          # Laplace continuous random variable
    'logistic',         # Logistic (or Sech-squared) continuous random variable
    'loggamma',         # Log gamma continuous random variable
    'loglaplace',       # Log-Laplace continuous random variable
    'lognorm',          # Lognormal continuous random variable
    'lomax',            # Lomax (Pareto of the second kind) continuous random variable
    'maxwell',          # Maxwell continuous random variable
    'mielke',           # Mielke's Beta-Kappa continuous random variable
    'nakagami',         # Nakagami continuous random variable
    'ncx2',             # Non-central chi-squared continuous random variable
    'ncf',              # Non-central F distribution continuous random variable
    'nct',              # Non-central Student's T continuous random variable
    'norm',             # Normal continuous random variable
    'pareto',           # Pareto continuous random variable
    'pearson3',         # Pearson type III continuous random variable
    'powerlaw',         # Power-function continuous random variable
    'powerlognorm',     # Power log-normal continuous random variable
    'powernorm',        # Power normal continuous random variable
    'rdist',            # R-distributed continuous random variable
    'reciprocal',       # Reciprocal continuous random variable
    'rayleigh',         # Rayleigh continuous random variable
    'rice',             # Rice continuous random variable
    'recipinvgauss',    # Reciprocal inverse Gaussian continuous random variable
    'semicircular',     # Semicircular continuous random variable
    't',                # Student's T continuous random variable
    'triang',           # Triangular continuous random variable
    'truncexpon',       # Truncated exponential continuous random variable
    'truncnorm',        # Truncated normal continuous random variable
    'tukeylambda',      # Tukey-Lamdba continuous random variable
    'uniform',          # Uniform continuous random variable
    'vonmises',         # Von Mises continuous random variable
    'wald',             # Wald continuous random variable
    'weibull_min',      # Frechet right (or Weibull minimum) continuous random variable
    'weibull_max',      # Frechet left (or Weibull maximum) continuous random variable
    'wrapcauchy'        # Wrapped Cauchy continuous random variable
)

# A subset of common distribution names in scipy.stats
common_cont_dist_names = (
    'alpha',            # Alpha continuous random variable
    'beta',             # Beta continuous random variable
    'cauchy',           # Cauchy continuous random variable
    'chi2',             # Chi-squared continuous random variable
    'expon',            # Exponential continuous random variable
    'exponweib',        # Exponentiated Weibull continuous random variable
    'f',                # F continuous random variable
    'fatiguelife',      # Fatigue-life (Birnbaum-Sanders) continuous random variable
    'genextreme',       # Generalized extreme value continuous random variable
    'gamma',            # Gamma continuous random variable
    'laplace',          # Laplace continuous random variable
    'lognorm',          # Lognormal continuous random variable
    'norm',             # Normal continuous random variable
    'powerlognorm',     # Power log-normal continuous random variable
    'powernorm',        # Power normal continuous random variable
    't',                # Student's T continuous random variable
    'tukeylambda',      # Tukey-Lamdba continuous random variable
    'uniform'           # Uniform continuous random variable
)


# The basic mechanism for searching through a list of distribution
# names for the one that fits best was inspired by a snippet of code
# posted in the following online Q/A forum question:
# http://stats.stackexchange.com/questions/74434/kolmogorov-smirnov-test-strange-output
def find_best_fit_distributions(data, dist_names=common_cont_dist_names,
                                goodness_of_fit_pvalue=0.10,
                                drop_dist_names=('erlang',),
                                verbose_mode=False,
                                logfilename='bestfitdist.log',
                                ignore_gof=False):
    """Find the probability distributions that best fit the given data.

    :Inputs:

        `data`: The input data in the form of a list or array of floating point numbers.

        `dist_names`: A list of distribution names as found in scipy.stats (defaults to `common_cont_dist_names`).

        `goodness_of_fit_pvalue`: The p-value for passing the Kolmogorov-Smirnov GoF test (default is 0.10).

        `drop_dist_names`: A list of distributions to leave off of considerations (module default is ["erlang"]).

        `verbose_mode`: A boolean that controls whether verbose output will be used (default is False).

        `logfilename`: The name of the logging file (default is "bestfitdist.log").

        `ignore_gof`: If True, the KS GoF test will be ignored and all results will be output (default is False).

    :Outputs:

        A list of best-fit results, sorted on KS GoF p-value ("best" to "worst"), where each result
        is itself a list consisting of three things:
           * distribution name (the string used in `scipy.stats`),
           * tuple of MLE parameter estimates (location & scale, resp., are the last two entries)
           * p-value for the Kolmogorov-Smirnov goodness-of-fit test

    :Distributions:

        The distributions used here are listed below.  Names in **bold** are members of `common_cont_dist_names`.

        **'alpha'**, 'anglit', 'arcsine', **'beta'**, 'betaprime', 'bradford', 'burr', **'cauchy'**, 'chi',
        **'chi2'**, 'cosine', 'dgamma', 'dweibull', 'erlang', **'expon'**, **'exponweib'**, 'exponpow', **'f'**,
        **'fatiguelife'**, 'fisk', 'foldcauchy', 'foldnorm', 'frechet_r', 'frechet_l', 'genlogistic', 'genpareto',
        'genexpon', **'genextreme'**, 'gausshyper', **'gamma'**, 'gengamma', 'genhalflogistic', 'gilbrat',
        'gompertz', 'gumbel_r', 'gumbel_l', 'halfcauchy', 'halflogistic', 'halfnorm', 'hypsecant', 'invgamma',
        'invgauss', 'invweibull', 'johnsonsb', 'johnsonsu', 'ksone', 'kstwobign', **'laplace'**, 'logistic',
        'loggamma', 'loglaplace', **'lognorm'**, 'lomax', 'maxwell', 'mielke', 'nakagami', 'ncx2', 'ncf', 'nct',
        **'norm'**, 'pareto', 'pearson3', 'powerlaw', **'powerlognorm'**, **'powernorm'**, 'rdist', 'reciprocal',
        'rayleigh', 'rice', 'recipinvgauss', 'semicircular', **'t'**, 'triang', 'truncexpon', 'truncnorm',
        **'tukeylambda'**, **'uniform'**, 'vonmises', 'wald', 'weibull_min', 'weibull_max', 'wrapcauchy'

    :Misc:

        Depending on the dataset, some distributions will encounter convergence issues during MLE fitting.
        Running `fitdist` with `verbose_mode` set to **True** will aid in finding which distributions have
        problems. On subsequent runs, the `drop_dist_names` option can be used to remove problem distributions
        from consideration. The 'erlang' distribution has been found to be problematic often, so it is dropped
        by default.
    """
    logging.basicConfig(filename=logfilename, filemode='w', level=logging.INFO)
    dist_names = set(dist_names) - set(drop_dist_names)
    if ignore_gof:
        logging.info('===> Scanning %d distributions, ignoring GoF p-value...',
                     len(dist_names))
    else:
        logging.info('===> Scanning %d distributions, using GoF p-value..., %f',
                     len(dist_names), goodness_of_fit_pvalue)
    if verbose_mode:
        if ignore_gof:
            print("Scanning {} distributions, ignoring GoF p-value...".format(len(dist_names)))
        else:
            print("Scanning {} distributions, using GoF pvalue, %f...".format(len(dist_names), goodness_of_fit_pvalue))
    best_fit_results = []
    for dist_name in dist_names:
        if verbose_mode:
            print(dist_name)
        logging.info('**** Distribution: %s', dist_name)
        dist = getattr(scipy.stats, dist_name)  # Gets a distribution object by name
        parameters = dist.fit(data)  # Computes the MLE of the distribution's parameters
        # Kolmogorov-Smirnov Goodness-of-Fit
        ks_stat, ks_p_value = scipy.stats.kstest(data, dist_name, args=parameters)
        logging.info('       Parameters: %s', str(parameters))
        logging.info('     KS Statistic: %f', ks_stat)
        logging.info('       KS P-Value: %f', ks_p_value)
        if (not np.isnan(ks_p_value)) and (ignore_gof or (ks_p_value >= goodness_of_fit_pvalue)):
            best_fit_results.append([dist_name, parameters, ks_p_value])
            logging.info('                   PASS')
        else:
            logging.info('                   FAIL')
    return sorted(best_fit_results, key=lambda result: result[2], reverse=True)


def make_fitted_pdf(distribution, parameters):
    """Given a scipy.stats distribution and its MLE parameters, return the
    corresponding probability density function (PDF).

    :Inputs:

        `distribution`: Either an **instance** of a scipy.stats continuous distribution
        or a **string** that corresponds to a scipy.stats distribution

        `parameters`: A tuple of MLE parameters for the input distribution in the order
        defined by scipy.stats

    :Outputs:

        A probability density function, f, such that f(x) --> y, where x & y are floating point values.

    :Example:

        >>> import fitdist as fd
        >>> normal_pdf = fd.make_fitted_pdf('norm',(0.0,1.0))
        >>> normal_pdf(1.0)
        >>> 0.24197072451914337

    """
    if type(distribution) == str:
        dist = getattr(scipy.stats, distribution)
    else:
        dist = distribution

    def pdf(x):
        num_parameters = len(parameters)
        if num_parameters > 2:  # Some distributions have 3 or 4 parameters
            return dist.pdf(x, *parameters[:-2], loc=parameters[-2], scale=parameters[-1])
        elif num_parameters == 2:
            return dist.pdf(x, loc=parameters[0], scale=parameters[1])
        elif num_parameters == 1:
            return dist.pdf(x, parameters[0])
        else:
            raise ValueError('There must be at least one parameter.')

    return pdf


def make_fitted_ppf(distribution, parameters):
    """Given a scipy.stats distribution and its MLE parameters, return the
    corresponding point percentage function (PPF) -- inverse of the CDF.

    :Inputs:

        `distribution`: Either an **instance** of a scipy.stats continuous distribution
        or a **string** that corresponds to a scipy.stats distribution

        `parameters`: A tuple of MLE parameters for the input distribution in the order
        defined by scipy.stats

    :Outputs:

        An point percentage function (PPF), G, such that G(x) --> y, where x & y are floating point values.
        Note: The PPF is the inverse Cumulative Distribution Function (CDF)

    :Example:

        >>> import fitdist as fd
        >>> normal_ppf = fd.make_fitted_ppf('norm',(0.0,1.0))
        >>> normal_ppf(0.95)
        >>> 1.6448536269514722

    """
    if type(distribution) == str:
        dist = getattr(scipy.stats, distribution)
    else:
        dist = distribution

    def ppf(x):
        num_parameters = len(parameters)
        if num_parameters > 2:  # Some distributions have 3 or 4 parameters
            return dist.ppf(x, *parameters[:-2], loc=parameters[-2], scale=parameters[-1])
        elif num_parameters == 2:
            return dist.ppf(x, loc=parameters[0], scale=parameters[1])
        elif num_parameters == 1:
            return dist.ppf(x, parameters[0])
        else:
            raise ValueError('There must be at least one parameter.')
    return ppf


def make_fitted_cdf(distribution, parameters):
    """Given a scipy.stats distribution and its MLE parameters, return the
    corresponding cumulative distribution function (CDF).

    :Inputs:

        `distribution`: Either an **instance** of a scipy.stats continuous distribution
        or a **string** that corresponds to a scipy.stats distribution

        `parameters`: A tuple of MLE parameters for the input distribution in the order
        defined by scipy.stats

    :Outputs:

        An cumulative distribution function (CDF), F, such that F(x) --> y, where x & y are floating point values.

    :Example:

        >>> import fitdist as fd
        >>> normal_cdf = fd.make_fitted_cdf('norm',(0.0,1.0))
        >>> normal_cdf(1.645)
        >>> 0.95001509446087862

    """
    if type(distribution) == str:
        dist = getattr(scipy.stats, distribution)
    else:
        dist = distribution

    def cdf(x):
        num_parameters = len(parameters)
        if num_parameters > 2:  # Some distributions have 3 or 4 parameters
            return dist.cdf(x, *parameters[:-2], loc=parameters[-2], scale=parameters[-1])
        elif num_parameters == 2:
            return dist.cdf(x, loc=parameters[0], scale=parameters[1])
        elif num_parameters == 1:
            return dist.cdf(x, parameters[0])
        else:
            raise ValueError('There must be at least one parameter.')

    return cdf


def best_fit_z_scores(best_fit_results, fitted_dist_p_value=0.05,
                      onesided=True, upper_pvalue=True):
    """Takes the output of *find_best_fit_distributions* and returns confidence intervals
    for each entry (i.e., distribution specification) in the results.

    :Inputs:

        `best_fit_results`: The list of results that is output by *find_best_fit_distributions*

        `fitted_dist_p_value`: A probability, p, where 1-p is the size of the
        confidence interval (default is 0.05, i.e., 95% confidence interval).

        `one_sided`: If True, then the output z-scores will correspond to one-sided
        confidence intervals, otherwise the z-scores will correspond to two-sided
        confidence intervals (default is True, i.e., one-sided).

        `upper_pvalue`: If True, then one-sided, *upper* confidence interval z-scores
        are returned, otherwise, a *lower* confidence interval z-score is returned
        (default is True).  Only takes effect when *one-sided* is True.

    :Outputs:

        A list of z-scores.  If one sided is True, the list will consist of floating
        point values, otherwise the list will consist of pairs (tuples) of floating
        point values, where the first float in each pair is the *lower z-score* and
        the second float is the *upper z-score*.

    :Example:

        TBD

    """
    z_scores = []
    if len(best_fit_results) > 0:
        for bf_result in best_fit_results:
            dist_name = bf_result[0]
            parameters = bf_result[1]
            dist = getattr(scipy.stats, dist_name)
            dist_ppf = make_fitted_ppf(dist, parameters)  # for computing z-scores
            if onesided:
                if upper_pvalue:
                    fitted_zscore = dist_ppf(1 - fitted_dist_p_value)
                else:
                    fitted_zscore = dist_ppf(fitted_dist_p_value)
                z_scores.append(fitted_zscore)
            else:  # divide probability evenly between both sides
                lower_zscore = dist_ppf(fitted_dist_p_value / 2.0)
                upper_zscore = dist_ppf(1 - (fitted_dist_p_value / 2.0))
                z_scores.append((lower_zscore, upper_zscore))
    else:
        print("There are no best-fit results.")
    return z_scores


def write_p_value_table(path_name=None, table_name="p_value_table.csv",
                        distribution='norm', mle_parameter_list=(0.0, 1.0),
                        probabilities=(0.50, 0.75, 0.90, 0.95, 0.975, 0.995)):
    """For a given probability distribution, the function writes out a one/two-sided confidence interval table."""
    if type(distribution) == str:
        dist = getattr(scipy.stats, distribution)
    else:
        dist = distribution
    dist_ppf = make_fitted_ppf(dist, mle_parameter_list)
    lower_z_scores = []
    upper_z_scores = []
    for prob in probabilities:
        lower_z_scores.append(dist_ppf((1 - prob) / 2.0))
        upper_z_scores.append(dist_ppf((1 + prob) / 2.0))
    result = pd.DataFrame({'Probability': probabilities,
                           'Z_Score_Lower': lower_z_scores,
                           'Z_Score_Upper': upper_z_scores
                           })
    if path_name:
        dir_name = path_name
    else:
        dir_name = os.getcwd()
    full_path_name = os.path.join(dir_name, table_name)
    result.to_csv(full_path_name, index=False)


# def sunspot_data(describe_sunspot_data=False):
#     """
#     Yearly Sunspot Data -- for test, development, and demonstration purposes.
#     """
#     if describe_sunspot_data:
#         print sm.datasets.sunspots.NOTE
#     sunspots = sm.datasets.sunspots.load_pandas().data
#     # Jan 1 doesn't work below.  Why not?
#     sunspots.index = pd.Index(map(lambda y: datetime.datetime(int(y), 12, 31), sunspots['YEAR']))
#     del sunspots['YEAR']
#     return sunspots


def output_results(fitdist_results, z_scores=None, onesided=True, upper_zscores=False, conf_int_size=0.95,
                   delimiter=',', noheader=False, verbose_mode=True, ignore_gof=False):
    """
    Formats the output of find_best_fit_distributions and resulting confidence intervals into CSV form.

    :Inputs:

        `fitdist_results`: The list of results that is output by *find_best_fit_distributions*

        `z_scores`: The list of results that is output by *best_fit_z_scores* (default is None).

        `one_sided`: If True, then the output z-scores will correspond to one-sided
        confidence intervals, otherwise the z-scores will correspond to two-sided
        confidence intervals (default is True, i.e., one-sided).

        `fitted_dist_p_value`: A probability, p, where 1-p is the size of the
        confidence interval (default is 0.05, i.e., 95% confidence interval).

        `upper_pvalue`: If True, then one-sided, *upper* confidence interval z-scores
        are returned, otherwise, a *lower* confidence interval z-score is returned
        (default is True).  Only takes effect when *one-sided* is True.

        `upper_zscores`: (default is False)

        `conf_int_size`: (default is 0.95)

        `delimiter`: (default is ',')

        `noheader`: (default is False)

        `verbose_mode`: (default is True)

        `ignore_gof`: (default is False)

        `fitted_dist_p_value`: A probability, p, where 1-p is the size of the
        confidence interval (default is 0.05, i.e., 95% confidence interval).

        `one_sided`: If True, then the output z-scores will correspond to one-sided
        confidence intervals, otherwise the z-scores will correspond to two-sided
        confidence intervals (default is True, i.e., one-sided).

        `upper_pvalue`: If True, then one-sided, *upper* confidence interval z-scores
        are returned, otherwise, a *lower* confidence interval z-score is returned
        (default is True).  Only takes effect when *one-sided* is True.

    :Outputs:

        A list of z-scores.  If one sided is True, the list will consist of floating
        point values, otherwise the list will consist of pairs (tuples) of floating
        point values, where the first float in each pair is the *lower z-score* and
        the second float is the *upper z-score*.

    :Example:

        TBD
    """
    if verbose_mode:
        if ignore_gof:
            print("\n{} distributions output.".format(len(fitdist_results)))
        else:
            print("\n{} distributions passed the goodness-of-fit test.".format(len(fitdist_results)))
        print("Here is the best one:")
        print("  Distribution Name: {}\n  MLE Parameter Estimates: {}".format(fitdist_results[0][0], fitdist_results[0][1]))
        print("\nHere are the fit results that were returned:\n")
    writer = csv.writer(sys.stdout, delimiter=delimiter)
    if not noheader:
        writer.writerow(["distribution", "goodness_of_fit", "lower_z_score", "upper_z_score",
                         "location", "scale", "parameter0", "parameter1", "parameter2"])
    if not z_scores:
        z_scores = best_fit_z_scores(fitdist_results, fitted_dist_p_value=(1-conf_int_size),
                                     onesided=onesided, upper_pvalue=upper_zscores)
    for z_score, result in zip(z_scores, fitdist_results):
        if onesided:
            if upper_zscores:
                lowerz = None
                upperz = z_score
            else:
                lowerz = z_score
                upperz = None
        else:
            lowerz = z_score[0]
            upperz = z_score[1]
        distname = result[0]
        parameters = result[1]
        scale = parameters[-1]
        location = parameters[-2]
        ps = [None, None, None, None]  # Placeholders for remaining parameters, if any
        for i, param in enumerate(parameters[:-2]):
            ps[i] = param
        gof = result[2]
        writer.writerow([distname, gof, lowerz, upperz, location, scale, ps[0], ps[1], ps[2], ps[3]])


if __name__ == "__main__":

    import argparse
    import textwrap

    # PARSE COMMAND LINE ARGUMENTS
    parser = argparse.ArgumentParser(
        description=textwrap.dedent('''\
        Find the probability distribution that best-fits a set of data. Eighteen common continuous distributions
        are examined for a best-fit using the default option, dist=common.  Setting dist=all results in over 80
        continuous distributions being examined.
        
        The Kolmogorov-Smirnov (KS) Goodness-of-Fit (GoF) test is used to determine whether the fit is suitable.
        If a KS GoF statistic is larger than the input (or default) threshold, fitpval, the distributions passes,
        otherwise it fails.  For those distributions that pass the following is output: distribution name, list
        of MLE parameter estimates, and KS GoF p-value.

        A log file of distribution names, their MLE parameter estimates, the KS GoF statistic and p-value, and
        whether they pass or fail is always written out (see the logfile option).

        The following third-party Python modules are required: numpy, scipy.stats, and pandas
        '''),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples:
            (NOTE: No whitespace is permitted in 'drop' lists)
            python fitdist.py --verbose --drop [\\'beta\\',\\'ksone\\'] mydata.txt    <--input file at end
            python fitdist.py mydata.txt --verbose --drop [\\'beta\\,\\'ksone\\'']    <--input file at beginning
            cat mydata.txt | python fitdist.py --drop [\\'beta\\',\\'ksone\\']        <--stdin
            python fitdist.py mydata.txt --dist custom --distlist [\\'norm\\',\\'t\\'] --delimiter ','  <--custom list
        """))
    parser.add_argument("infile",
                        help="Name of an input file of floats; one float per line, no header. \
                        Leave this name off if using stdin instead.",
                        nargs='?',
                        type=argparse.FileType('r'),
                        default=sys.stdin)
    parser.add_argument("--verbose",
                        help="Increase output verbosity",
                        action="store_true")
    parser.add_argument("--dist",
                        help="Specify built-in distribution set to use; default is \'common\'",
                        type=str,
                        choices=['all', 'common', 'custom'])
    parser.add_argument("--distlist",
                        help='''List of distributions to use if when the \'dist\' option
                        is \'custom\'; use the same type of formatting used for the \'drop\'
                        option; when using this, the \'results\' option is automatically set
                        to \'all\'.''')
    parser.add_argument("--drop",
                        help='''List of distribution names to drop from consideration;
                        names must be formatted as shown in the examples, below.
                        WARNING: Whitespace in drop lists is NOT permitted! ''')
    parser.add_argument("--results",
                        help='''Ignore the \'fitpval\' option and output either
                        all results, or the single best result''',
                        type=str,
                        choices=['all', 'one'])
    parser.add_argument("--delimiter",
                        help='''Delimiter to use when writing output results; the default is tabs;
                        set this option to \',\' for commas''',
                        type=str,
                        #default=',')
                        default='\t')
    parser.add_argument("--noheader",
                        help="Don't include a header at the start of the output results",
                        action="store_true")
    parser.add_argument("--logfile",
                        help='''Filename to write logging statements to; default is
                        \"bestfitdist.log\"''',
                        type=str)
    parser.add_argument("--fitpval",
                        help='''P-value to use for the KS GoF test on distributions;
                        must be >= 0.0 and < 1.0; default is 0.25;
                        using 0.0 will output all results.''',
                        type=float)
    parser.add_argument("--zscores",
                        help='''The size of a confidence interval (probability)
                        to output z-scores for; must be > 0.0 and < 1.0; default is 95%%.''',
                        type=float)
    parser.add_argument("--upper",
                        help='''output one-sided, upper z-score; only makes
                        sense when used with the zscores option; do not use with
                        \"lower\" option''',
                        action="store_true")
    parser.add_argument("--lower",
                        help='''output one-sided, lower z-score; only makes
                        sense when used with the zscores option; do not use with
                        \"upper\" option''',
                        action="store_true")
    args = parser.parse_args()

    # SELECT THE SET OF DISTRIBUTIONS TO USE
    if (not args.dist) or (args.dist == 'common'):  # DEFAULT OPTION
        dist_name_list = common_cont_dist_names
    elif args.dist == 'all':
        dist_name_list = cont_dist_names
    elif args.dist == 'custom':
        if args.distlist:
            dist_name_list = eval(args.distlist)
            if args.verbose:
                print("Using custom distribution list: {}".format(str(dist_name_list)))
        else:
            msg = "\'dist\' = \'custom\', but no distributions were provided in the \'distlist\' option."
            raise argparse.ArgumentTypeError(msg)
    else:
        # The way argparse is setup, we should never get here, but just in case...
        msg = "{} is not an option for selecting the set of distributions" % args.dist
        raise argparse.ArgumentTypeError(msg)

    # CHECK IF 'RESULTS' IS SET TO 'ALL' OR 'ONE'
    if (args.results == 'all') or (args.dist == 'custom'):
        results_all = True  # pvalue will be ignored in 'find_best_fit_distributions'
        results_one = False
    elif args.results == 'one':
        results_all = True
        results_one = True  # Only the top scoring distribution will be output
    else:
        results_all = False
        results_one = False

    # IF INPUT DATA THEN READ IT
    dataset = []
    for line in args.infile:
        dataset.append(float(line))
    if args.verbose:
        print("Processing %d values in file \'{}\'.".format((len(dataset), args.infile.name)))

    # SETUP DISTRIBUTIONS TO DROP FROM CONSIDERATION
    if args.drop:
        drop_names = eval(args.drop)
        if args.verbose:
            print("Dropping distributions: {}".format(str(drop_names)))
    else:
        drop_names = []
        if args.verbose:
            print("Dropping distributions: None")

    # SETUP THE LOGGING FILE
    logfile = 'bestfitdist.log'  # Default
    if args.logfile:
        logfile = args.logfile

    # SETUP THE KS GOF P-VALUE
    gof_pvalue = 0.25  # Default
    if args.fitpval:
        if (args.fitpval >= 0.0) and (args.fitpval < 1.0):
            gof_pvalue = args.fitpval
        else:
            raise ValueError("The \'fitpval\', %f, is not strictly between 0.0 and 1.0" % args.fitpval)

    # SETUP Z-SCORES
    zscores_pvalue = 0.05  # Default (one minus the size of the desired confidence interval)
    if args.zscores:
        if 0.0 < args.zscores < 1.0:
            # z-scores calc uses tail probs, not the size of the confidence interval
            zscores_pvalue = 1 - args.zscores
        else:
            raise ValueError("The \'zscores\' option value, %f, is not strictly between 0.0 and 1.0" % args.zscores)

    # DO THE BEST-FIT CALCULATION
    if args.verbose:
        print("\n*** FINDING BEST-FIT DISTRIBUTIONS...\n")
    fit_results = find_best_fit_distributions(dataset,
                                              dist_names=dist_name_list,
                                              goodness_of_fit_pvalue=gof_pvalue,
                                              drop_dist_names=drop_names,
                                              verbose_mode=args.verbose,
                                              logfilename=logfile,
                                              ignore_gof=results_all)

    # IF ONLY THE SINGLE BEST FIT IS DESIRED, THEN SELECT ONLY IT
    if results_one:
        fit_results = [fit_results[0]]

    # CALCULATE THE Z-SCORES
    one_sided = False
    if args.upper or args.lower:
        one_sided = True
    zscores = best_fit_z_scores(fit_results, fitted_dist_p_value=zscores_pvalue, onesided=one_sided,
                                upper_pvalue=args.upper)

    output_results(fit_results, zscores, one_sided, args.upper, delimiter=args.delimiter, noheader=args.noheader,
                   verbose_mode=args.verbose, ignore_gof=results_all)
    
    # ------------
    # END OF FILE
    # ------------
