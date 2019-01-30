Fit Distribution Module/Script
==============================

The *fitdist* module/script finds the probability distributions that
best-fit a set of data. By default, eighteen common continuous
distributions are examined for a best-fit.  Setting *dist=all*
results in over 80 continuous distributions being examined.

The Kolmogorov-Smirnov (KS) Goodness-of-Fit (GoF) test is used to
determine whether a fit is suitable.  If the KS GoF statistic is
larger than the input (or default) threshold, *fitpval*, the
distribution passes, otherwise it fails.  For those distributions that
pass the following is output: distribution name, list of MLE parameter
estimates (in scipy.stat's defined order), and the KS GoF p-value.

A log file of distribution names, their MLE parameter estimates, the
KS GoF statistic and p-value, and whether they pass or fail is always
written out (see the logfile option).

The following Python modules are required: numpy, scipy, matplotlib,
pandas, statsmodels

See
http://docs.scipy.org/doc/scipy/reference/tutorial/stats/continuous.html
for information on the distributions supported by *fitdist*.

Module API
----------

.. autofunction:: fitdist.find_best_fit_distributions
.. autofunction:: fitdist.make_fitted_pdf
.. autofunction:: fitdist.make_fitted_cdf
.. autofunction:: fitdist.make_fitted_ppf
.. autofunction:: fitdist.best_fit_z_scores
.. autofunction:: fitdist.write_p_value_table
.. autofunction:: fitdist.output_results

Script Usage
------------

For help on using *fitdist* as a script, enter **python fitdist.py
--help** in a terminal window , as shown below:

::
    
    prompt> python fitdist.py --help
    usage: fitdist.py [-h] [--verbose] [--dist {all,common,custom}]
                      [--distlist DISTLIST] [--drop DROP] [--results {all,one}]
                      [--delimiter DELIMITER] [--noheader] [--logfile LOGFILE]
                      [--fitpval FITPVAL] [--zscores ZSCORES] [--upper] [--lower]
                      [infile]
    
