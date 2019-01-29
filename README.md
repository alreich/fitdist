Fit Distribution Module/Script (fitdist)
========================================

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
