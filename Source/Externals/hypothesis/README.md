# hypothesis.h
## A collection of quantiles and utility functions for running Z, Chi^2, and Student's T hypothesis tests

A variety of quantile functions are needed to perform statistical hypothesis
tests, but these are missing from the C++ standard library. This compact header
file-only library contains the most important quantiles; it is mostly a wrapper
around a C++ port of the relevant functions from the Cephes math library.
