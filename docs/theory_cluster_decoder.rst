Theory behind Cluster Decoder
=============================

==================================
Regression Method and EM Algorithm
==================================

===================
Hierarchical Method
===================

=================
Sequential Method
=================

The sequential method is characterized by a first strong hypothesis which
supposes that the k states of the signal will follow one after the other
without ever reappearing, even if each ones of them can have a different duration.
For example, we could have a signal where the states follow each other in this way
1-2-3-4-... but not such as 1-2-1-3-4-2-3-...

The fit method of the sequential algorithm proceeds as follows:

We initialize an error e0 by first assuming an uniform distribution in time of the k states.
Thus for each state we will take into account only the relevant part of the X and y
which allows us to calculate a decoding matrix for each state. Once this is done,
we then calculate the cumulative error on each state by computing the distance between
y and X*Wk (Wk the decoding matrix).
After that we launch a loop that will do the same thing as before but choosing random
distributions of states in time. If the error is smaller we save this distribution
then we continue the loop. At the end we obtain a distribution as well as decoding matrices
that have minimized the error on Max_iter random draws.