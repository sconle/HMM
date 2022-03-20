User Guide
==========

Installation
------------

To install the neuroHMM library, you can copy the following command line into your terminal::

 git clone https://github.com/sconle/HMM.git

or download the ZIP repository on GitHub.

To be able to use our library, you will also need to install the necessary packages by using the following command line into your terminal::

 pip install -r requirements.txt

Using our Estimators
--------------------

Classifiers and Regressors are subclasses of the scikit-learn Estimator class, and correspond to supervised or unsupervised classification or regression models. All Estimators of scikit-learn and of the libraries which are inspired by them, including ours, have a structure and methods in common to unify their use. In particular, here are the basic lines of code that can be used with our Estimators::

 model = Cluster_Decoder(n_components=n_states, n_iter=n_iter, covariance_type=covariance_type, tol=tol)

This line of instructions is used to create the ``Cluster_Decoder`` Estimator according to the model and constraints specific to the cluster decoding method, and to define the hyperparameters of this Estimator.

::

 model.fit(X, y)    # or model.fit(X)

This line of code is used to adjust the parameters of the Estimator and to find those that best fit the training data (X, y) - or X for unlabelled data, depending on the Estimator used - using the inference algorithm implemented in the Estimator.

::

 y_predict = model.predict(X)

This method returns the estimate y_predict obtained from the data X and the Estimator's own parameters.

::

 proba = model.predict_proba(X)

This method returns the likelihood of each element of X belonging to each cluster according to the Estimator's own parameters.