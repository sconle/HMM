Theory behind TDE-HMM
=====================

  (Description, Développement, Intérêt et exemples)

Le time-delay embedded hidden Markov model (TDE-HMM) est la méthode de clustering des papiers d’Oxford que nous maîtrisions le mieux.
En effet, il s’agit de la méthode utilisée lors du stage de Timothée à l’INT pour effectuer la détection de
bouffées de signal dans les données d’expérience présentées plus haut, caractérisées par des spectres fréquentiels
différents. Elle repose sur un modèle de Markov caché dont le signal observé(les composantes indépendantes extraites
de l’EEG) est représenté par le modèle d’observation décrit dans le papier de [Seedat et al., 2020] et que l’on résume
dans la figure ci-contre.

Dans ce modèle d’observation, les données [y_t-N, …, y_t+N], une fenêtre d’observation de taille 2N+1, sont
supposées générées par une loi gaussienne vectorielle, dont les paramètres (le vecteur moyenne et la matrice
de covariance) dépendent de l’état caché x_t.

