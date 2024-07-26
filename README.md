# Anderson-model-on-graphs

Script for creating graphs (random regular graph, Erdos-Reyni graph, small world network and uniformly distributed random graph), applying Anderson model on the graph, using exact diagonalization to extract eigenvalues and eigenvectors and finally calculating ergodicity indicators (averaged level separation ratio, fractal dimensionality and entangelment entropy). I also included simple calucation of level spacings. It is also possible to calculate expected values of three different observables and following that the fluctuations of the diagonal matrix elements. I used the following libraries: networkx for graph creation, numpy for calcuation, qutip for calculating partial trace and entangelment entropy and some other standard python libraries.
