import numpy as np
import numpy.typing as npt

from tqdm import tqdm
from typing import Optional, List

from .kutils import create_centroids, map_points_to_centroids, improve_centroids

class KMeans:

    def __init__(self, k: int, random_state: Optional[int] = 12345, max_iter: Optional[int] = 1000) -> None:
        """
        Creates a kmeans object, initialized with the input data.

        Follows a similar interface to sklearn's KMeans class.

        Order of use:
            Initialize this object
            Run the .fit() method to solve the KMeans problem
            Read the subparameters for the output.

        Input:
            k:
                An integer in the range 1 <= k <= N, where N is the length of the training set. Will be the number of clusters
                found by the K-Means algorithm.
            random_state:
                An integer used for initializing any subparameters that rely on random number generation.
                Optional.
            max_iter:
                The maximum number of iterations to do for training the kmeans solution.

        Output:
            None
        """

        #Initialize the parameters
        self.k = k
        self.random_state = random_state
        self.max_iter = max_iter


        self.labels_ = None
        self.cluster_centers_ = None

    def fit(self, point_matrix: npt.ArrayLike) -> None:
        """
        Runs the KMeans algorithm on the given training data.

        Input:
            point_matrix:
                A numpy ArrayLike, expected to be of shape (N, D), where N is the length of the training set,
                and D is the dimensionality of the points. The values inside the matrix should be numerical.

        Output:
            None
        """

        #Check some common input parameter error conditions
        try:
            self.point_matrix = point_matrix.reshape((point_matrix.shape[0], point_matrix.shape[1]))
            self.N = self.point_matrix.shape[0]
            self.D = self.point_matrix.shape[1]
        except:
            self.point_matrix = None
            self.N = None
            self.D = None

            raise RuntimeError(f"point_matrix was not structured correctly. Expecting a numpy ArrayLike, of shape (N, D) where N is the length of the training set, and D is the dimensionality of the points.")

        if not (1 <= self.k <= self.N):
            raise RuntimeError(f"k must be an integer number in the range [1, {self.N}] for this training data!")

        #Initialze the centroids
        self.cluster_centers_ = create_centroids(self.point_matrix, self.k, self.random_state)

        #Main training loop
        converged = False
        for i in range(self.max_iter):
            #Map the training points to the centroids
            self.labels_ = map_points_to_centroids(self.point_matrix, self.cluster_centers_)

            #Generate the next set of centroids:
            new_centroids = improve_centroids(self.point_matrix, self.labels_, self.cluster_centers_)

            #Move centroids to next step
            difference = np.abs(self.cluster_centers_ - new_centroids)
            self.cluster_centers_ = new_centroids

            if difference.mean() < 0.001:
                converged = True
                break

        #Check for convergence
        if converged:
            #One last label mapping to be technically correct
            self.labels = map_points_to_centroids(self.point_matrix, self.cluster_centers_)
        else:
            raise RuntimeError(f"Training loop failed to converge! Current centroid distance: {difference.mean()}")

    def predict(self, point_matrix: npt.ArrayLike) -> npt.ArrayLike:
        """
        Computes the closest labels for each point given, using the precomputed centroids

        Input:
            point_matrix:
                A numpy ArrayLike, expected to be of shape (N, D), where N is the length of the training set,
                and D is the dimensionality of the points. The values inside the matrix should be numerical.

                D must match the dimensionality of the data given at training time.

        Output:
            labels:
                The labels predicted by the model. A numpy ArrayLike of shape (N, 1) where N matches the points given in this function

        """

        if point_matrix.shape[1] != self.D:
            raise RuntimeError(f"Expected an input matrix of shape (N, {self.D}), not (N, {point_matrix.shape[1]})")

        labels = map_points_to_centroids(point_matrix, self.cluster_centers_)

        return labels
