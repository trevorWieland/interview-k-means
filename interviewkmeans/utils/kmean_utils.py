import numpy as np
import numpy.typing as npt

from typing import Optional, List

def create_centroids(point_matrix: npt.ArrayLike, k: int, random_state: Optional[int] = 12345) -> npt.ArrayLike:
    """
    Creates initial centroids

    Input
        point_matrix:
            A numpy ArrayLike, expected to be of shape (N, D), where N is the length of the training set,
            and D is the dimensionality of the points. The values inside the matrix should be numerical.
        k:
            An integer in the range 1 <= k <= N, where N is the length of the training set. Will be the number of clusters
            found by the K-Means algorithm.

    Output:
        centroids:
            A numpy array of shape (k, D) where k is the number of centroids requested, and D is the dimensionality of the input data.

    """

    #Create a numpy generator for repeatability
    rng = np.random.default_rng(random_state)

    #Shuffle the points randomly
    rng.shuffle(point_matrix, axis=0)

    #Assign the first k rows to the centroids
    centroids = point_matrix[:k, :]

    return centroids

def map_points_to_centroids(point_matrix: npt.ArrayLike, centroids: npt.ArrayLike) -> npt.ArrayLike:
    """
    Map each point to their closest centroid

    Input
        point_matrix:
            A numpy ArrayLike, expected to be of shape (N, D), where N is the length of the training set,
            and D is the dimensionality of the points. The values inside the matrix should be numerical.
        centroids:
            A numpy array of shape (k, D) where k is the number of centroids requested, and D is the dimensionality of the input data.

    Output:
        labels:
            A numpy ArrayLike of shape (N, 1) where N is the length of the training data, and the single value is the integer label of the closest centroid
    """


    #Calculate the euclidean distance sqrt((A - B)^2) but in a cool numpy way
    #Credit to https://flothesof.github.io/k-means-numpy.html for the method, with more time I could probably have found this, but
    #decided to just search for a fast solution
    distances = np.sqrt(((point_matrix - centroids[:, np.newaxis])**2).sum(axis=2))

    #Calculate the minimum distance index for each point, which is the closest centroid
    labels = np.argmin(distances, axis=0)

    return labels

def improve_centroids(point_matrix: npt.ArrayLike, labels: npt.ArrayLike, centroids: npt.ArrayLike):
    """
    Improves each centroid slightly, by adjusting them to be the new means of their currently labeled set

    Input:
        point_matrix:
            A numpy ArrayLike, expected to be of shape (N, D), where N is the length of the training set,
            and D is the dimensionality of the points. The values inside the matrix should be numerical.
        centroids:
            A numpy array of shape (k, D) where k is the number of centroids requested, and D is the dimensionality of the input data.
        labels:
            A numpy ArrayLike of shape (N, 1) where N is the length of the training data, and the single value is the integer label of the closest centroid

    Output:
        centroids:
            A numpy array of shape (k, D) where k is the number of centroids requested, and D is the dimensionality of the input data.
            Has now been adjusted to be closer to optimal

    """

    #For enforcing that nothing weird happens after this loop and reshaping
    k = centroids.shape[0]
    D = centroids.shape[1]

    #Iterate through each k-set
    new_centroids = []
    for k_label in range(centroids.shape[0]):

        #Find the relevant points to this k-set
        relevant_points = point_matrix[labels==k]

        #Calculate the mean of this relevant set
        relevant_mean = relevant_points.mean(axis=0)

        #Append this new centroid to the list
        new_centroids.append(relevant_mean)

    #Reshape the list into a numpy array with same shape as before
    new_centroids = np.array(new_centroids)
    new_centroids = new_centroids.reshape((k, D))

    return new_centroids
