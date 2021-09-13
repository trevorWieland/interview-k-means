from interviewkmeans import KMeans
import numpy as np

def test_correctness():
    """
    Runs a test case that should converge easily
    """

    rng = np.random.default_rng(12345)

    group_50 = rng.standard_normal(size=(25,2)) + (50, 50)
    group_20 = rng.standard_normal(size=(25,2)) + (20, 20)

    input_data = np.vstack((group_50, group_20))

    kmeans = KMeans(2, 12345)
    kmeans.fit(input_data)

    expected_answer = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0]

    assert all((kmeans.labels_ - expected_answer) == 0)

    assert (kmeans.cluster_centers_[0][0] - 19.88) < 0.01
    assert (kmeans.cluster_centers_[0][0] - 19.88) < 0.01
    assert (kmeans.cluster_centers_[0][0] - 50.04) < 0.01
    assert (kmeans.cluster_centers_[0][0] - 50.07) < 0.01
