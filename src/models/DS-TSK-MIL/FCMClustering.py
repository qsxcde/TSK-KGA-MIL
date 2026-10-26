import numpy as np

# Relative distance below which a prototype is considered to coincide with the
# common mean of the standardised instances.  The default keeps the check
# inactive unless the prototypes have effectively merged into a single point.
DEFAULT_PROTOTYPE_TOL = 1e-2


class FCMClustering:
    """Fuzzy c-means used to initialise the TSK antecedents.

    The fixed-point iteration admits a solution in which every prototype
    coincides with the mean of the instances.  With standardised features that
    mean is the origin, so the solution is numerically indistinguishable from an
    all-zero antecedent matrix and the resulting rule memberships become
    uniform, i.e. the rule layer degenerates into a single linear map.  The
    solution is attracting when the instances live in a high-dimensional space
    with heavy-tailed pairwise distances, and it does not depend on the
    initialisation, so re-seeding the memberships alone does not avoid it.

    For this reason the prototypes are checked at every iteration.  If they have
    merged into a single point, they are re-estimated with K-Means (which by
    construction returns separated centroids) and the iteration stops.  On
    instances where the iteration behaves normally the check never fires and the
    output is unchanged.
    """

    def __init__(
        self,
        n_clusters=8,
        m=2.0,
        max_iter=150,
        tol=1e-5,
        random_state=None,
        prototype_tol=DEFAULT_PROTOTYPE_TOL,
    ):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.prototype_tol = prototype_tol
        self.reinitialised_ = False

    def _membership_from_centers(self, X, centers):
        dist = np.zeros((len(X), self.n_clusters))
        for cluster_idx in range(self.n_clusters):
            diff = X - centers[cluster_idx]
            dist[:, cluster_idx] = np.sqrt(np.sum(diff ** 2, axis=1))
        dist = np.maximum(dist, 1e-8)
        dist_power = dist ** (2.0 / (self.m - 1))
        return 1.0 / (
            dist_power * np.sum(1.0 / dist_power, axis=1, keepdims=True)
        )

    def _kmeans_prototypes(self, X):
        from sklearn.cluster import KMeans

        kmeans = KMeans(
            n_clusters=self.n_clusters,
            n_init=10,
            random_state=self.random_state,
        ).fit(X)
        centers = kmeans.cluster_centers_.astype(np.float64)
        membership = self._membership_from_centers(X, centers)
        return centers, membership

    def fit(self, X):
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)

        membership = rng.uniform(0, 1, (n_samples, self.n_clusters))
        membership = membership / np.sum(membership, axis=1, keepdims=True)

        scale = float(np.std(X)) if np.std(X) > 0 else 1.0
        threshold = self.prototype_tol * scale

        for iteration in range(self.max_iter):
            membership_old = membership.copy()

            membership_power = membership ** self.m
            centers = np.dot(membership_power.T, X) / np.sum(
                membership_power, axis=0, keepdims=True
            ).T

            if np.abs(centers).max() < threshold:
                centers, membership = self._kmeans_prototypes(X)
                self.reinitialised_ = True
                break

            dist = np.zeros((n_samples, self.n_clusters))
            for cluster_idx in range(self.n_clusters):
                diff = X - centers[cluster_idx]
                dist[:, cluster_idx] = np.sqrt(np.sum(diff ** 2, axis=1))

            dist = np.maximum(dist, 1e-8)
            exponent = 2.0 / (self.m - 1)
            dist_power = dist ** exponent
            membership = 1.0 / (
                dist_power * np.sum(1.0 / dist_power, axis=1, keepdims=True)
            )

            if np.linalg.norm(membership - membership_old) < self.tol:
                print(f"FCM converged at iteration {iteration}")
                break

        self.centers_ = centers
        self.U_ = membership
        self.widths_ = np.zeros((self.n_clusters, n_features))
        global_std = np.std(X, axis=0)
        adaptive_floor = np.maximum(0.1 * global_std, 1e-2)
        for cluster_idx in range(self.n_clusters):
            weights = membership[:, cluster_idx] ** self.m
            weight_sum = np.sum(weights) + 1e-8
            weighted_var = np.sum(
                weights[:, None] * (X - centers[cluster_idx]) ** 2,
                axis=0,
            ) / weight_sum
            cluster_width = np.sqrt(weighted_var + 1e-8)
            self.widths_[cluster_idx] = np.maximum(cluster_width, adaptive_floor)

        return self.centers_, self.widths_
