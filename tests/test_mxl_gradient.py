import numpy as np
import pytest
import time
from xlogit.mixed_logit import MixedLogit


@pytest.fixture
def small_dataset():
    """Generate a small synthetic dataset for basic testing."""
    np.random.seed(42)
    N = 100  # choice situations
    J = 3  # alternatives
    K = 4  # variables

    # Generate synthetic data
    X = np.random.randn(N, J, K)
    X[:, 0, :] += 0.5  # Make first alternative more attractive

    # Create choice probabilities and sample choices
    probs = np.exp(X.sum(axis=2))
    probs = probs / probs.sum(axis=1, keepdims=True)

    # Create choice matrix
    y = np.zeros((N, J))
    for i in range(N):
        choice = np.random.choice(J, p=probs[i])
        y[i, choice] = 1

    # Reshape to expected format
    y = y.reshape(N, J, 1)

    # Create draws
    R = 10  # Small number of draws for testing
    draws = np.random.randn(N, 2, R)

    return {"X": X, "y": y, "draws": draws, "N": N, "J": J, "K": K, "R": R}


@pytest.fixture
def distributions_dataset():
    """Generate a dataset for testing different distributions."""
    np.random.seed(42)
    N, J, K, R = 50, 3, 3, 10
    X = np.random.randn(N, J, K)
    y = np.zeros((N, J, 1))
    y[:, 0, 0] = 1  # All choose first alternative for simplicity
    draws = np.random.randn(N, 3, R)

    distributions = [
        ["n", "n", "n"],  # All normal
        ["ln", "ln", "ln"],  # All lognormal
        ["tn", "tn", "tn"],  # All truncated normal
        ["n", "ln", "tn"],  # Mixed distributions
    ]

    return {
        "X": X,
        "y": y,
        "draws": draws,
        "distributions": distributions,
        "N": N,
        "J": J,
        "K": K,
        "R": R,
    }


@pytest.fixture
def panel_dataset():
    """Generate a panel dataset."""
    np.random.seed(42)
    P = 20  # Number of individuals
    Tp = 5  # Choices per individual
    N = P * Tp  # Total choice situations
    J = 3  # Alternatives per choice
    K = 4  # Variables
    R = 10  # Number of draws

    # Generate synthetic data
    X = np.random.randn(N, J, K)
    y = np.zeros((N, J, 1))
    panels = np.repeat(np.arange(P), Tp)

    # Convert panel ids to indices
    panels_idx = np.empty(N)
    for i, u in enumerate(np.unique(panels)):
        panels_idx[np.where(panels == u)] = i
    panels = panels_idx.astype(int)

    # Weights with one weight per individual
    weights = np.ones(N)
    panel_change_idx = np.concatenate(([0], np.where(panels[:-1] != panels[1:])[0] + 1))
    weights = weights[panel_change_idx]

    # Create preference heterogeneity
    individual_prefs = np.random.randn(P, K)

    # Generate choices based on individual preferences
    for p in range(P):
        p_idx = np.where(panels == p)[0]
        for t in range(len(p_idx)):
            idx = p_idx[t]
            util = np.sum(X[idx] * (0.5 + 0.5 * individual_prefs[p]), axis=1)
            util += np.random.gumbel(size=J)
            choice = np.argmax(util)
            y[idx, choice, 0] = 1

    # Create panel-consistent draws
    draws_by_panel = np.random.randn(P, 2, R)
    draws = np.zeros((N, 2, R))

    for p in range(P):
        p_idx = np.where(panels == p)[0]
        for idx in p_idx:
            draws[idx] = draws_by_panel[p]

    return {
        "X": X,
        "y": y,
        "panels": panels,
        "draws": draws,
        "draws_by_panel": draws_by_panel,
        "weights": weights,
        "P": P,
        "N": N,
        "J": J,
        "K": K,
        "R": R,
    }


@pytest.fixture
def large_dataset():
    """Generate a large dataset for performance testing."""
    np.random.seed(123)
    P = 50  # Number of individuals
    Tp = 10  # Choices per individual
    N = P * Tp  # Total observations
    J = 5  # Number of alternatives
    K = 8  # Variables (5 random, 3 fixed)
    R = 100  # Number of draws

    # Generate data
    X = np.random.randn(N, J, K)

    # Add alternative-specific constants
    for j in range(1, J):
        X[:, j, 0] += j * 0.5

    # Add correlation between variables
    correlation_matrix = np.array(
        [
            [1.0, 0.6, 0.2, -0.3, 0.1, 0.0, 0.0, 0.0],
            [0.6, 1.0, 0.4, -0.2, 0.3, 0.0, 0.0, 0.0],
            [0.2, 0.4, 1.0, 0.5, 0.1, 0.0, 0.0, 0.0],
            [-0.3, -0.2, 0.5, 1.0, 0.2, 0.0, 0.0, 0.0],
            [0.1, 0.3, 0.1, 0.2, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.3, 0.2],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 1.0, 0.4],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.4, 1.0],
        ]
    )

    L = np.linalg.cholesky(correlation_matrix)
    for i in range(N):
        for j in range(J):
            X[i, j, :] = L @ X[i, j, :]

    # Generate utilities and choices
    true_betas = np.array([0.8, -0.6, 0.4, 1.2, -0.9, 0.5, -0.7, 0.3])
    utilities = np.sum(X * true_betas[None, None, :], axis=2) + np.random.gumbel(
        size=(N, J)
    )
    choices = np.argmax(utilities, axis=1)

    y = np.zeros((N, J, 1))
    for i, choice in enumerate(choices):
        y[i, choice, 0] = 1

    # Create panel structure
    panels = np.repeat(np.arange(P), Tp)
    panels_idx = np.empty(N)
    for i, u in enumerate(np.unique(panels)):
        panels_idx[np.where(panels == u)] = i
    panels = panels_idx.astype(int)

    # Create draws for different distributions
    draws_n = np.random.randn(N, 2, R)
    draws_ln = np.random.randn(N, 2, R)
    draws_tn = np.abs(np.random.randn(N, 1, R))
    draws = np.concatenate([draws_n, draws_ln, draws_tn], axis=1)

    # Create availability and weights
    avail = np.ones((N, J))
    for i in range(N):
        if np.random.random() < 0.2:
            non_chosen_alts = [j for j in range(J) if y[i, j, 0] == 0]
            if non_chosen_alts:
                unavail_alt = np.random.choice(non_chosen_alts)
                avail[i, unavail_alt] = 0

    weights = np.ones(N)
    weights[np.random.choice(N, size=int(N * 0.3))] = 2.0
    panel_change_idx = np.concatenate(([0], np.where(panels[:-1] != panels[1:])[0] + 1))
    weights = weights[panel_change_idx]

    return {
        "X": X,
        "y": y,
        "panels": panels,
        "draws": draws,
        "avail": avail,
        "weights": weights,
        "P": P,
        "N": N,
        "J": J,
        "K": K,
        "R": R,
    }


@pytest.fixture
def batch_dataset():
    """Generate a very large dataset for batch processing tests."""
    np.random.seed(456)
    N = 5000  # Large number of choice situations
    J = 4  # Alternatives
    K = 6  # Variables (4 random, 2 fixed)
    R = 50  # Draws

    X = np.random.randn(N, J, K)

    # Generate utilities and choices
    true_betas = np.array([1.0, -0.5, 0.8, -0.3, 0.6, -0.2])
    utilities = np.sum(X * true_betas[None, None, :], axis=2) + np.random.gumbel(
        size=(N, J)
    )
    choices = np.argmax(utilities, axis=1)

    y = np.zeros((N, J, 1))
    for i, choice in enumerate(choices):
        y[i, choice, 0] = 1

    # Create draws
    draws = np.random.randn(N, 4, R)

    return {"X": X, "y": y, "draws": draws, "N": N, "J": J, "K": K, "R": R}


@pytest.fixture
def mixed_logit_basic():
    """Create a basic MixedLogit model with 2 random parameters."""
    model = MixedLogit()
    model._rvidx = np.array([True, True, False, False])
    model._rvdist = ["n", "n"]
    return model


@pytest.fixture
def mixed_logit_all_random():
    """Create a MixedLogit model with all parameters random."""
    model = MixedLogit()
    model._rvidx = np.array([True, True, True])
    model._rvdist = ["n", "n", "n"]
    return model


@pytest.fixture
def mixed_logit_large():
    """Create a MixedLogit model with mixed distributions."""
    model = MixedLogit()
    model._rvidx = np.array([True, True, True, True, True, False, False, False])
    model._rvdist = ["n", "ln", "tn", "n", "ln"]
    return model


@pytest.fixture
def mixed_logit_batch():
    """Create a MixedLogit model for batch processing."""
    model = MixedLogit()
    model._rvidx = np.array([True, True, True, True, False, False])
    model._rvdist = ["n", "ln", "n", "tn"]
    return model


def test_loglik_gradient_accuracy(small_dataset, mixed_logit_basic):
    """Test that analytical gradients match finite difference approximation."""
    data = small_dataset
    model = mixed_logit_basic

    # Test parameters
    betas = np.array([0.5, -0.3, 0.2, 0.1, 0.8, 0.6])

    # Get analytical gradient
    loglik, analytical_grad, grad_n = model._loglik_gradient(
        betas,
        data["X"],
        None,
        data["draws"],
        np.ones(data["N"]),
        np.ones((data["N"], data["J"])),
        None,
        None,
        None,
        None,
        return_gradient=True,
    )

    # Compute finite difference approximation
    epsilon = 1e-6
    fd_grad = np.zeros_like(betas)

    for i in range(len(betas)):
        betas_plus = betas.copy()
        betas_plus[i] += epsilon

        loglik_plus = model._loglik_gradient(
            betas_plus,
            data["X"],
            None,
            data["draws"],
            np.ones(data["N"]),
            np.ones((data["N"], data["J"])),
            None,
            None,
            None,
            None,
            return_gradient=False,
        )

        fd_grad[i] = (loglik_plus - loglik) / epsilon

    # Check that analytical and finite difference gradients are close
    np.testing.assert_allclose(
        analytical_grad,
        fd_grad,
        rtol=1e-3,
        atol=1e-4,
        err_msg=f"Analytical gradient {analytical_grad} doesn't match finite difference {fd_grad}",
    )

    # Verify the shape of grad_n
    assert grad_n.shape == (data["N"], len(betas)), (
        f"Expected grad_n shape {(data['N'], len(betas))}, got {grad_n.shape}"
    )

    # Check that summing grad_n equals the negative of the full gradient
    np.testing.assert_allclose(
        analytical_grad,
        -grad_n.sum(axis=0),
        rtol=1e-10,
        atol=1e-10,
        err_msg=f"Sum of per-observation gradients {grad_n.sum(axis=0)} doesn't match -analytical_grad {-analytical_grad}",
    )


@pytest.mark.parametrize("dist_idx", [0, 1, 2, 3])
def test_loglik_gradient_distributions(distributions_dataset, dist_idx):
    """Test gradients for different distributions (n, ln, tn)."""
    data = distributions_dataset

    # Create mixed logit model
    model = MixedLogit()
    model._rvidx = np.array([True, True, True])
    model._rvdist = data["distributions"][dist_idx]

    # Parameters: [mean1, mean2, mean3, sd1, sd2, sd3]
    betas = np.array([0.5, -0.3, 0.2, 0.8, 0.6, 0.4])

    # Get analytical gradient
    loglik, analytical_grad, grad_n = model._loglik_gradient(
        betas,
        data["X"],
        None,
        data["draws"],
        np.ones(data["N"]),
        np.ones((data["N"], data["J"])),
        None,
        None,
        None,
        None,
        return_gradient=True,
    )

    # Compute finite difference approximation
    epsilon = 1e-6
    fd_grad = np.zeros_like(betas)

    for i in range(len(betas)):
        betas_plus = betas.copy()
        betas_plus[i] += epsilon

        loglik_plus = model._loglik_gradient(
            betas_plus,
            data["X"],
            None,
            data["draws"],
            np.ones(data["N"]),
            np.ones((data["N"], data["J"])),
            None,
            None,
            None,
            None,
            return_gradient=False,
        )

        fd_grad[i] = (loglik_plus - loglik) / epsilon

    # Check gradient accuracy for this distribution set
    np.testing.assert_allclose(
        analytical_grad,
        fd_grad,
        rtol=1e-3,
        atol=1e-4,
        err_msg=f"Gradient incorrect for distributions {data['distributions'][dist_idx]}",
    )

    # Verify that grad_n sums to negative of analytical_grad
    np.testing.assert_allclose(
        -analytical_grad,
        grad_n.sum(axis=0),
        rtol=1e-10,
        atol=1e-10,
        err_msg=f"For {data['distributions'][dist_idx]}, sum of grad_n doesn't match -analytical_grad",
    )


def test_loglik_gradient_panel_data(panel_dataset, mixed_logit_basic):
    """Test gradients with panel data structure."""
    data = panel_dataset
    model = mixed_logit_basic

    # Test parameters
    betas = np.array([0.5, -0.3, 0.2, 0.1, 0.8, 0.6])

    # Test with panel structure
    loglik_panel, grad_panel, grad_n_panel = model._loglik_gradient(
        betas,
        data["X"],
        data["panels"],
        data["draws"],
        data["weights"],
        np.ones((data["N"], data["J"])),
        None,
        None,
        None,
        None,
        return_gradient=True,
    )

    # Test without panel structure
    loglik_no_panel, grad_no_panel, grad_n_no_panel = model._loglik_gradient(
        betas,
        data["X"],
        None,
        data["draws"],
        np.ones(data["N"]),
        np.ones((data["N"], data["J"])),
        None,
        None,
        None,
        None,
        return_gradient=True,
    )

    # Results should differ when panel structure is considered
    assert loglik_panel != loglik_no_panel, (
        "Panel and non-panel log-likelihood should differ"
    )

    # Verify gradients match finite difference
    epsilon = 1e-6
    fd_grad_panel = np.zeros_like(betas)

    for i in range(len(betas)):
        betas_plus = betas.copy()
        betas_plus[i] += epsilon

        loglik_plus = model._loglik_gradient(
            betas_plus,
            data["X"],
            data["panels"],
            data["draws"],
            data["weights"],
            np.ones((data["N"], data["J"])),
            None,
            None,
            None,
            None,
            return_gradient=False,
        )

        fd_grad_panel[i] = (loglik_plus - loglik_panel) / epsilon

    # Check that analytical gradient matches finite difference for panel data
    np.testing.assert_allclose(
        grad_panel,
        fd_grad_panel,
        rtol=1e-3,
        atol=1e-4,
        err_msg="Panel data: analytical gradient doesn't match finite difference",
    )

    # Verify that panel gradients sum to negative of analytical gradient
    np.testing.assert_allclose(
        -grad_panel,
        grad_n_panel.sum(axis=0),
        rtol=1e-10,
        atol=1e-10,
        err_msg="Panel data: sum of grad_n doesn't match -analytical_grad",
    )


def test_loglik_gradient_large_scale(large_dataset, mixed_logit_large):
    """Test gradient computation with a large, complex dataset."""
    data = large_dataset
    model = mixed_logit_large

    # Large parameter vector: 5 means + 3 fixed + 5 std deviations = 13 parameters
    betas = np.array(
        [0.5, -0.3, 0.2, 0.8, -0.4, 0.6, -0.7, 0.3, 0.9, 0.7, 0.5, 1.2, 0.6]
    )

    # Calculate analytical gradient
    start_grad = time.time()
    loglik, analytical_grad, grad_n = model._loglik_gradient(
        betas,
        data["X"],
        data["panels"],
        data["draws"],
        data["weights"],
        data["avail"],
        None,
        None,
        None,
        None,
        return_gradient=True,
    )
    grad_time = time.time() - start_grad

    # Compute finite difference approximation
    start_fd = time.time()
    epsilon = 1e-6
    fd_grad = np.zeros_like(betas)

    for i in range(len(betas)):
        betas_plus = betas.copy()
        betas_plus[i] += epsilon

        loglik_plus = model._loglik_gradient(
            betas_plus,
            data["X"],
            data["panels"],
            data["draws"],
            data["weights"],
            data["avail"],
            None,
            None,
            None,
            None,
            return_gradient=False,
        )

        fd_grad[i] = (loglik_plus - loglik) / epsilon

    fd_time = time.time() - start_fd

    # Check analytical vs finite difference accuracy
    np.testing.assert_allclose(
        analytical_grad,
        fd_grad,
        rtol=2e-2,
        atol=1e-4,
        err_msg="Large-scale test: analytical gradient doesn't match finite difference",
    )

    # Verify that grad_n sums to negative of analytical_grad
    np.testing.assert_allclose(
        -analytical_grad,
        grad_n.sum(axis=0),
        rtol=1e-10,
        atol=1e-10,
        err_msg="Large-scale test: sum of grad_n doesn't match -analytical_grad",
    )

    # Performance check
    assert grad_time < fd_time, (
        "Analytical gradient should be faster than finite difference"
    )

    # Check for NaN/Inf values
    assert not np.any(np.isnan(analytical_grad)), (
        "NaN values found in analytical gradient"
    )
    assert not np.any(np.isnan(grad_n)), "NaN values found in per-observation gradient"
    assert not np.any(np.isinf(analytical_grad)), (
        "Inf values found in analytical gradient"
    )
    assert not np.any(np.isinf(grad_n)), "Inf values found in per-observation gradient"


def test_loglik_gradient_batch_processing(batch_dataset, mixed_logit_batch):
    """Test gradient computation with batch processing for large datasets."""
    data = batch_dataset
    model = mixed_logit_batch

    # Set up parameters
    betas = np.array([0.8, -0.4, 0.6, -0.2, 0.5, -0.3, 0.7, 0.9, 0.5, 0.3])

    # Run with full batch (no batching)
    start_full = time.time()
    loglik_full, grad_full, grad_n_full = model._loglik_gradient(
        betas,
        data["X"],
        None,
        data["draws"],
        np.ones(data["N"]),
        np.ones((data["N"], data["J"])),
        None,
        None,
        None,
        None,
        return_gradient=True,
    )
    full_time = time.time() - start_full

    # Run with batch processing
    batch_size = 500
    start_batch = time.time()
    loglik_batch, grad_batch, grad_n_batch = model._loglik_gradient(
        betas,
        data["X"],
        None,
        data["draws"],
        np.ones(data["N"]),
        np.ones((data["N"], data["J"])),
        None,
        None,
        None,
        batch_size,
        return_gradient=True,
    )
    batch_time = time.time() - start_batch

    # Compare results
    np.testing.assert_allclose(
        loglik_full,
        loglik_batch,
        rtol=1e-5,
        atol=1e-6,
        err_msg="Log-likelihood differs between full and batched computation",
    )

    np.testing.assert_allclose(
        grad_full,
        grad_batch,
        rtol=1e-5,
        atol=1e-6,
        err_msg="Gradient differs between full and batched computation",
    )

    # Ensure the per-observation gradients match in shape
    assert grad_n_full.shape == grad_n_batch.shape, (
        f"Shape mismatch: {grad_n_full.shape} vs {grad_n_batch.shape}"
    )

    # Check some random observations' gradients
    sample_indices = np.random.choice(data["N"], size=10, replace=False)
    np.testing.assert_allclose(
        grad_n_full[sample_indices],
        grad_n_batch[sample_indices],
        rtol=1e-5,
        atol=1e-6,
        err_msg="Per-observation gradients differ between full and batched computation",
    )
