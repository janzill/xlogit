import numpy as np
import pytest
from scipy.optimize import minimize
from xlogit._optimize import _bfgs


def quadratic_loglik(x, *args, **kwargs):
    # f(x) = 0.5 * x^T x, minimum at x=0
    return_gradient = kwargs.get("return_gradient", False)
    val = 0.5 * np.dot(x, x)
    grad = x
    grad_n = grad.reshape(1, -1)
    if return_gradient:
        return val, grad, grad_n
    else:
        return val


def shifted_quadratic_loglik(x, *args, **kwargs):
    # f(x) = 0.5 * (x - c)^T (x - c), minimum at x=c
    c = args[0]
    diff = x - c
    val = 0.5 * np.dot(diff, diff)
    grad = diff
    grad_n = grad.reshape(1, -1)
    if kwargs.get("return_gradient", False):
        return val, grad, grad_n
    else:
        return val


def rosenbrock_loglik(x, *args, **kwargs):
    # Rosenbrock function: minimum at [1, 1]
    val = 100.0 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
    grad = np.array(
        [-400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]), 200 * (x[1] - x[0] ** 2)]
    )
    grad_n = grad.reshape(1, -1)
    if kwargs.get("return_gradient", False):
        return val, grad, grad_n
    else:
        return val


@pytest.mark.parametrize(
    "x0", [np.array([5.0, -3.0]), np.array([0.1, 0.1]), np.array([-10.0, 10.0])]
)
def test_bfgs_quadratic(x0):
    args = ()
    result = _bfgs(
        quadratic_loglik,
        x0,
        args,
        maxiter=100,
        tol=1e-8,
        gtol=1e-6,
        disp=False,
        restart=True,
    )
    assert result["success"], f"BFGS did not converge: {result['message']}"
    np.testing.assert_allclose(result["x"], np.zeros_like(x0), atol=1e-5)
    np.testing.assert_allclose(result["grad"], np.zeros_like(x0), atol=1e-5)


@pytest.mark.parametrize("c", [np.array([2.0, -2.0]), np.array([-1.5, 3.5])])
def test_bfgs_shifted_quadratic(c):
    x0 = np.array([0.0, 0.0])
    args = (c,)
    result = _bfgs(
        shifted_quadratic_loglik,
        x0,
        args,
        maxiter=100,
        tol=1e-8,
        gtol=1e-6,
        disp=False,
        restart=True,
    )
    assert result["success"], f"BFGS did not converge: {result['message']}"
    np.testing.assert_allclose(result["x"], c, atol=1e-5)
    np.testing.assert_allclose(result["grad"], np.zeros_like(c), atol=1e-5)


def test_bfgs_rosenbrock():
    x0 = np.array([-1.2, 1.0])
    args = ()
    result = _bfgs(
        rosenbrock_loglik,
        x0,
        args,
        maxiter=5000,
        tol=1e-10,
        gtol=1e-4,
        disp=False,
        restart=True,
    )
    assert result["success"], f"BFGS did not converge: {result['message']}"

    # Scipy BFGS
    def fun(x, *args):
        return rosenbrock_loglik(x, *args, return_gradient=True)[0]

    def jac(x, *args):
        return rosenbrock_loglik(x, *args, return_gradient=True)[1]

    scipy_result = minimize(
        fun,
        x0,
        args=args,
        jac=jac,
        method="BFGS",
        tol=1e-10,
        options={"gtol": 1e-4, "maxiter": 1000},
    )

    print(result["x"], result["grad"], result["fun"], result["message"])
    np.testing.assert_allclose(result["grad"], scipy_result.jac, atol=1e-4)
    np.testing.assert_allclose(result["fun"], scipy_result.fun, atol=1e-6)
    print(scipy_result.x, scipy_result.jac, scipy_result.fun)

    np.testing.assert_allclose(result["x"], np.ones_like(x0), atol=1e-3)
    np.testing.assert_allclose(result["grad"], np.zeros_like(x0), atol=1e-3)


def himmelblau_loglik(x, *args, **kwargs):
    # Himmelblau's function: four identical minima
    # f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
    val = (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2
    grad = np.array(
        [
            4 * x[0] * (x[0] ** 2 + x[1] - 11) + 2 * (x[0] + x[1] ** 2 - 7),
            2 * (x[0] ** 2 + x[1] - 11) + 4 * x[1] * (x[0] + x[1] ** 2 - 7),
        ]
    )
    grad_n = grad.reshape(1, -1)
    if kwargs.get("return_gradient", False):
        return val, grad, grad_n
    else:
        return val


def test_bfgs_vs_scipy_himmelblau():
    x0 = np.array([6.0, 6.0])
    args = ()
    # Our BFGS
    result = _bfgs(
        himmelblau_loglik,
        x0,
        args,
        maxiter=1000,
        tol=1e-8,
        gtol=1e-6,
        disp=False,
        restart=True,
    )

    # Scipy BFGS
    def fun(x, *args):
        return himmelblau_loglik(x, *args, return_gradient=True)[0]

    def jac(x, *args):
        return himmelblau_loglik(x, *args, return_gradient=True)[1]

    scipy_result = minimize(
        fun,
        x0,
        args=args,
        jac=jac,
        method="BFGS",
        tol=1e-8,
        options={"gtol": 1e-6, "maxiter": 1000},
    )

    # Both should converge to a minimum (one of the four)
    assert result["success"], f"Our BFGS did not converge: {result['message']}"
    assert scipy_result.success, f"SciPy BFGS did not converge: {scipy_result.message}"

    # The solutions should be close to each other and to a known minimum
    # Actually, as long as we find a minimum we are good, does not have to the same
    # as scipy's result, since there are multiple minima.
    # np.testing.assert_allclose(result["x"], scipy_result.x, atol=1e-4)
    np.testing.assert_allclose(result["grad"], scipy_result.jac, atol=1e-4)
    np.testing.assert_allclose(result["fun"], scipy_result.fun, atol=1e-6)

    # Optionally, check closeness to a known minimum (e.g., [3, 2])
    known_minima = [
        np.array([3.0, 2.0]),
        np.array([-2.805118, 3.131312]),
        np.array([-3.779310, -3.283186]),
        np.array([3.584428, -1.848126]),
    ]
    assert any(np.allclose(result["x"], m, atol=1e-3) for m in known_minima), (
        f"Result {result['x']} not close to any known minimum"
    )
