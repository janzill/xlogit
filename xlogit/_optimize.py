import numpy as np
from scipy.optimize import minimize, approx_fprime, line_search


def _bfgs(
    loglik_fn,
    x,
    args,
    maxiter=2000,
    tol=1e-10,
    gtol=1e-6,
    step_tol=1e-10,
    disp=False,
    hinv_init="",  # identity
    restart=False,
):
    """BFGS optimization routine."""

    res, g, grad_n = loglik_fn(x, *args, **{"return_gradient": True})

    if hinv_init == "identity":
        Hinv = np.eye(len(g))
    else:
        Hinv = np.linalg.pinv(np.dot(grad_n.T, grad_n))

    step_tol_failed = False
    convergence = False
    nit, nfev, njev = 0, 1, 1
    message = ""

    while True:
        old_g = g.copy()
        old_res = res

        # search direction
        d = -Hinv.dot(g)

        # Define functions for scipy's line_search
        def f(x):
            return loglik_fn(x, *args, **{"return_gradient": False})

        def fprime(x):
            _, grad, _ = loglik_fn(x, *args, **{"return_gradient": True})
            return grad

        # Perform line search along direction d
        try:
            ls_result = line_search(f, fprime, x, d, g)
            step = ls_result[0]  # alpha

            if step is None or step < step_tol:
                step_tol_failed = True
                message = "Local search could not find a higher log likelihood value"
                break

            # update position
            s = step * d
            x = x + s
            # Evaluate at new position
            res, g, grad_n = loglik_fn(x, *args, **{"return_gradient": True})
            njev += 1
            nfev += ls_result[3] if ls_result[3] is not None else 1
        except Exception as e:
            step_tol_failed = True
            message = f"Line search failed: {e}"
            break

        nit += 1

        # Check for NaN/Inf
        if not np.isfinite(res) or not np.all(np.isfinite(g)):
            convergence = False
            message = "Encountered NaN or Inf values"
            break

        gproj = np.abs(np.dot(d, old_g))
        g_norm = np.linalg.norm(g)
        if disp:
            print(
                f"Iteration: {nit} \t Log-Lik.= {res:.3f} \t |proj g|= {gproj:e} \t norm(g) = {g_norm:e}"
            )
            if nit % 50 == 0:
                print(f"Current parameter values: {x}")

        # if g_norm < gtol:
        if gproj < gtol:
            convergence = True
            message = "The gradients are close to zero"
            break

        if np.abs(res - old_res) < tol:
            convergence = False
            message = "Successive log-likelihood values within tolerance limits"
            break

        if nit > maxiter:
            convergence = False
            message = "Maximum number of iterations reached without convergence"
            break

        # BFGS update
        delta_g = g - old_g
        # Check for positive curvature condition
        s_dot_y = s.dot(delta_g)
        if s_dot_y <= 1e-14:
            if disp:
                print("Warning: Skipping BFGS update due to non-positive curvature")
            continue

        ###
        # Hinv = (
        #     Hinv
        #     + (
        #         ((s_dot_y + (delta_g[None, :].dot(Hinv)).dot(delta_g)) * np.outer(s, s))
        #         / (s_dot_y) ** 2
        #     )
        #     - (
        #         (np.outer(Hinv.dot(delta_g), s) + (np.outer(s, delta_g)).dot(Hinv))
        #         / s_dot_y
        #     )
        # )
        ###
        # # Standard BFGS update formula
        rho = 1.0 / s_dot_y
        Id = np.eye(len(x))
        V = Id - rho * np.outer(s, delta_g)
        Hinv = V @ Hinv @ V.T + rho * np.outer(s, s)
        ###

    if step_tol_failed:
        convergence = False

    if disp:
        print(f"Convergence = {convergence}. {message}.")

    # Hinv = np.linalg.pinv(np.dot(grad_n.T, grad_n))
    results = {
        "success": convergence,
        "x": x,
        "fun": res,
        "message": message,
        "hess_inv": Hinv,
        "grad_n": grad_n,
        "grad": g,
        "nit": nit,
        "nfev": nfev,
        "njev": njev,
    }

    if not convergence and restart:
        print("Restarting optimization with I as initial Hessian.")
        results = _bfgs(
            loglik_fn, x, args, maxiter, tol, gtol, step_tol, disp, hinv_init="identity"
        )

    return results


def _minimize(loglik_fn, x, args, method, tol, options, bounds=None):
    if method == "BFGS":
        if bounds is not None:
            print(
                "Bounds set but optimization method BFGS will ignore these. Use L-BFGS-B for constrained optimization."
            )
        return _bfgs(loglik_fn, x, args=args, tol=tol, **options)
    elif method == "L-BFGS-B":
        return minimize(
            loglik_fn,
            x,
            args=args,
            jac=True,
            method="L-BFGS-B",
            tol=tol,
            options=options,
            bounds=bounds,
        )
    elif method == "BFGS-scipy":
        return minimize(
            loglik_fn,
            x,
            args=args,
            jac=True,
            method="BFGS",
            options=options,
        )
    else:
        raise ValueError(f"Unknown optimization method: {method}")


def _numerical_hessian(x, fn, args):
    H = np.empty((len(x), len(x)))
    eps = 1.4901161193847656e-08  # From scipy 1.8 defaults

    for i in range(len(x)):
        fn_call = lambda x_: fn(x_, *args)[1][i]
        hess_row = approx_fprime(x, fn_call, epsilon=eps)
        H[i, :] = hess_row

    Hinv = np.linalg.inv(H)
    return Hinv
