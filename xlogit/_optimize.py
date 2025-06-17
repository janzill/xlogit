import numpy as np
from scipy.optimize import minimize, approx_fprime, line_search

def _bfgs(loglik_fn, x, args, maxiter=2000, tol=1e-10, gtol=1e-6, step_tol=1e-10, disp=False):
    """BFGS optimization routine."""
    
    res, g, grad_n = loglik_fn(x, *args, **{'return_gradient': True})

    Hinv = np.linalg.pinv(np.dot(grad_n.T, grad_n))
    # Hinv = np.eye(len(g))

    convergence = False
    step_tol_failed = False
    nit, nfev, njev = 0, 1, 1
    while True:
        old_g = g

        d = -Hinv.dot(g)

        # Define functions for scipy's line_search
        def f(x):
            return loglik_fn(x, *args, **{"return_gradient": False})

        def fprime(x):
            _, grad, _ = loglik_fn(x, *args, **{"return_gradient": True})
            return grad

        # Perform line search along direction d
        ls_result = line_search(f, fprime, x, d, g)
        step = ls_result[0]  # alpha

        if step is None or step < step_tol:
            convergence = False
            message = "Local search could not find a higher log likelihood value"
            # step_tol_failed = True
            break

        s = step * d
        x = x + s
        resnew, gnew, grad_n = loglik_fn(x, *args, **{"return_gradient": True})
        njev += 1
        nfev += ls_result[3] if ls_result[3] is not None else 1

        nit += 1

        # if step_tol_failed:
        #     convergence = False
        #     message = "Local search could not find a higher log likelihood value"
        #     break
        
        old_res = res
        res = resnew
        g = gnew
        gproj = np.abs(np.dot(d, old_g))
        g_norm = np.linalg.norm(g)
        
        if disp:
            print(
                f"Iteration: {nit} \t Log-Lik.= {resnew:.3f} \t |proj g|= {gproj:e} \t norm(g) = {g_norm:e}"
            )

        # if gproj < gtol:
        if g_norm < gtol:
            convergence = True
            message = "The gradients are close to zero"
            break

        if np.abs(res - old_res) < tol:
            convergence = True
            message = "Successive log-likelihood values within tolerance limits"
            break

        if nit > maxiter:
            convergence = False
            message = "Maximum number of iterations reached without convergence"
            break

        delta_g = g - old_g

        Hinv = Hinv + (((s.dot(delta_g) + (delta_g[None, :].dot(Hinv)).dot(
            delta_g))*np.outer(s, s)) / (s.dot(delta_g))**2) - ((np.outer(
                Hinv.dot(delta_g), s) + (np.outer(s, delta_g)).dot(Hinv)) /
                (s.dot(delta_g)))

    # Hinv = np.linalg.pinv(np.dot(grad_n.T, grad_n))
    if disp:
        print(f"Convergence = {convergence}. {message}.")

    return {'success': convergence, 'x': x, 'fun': res, 'message': message,
            'hess_inv': Hinv, 'grad_n':grad_n, 'grad':g, 'nit': nit, 'nfev': nfev, 'njev': njev}
    
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
    eps = 1.4901161193847656e-08 # From scipy 1.8 defaults
    
    for i in range(len(x)):
        fn_call = lambda x_: fn(x_, *args)[1][i]
        hess_row = approx_fprime(x, fn_call, epsilon=eps)
        H[i, :] = hess_row
    
    Hinv = np.linalg.inv(H)
    return Hinv
