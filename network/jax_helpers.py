"""
Contains the implementation of functions that do not natively
exist in JAX.
"""
import functools as ft
import jax
import jax.numpy as jnp
import scipy.integrate


@ft.partial(jax.custom_vjp, nondiff_argnums=(0,))
def quad(func, a, b, args=()):
    """
    Calculates the integral
    int_a^b func(t, *args) dt
    """
    result, _ = scipy.integrate.quad(func, a, b, args)

    return result


def quad_fwd(func, a, b, args=()):
    result = quad(func, a, b, args)
    aux = (a, b, args)

    return result, aux


def quad_bwd(func, aux, grad):
    a, b, args = aux

    grad_a = -grad * func(a, *args)
    grad_b = grad * func(b, *args)

    grad_args = []
    for i in range(len(args)):

        def _vjp_func(_t, *_args):
            return jax.grad(func, i)(_t, *_args)

        grad_args.append(grad * quad(_vjp_func, a, b, args))
    grad_args = tuple(grad_args)

    return grad_a, grad_b, grad_args


"""
Custom VJP functions wrapped around scipy.integrate.quad
See:
https://github.com/google/jax/issues/9014#issuecomment-998989965
"""
quad.defvjp(quad_fwd, quad_bwd)
