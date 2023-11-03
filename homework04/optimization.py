import numpy as np
from datetime import datetime
from collections import defaultdict

class LineSearchTool:
    def __init__(self, method='Backtracking', **kwargs):
        self._method = method
        if self._method == 'Backtracking':
            self.rho = kwargs.get('rho', 0.5)  # Коэффициент уменьшения шага (обычно 0.5)
            self.c = kwargs.get('c', 1e-4)    # Параметр условия Армихо

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        alpha = 1.0 if previous_alpha is None else previous_alpha

        while True:
            f_val = oracle.func(x_k + alpha * d_k)
            grad_val = oracle.grad(x_k + alpha * d_k)

            if f_val <= oracle.func(x_k) + self.c * alpha * np.dot(grad_val, d_k):
                return alpha  # Успешно найден подходящий шаг

            alpha *= self.rho  # Уменьшаем шаг и продолжаем итерации

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def backtracking_line_search(self, oracle, x_k, d_k, previous_alpha=None):
        alpha = 1.0 if previous_alpha is None else previous_alpha
        rho = 0.5  # Коэффициент уменьшения шага (обычно 0.5)
        c = 1e-4   # Параметр условия Армихо

        while True:
            f_val = oracle.func(x_k + alpha * d_k)
            grad_val = oracle.grad(x_k + alpha * d_k)

            if f_val <= oracle.func(x_k) + c * alpha * np.dot(grad_val, d_k):
                return alpha  # Успешно найден подходящий шаг

            alpha *= rho  # Уменьшаем шаг и продолжаем итерации

def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if isinstance(line_search_options, LineSearchTool):
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000, line_search_options=None, trace=False, display=False):
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    history = defaultdict(list) if trace else None

    for iteration in range(max_iter):
        gradient = oracle.grad(x_k)
        alpha = line_search_tool.line_search(oracle, x_k, -gradient)
        x_k -= alpha * gradient

        if trace:
            history['time'].append(datetime.now())
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(np.linalg.norm(gradient))
            history['x'].append(x_k.copy())

        if np.linalg.norm(gradient) < tolerance:
            if display:
                print(f"Gradient Descent successfully converged after {iteration} iterations.")
            return x_k, 'success', history

    if display:
        print("Gradient Descent did not converge.")
    return x_k, 'iterations_exceeded', history or defaultdict(list)  # Ensure history is a dictionary even if no iterations were performed


# Пример использования:

class Oracle:
    def func(self, x):
        return np.sum(x ** 2)

    def grad(self, x):
        return 2 * x

    def func_directional(self, x, d, alpha):
        return self.func(x + alpha * d)

    def grad_directional(self, x, d, alpha):
        return np.dot(self.grad(x + alpha * d), d)

oracle = Oracle()
x_0 = np.zeros(10)
result, status, history = gradient_descent(oracle, x_0, trace=True, display=True)


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    start_time = datetime.now()

    for iteration in range(max_iter):
        # Compute the gradient at the current point
        gradient = oracle.grad(x_k)
        grad_norm = np.linalg.norm(gradient)
        
        # Stopping criterion: check if the norm of the gradient is small enough
        if grad_norm < tolerance:
            if trace:
                history['time'].append((datetime.now() - start_time).total_seconds())
                history['func'].append(oracle.func(x_k))
                history['grad_norm'].append(grad_norm)
                history['x'].append(x_k.copy())
            return x_k, 'success', history
        
        # Compute the search direction using the negative gradient
        search_direction = -gradient
        
        # Perform line search to find the optimal step size
        alpha = line_search_tool.line_search(oracle, x_k, search_direction)
        
        # Update the current point
        x_k += alpha * search_direction
        
        if trace:
            history['time'].append((datetime.now() - start_time).total_seconds())
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(grad_norm)
            history['x'].append(x_k.copy())

    return x_k, 'iterations_exceeded', history


def newton(oracle, x_0, tolerance=1e-5, max_iter=100, line_search_options=None, trace=False, display=False):
    # ... ваша реализация ...

    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    start_time = datetime.now()

    for iteration in range(max_iter):
        # Compute the gradient and Hessian at the current point
        gradient = oracle.grad(x_k)
        hessian = oracle.hess(x_k)
        grad_norm = np.linalg.norm(gradient)
        
        # Stopping criterion: check if the norm of the gradient is small enough
        if grad_norm < tolerance:
            if trace:
                history['time'].append((datetime.now() - start_time).total_seconds())
                history['func'].append(oracle.func(x_k))
                history['grad_norm'].append(grad_norm)
                history['x'].append(x_k.copy())
            return x_k, 'success', history
        
        try:
            # Solve the Newton system for the search direction
            search_direction = np.linalg.solve(hessian, -gradient)
        except LinAlgError:
            # Newton system is not solvable (non-invertible Hessian)
            if trace:
                history['time'].append((datetime.now() - start_time).total_seconds())
                history['func'].append(oracle.func(x_k))
                history['grad_norm'].append(grad_norm)
                history['x'].append(x_k.copy())
            return x_k, 'newton_direction_error', history
        
        # Perform line search to find the optimal step size
        alpha = line_search_tool.line_search(oracle, x_k, search_direction)
        
        # Update the current point
        x_k += alpha * search_direction
        
        if trace:
            history['time'].append((datetime.now() - start_time).total_seconds())
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(grad_norm)
            history['x'].append(x_k.copy())

    return x_k, 'iterations_exceeded', history
