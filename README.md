# ELM-Differential-Equations
Extreme Learning Machine For solving Second Order Inhomogeneous Differential Equations Implemented in python
# Extreme Learning Machine (ELM) for Solving Second-Order Inhomogeneous Differential Equations

This repository contains Python code for approximating the solution of a second-order inhomogeneous ordinary differential equation (ODE) with boundary conditions using an Extreme Learning Machine (ELM). It also provides a comparison with a numerical solution obtained using `scipy.integrate.solve_bvp` and analyzes the convergence rate of the ELM.

## Problem Description

The code is designed to solve ODEs of the form:

$$ a_2(x)y''(x) + a_1(x)y'(x) + a_0(x)y(x) = f(x, y(x)) $$

with given boundary conditions:

$$ y(a) = A, \quad y(b) = B $$

Specifically, the provided code demonstrates the solution for the following ODE and boundary conditions:

$$x^2y'' - 2xy' + 2y = 6x^4$$

$$y(0.1) = 1, y(1) = 2$$


## Overview

The core of this project is the implementation of an ELM to approximate the solution `y(x)` of the ODE. The workflow involves:

1.  **Numerical Solution:**  A high-precision numerical solution is obtained using `scipy.integrate.solve_bvp` as a reference for comparison.
2.  **ELM Approximation:** The ODE is transformed into a neural network problem and solved using the ELM algorithm.
3. **Error Analysis:** The mean absolute error (MAE), root mean squared error (RMSE), and the integral of the absolute error between the true and approximated solutions are calculated.
4.  **Heatmap Visualization:** A heatmap shows the log10 of RMSE for various combinations of hidden neurons (N) and training points (m). The heatmap includes a rank overlay of lowest (1) to highest RMSE value.
5.  **Convergence Analysis:** A comparison of the convergence rates based on the MAE and RMSE is performed, showing how the error decreases with increasing training points for a fixed number of hidden neurons.
6.  **Plotting and Tabular Output:** The true and approximated solutions, the absolute error, and tabular values are plotted and printed.

## Code Files

-   `elm_ode_solver.py`: The main Python script containing the ELM solver, numerical solution, and analysis code.
-   `README.md`: This file (describing the project).

## Dependencies

*   `numpy`
*   `matplotlib`
*   `pandas`
*   `scipy`
*   `seaborn`

You can install these packages using pip:

```bash
pip install numpy matplotlib pandas scipy seaborn
