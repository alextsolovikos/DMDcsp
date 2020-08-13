# DMDcsp
Code developed for the paper *A. Tsolovikos, E. Bakolas, S. Suryanarayanan and D. Goldstein, "Estimation and Control of Fluid Flows Using Sparsity-Promoting Dynamic Mode Decomposition," in IEEE Control Systems Letters, doi: 10.1109/LCSYS.2020.3015776.*

The paper can be found [here](https://ieeexplore.ieee.org/document/9164896).

### Abstract:

Control and estimation of fluid systems is a challenging problem that requires approximating high-dimensional, nonlinear dynamics with computationally tractable models. A number of techniques, such as proper orthogonal decomposition (POD) and dynamic mode decomposition (DMD) have been developed to derive such reduced-order models. In this paper, the problem of selecting the dynamically important modes of dynamic mode decomposition with control (DMDc) is addressed. Similar to sparsity-promoting DMD, the method described in this work solves a convex optimization problem in order to determine the most important modes. The proposed algorithm produces sparse dynamical models for systems with inputs by solving a regularized least-squares problem that minimizes the reweighted L1 norm of relative mode weights and can work even with snapshot data that are not sequential. In addition, the process of estimating the modeling errors and designing a Kalman filter for flow estimation from limited measurements is presented. The method is demonstrated in the control and estimation of the unsteady wake past an inclined flat plate in a high-fidelity direct numerical simulation.


### Results:

- Stabilizing the wake behind a flat plate at angle a = 20deg and Reynolds number Re = 250:

![](animations/flat_plate_wake.gif)

