#pragma once

#include <vector>

class MMA {
public:

    MMA(int n_vars, int m_constrs);

    ~MMA();

    void SetArtificialParameters(double a0_param,
                                 std::vector<double> a_params,
                                 std::vector<double> c_params,
                                 std::vector<double> d_params) {
        a0 = a0_param;  // Default 1.0
        ai = a_params;  // Default 0.0
        ci = c_params;  // Default 1000.0
        di = d_params;  // Default 1.0
    }

    void SetAsymptoteParameters(double init, double dec, double inc) {
        asyminit = init; // Default 0.5
        asymdec = dec;   // Default 0.7
        asyminc = inc;   // Default 1.2
    }

    void SetMoveFactor(double move_factor, double albefa_factor) {
        move = move_factor;     // Default 0.5
        albefa = albefa_factor; // Default 0.1
    }

    void Update(std::vector<double>& xval,
                const std::vector<double>& dfdx,
                const std::vector<double>& gval,
                const std::vector<double>& dgdx,
                const std::vector<double>& xmin,
                const std::vector<double>& xmax);

    int GetIter() const { return iter; }

private:
    int n;      // Design variables
    int m;      // Constraints
    int iter = 0;   // Iteration counter

    double raa0 = 1e-5;     // raa0
	double move = 0.5;      // move factor of alpha and beta
    double albefa = 0.1;    // albefa factor of alpha and beta
    
    double asyminit = 0.5;    // Initial asymptote expansion
    double asymdec = 0.7;     // Asymptote decrease expansion
    double asyminc = 1.2;     // Asymptote increase expansion

    double a0 = 1.0;            // Artificial parameter sub-problem a0 > 0
    std::vector<double> ai;     // Artificial parameters sub-problem ai >= 0
    std::vector<double> ci;     // Artificial parameters sub-problem ci >= 0, ai*ci > 0
    std::vector<double> di;     // Artificial parameters sub-problem di >= 0, ci + di > 0
 
    // State vectors
    std::vector<double> xold1, xold2;
    std::vector<double> low, upp;
    std::vector<double> alpha, beta;

    // Calculation vectors (Sub-problem)
    std::vector<double> p0, q0;
    std::vector<double> pij, qij;
    std::vector<double> b;

    // Dual vectors
    std::vector<double> lam, y, eta, s;
    std::vector<double> grad, hess;
    double z;

    void GenSub(const std::vector<double>& xval,
                const std::vector<double>& dfdx,
                const std::vector<double>& gval,
                const std::vector<double>& dgdx,
                const std::vector<double>& xmin,
                const std::vector<double>& xmax);

    void SolveDIP(std::vector<double>& xval);       // Dual Interior Point Solver
    void XYZofLAMBDA(std::vector<double>& lam, std::vector<double>& xval);
    void DualGrad(const std::vector<double>& xval);
    void DualHess(const std::vector<double>& xval);
    void FactorizeCholeskyRobust(std::vector<double>& K, int m); // Cholesky factorization
    void SolveCholesky(const std::vector<double>& K, std::vector<double>& s, int m);
    void DualLineSearch();
    double DualResidual(const std::vector<double>& xval, double epsi);

    void AllocateWorkspace();
    void FreeWorkspace();

    void FactorizeLU(std::vector<double>& K, int m);
    void SolveLU(const std::vector<double>& K, std::vector<double>& x, int m);
};