#include "MMA.h"

#include <cmath>
#include <iostream>

// Constructor
MMA::MMA(int n_vars, int m_constrs)
      : n(n_vars),
        m(m_constrs),
        // Default values :
        ai(m, 0.0),
        ci(m, 1000.0),  // or 1e4
        di(m, 1.0),

        // Sizing
        xold1(n),
        xold2(n),
        low(n),
        upp(n),
        alpha(0),
        beta(0),
        p0(0),
        q0(0),
        pij(0),
        qij(0),
        b(m, 0.0),
        lam(m),
        y(m),
        eta(m),
        s(m * 2),
        grad(m, 0.0),
        hess(m * m, 0.0) {
}


// Destructor
MMA::~MMA() {
}

// ============================================================================
// Public
// ============================================================================

void MMA::Update(std::vector<double>& xval,
                 const std::vector<double>& dfdx,
                 const std::vector<double>& gval,
                 const std::vector<double>& dgdx,
                 const std::vector<double>& xmin,
                 const std::vector<double>& xmax) {

    iter++;

    AllocateWorkspace();

    GenSub(xval, dfdx, gval, dgdx, xmin, xmax);

    xold2 = xold1;
    xold1 = xval;

    SolveDIP(xval);

    FreeWorkspace();
}

// ============================================================================
// Private
// ============================================================================

void MMA::GenSub(const std::vector<double>& xval,
                 const std::vector<double>& dfdx,
                 const std::vector<double>& gval,
                 const std::vector<double>& dgdx,
                 const std::vector<double>& xmin,
                 const std::vector<double>& xmax) {
    
    std::fill(b.begin(), b.end(), 0.0);

    // Set asymptotes
    if (iter < 3) {
		for (int i = 0; i < n; i++) {
			low[i] = xval[i] - asyminit * (xmax[i] - xmin[i]);
			upp[i] = xval[i] + asyminit * (xmax[i] - xmin[i]);
        }
    } else {
        for (int i = 0; i < n; i++) {
			double dir = (xval[i] - xold1[i]) * (xold1[i] - xold2[i]);
			double gamma;
			if (dir < 0.0) {
				gamma = asymdec;
			} else if (dir > 0.0) {
				gamma = asyminc;
			} else {
				gamma = 1.0;
			}

			low[i] = xval[i] - gamma * (xold1[i] - low[i]);
			upp[i] = xval[i] + gamma * (upp[i] - xold1[i]);

			double xmami = std::max(1.0e-5, xmax[i] - xmin[i]); // if forced value : xmax = xmin;

			low[i] = std::max(low[i], xval[i] - 100.0 * xmami);
			low[i] = std::min(low[i], xval[i] - 1.0e-5 * xmami);
			upp[i] = std::max(upp[i], xval[i] + 1.0e-5 * xmami);
			upp[i] = std::min(upp[i], xval[i] + 100.0 * xmami);

			double xmi = xmin[i] - 1.0e-6;
			double xma = xmax[i] + 1.0e-6;

			if (xval[i] < xmi) {
				low[i] = xval[i] - (xma - xval[i]) / 0.9;
				upp[i] = xval[i] + (xma - xval[i]) / 0.9;
			}
			if (xval[i] > xma) {
				low[i] = xval[i] - (xval[i] - xmi) / 0.9;
				upp[i] = xval[i] + (xval[i] - xmi) / 0.9;
			}
        }    
    }

    // Set bounds and the coefficients for the approximation
	for (int i = 0; i < n; ++i) {
        // Precompute
        const double xmamiinv = 1.0 / std::max(1.0e-5, xmax[i] - xmin[i]);
        const double ux = upp[i] - xval[i];
        const double xl = xval[i] - low[i];
        const double ux2 = ux * ux;
        const double xl2 = xl * xl;

		// Compute bounds alpha and beta
		alpha[i] = std::max(xmin[i], low[i] + albefa * xl);
		alpha[i] = std::max(alpha[i], xval[i] - move * (xmax[i] - xmin[i]));
		alpha[i] = std::min(alpha[i], xmax[i]);
		beta[i] = std::min(xmax[i], upp[i] - albefa * ux);
		beta[i] = std::min(beta[i], xval[i] + move * (xmax[i] - xmin[i]));
		beta[i] = std::max(beta[i], xmin[i]);

        // Objective function
        double dfdxp = std::max(0.0, dfdx[i]);
        double dfdxm = std::max(0.0, -1.0 * dfdx[i]);
        p0[i]=ux2 * (1.001 * dfdxp + 0.001 * dfdxm + raa0 * xmamiinv);
        q0[i]=xl2 * (0.001 * dfdxp + 1.001 * dfdxm + raa0 * xmamiinv);

		// Constraints
		for (int j = 0; j < m; j++) {
			double dgdxp = std::max(0.0, dgdx[i * m + j]);
			double dgdxm = std::max(0.0, -1.0 * dgdx[i * m + j]);
            pij[i * m + j]=ux2 * (1.001 * dgdxp + 0.001 * dgdxm + raa0 * xmamiinv);
            qij[i * m + j]=xl2 * (0.001 * dgdxp + 1.001 * dgdxm + raa0 * xmamiinv);

            // Constant for constraints
            b[j] += pij[i * m + j] / ux + qij[i * m + j] / xl;
		}
	}

    // Constant for the constraints
    for (int i = 0; i < m; i++) {
        b[i] -= gval[i];
    }
}

void MMA::SolveDIP(std::vector<double>& xval) {

    for (int j = 0; j < m; j++) {
        lam[j] = ci[j] / 2.0;
        eta[j] = 1.0;
    }

    double tol = 1.0e-9 * std::sqrt(m + n);
    double mu = 1.0;
    double err = 1.0;
    int loop;
    int max_inner_iter = 100;

    while (mu > tol) {

        loop = 0;

        while (err > 0.9 * mu && loop < max_inner_iter) {
            loop++;

            XYZofLAMBDA(lam, xval);

            DualGrad(xval);

            for (int i = 0; i < m; i++) {
                grad[i] = 1.0 * grad[i] + mu / lam[i]; // Set positive for Cholesky decomposition
            }

            DualHess(xval);

            if (m==1) {
                grad[0] = grad[0] / hess[0];
            } else {
                FactorizeCholeskyRobust(hess, m);
                SolveCholesky(hess, grad, m);
            }

            for (int i = 0; i < m; i++) {
                s[i] = grad[i];
            }

            for (int i = 0; i < m; i++) {
                double dlam = s[i];
                s[m + i] = -eta[i] + mu / lam[i] - dlam * eta[i] / lam[i];
            }
            
            DualLineSearch();

            XYZofLAMBDA(lam, xval);

            err = DualResidual(xval, mu);

        }

        mu = mu * 0.1;
    }
}

void MMA::XYZofLAMBDA(std::vector<double>& lam_val, std::vector<double>& x) {

    // Compute y and lamai
	double lamai = 0.0;
	for (int i = 0; i < m; i++) {
		if (lam_val[i] < 0.0) {
			lam_val[i] = 0;
		}
		y[i] = std::max(0.0, (lam_val[i] - ci[i]) / di[i]);
		lamai += lam_val[i] * ai[i];
	}

    // Compute z
	z = std::max(0.0, 10.0 * (lamai - a0)); 

    // Compute x
	for (int i = 0; i < n; i++) {
		double pjlam = p0[i];
		double qjlam = q0[i];

		for (int j = 0; j < m; j++) {
			pjlam += pij[i * m + j] * lam_val[j];
			qjlam += qij[i * m + j] * lam_val[j];
		}

		x[i] = (sqrt(pjlam) * low[i] + sqrt(qjlam) * upp[i]) / (sqrt(pjlam) + sqrt(qjlam));

        if (x[i] < alpha[i]) x[i] = alpha[i];
        if (x[i] > beta[i])  x[i] = beta[i];
	}
}

void MMA::DualGrad(const std::vector<double>& xval) {

    std::fill(grad.begin(), grad.end(), 0.0);

    for (int i = 0; i < n; i++) {
        double ux_inv = 1.0 / (upp[i] - xval[i]);
        double xl_inv = 1.0 / (xval[i] - low[i]);

        for (int j = 0; j < m; j++) {
            grad[j] += pij[i * m + j] * ux_inv + qij[i * m + j] * xl_inv;
        }
    }

    for (int j = 0; j < m; j++) {
        grad[j] += -b[j] - ai[j]*z - y[j];
    }
}

void MMA::DualHess(const std::vector<double>& xval) {

    std::fill(hess.begin(), hess.end(), 0.0);

    std::vector<double> pq_i(m);

    // Hessian contribution from x
    for (int j = 0; j < n; j++) {
        double pjlam = p0[j];
        double qjlam = q0[j];

        for (int i = 0; i < m; i++) {
            pjlam += pij[j * m + i] * lam[i];
            qjlam += qij[j * m + i] * lam[i];
        }

        double ux = upp[j] - xval[j];
        double xl = xval[j] - low[j];
        double u_inv = 1.0 / ux;
        double l_inv = 1.0 / xl;
        double u2_inv = u_inv * u_inv;
        double l2_inv = l_inv * l_inv;
        double u3_inv = u2_inv * u_inv;
        double l3_inv = l2_inv * l_inv;

        for (int i = 0; i < m; i++) {
            pq_i[i] = pij[j * m + i] * u2_inv - qij[j * m + i] * l2_inv;
        }

        double d2L = 2.0 * (pjlam * u3_inv + qjlam * l3_inv);
        double df2 = -1.0 / d2L;

        double sqrt_p = std::sqrt(pjlam);
        double sqrt_q = std::sqrt(qjlam);
        double xp = (sqrt_p * low[j] + sqrt_q * upp[j]) / (sqrt_p + sqrt_q);
        if (xp < alpha[j] || xp > beta[j]) {
            df2 = 0.0;
        }

        for (int i = 0; i < m; i++) {
            double contribution_j = df2 * pq_i[i]; 
            for (int k = 0; k < m; k++) {
                hess[i * m + k] -= contribution_j * pq_i[k]; // Set negative for Cholesky decomposition
            }
        }
    }

    // Hessian contribution from z
    double lamai = 0.0;
    for (int i =0; i < m; i++) {
        lamai += lam[i] * ai[i];
    }
    if (lamai > 0.0) {
        for (int j = 0; j < m; j++) {
            for (int k = 0; k < m; k++) {
                hess[j * m + k] -= -10.0 * ai[j] * ai[k]; // Set negative for Cholesky decomposition
            }
        }
    }

    // Hessian contribution from y
    for (int i = 0; i < m; i++) {
        if (lam[i] > ci[i]) {
            hess[i * m + i] -= -1.0 / di[i]; // Set negative for Cholesky decomposition
        }
    }

    // Dual Regularization
    for (int i = 0; i < m; i++) {
        if (lam[i] < 0.0) lam[i] = 0.0;
  
        hess[i * m + i] -= -eta[i] / lam[i]; // Set negative for Cholesky decomposition
    }
    
    // Robustification (avoid singular hessian)
    double HessTrace = 0.0;
    for (int i = 0; i < m; i++) HessTrace += hess[i * m + i]; // Positive value
    double HessCorr = 1e-4 * HessTrace / m;
    if (HessCorr < 1.0e-7) HessCorr = 1.0e-7;
    
    for (int i = 0; i < m; i++) {
        hess[i * m + i] += HessCorr; 
    }

}

void MMA::FactorizeCholeskyRobust(std::vector<double>& K, int m) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = K[i * m + j];

            for (int k = 0; k < j; k++) {
                sum -= K[i * m + k] * K[j * m + k];
            }

            if (i == j) {
                if (sum <= 1.0e-14) {
                    sum = 1.0e-9;
                }
                K[i * m + i] = std::sqrt(sum);
            } else {
                K[i * m + j] = sum / K[j * m + j];
            }
        }
    }
}

void MMA::SolveCholesky(const std::vector<double>& K, std::vector<double>& x, int m) {
    for (int i = 0; i < m; i++) {
        double sum = x[i];
        for (int k = 0; k < i; k++) {
            sum -= K[i * m + k] * x[k];
        }
        x[i] = sum / K[i * m + i];
    }

    for (int i = m - 1; i >= 0; i--) {
        double sum = x[i];
        for (int k = i + 1; k < m; k++) {
            sum -= K[k * m + i] * x[k];
        }
        x[i] = sum / K[i * m + i];
    }
}

void MMA::DualLineSearch() {

    double theta_inv = 1.005; 

    for (int i = 0; i < m; i++) {
        if (theta_inv < -1.01 * s[i] / lam[i]) {
            theta_inv = -1.01 * s[i] / lam[i];
        }

        if (theta_inv < -1.01 * s[m + i] / eta[i]) {
            theta_inv = -1.01 * s[m + i] / eta[i];
        }
    }

    double step = 1.0 / theta_inv;

    for (int i = 0; i < m; i++) {
        lam[i] = lam[i] + step * s[i];
        eta[i] = eta[i] + step * s[m + i];
    }
}

double MMA::DualResidual(const std::vector<double>& xval, double mu) {
    std::vector<double> res(2 * m, 0.0);

    for (int j = 0; j < n; j++) {
        double ux = upp[j] - xval[j];
        double xl = xval[j] - low[j];
        
        double inv_ux = 1.0 / ux;
        double inv_xl = 1.0 / xl;

        for (int i = 0; i < m; i++) {
            res[i] += pij[j * m + i] * inv_ux + qij[j * m + i] * inv_xl;
        }
    }

    std::vector<double> global_res(2 * m);

    for(int i=0; i<m; i++) global_res[i] = res[i];

    for (int i = 0; i < m; i++) {

        global_res[i] += -b[i] - ai[i] * z - y[i] + eta[i]; 

        global_res[m + i] = lam[i] * eta[i] - mu;
    }

    double nrI = 0.0;
    for (int i = 0; i < 2 * m; i++) {
        double abs_val = std::abs(global_res[i]);
        if (abs_val > nrI) {
            nrI = abs_val;
        }
    }

    return nrI;
}

void MMA::FactorizeLU(std::vector<double>& K, int m) {
    for (int k = 0; k < m - 1; k++) { 
        for (int i = k + 1; i < m; i++) { 
            K[i * m + k] /= K[k * m + k];
            
            for (int j = k + 1; j < m; j++) {
                K[i * m + j] -= K[i * m + k] * K[k * m + j];
            }
        }
    }
}

void MMA::SolveLU(const std::vector<double>& K, std::vector<double>& x, int m) {
    // Forward Substitution
    for (int i = 1; i < m; i++) {
        double sum = 0.0;
        for (int j = 0; j < i; j++) {
            sum -= K[i * m + j] * x[j];
        }
        x[i] += sum;
    }

    // Backward Substitution
    x[m - 1] /= K[(m - 1) * m + (m - 1)];
    
    for (int i = m - 2; i >= 0; i--) {
        double sum = x[i];
        for (int j = i + 1; j < m; j++) {
            sum -= K[i * m + j] * x[j];
        }
        x[i] = sum / K[i * m + i];
    }
}

// MEMORY 
void MMA::AllocateWorkspace() {

    if (pij.size() != static_cast<size_t>(n * m)) pij.resize(n * m);
    if (qij.size() != static_cast<size_t>(n * m)) qij.resize(n * m);

    if (p0.size() != static_cast<size_t>(n)) p0.resize(n);
    if (q0.size() != static_cast<size_t>(n)) q0.resize(n);

    if (alpha.size() != static_cast<size_t>(n)) alpha.resize(n);
    if (beta.size() != static_cast<size_t>(n)) beta.resize(n);

}

void MMA::FreeWorkspace() {
    pij.clear(); pij.shrink_to_fit(); 
    qij.clear(); qij.shrink_to_fit();

    p0.clear(); p0.shrink_to_fit();
    q0.clear(); q0.shrink_to_fit();

    alpha.clear(); alpha.shrink_to_fit();
    beta.clear(); beta.shrink_to_fit();

}