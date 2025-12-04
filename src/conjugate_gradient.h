// Unique include guard for this header (avoid clash with linalg.h)
#ifndef CONJUGATE_GRADIENT_H
#define CONJUGATE_GRADIENT_H

#include <vector>
#include <cstdint>
#include <cmath>
#include <limits>
#include <iostream>
#include <algorithm>
#include <numeric>

#include "linalg.h"

namespace fem {

struct CGResult {
    bool converged = false;
    uint32_t iters = 0;
    double final_res_norm = std::numeric_limits<double>::infinity();
};

inline double dot(const std::vector<double>& a, const std::vector<double>& b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

inline double l2norm(const std::vector<double>& a) {
    return std::sqrt(dot(a, a));
}

inline void axpy(double alpha, const std::vector<double>& x, std::vector<double>& y) {
    const size_t n = x.size();
    for (size_t i = 0; i < n; ++i) y[i] += alpha * x[i];
}

inline void copy_to(const std::vector<double>& src, std::vector<double>& dst) {
    // Ensure dst becomes an exact copy of src. Simpler and safer than
    // assuming dst already has the right size and valid iterators.
    dst = src;
}

// SpMV en place: y = A * x
inline void spmv_csr(const SparseMatrixCSR& A,
                     const std::vector<double>& x,
                     std::vector<double>& y)
{
    // Delegate to CSR multiply implementation (may be optimized there)
    y = A.multiply(x);
}

// Préconditionneur Jacobi: z = D^{-1} r (si diag disponible), sinon z = r
inline void apply_jacobi(const SparseMatrixCSR& A,
                         const std::vector<double>& r,
                         std::vector<double>& z)
{
    const uint32_t n = A.nrows;
    z.resize(n);
    if (A.diag_ptr.size() == n) {
        const double eps = std::numeric_limits<double>::epsilon();
        const size_t vals_sz = A.values.size();
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t dp = A.diag_ptr[i];
            if (dp < vals_sz) {
                const double d = A.values[dp];
                // use a small threshold to consider near-zero diagonal
                if (std::abs(d) > eps) z[i] = r[i] / d;
                else z[i] = r[i];
            } else {
                // diag pointer invalid -> fallback to identity preconditioner
                z[i] = r[i];
            }
        }
    } else {
        // Pas de diag connue: identité
        copy_to(r, z);
    }
}

// CG pour SPD: résout A x = b
inline CGResult conjugate_gradient(const SparseMatrixCSR& A,
                                   const std::vector<double>& b,
                                   std::vector<double>& x,
                                   double rtol = 1e-8,
                                   uint32_t max_iters = 1000,
                                   bool use_jacobi = true,
                                   bool verbose = false)
{
    const uint32_t n = A.nrows;
    if (n == 0 || A.ncols != n || b.size() != n || x.size() != n) {
        std::cerr << "CG: dimensions incompatibles.\n";
        return {};
    }

    // Buffers réutilisés
    std::vector<double> r(n, 0.0), z(n, 0.0), p(n, 0.0), Ap(n, 0.0);

    // r0 = b - A x0
    spmv_csr(A, x, Ap);
    for (uint32_t i = 0; i < n; ++i) r[i] = b[i] - Ap[i];

    const double b_norm = std::max(l2norm(b), 1e-30);
    double r_norm = l2norm(r);

    if (verbose) {
        std::cout << "CG: ||r0||/||b|| = " << (r_norm / b_norm) << "\n";
    }

    if (r_norm / b_norm <= rtol) {
        return {true, 0, r_norm};
    }

    // z0 = M^{-1} r0
    if (use_jacobi) apply_jacobi(A, r, z);
    else copy_to(r, z);

    copy_to(z, p);

    double rz_old = dot(r, z);
    if (!std::isfinite(rz_old) || std::abs(rz_old) < 1e-300) {
        std::cerr << "CG: breakdown initial (r^T M^{-1} r ~ 0).\n";
        return {};
    }

    CGResult result;

    for (uint32_t k = 1; k <= max_iters; ++k) {
        // Ap = A p
        spmv_csr(A, p, Ap);

        const double pAp = dot(p, Ap);
        if (!std::isfinite(pAp) || std::abs(pAp) < 1e-300) {
            std::cerr << "CG: breakdown p^T A p ~ 0.\n";
            break;
        }

        const double alpha = rz_old / pAp;

        // x_{k+1} = x_k + alpha * p
        axpy(alpha, p, x);

        // r_{k+1} = r_k - alpha * Ap
        axpy(-alpha, Ap, r);

        r_norm = l2norm(r);
        const double rel = r_norm / b_norm;
        if (verbose) {
            std::cout << "CG iter " << k << " : ||r||/||b|| = " << rel << "\n";
        }
        if (rel <= rtol) {
            result = {true, k, r_norm};
            break;
        }

        // z_{k+1} = M^{-1} r_{k+1}
        if (use_jacobi) apply_jacobi(A, r, z);
        else copy_to(r, z);

        const double rz_new = dot(r, z);
        if (!std::isfinite(rz_new) || std::abs(rz_new) < 1e-300) {
            std::cerr << "CG: breakdown r^T M^{-1} r ~ 0.\n";
            break;
        }

        const double beta = rz_new / rz_old;
        rz_old = rz_new;

        // p_{k+1} = z_{k+1} + beta * p_k
        for (uint32_t i = 0; i < n; ++i) p[i] = z[i] + beta * p[i];

        result = {false, k, r_norm}; // mis à jour au fil de l’eau
    }

    return result;
}

}   // namespace fem

#endif