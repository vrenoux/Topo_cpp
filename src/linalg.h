#ifndef LINALG_H
#define LINALG_H

#include <vector>
#include <cstdint>
#include <numeric>
#include <algorithm>

namespace fem
{

    // COO format
    struct SparseMatrixCOO
    {
        std::vector<uint32_t> rows, cols;
        std::vector<double> vals;
        uint32_t n; // matrice carrée n x n

        explicit SparseMatrixCOO(uint32_t n_);
        void add(uint32_t i, uint32_t j, double v);
        size_t get_nnz() const;
    };

    // CSR format

    struct SparseMatrixCSR
    {
        uint32_t nrows = 0;
        uint32_t ncols = 0;

        std::vector<uint32_t> row_offsets; // size = nrows + 1
        std::vector<uint32_t> col_indices; // size = nnz
        std::vector<double> values;        // size = nnz

        std::vector<uint32_t> diag_ptr; // index de la diagonale pour chaque ligne

        SparseMatrixCSR();

        // Conversion COO -> CSR avec tri et fusion
        void build_from_coo(const SparseMatrixCOO &coo, bool sum_duplicates = true);
        
        // Calcul des pointeurs vers la diagonale
        void compute_diag_ptr();

        // Réduit la capacité des vecteurs au minimum nécessaire
        void shrink_to_fit_all()
        {
            col_indices.shrink_to_fit();
            values.shrink_to_fit();
        }

        // Multiplication matrice-vecteur y = A*x
        std::vector<double> multiply(const std::vector<double> &x) const;

        // Apply Dirichlet on the CSR using hard constraints
        void apply_dirichlet_batch(const std::vector<uint32_t> &dofs);
        
    };

}

#endif
