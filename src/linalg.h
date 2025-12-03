#ifndef LINALG_H
#define LINALG_H

#include <vector>
#include <cstdint>
#include <numeric>
#include <algorithm>
#include <iomanip> 
#include <iostream>

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


    struct VectorDense
    {
        std::vector<double> data;

        explicit VectorDense(size_t n) : data(n, 0.0) {}

        size_t size() const { return data.size(); }

        double &operator[](size_t i) { return data[i]; }
        const double &operator[](size_t i) const { return data[i]; }
    };

inline void print_dense(const SparseMatrixCSR& A, std::ostream& os = std::cout) {
    const uint32_t nrows = A.nrows;
    const uint32_t ncols = A.ncols;

    for (uint32_t i = 0; i < nrows; ++i) {
        const uint32_t start = A.row_offsets[i];
        const uint32_t end   = A.row_offsets[i + 1];

        // Pointeur dans la bande CSR de la ligne i
        uint32_t p = start;

        for (uint32_t j = 0; j < ncols; ++j) {
            double val = 0.0;

            // Si on est toujours dans la bande et que la colonne matche, on lit
            if (p < end && A.col_indices[p] == j) {
                val = A.values[p];
                ++p; // avance au prochain NNZ de la ligne
            }

            os << val << (j + 1 < ncols ? " " : "");
        }
        os << '\n';
    }
}

inline void print_vector_dense(const VectorDense& v, std::ostream& os = std::cout) {
    for (size_t i = 0; i < v.size(); ++i) {
        os << v[i] << (i + 1 < v.size() ? " " : "");
    }
    os << '\n';
}

} // namespace fem

#endif
