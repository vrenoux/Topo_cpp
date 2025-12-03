#include "linalg.h"
#include <algorithm>
#include <numeric>
#include <cassert>
#include <iostream>

namespace fem {

// SparseMatrixCOO
SparseMatrixCOO::SparseMatrixCOO(uint32_t n_) : n(n_) {
        size_t estimated_nnz = n * 10; // exemple : 10 entrées par ligne
        rows.reserve(estimated_nnz);
        cols.reserve(estimated_nnz);
        vals.reserve(estimated_nnz);
}

void SparseMatrixCOO::add(uint32_t i, uint32_t j, double v) {
    rows.push_back(i);
    cols.push_back(j);
    vals.push_back(v);
}

size_t SparseMatrixCOO::get_nnz() const { return vals.size(); }


// SparseMatrixCSR
SparseMatrixCSR::SparseMatrixCSR() = default;

void SparseMatrixCSR::build_from_coo(const SparseMatrixCOO& coo, bool sum_duplicates) {
    nrows = coo.n;
    ncols = coo.n; // carré

    const size_t nnz_in = coo.get_nnz();
    row_offsets.assign(nrows + 1, 0);
    col_indices.clear();
    values.clear();

    if (nnz_in == 0) {
        return;
    }

    // Sanity check indices
    for (size_t k = 0; k < nnz_in; ++k) {
        assert(coo.rows[k] < nrows && "row index out of bounds");
        assert(coo.cols[k] < ncols && "col index out of bounds");
    }

    // (1) Permutation pour trier par (row, col)
    std::vector<size_t> idx(nnz_in);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
        if (coo.rows[a] != coo.rows[b]) return coo.rows[a] < coo.rows[b];
        return coo.cols[a] < coo.cols[b];
    });

    // (2) Fusion des doublons consécutifs
    std::vector<uint32_t> fused_cols;
    std::vector<double>   fused_vals;
    std::vector<uint32_t> row_counts(nrows, 0);

    fused_cols.reserve(nnz_in);
    fused_vals.reserve(nnz_in);

    uint32_t curr_row = coo.rows[idx[0]];
    uint32_t curr_col = coo.cols[idx[0]];
    double   curr_val = coo.vals[idx[0]];

    auto flush_entry = [&](uint32_t r, uint32_t c, double v) {
        fused_cols.push_back(c);
        fused_vals.push_back(v);
        row_counts[r] += 1;
    };

    for (size_t t = 1; t < nnz_in; ++t) {
        const uint32_t r = coo.rows[idx[t]];
        const uint32_t c = coo.cols[idx[t]];
        const double   v = coo.vals[idx[t]];

        if (r == curr_row && c == curr_col) {
            if (sum_duplicates) {
                curr_val += v;
            } else {
                flush_entry(curr_row, curr_col, curr_val);
                curr_row = r; curr_col = c; curr_val = v;
            }
        } else {
            flush_entry(curr_row, curr_col, curr_val);
            curr_row = r; curr_col = c; curr_val = v;
        }
        std::cout << "Processing entry " << t + 1 << " / " << nnz_in << "\r" << std::flush;
    }
    flush_entry(curr_row, curr_col, curr_val);

    // (3) Construire row_offsets par cumul
    row_offsets[0] = 0;
    for (uint32_t i = 0; i < nrows; ++i) {
        row_offsets[i + 1] = row_offsets[i] + row_counts[i];
    }

    // (4) Déplacer les données
    col_indices = std::move(fused_cols);
    values      = std::move(fused_vals);

    // Calcul des pointeurs vers la diagonale
    compute_diag_ptr();

    // Invariants
    assert(row_offsets.size() == nrows + 1);
    assert(row_offsets.back() == col_indices.size());
    assert(col_indices.size() == values.size());

    #ifndef NDEBUG
    for (uint32_t r = 0; r < nrows; ++r) {
        const uint32_t start = row_offsets[r];
        const uint32_t end   = row_offsets[r+1];
        for (uint32_t k = start + 1; k < end; ++k) {
            assert(col_indices[k-1] <= col_indices[k] && "cols must be non-decreasing within rows");
        }
    }
    #endif
}

void SparseMatrixCSR::compute_diag_ptr() {
    diag_ptr.resize(nrows);
    for (uint32_t r = 0; r < nrows; ++r) {
        uint32_t start = row_offsets[r];
        uint32_t end   = row_offsets[r+1];
        for (uint32_t k = start; k < end; ++k) {
            if (col_indices[k] == r) {
                diag_ptr[r] = k;
                break;
            }
        }
    }
}

void SparseMatrixCSR::apply_dirichlet_batch(const std::vector<uint32_t>& dofs) {
    std::vector<bool> is_constrained(nrows, false);
    for (auto d : dofs) {
        assert(d < nrows);
        is_constrained[d] = true;
    }

    for (uint32_t r = 0; r < nrows; ++r) {
        uint32_t start = row_offsets[r];
        uint32_t end   = row_offsets[r+1];
        for (uint32_t k = start; k < end; ++k) {
            uint32_t c = col_indices[k];
            if (is_constrained[r] || is_constrained[c]) {
                values[k] = 0.0;
            }
        }
    }

    for (auto d : dofs) {
        values[diag_ptr[d]] = 1.0;
    }
    // Nettoyage mémoire : après dirichlet fin de modifications de la CSR et résolution
    shrink_to_fit_all();
}

std::vector<double> SparseMatrixCSR::multiply(const std::vector<double> &x) const {
    assert(x.size() == static_cast<size_t>(ncols));
    std::vector<double> y(nrows, 0.0);

    // Parcours des lignes
    for (uint32_t r = 0; r < nrows; ++r)
    {
        const uint32_t start = row_offsets[r];
        const uint32_t end = row_offsets[r + 1];

        double acc = 0.0;
        // Accumule la contribution de la ligne r
        for (uint32_t k = start; k < end; ++k)
        {
            acc += values[k] * x[col_indices[k]];
        }
        y[r] = acc;
    }
    return y;
}



} // namespace fem
