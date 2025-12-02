// ----------------------------------------------------------------------------
// main pour tester linalg
// ----------------------------------------------------------------------------
#include "../src/linalg.h"
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>


using namespace fem;

// Fonction pour comparer deux vecteurs avec tolérance
bool compare_vectors(const std::vector<double>& a, const std::vector<double>& b, double tol = 1e-9) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::fabs(a[i] - b[i]) > tol) return false;
    }
    return true;
}


int main() {
    std::cout << "=== Test SparseMatrixCOO ===\n";

    // Test 1 : Créer une matrice COO 3x3
    SparseMatrixCOO coo(3);
    assert(coo.n == 3 && "Erreur: n incorrect");
    std::cout << "✓ Matrice COO créée (3x3)\n";

    // Test 2 : Ajouter des éléments
    coo.add(0, 0, 4.0);
    coo.add(0, 1, 1.0);
    coo.add(1, 0, 1.0);
    coo.add(1, 1, 3.0);
    coo.add(1, 2, 1.0);
    coo.add(2, 1, 1.0);
    coo.add(2, 2, 2.0);
    assert(coo.get_nnz() == 7 && "Erreur: nombre d'éléments incorrect");
    std::cout << "✓ " << coo.get_nnz() << " éléments ajoutés\n";
    
    // Test 3 : Conversion en CSR
    std::cout << "\n=== Test Conversion COO -> CSR ===\n";
    SparseMatrixCSR csr;
    csr.build_from_coo(coo, true);
    assert(csr.nrows == 3 && csr.ncols == 3 && "Erreur: dimensions CSR incorrectes");
    assert(csr.values.size() == 7 && "Erreur: nombre de valeurs CSR incorrect");
    std::cout << "✓ Conversion COO -> CSR réussie\n";

    // Test 4 : Multiplication A * x
    std::cout << "\n=== Test Multiplication Matrice-Vecteur ===\n";
    std::vector<double> x = {1.0, 2.0, 3.0};
    std::vector<double> y = csr.multiply(x);
    std::vector<double> expected_y = {6.0, 10.0, 8.0};
    assert(compare_vectors(y, expected_y) && "Erreur: résultat de la multiplication incorrect");
    std::cout << "✓ Multiplication réussie\n";

    // Test 5 : Appliquer Dirichlet
    std::cout << "\n=== Test apply_dirichlet_batch ===\n";

    SparseMatrixCOO coo2(3);
    coo2.add(0, 0, 4.0);
    coo2.add(0, 1, 1.0);
    coo2.add(1, 0, 1.0);
    coo2.add(1, 1, 3.0);
    coo2.add(1, 2, 1.0);
    coo2.add(2, 1, 1.0);
    coo2.add(2, 2, 2.0);

    SparseMatrixCSR csr2;
    csr2.build_from_coo(coo2, true);
    std::vector<uint32_t> constrained_dofs = {0};
    csr2.apply_dirichlet_batch(constrained_dofs);

    // Vérifier que la première ligne est bien modifiée (par exemple, diagonale = 1)
    assert(std::fabs(csr2.values[0] - 1.0) < 1e-9 && "Erreur: Dirichlet non appliqué correctement");
    std::cout << "✓ Dirichlet appliqué\n";

    std::cout << "\n=== Tous les tests sont passés ! ===\n";
    return 0;
}
