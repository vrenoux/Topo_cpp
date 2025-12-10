import sys
import matplotlib.pyplot as plt
from petsc4py import PETSc
import petsc4py
import numpy as np

# --- 0) Initialisation PETSc
petsc4py.init(sys.argv)  # initialise PETSc/petsc4py (options en ligne de commande si besoin)
comm = PETSc.COMM_WORLD
rank = comm.getRank()
size = comm.getSize()


FILENAME = "A.dat"  # adapte le nom/chemin si nécessaire

viewer = PETSc.Viewer().createBinary(FILENAME, mode='r', comm=comm)

A = PETSc.Mat().create(comm=comm)
A.setFromOptions()
A.load(viewer)  # charge la matrice depuis le binaire

m, n = A.getSize()
mrows = list(range(m))
ncols = list(range(n))

# On tente d'abord la voie "sparse AIJ" -> SciPy CSR -> dense (rapide et standard)
A_type = A.getType()
A_numpy = None          # ndarray dense pour Spyder
M_scipy = None          # CSR SciPy pour le spy plot (si AIJ)

indptr, indices, data = A.getValuesCSR()
from scipy.sparse import csr_matrix
M_scipy = csr_matrix((data, indices, indptr), shape=(m, n))
nnz_per_row = np.diff(M_scipy.indptr)
A_numpy = M_scipy.toarray() 


plt.figure(figsize=(6, 6))
if M_scipy is not None:
    plt.spy(M_scipy, markersize=0.5)
else:
    # Matplotlib spy accepte aussi un ndarray; les éléments != 0 seront marqués
    plt.spy(A_numpy, markersize=0.5)

plt.title("Spy(A) – structure des non‑zéros")
plt.xlabel("Colonnes")
plt.ylabel("Lignes")
plt.tight_layout()
plt.show()

print(f"A lue depuis '{FILENAME}': taille = {m} x {n}, type PETSc = {A_type}")
print(f"A_numpy (dense) pour Spyder: shape={A_numpy.shape}, dtype={A_numpy.dtype}")
