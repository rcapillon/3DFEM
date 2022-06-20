class Solver:
    def __init__(self, structure, neumann_BC=None, initial_conditions=None):
        self.structure = structure
        self.neumann_BC = neumann_BC
        self.initial_conditions = initial_conditions

        self.n_total_dofs = structure.n_total_dofs
        self.dirichlet_dofs = structure.n_dirichlet_dofs
        self.n_dirichlet_dofs = structure.n_dirichlet_dofs
        self.free_dofs = structure.free_dofs
        self.n_free_dofs = structure.n_free_dofs
        self.observed_dofs = structure.observed_dofs
        self.n_observed_dofs = structure.n_observed_dofs
