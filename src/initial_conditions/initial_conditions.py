class InitialConditions:
    def __init__(self, U0=None, V0=None, A0=None):
        self.U0 = U0
        self.V0 = V0
        self.A0 = A0

        self.U0_f = None
        self.V0_f = None
        self.A0_f = None

    def apply_dirichlet(self, free_dofs):
        if self.U0 is not None:
            self.U0_f = self.U0[free_dofs]
        if self.V0 is not None:
            self.V0_f = self.V0[free_dofs]
        if self.A0 is not None:
            self.A0_f = self.A0[free_dofs]
