from dataclasses import dataclass

@dataclass
class LSG:
    # Class for storing Location Selection Games

    # Sets
    incumbents: list # I
    customer_locs: list # J
    retail_locs: list # K
    potential_locs: dict # K_i \subset K for i \in I
    retail_strats: dict # i \in I
    A: list # possible parameters for alpha
    B: list # possible parameters for beta

    # Exogenous parameters
    pop_count: list # (j)
    margin: dict # (i, j)
    costs: dict # (i, k)
    max_dist: int
    distance: dict # (j, k)
    convenience: list # (k)
    max_convenience: int
    observed_locs: list # (i)

    # Endogenous variables
    utility: dict # (i, j, k)
    alpha: int # normalised sensitivity to distance
    beta: list # (i) brand attractiveness
    delta: list # (i) unilateral imporvement potentials

    def patreons(i, j):
        return 0

