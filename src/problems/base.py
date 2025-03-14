from enum import Enum
from typing import Optional

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
import numpy as np
import gurobipy as gp
from gurobipy import GRB

allow_nparray = ConfigDict(arbitrary_types_allowed=True)


@dataclass(config=allow_nparray)
class IPGResult:
    PNE: bool
    X: np.ndarray
    ObjVal: float
    runtime: float
    phi: float
    timelimit_reached: bool = False


@dataclass
class ApproxOptions:
    allow_phi_ne: bool = False
    timelimit: Optional[int] = None
    allow_timelimit_reached: bool = False

    def valid_problem(self, result: IPGResult) -> bool:
        if result.PNE:
            return True
        elif not result.PNE and not self.allow_phi_ne:
            return False
        elif result.timelimit_reached and not self.allow_timelimit_reached:
            return False
        else:
            return True


class Target(Enum):
    PAYOFFS = 1
    WEIGHTS = 2
    PARAMS = 3


def early_stopping(model: gp.Model, where: GRB.Callback):
    # Requires model._timelimit and model._current_obj to be set.
    if where == GRB.Callback.MIP:
        time = model.cbGet(GRB.Callback.RUNTIME)
        best = model.cbGet(GRB.Callback.MIP_OBJBST)

        if time > model._timelimit and best > model.current_obj:
            print("Terminate with approximation")
            model.terminate()
