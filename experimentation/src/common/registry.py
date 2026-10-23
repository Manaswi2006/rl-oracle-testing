"""Name -> constructor for learned structural-reward modules."""


def get_ext(name):
    if name == "psvm":
        from src.psvm.psvm_ext import PSVMExt
        return PSVMExt
    if name == "outcome":
        from src.psvm.outcome_ext import OutcomeExt
        return OutcomeExt
    if name == "facr":
        from src.facr.facr_ext import FACRExt
        return FACRExt
    if name == "rlf":
        from src.falsification.rlf_ext import RLFExt
        return RLFExt
    raise KeyError(name)
