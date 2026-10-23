"""Snapshot / restore of a MetaWorld env so that several simulator continuations can be branched from one state."""
import copy
import numpy as np
import mujoco

_DATA_FIELDS = ("qpos", "qvel", "qacc", "qacc_warmstart", "ctrl", "act", "mocap_pos", "mocap_quat", "xfrc_applied", "qfrc_applied")
_PY_ATTRS = ("curr_path_length", "_last_stable_obs", "_prev_obs", "_partially_observable", "isV2", "_last_rand_vec", "_target_pos",
             "obj_init_pos", "init_tcp", "init_left_pad", "init_right_pad", "maxPullDist", "_obj_to_target_init", "_obj_to_target_init")


def _real(env):
    u = env.unwrapped
    while hasattr(u, '_wrapped_env'):
        u = u._wrapped_env
    return u


def snapshot(env):
    u = _real(env)
    d = {f: np.copy(getattr(u.data, f)) for f in _DATA_FIELDS}
    d["time"] = float(u.data.time)
    attrs = {k: copy.deepcopy(getattr(u, k)) for k in dir(u) if (k in _PY_ATTRS or k.startswith("_obj") or k in ("target_pos", "goal", "_state_goal", "pickCompleted", "placeCompleted", "pickCompleted"))
             and not callable(getattr(u, k, None)) and hasattr(u, k)}
    return dict(data=d, attrs=attrs, elapsed=getattr(env, "_elapsed_steps", None))


def restore(env, snap):
    u = _real(env)
    for f in _DATA_FIELDS:
        getattr(u.data, f)[:] = snap["data"][f]
    u.data.time = snap["data"]["time"]
    for k, v in snap["attrs"].items():
        try:
            setattr(u, k, copy.deepcopy(v))
        except AttributeError:
            pass
    if snap["elapsed"] is not None:
        env._elapsed_steps = snap["elapsed"]
    mujoco.mj_forward(u.model, u.data)
