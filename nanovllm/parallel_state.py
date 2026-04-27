import torch.distributed as dist


_TP_GROUP = None
_EP_GROUP = None
_TP_RANK = 0
_TP_SIZE = 1
_EP_RANK = 0
_EP_SIZE = 1
_WORLD_SIZE = 1
_WORLD_RANK = 0


def initialize_parallel_state(tp_size: int, ep_size: int, rank: int):
    global _TP_GROUP, _EP_GROUP
    global _TP_RANK, _TP_SIZE, _EP_RANK, _EP_SIZE
    global _WORLD_RANK, _WORLD_SIZE
    assert tp_size >= 1 and ep_size >= 1
    world_size = tp_size * ep_size
    assert world_size == dist.get_world_size()
    assert 0 <= rank < world_size
    _WORLD_RANK = rank
    _WORLD_SIZE = world_size
    _TP_SIZE = tp_size
    _EP_SIZE = ep_size
    _TP_RANK = rank % tp_size
    _EP_RANK = rank // tp_size

    tp_group_ranks = list(range(_EP_RANK * tp_size, (_EP_RANK + 1) * tp_size))
    ep_group_ranks = list(range(_TP_RANK, world_size, tp_size))
    _TP_GROUP = dist.new_group(tp_group_ranks)
    _EP_GROUP = dist.new_group(ep_group_ranks)


def get_world_rank() -> int:
    return _WORLD_RANK


def get_world_size() -> int:
    return _WORLD_SIZE


def get_tp_rank() -> int:
    return _TP_RANK


def get_tp_size() -> int:
    return _TP_SIZE


def get_tp_group():
    return _TP_GROUP


def get_ep_rank() -> int:
    return _EP_RANK


def get_ep_size() -> int:
    return _EP_SIZE


def get_ep_group():
    return _EP_GROUP
