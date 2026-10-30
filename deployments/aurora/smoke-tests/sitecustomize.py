"""Install Aurora's TorchComms rank compatibility in every Python worker."""

import os

if os.environ.get("TORCH_DISTRIBUTED_USE_TORCHCOMMS") == "1":
    import torch.distributed.distributed_c10d as c10d
    from torchcomms_rank_patch import install_torchcomms_group_rank_patch

    install_torchcomms_group_rank_patch(c10d)