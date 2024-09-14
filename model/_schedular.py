from torch import optim


def get_schedular(optimizer, **kwargs):
    name = kwargs["name"]
    setting = kwargs["setting"]
    
    if name == "steplr":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            **setting
        )
    elif name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            **setting
        )
    
    return scheduler