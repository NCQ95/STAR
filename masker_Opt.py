from torch import optim

def get_optim_and_scheduler(network):
    params = network.parameters()
    epoch = 300
    optimizer = optim.SGD(params,
                          weight_decay=5e-4,
                          momentum=0.9,
                          nesterov=True,
                          lr=0.00154)
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=int(epoch * 0.8),
                                          gamma=0.1)
    return optimizer, scheduler