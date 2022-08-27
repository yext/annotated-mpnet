"""
Module containing the polynomial decay LR scheduler with warmup step parameter
"""


class PolynomialDecayLRScheduler(object):
    """
    This class will handle the learning rate schedule required by pretrained language models (or at
    least according to the original "Attention is all you need" paper). This means we need to "warm
    up" the learning rate for a number of steps and then decay it polynomially
    """

    def __init__(self, args, optimizer) -> None:
        """
        Initialize the scheduler

        Args:
            args: the pass in from argparse containing all the args (including the ones we need for
                defining this LR scheduler)
            optimizer: the optimizer we're using, will probably always be Adam
        """
        super().__init__()

        self.args = args
        self.optimizer = optimizer

        # Unpack the learning rate
        self.lr = args.lr

        # Create a "warmup factor" which will step the learning rate up by 1 / warmup_updates
        if args.warmup_updates > 0:
            self.warmup_factor = 1.0 / args.warmup_updates
        else:
            self.warmup_factor = 1

        # Extract the end learning rate. Will usually always be 0.0
        self.end_learning_rate = args.end_learning_rate

        # Extract the total updates so we can just the polynomial decay
        self.total_updates = args.total_updates

        # Extract the power of the polynomial factor
        self.power = args.power

        # Set the inital learning rate
        self.set_lr(self.warmup_factor * self.lr)

    def set_lr(self, lr):
        """
        Set the learning rate of the param groups
        """
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self):
        """
        Get the current learning rate of the param groups
        """
        return self.optimizer.param_groups[0]["lr"]

    def step(self, num_updates):
        """
        The step function of the scheduler. Will update with either a linea increase if num_updates
        is less than warmup_updates or will decrease polynomially otherwise
        """

        # Branch first to the linear increase using the warmup factor
        if self.args.warmup_updates > 0 and num_updates <= self.args.warmup_updates:
            self.warmup_factor = num_updates / float(self.args.warmup_updates)
            lr = self.warmup_factor * self.lr
        # Branch to end learning rate
        elif num_updates > self.total_updates:
            lr = self.end_learning_rate
        # Branch to polynomial decay
        else:
            warmup = self.args.warmup_updates

            # Create a range from peak LR to end LR
            lr_range = self.lr - self.end_learning_rate

            # Create a pct_remaining factor that calculates how to move the polynomial factor
            pct_remaining = 1 - (num_updates - warmup) / (self.total_updates - warmup)

            # Finally use the power arg to calculate the polynomial factor
            lr = lr_range * pct_remaining ** (self.power) + self.end_learning_rate

        # Finally set the new LR
        self.set_lr(lr)

        # Step with the new LR
        self.optimizer.step()

        # Return this new LR so we can log it in tensorboard
        return self.get_lr()

    # A couple helper functions that seem to be required for all LR schedulers below
    def state_dict(self):
        """Return the optimizer's state dict."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict, optimizer_overrides=None):
        """Load an optimizer state dict.
        In general we should prefer the configuration of the existing optimizer
        instance (e.g., learning rate) over that found in the state_dict. This
        allows us to resume training from a checkpoint using a new set of
        optimizer args.
        """
        self.optimizer.load_state_dict(state_dict)

        if optimizer_overrides is not None and len(optimizer_overrides) > 0:
            # override learning rate, momentum, etc. with latest values
            for group in self.optimizer.param_groups:
                group.update(optimizer_overrides)
