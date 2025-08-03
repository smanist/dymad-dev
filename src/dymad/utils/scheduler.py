import logging

logger = logging.getLogger(__name__)

class SchedulerBase:
    """
    Base class for schedulers.

    As wrapper of standard schedulers, with customized inputs and outputs.
    Also for other custom schedulers.
    """
    def step(self, **kwargs) -> bool:
        """
        Step through the scheduler logic.

        Returns:
            bool: True if the scheduler indicates to stop training, False by default.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def diagnostic_info(self):
        """
        Return diagnostic information about the scheduler.

        Returns:
            str: Diagnostic information.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def state_dict(self) -> dict:
        """Return the state dictionary for saving."""
        raise NotImplementedError("Subclasses should implement this method.")

    def load_state_dict(self, state_dict: dict):
        """Load the state dictionary."""
        raise NotImplementedError("Subclasses should implement this method.")

class StandardScheduler(SchedulerBase):
    """
    Standard scheduler that wraps a PyTorch scheduler.

    Args:
        scheduler (torch.optim.lr_scheduler._LRScheduler): PyTorch scheduler instance.
    """
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def diagnostic_info(self):
        return str(self.state_dict())

    def step(self, **kwargs) -> bool:
        """Step through the wrapped scheduler."""
        self.scheduler.step()
        return False

    def state_dict(self) -> dict:
        """Return the state dictionary of the wrapped scheduler."""
        return self.scheduler.state_dict()
    
    def load_state_dict(self, state_dict: dict):
        """Load the state dictionary into the wrapped scheduler."""
        self.scheduler.load_state_dict(state_dict)

class SweepScheduler(SchedulerBase):
    """
    Scheduler to manage sweep lengths during training.
    Cycles through predefined sweep lengths and tolerances.

    It operates in four modes:

        - No sweep_lengths or sweep_tols: Does nothing.
        - Only sweep_lengths: Change lengths every `epoch_step`.
        - sweep_lengths AND sweep_tols: Change lengths either every `epoch_step` or a tolerance is reached,
        whichever comes first.  In particular, when the last tolerance is reached,

            - If mode='full': the scheduler resets to the initial length.
            - If mode='skip': the scheduler keeps using the current length.

    Note:
        - If `sweep_tols` is set, the optimization ends when the last tolerance is reached.

    Args:
        sweep_lengths (list): List of sweep lengths to cycle through.
        sweep_tols (list, optional): List of tolerances for each sweep length.
        epoch_step (int): Number of epochs after which to switch to the next sweep length.
        mode (str): Mode of operation ('full' or 'skip'). Default is 'skip'.
    """

    def __init__(self, sweep_lengths: list = None, sweep_tols: list = None, epoch_step: int = 10, mode: str = 'skip'):
        assert epoch_step > 0, "epoch_step must be a positive integer."

        self.sweep_lengths = sweep_lengths
        self.epoch_step    = epoch_step
        if sweep_tols is not None:
            self.sweep_tols = [float(tol) for tol in sweep_tols]
        else:
            self.sweep_tols = None
        self.mode          = mode.lower()
        self.current_epoch = 0
        self.sweep_epoch   = 0
        self.current_len   = 0
        self.current_tol   = 0

        self._init_params()

    def _init_params(self):
        self._Nlen = len(self.sweep_lengths) if self.sweep_lengths is not None else 0
        self._Ntol = len(self.sweep_tols) if self.sweep_tols is not None else 0

        if self.mode == 'full':
            self._step_with_tolerance = self._step_with_tolerance_full
        elif self.mode == 'skip':
            self._step_with_tolerance = self._step_with_tolerance_skip
        else:
            raise ValueError(f"Unknown mode '{self.mode}'. Supported modes are 'full' and 'skip'.")

    def diagnostic_info(self):
        return (f"SweepScheduler(sweep_lengths={self.sweep_lengths}, "
                f"epoch_step={self.epoch_step}, "
                f"sweep_tols={self.sweep_tols})")

    def step(self, eploss: float = None) -> bool:
        self.current_epoch += 1

        flag = False
        if self._Nlen > 0:
            if self._Ntol == 0:
                self._step_no_tolerance()
            elif self._Ntol > 0:
                self._step_with_tolerance(eploss)

                if (self.current_len == self._Nlen-1 and self.current_tol == self._Ntol-1):
                    if eploss < self.sweep_tols[-1]:
                        logger.info("Reached the last sweep length and tolerance. Stopping sweep.")
                        flag = True
        else:
            # Do nothing if no sweep lengths are defined
            pass

        return flag

    def _step_no_tolerance(self) -> None:
        """Handle stepping when no tolerances are provided."""
        index = self.current_epoch // self.epoch_step
        old_index = self.current_len
        self.current_len = min(index, self._Nlen-1)
        if old_index != self.current_len:
            logger.info(f"Switching to sweep length {self.sweep_lengths[self.current_len]} at epoch {self.current_epoch}")

    def _step_with_tolerance_skip(self, eploss) -> None:
        self.sweep_epoch += 1
        current_tol = float(self.sweep_tols[self.current_tol])

        if self.sweep_epoch >= self.epoch_step or eploss < current_tol:
            self.sweep_epoch = 0
            self.current_len += 1
            if self.current_len >= self._Nlen:
                self.current_len = self._Nlen-1
                if self.current_tol < self._Ntol-1:
                    self.current_tol += 1
                else:
                    logger.info("Reached end of sweep lengths and tolerances. Stopping sweep.")
            logger.info(f"Switching to sweep length {self.sweep_lengths[self.current_len]} "
                        f"at epoch {self.current_epoch} with loss {eploss:.4e} with tolerance {current_tol:.4e}")

    def _step_with_tolerance_full(self, eploss) -> None:
        self.sweep_epoch += 1
        current_tol = float(self.sweep_tols[self.current_tol])

        if self.sweep_epoch >= self.epoch_step or eploss < current_tol:
            self.sweep_epoch = 0
            self.current_len += 1
            if self.current_len >= self._Nlen:
                self.current_len = 0
                if self.current_tol < self._Ntol-1:
                    self.current_tol += 1
                    logger.info("Resetting to first sweep length after reaching end of list. " \
                                f"Current tolerance {self.sweep_tols[self.current_tol]}")
                else:
                    logger.info("Reached end of sweep lengths and tolerances. Stopping sweep.")
            logger.info(f"Switching to sweep length {self.sweep_lengths[self.current_len]} "
                        f"at epoch {self.current_epoch} with loss {eploss:.4e} with tolerance {current_tol:.4e}")

    def get_length(self) -> int:
        return self.sweep_lengths[self.current_len]

    def state_dict(self) -> dict:
        """Return the state dictionary for saving."""
        return {
            'epoch_step':    self.epoch_step,
            'sweep_tols':    self.sweep_tols,
            'sweep_epoch':   self.sweep_epoch,
            'sweep_lengths': self.sweep_lengths,
            'current_epoch': self.current_epoch,
            'current_len':   self.current_len,
            'current_tol':   self.current_tol,
            'mode':          self.mode
        }

    def load_state_dict(self, state_dict: dict):
        """Load the state dictionary."""
        self.epoch_step    = state_dict['epoch_step']
        self.sweep_tols    = state_dict['sweep_tols']
        self.sweep_epoch   = state_dict['sweep_epoch']
        self.sweep_lengths = state_dict['sweep_lengths']
        self.current_epoch = state_dict['current_epoch']
        self.current_len   = state_dict['current_len']
        self.current_tol   = state_dict['current_tol']
        self.mode          = state_dict['mode']

        self._init_params()

def make_scheduler(scheduler=None, scheduler_type: str="", **kwargs) -> SchedulerBase:
    """
    Factory function to create a scheduler instance.

    Args:
        scheduler_type (str): Type of the scheduler to create.
        **kwargs: Additional arguments for the scheduler.

    Returns:
        SchedulerBase: An instance of the specified scheduler type.
    """
    if scheduler is not None:
        return StandardScheduler(scheduler)
    elif scheduler_type == 'sweep':
        return SweepScheduler(**kwargs)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
