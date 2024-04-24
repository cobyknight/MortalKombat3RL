import os 
from stable_baselines3.common.callbacks import BaseCallback

class Callbacks(BaseCallback):
    """
    Custom callback for saving models periodically during training.
    """

    def __init__(self, check_freq, save_path, verbose=1):
        """
        Initialize the callback.

        :param check_freq: Frequency of model saving (in number of steps)
        :param save_path: Path to save the models
        :param verbose: Verbosity level (0: no output, 1: some output, 2: detailed output)
        """
        super(Callbacks, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        """
        Initialize the callback. Create the directory to save models if it does not exist.
        """
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        """
        Perform the callback action on each step.

        :return: Boolean indicating whether the callback should continue
        """
        # Save the model if the number of calls is a multiple of check_freq
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'new_best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True
