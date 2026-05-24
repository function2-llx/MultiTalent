from multitalent.training.nnUNetTrainer.project_specific.MultiTalent_native.MultiTalent_trainer import MultiTalent_trainer


class MultiTalent_trainer_v1repro(MultiTalent_trainer):
    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, unpack_dataset: bool = True,
                 device=None):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset, device)
        self.initial_lr = 0.01
        self.num_epochs = 2000
