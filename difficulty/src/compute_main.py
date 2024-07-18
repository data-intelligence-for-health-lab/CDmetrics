import pandas as pd
from d_case_difficulty_metrics.src.metrics.CDmc import CDmc_run
from d_case_difficulty_metrics.src.metrics.CDdm import CDdm_run
from d_case_difficulty_metrics.src.metrics.CDpu import hyperparm_searching, CDpu_run


class compute_case_difficulty:
    def __init__(
        self, file_name: str, data: pd.DataFrame, processing: any, target_column: str
    ):
        self.file_name = file_name
        self.data = data
        self.processing = processing
        self.target_column = target_column

    def CDmc(self, number_of_NNs: int = 10):
        if __name__ == "__main__":
            CDmc_run(
                self.file_name,
                self.data,
                self.processing,
                self.target_column,
                number_of_NNs,
            )

    def CDdm(self, max_eval_a: int = 10, max_eval_b: int = 5):
        CDdm_run(
            self.file_name,
            self.data,
            max_eval_a,
            max_eval_b,
            self.processing,
            self.target_column,
        )

    def CDpu(
        self,
        hyper_file_name: str,
        number_of_predictions: int = 10,
        number_of_cpu: int = 1,
    ):
        hyperparm_searching(
            hyper_file_name, self.data, self.processing, self.target_column
        )
        if __name__ == "__main__":
            CDpu_run(
                self.file_name,
                hyper_file_name,
                self.data,
                self.processing,
                self.target_column,
                number_of_predictions,
                number_of_cpu,
            )
