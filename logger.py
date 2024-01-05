import typing
from abc import ABC, abstractmethod
from typing import List
import os
import wandb
import pandas as pd


class BaseLogger(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def log(self, payload: dict, step: int) -> None:
        pass

    @abstractmethod
    def flush(self) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @abstractmethod
    def _get_fitting_summary(self):
        pass


class DummyPrintLogger(BaseLogger):
    def __init__(self) -> None:
        super().__init__()

    def log(self, payload: dict, step: int) -> None:
        print(f"Step {step}: {payload}")

    def flush(self) -> None:
        pass

    def close(self) -> None:
        del self
        
    def _get_fitting_summary(self):
        return None


class CSVLogger(BaseLogger):
    def __init__(self, name: str, saving_dir: str, buffer_interval: int = 1000) -> None:
        super().__init__()
        self.saving_dir = saving_dir
        self.name = name
        self.buffer_interval = buffer_interval
        self.path = os.path.join(self.saving_dir, self.name)
        self.current_step = -1
        self.header = ["step"]
        self.rows = []
        self.all_data = pd.DataFrame(columns=self.header)

        if not os.path.exists(saving_dir):
            os.makedirs(saving_dir)
        self._check_exists()

    def _check_exists(self):
        if os.path.exists(self.path):
            print(f"File {self.path} already exists. It will be overwritten.")
            os.remove(self.path)

    def log(self, payload: dict, step: int) -> None:
        assert step >= self.current_step, "Step must be monotonically increasing"
        self.current_step = step

        # Convert tensors to numbers
        payload = {
            k: (v.item() if hasattr(v, "item") else v) for k, v in payload.items()
        }
        payload["step"] = step

        # Update header if new keys are found
        new_keys = set(payload.keys()) - set(self.header)
        if new_keys:
            self.header.extend(new_keys)
            self._rewrite_file_with_new_header()

        # Append the new row
        self.rows.append(payload)

        # Flush if interval is reached
        if len(self.rows) >= self.buffer_interval:
            self.flush()

    def _rewrite_file_with_new_header(self):
        # Append existing rows to the all_data DataFrame
        if self.rows:
            self.all_data = pd.concat(
                [self.all_data, pd.DataFrame(self.rows)], sort=False
            )
            self.rows = []

        # Reindex the all_data DataFrame to include new columns with NaNs
        self.all_data = self.all_data.reindex(columns=self.header)
        self.flush(force_rewrite=True)

    def flush(self, force_rewrite: bool = False) -> None:
        if force_rewrite:
            # Rewrite the entire file
            self.all_data.to_csv(self.path, mode="w", header=True, index=False)
        else:
            # Append new rows to the file or create if doesn't exist
            if self.rows:
                df_to_write = pd.DataFrame(self.rows, columns=self.header)
                if os.path.exists(self.path):
                    df_to_write.to_csv(self.path, mode="a", header=False, index=False)
                else:
                    df_to_write.to_csv(self.path, mode="w", header=True, index=False)
                self.rows.clear()

    def close(self) -> None:
        self.flush()

    def _get_fitting_summary(self):
        df = pd.read_csv(self.path)
        if df is None or df.empty:
            raise Exception("No data available or DataFrame is empty.")
        summary = {}
        for col in df.columns:
            non_nan_values = df[col].dropna()
            if not non_nan_values.empty:
                summary[col] = non_nan_values.iloc[-1]
            else:
                summary[col] = None
        return summary

class WandbLogger(BaseLogger):
    def __init__(self, project_name: str, run_name: str):
        super().__init__()
        self.project_name = project_name
        self.run_name = run_name
        # check if wandb is initialized
        if not wandb.run:
            try:
                run = wandb.init(project=self.project_name, name=self.run_name)
                self.run = run
            except Exception as e:
                print(f"Failed to initialize wandb: {e}")

    def log(self, payload: dict, step: int) -> None:
        if not wandb.run:
            print("wandb is not initialized. Skipping logging.")
            run = wandb.init(project=self.project_name, name=self.run_name)
            self.run = run
        # log to wandb
        wandb.log(payload, step=step)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        # stop wandb run
        wandb.finish()

    def _get_wandb_run(self):
        assert self.run is not None, "Wandb run is not initialized."
        return self.run
    
    def _get_fitting_summary(self):
        assert self.run is not None, "Wandb run is not initialized."
        return self.run.summary._as_dict()


class LoggerCollection(BaseLogger,  typing.Iterable):
    def __init__(self, loggers: List[BaseLogger]) -> None:
        super().__init__()
        assert len(loggers) > 0, "Must provide at least one logger"
        assert isinstance(loggers, list), "loggers must be a list of loggers"
        self.loggers = loggers
        self._logger_iter = iter(self.loggers)

    def log(self, payload: dict, step) -> None:
        for logger in self.loggers:
            logger.log(payload, step)

    def flush(self) -> None:
        for logger in self.loggers:
            logger.flush()

    def close(self) -> None:
        for logger in self.loggers:
            logger.close()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._logger_iter)

    def _get_fitting_summary(self) -> dict:
        summary = {}

        # Check for WandbLogger
        wandb_logger = next((logger for logger in self.loggers if isinstance(logger, WandbLogger)), None)
        if wandb_logger:
            wandb_run = wandb_logger._get_wandb_run()
            if wandb_run is None:
                raise Exception("WandbLogger found, but error in getting the wandb run.")
            summary = wandb_run.summary._as_dict()
            if isinstance(summary, dict):
                return self._format_wandb_summary(summary)

        # Check for CSVLogger
        csv_logger = next((logger for logger in self.loggers if isinstance(logger, CSVLogger)), None)
        if csv_logger:
            summary = csv_logger._get_fitting_summary()
            return summary
        # No valid logger found
        raise Exception("No valid logger (WandbLogger or CSVLogger) found in the loggers collection")

    def _format_wandb_summary(self, summary: dict) -> dict:
        formatted_summary = {}
        for key, value in summary.items():
            if key.startswith("_"):
                trimed_key = key[1:]
                formatted_summary[trimed_key] = value
            else:
                formatted_summary[key] = value
        return formatted_summary


if __name__ == "__main__":
    import pandas as pd

    metrics_to_monitor = ["metric1", "metric2", "metric3"]
    csv_logger = CSVLogger(name="test.csv", saving_dir=".")
    wandb_logger = WandbLogger(project_name="test", run_name="test")
    loggers = LoggerCollection([csv_logger, wandb_logger])
    try:
        # Simulate some data logging
        for step in range(5):
            results = {metric: step * 0.1 for metric in metrics_to_monitor}
            loggers.log(results, step)
        for step in range(5, 7):
            metrics_subset = {metric: step * 0.1 for metric in metrics_to_monitor[:2]}
            loggers.log(metrics_subset, step)
        for step in range(7, 10):
            metrics = {
                metric_name: step * 0.1
                for metric_name in ["hello", "world", "foo", "bar"]
            }
            loggers.log(metrics, step)

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        csv_logger.close()
        df = pd.read_csv("test.csv")
        print(df)

    # test passed
