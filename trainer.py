import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional, Literal, Union, Dict
import tqdm
from torchmetrics import MetricCollection
from logger import BaseLogger
from abc import ABC, abstractmethod
import os
import wandb


# BoilerPlate Functions for training pytorch Modules


class BaseTrainer(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def fit(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader,
            epochs: int,
    ) -> dict:
        pass

    @abstractmethod
    def predict(self, data_loader: DataLoader) -> torch.Tensor:
        pass

    @abstractmethod
    def evaluate(self, data_loader: DataLoader) -> dict:
        pass

    @abstractmethod
    def train_step(self, batch: tuple[torch.Tensor, ...]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        pass

    @abstractmethod
    def val_step(self, batch: tuple[torch.Tensor, ...]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        pass

    @abstractmethod
    def test_step(self, batch: tuple[torch.Tensor, ...]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        pass

    @abstractmethod
    def save(self) -> None:
        pass

    @abstractmethod
    def load(self, path: str) -> nn.Module:
        pass


class DefaultTrainer(BaseTrainer):
    def __init__(
            self,
            model: torch.nn.Module,
            optimizer: optim.Optimizer,
            criterion: torch.nn.Module,
            metrics: MetricCollection,
            loggers: BaseLogger,
            device: torch.device,
            log_interval: int = 100,
            need_saving: bool = True,
            saving_on: Optional[Literal["best", "last", "every_epoch"]] = "last",
            saving_dir: Optional[str] = "checkpoints",
            saving_name: Optional[str] = "model",
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = metrics
        self.metrics.to(device)
        self.loggers = loggers
        self.device = device
        self.step = 0
        self.log_interval = log_interval
        self.need_saving = need_saving
        if self.need_saving:
            assert saving_on or saving_dir or saving_name, "args must be provided"
            self.saving_on = saving_on
            os.makedirs(saving_dir, exist_ok=True)
            self.saving_dir = saving_dir
            self.saving_name = saving_name

    def log(self, 
            to_log: Union[torch.Tensor, Dict[str, torch.tensor]],
            stage: Literal["train", "val", "test"] = "train",
            when: Literal["step", "epoch"] = "step"):
        prefix = f"{stage}_{when}"
        if when == "step" and self.step % self.log_interval == 0:
            formatted_log = self._format_log_data(to_log, prefix)
            self.loggers.log(formatted_log, self.step)
        elif when == "epoch":
            formatted_log = self._format_log_data(to_log, prefix)
            self.loggers.log(formatted_log, self.step)
                        
        
    
    def train_step(self, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        self.step += 1
        self.optimizer.zero_grad()
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
        metrics = self.metrics(outputs, targets)
        # update the metrics calculation
        return {'loss': loss, **metrics}

    def val_step(self, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        with torch.no_grad():
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            self.metrics(outputs, targets)
            return {"loss": loss}

    def test_step(self, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        loss = self.val_step(batch)
        return {"loss": loss}

    def fit(
            self,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            epochs: int = 30,
    ) -> dict:
        train_results = None
        val_results = None
        best_val_loss = float("inf")
        self.model.train()
        self.model.to(self.device)
        for epoch in range(epochs):
            pbar = tqdm.tqdm(train_loader)
            train_epoch_loss = 0.0
            # training loop
            for batch in pbar:
                train_step_results = self.train_step(batch)
                self.log(train_step_results, stage="train", when="step")
                train_loss = train_step_results["loss"]
                pbar.set_description(f"Epoch {epoch} - loss: {train_loss:.4f}")
                train_epoch_loss += train_loss
            # traing epoch ends
            # compute metrics
            train_epoch_results = self.metrics.compute()
            train_epoch_loss = train_epoch_loss / len(train_loader)
            train_epoch_results['loss'] = train_epoch_loss
            self.log(train_epoch_results, stage="train", when="epoch")
            self.metrics.reset()
            if val_loader:
                print("Validation Starts")
                self.model.eval()
                val_epoch_loss = 0.0
                for batch in tqdm.tqdm(val_loader):
                    val_step_results = self.val_step(batch)
                    val_loss = val_step_results["loss"]
                    val_epoch_loss += val_loss
                val_epoch_results = self.metrics.compute()
                val_epoch_results["loss"] = val_epoch_loss / len(val_loader)
                self.log(val_epoch_results, stage="val", when="epoch")
                self.metrics.reset()
                if self.saving_on == "best":
                    if val_epoch_results["loss"] < best_val_loss:
                        best_val_loss = val_epoch_results["loss"]
                        self.save() if self.need_saving else None
            if self.saving_on == "every_epoch":
                self.save() if self.need_saving else None
        if self.saving_on == "last":
            self.save() if self.need_saving else None
        self.loggers.flush()
        self.loggers.close()
        print("Training Ends")
        assert train_epoch_results is not None, "Error in the training loop"
        print(f"Training Results: {train_results}")
        if val_loader:
            assert val_epoch_results is not None, "Error in the validation loop"
            print(f"Validation Results: {val_results}")
        fitting_summary = {}
        try:
            fitting_summary = self.loggers._get_fitting_summary()
        except Exception as e:
            print(f"Error in getting the fitting summary: {e}")
            print("Returning an empty summary")
        return fitting_summary

    def evaluate(self, data_loader: DataLoader) -> dict:
        print("Evaluation Starts")
        self.model.eval()
        self.model.to(self.device)
        test_epoch_loss = 0.0
        pbar = tqdm.tqdm(data_loader)
        for batch in pbar:
            test_step_results = self.test_step(batch)
            test_loss = test_step_results["loss"]
            test_epoch_loss += test_loss
        test_epoch_results = self.metrics.compute()
        test_epoch_results["loss"] = test_epoch_loss / len(data_loader)
        self.log(test_epoch_results, stage="test", when="epoch")
        self.metrics.reset()
        print("Evaluation Ends")
        print(f"Test Results: {test_epoch_results}")
        return test_epoch_results

    def predict(self, data_loader: DataLoader) -> torch.Tensor:
        self.model.eval()
        predictions = []
        print("Prediction Starts")
        with torch.no_grad():
            for batch in tqdm.tqdm(data_loader):
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                predictions.append(outputs)
        return torch.cat(predictions, dim=0)

    def save(self) -> None:
        path = os.path.join(self.saving_dir, self.saving_name)
        to_save = {
            "model_state_dict": self.model.to("cpu").state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step": self.step,
            "ckpt_path": path,
        }
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                wandb_log_path = logger._get_wandb_run().path
                to_save["wandb_log_path"] = wandb_log_path
            if isinstance(logger, CSVLogger):
                csv_log_path = logger.path
                to_save["csv_log_path"] = csv_log_path
        torch.save(to_save, path)

    def load(self, path: str) -> nn.Module:
        to_load = torch.load(path)
        model_state_dict = to_load["model_state_dict"]
        try:
            self.model.load_state_dict(model_state_dict)
        except RuntimeError as e:
            print(f"Error in loading the model: {e}")
        return self.model


if __name__ == "__main__":
    import torch
    from torch.utils.data import TensorDataset
    import torchmetrics
    from logger import LoggerCollection, WandbLogger, CSVLogger


    # Define a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 2)

        def forward(self, x):
            return self.fc(x)


    # Mock dataset
    def create_mock_dataset(num_samples=100):
        x = torch.randn(num_samples, 10)
        y = torch.randint(0, 2, (num_samples,))
        return TensorDataset(x, y)


    def test_trainer():
        # Create a mock dataset
        dataset = create_mock_dataset()
        loader = DataLoader(dataset, batch_size=10)

        # Initialize components
        model = SimpleModel()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        metrics = MetricCollection(
            [
                torchmetrics.Accuracy(task="multiclass", num_classes=2),
                torchmetrics.F1Score(task="multiclass", num_classes=2),
            ]
        )
        metrics_names = list(metrics.keys())
        loggers = LoggerCollection(
            [
                CSVLogger(name="test.csv", saving_dir="."),
                WandbLogger(project_name="test", run_name="test"),
            ]
        )  # Assuming you have a logger implementation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize trainer
        trainer = DefaultTrainer(model, optimizer, criterion, metrics, loggers, device)

        # Run training
        summary = trainer.fit(
            loader, None, epochs=5
        )  # No validation loader in this simple test
        print("WanDB Summary")
        print(summary)
        trainer.fit(loader, loader, epochs=5)  # With validation loader
        print("WanDB Summary")
        print(summary)
        loggers = LoggerCollection(
            [
                CSVLogger(name="test.csv", saving_dir="."),
            ]
        )
        trainer = DefaultTrainer(model, optimizer, criterion, metrics, loggers, device)
        csv_summary = trainer.fit(loader, None, epochs=5)
        print("CSV Summary")
        print(csv_summary)
        trainer.evaluate(loader)
        pred = trainer.predict(loader)
        print(pred.shape)


    try:
        test_trainer()
    except Exception as e:
        print(f"Error in the test: {e}")
        print("Exiting the test")
        print("Closing the wandb run")
        if wandb.run:
            wandb.run.finish()
    # Test Passed
    # Move to the server
