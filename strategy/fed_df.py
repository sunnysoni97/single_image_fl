from flwr.common import (
    Parameters,
    FitRes,
    FitIns,
    EvaluateRes,
    EvaluateIns,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    bytes_to_ndarray,
    NDArray
)

from typing import Optional, Tuple, List, Union, Dict
from flwr.server.strategy import Strategy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from models import init_model, get_parameters, set_parameters
from fed_df_data_loader.common import make_data_loader
from common import test_model

from flwr.common.logger import log
from logging import WARNING

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FedDF_strategy(Strategy):
    def __init__(self,
                 distillation_dataloader: DataLoader,
                 evaluation_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 model_type: str,
                 model_n_classes: int,
                 device: torch.device,
                 fraction_fit: float = 1.0,
                 fraction_evaluate: float = 1.0,
                 min_fit_clients: int = 2,
                 min_evaluate_clients: int = 2,
                 min_available_clients: int = 2,
                 initial_parameters: Optional[Parameters] = None,
                 fit_metrics_aggregation_fn=None,
                 evaluate_metrics_aggregation_fn=None,
                 on_fit_config_fn_client=None,
                 on_fit_config_fn_server=None,
                 evaluate_fn=None,
                 logger_fn=None,
                 ) -> None:
        super().__init__()

        # DataLoader for distillation, valuation and evaluation set
        self.distillation_dataloader = distillation_dataloader
        self.evaluation_dataloader = evaluation_dataloader
        self.val_dataloader = val_dataloader

        # Neural Net specifications
        self.model_type = model_type
        self.model_n_classes = model_n_classes

        # fraction of participating clients
        self.min_available_clients = min_available_clients
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients

        # initial parameters
        self.initial_parameters = initial_parameters

        # metrics aggregation functions
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn

        # configuration function for clients/server for training round
        self.on_fit_config_fn_client = on_fit_config_fn_client
        self.on_fit_config_fn_server = on_fit_config_fn_server

        # server side evaluation fn
        self.evaluate_fn = evaluate_fn

        # logger fn
        self.logger_fn = logger_fn

        # cpu/gpu selection
        self.device = device

    def __repr__(self) -> str:
        rep = f'FedDF'
        return rep

    # FUNCTIONS FOR SAMPLING CLIENTS FOR FIT AND EVALUATE (SAME AS FED AVG)

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available
        clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    # FUNCTIONS FOR OUR FEDDF STRATEGY

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        initial_parameters = self.initial_parameters
        self.initialize_parameters = None
        return initial_parameters

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn_client is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn_client(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, EvaluateIns]]:
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        config = {}

        if(self.on_fit_config_fn_server is not None):
            config = self.on_fit_config_fn_server(server_round)

        logits_results = [
            (bytes_to_ndarray(
                fit_res.metrics['preds']), fit_res.metrics['preds_number'])
            for _, fit_res in results
        ]

        logits_aggregated = np.array(aggregate(logits_results))

        # Distilling student model using Average Logits

        old_parameters = parameters_to_ndarrays(results[0][1].parameters)
        parameters_aggregated, fusion_metrics = self.__fuse_models(global_parameters=old_parameters, preds=logits_aggregated,
                                                                   config=config, dataloader=self.distillation_dataloader, val_dataloader=self.val_dataloader, model_type=self.model_type, model_n_classes=self.model_n_classes, DEVICE=self.device)

        # Aggregate custom metrics if aggregation fn was provided

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics)
                           for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        if(self.logger_fn is not None):
            self.logger_fn(server_round, metrics_aggregated, "fit", "client")
            self.logger_fn(server_round, fusion_metrics, "fit", "server")

        parameters_aggregated = ndarrays_to_parameters(parameters_aggregated)
        return parameters_aggregated, {}

    def aggregate_evaluate(self, server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]], failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics)
                            for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(
                eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        if(self.logger_fn is not None):
            self.logger_fn(server_round, metrics_aggregated,
                           "evaluate", "client")

        return loss_aggregated, metrics_aggregated

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(model_params=parameters_ndarrays, model_name=self.model_type,
                                    n_classes=self.model_n_classes, test_loader=self.evaluation_dataloader, device=self.device)
        if eval_res is None:
            return None
        loss, metrics = eval_res
        if(self.logger_fn is not None):
            self.logger_fn(server_round, metrics, "evaluate", "server")
        return loss, metrics

    @staticmethod
    def __fuse_models(global_parameters: NDArray, preds: NDArray, config: Dict[str, float], dataloader: DataLoader, val_dataloader: DataLoader, model_type: str, model_n_classes: int, DEVICE: torch.device) -> Parameters:

        print("Performing server side distillation training...")
        net = init_model(model_name=model_type, n_classes=model_n_classes)
        set_parameters(net, global_parameters)
        criterion = nn.KLDivLoss(reduction='sum')
        temperature = config['temperature']
        optimizer = torch.optim.Adam(params=net.parameters(), lr=config['lr'])
        # optimizer = torch.optim.SGD(params=net.parameters(
        # ), lr=config['lr'], momentum=config['momentum'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=config['steps'])
        net.train()
        net.to(DEVICE)

        train_loader = make_data_loader(
            dataloader, preds=preds, batch_size=dataloader.batch_size, n_workers=dataloader.num_workers)

        cur_step = 0
        plateau_step = 0
        best_val_acc = 0.0
        total_step_acc = []
        total_step_loss = []

        while(cur_step < config['steps'] and plateau_step < config['early_stopping_steps']):
            for images, labels in train_loader:
                if(cur_step == config['steps'] or plateau_step == config['early_stopping_steps']):
                    break

                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = net(images)
                outputs = F.log_softmax(outputs/temperature, dim=1)
                labels = F.softmax(labels/temperature, dim=1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                total = 0
                correct = 0
                net.eval()
                with torch.no_grad():
                    for images2, labels2 in val_dataloader:
                        images2, labels2 = images2.to(
                            DEVICE), labels2.to(DEVICE)
                        outputs2 = net(images2)
                        total += labels2.size(0)
                        correct += (torch.max(outputs2.detach(), 1)
                                    [1] == labels2.detach()).sum().item()
                net.train()
                step_val_acc = correct/total
                plateau_step += 1
                if(step_val_acc >= best_val_acc+1e-2):
                    plateau_step = 0
                    best_val_acc = step_val_acc

                total_step_acc.append(step_val_acc)
                total_step_loss.append(loss.item())

                cur_step += 1
                if(cur_step % 10 == 0):
                    print(f"step {cur_step}, val_acc : {step_val_acc}")

        print(f'Distillation training stopped at step number : {cur_step}')

        new_parameters = get_parameters(net)

        train_res = {'fusion_loss': np.mean(
            total_step_loss), 'fusion_acc': np.mean(total_step_acc)}

        print(
            f'Fusion training loss : {train_res["fusion_loss"]}, val accuracy : {train_res["fusion_acc"]}')

        return (new_parameters, train_res)


# callback functions for fed df

class fed_df_fn:
    @staticmethod
    def on_fit_config_fn_client(server_round: int) -> Dict[str, float]:
        lr = 1e-1
        config = {
            'lr': lr,
            'epochs': 1,
            # 'momentum': 0.9
        }
        return config

    @staticmethod
    def on_fit_config_fn_server(server_round: int) -> Dict[str, float]:
        lr = 1e-3
        config = {
            "steps": 1e2,
            "early_stopping_steps": 5e1,
            "lr": lr,
            # "momentum": 0.9,
            "temperature": 1,
        }
        return config

    @staticmethod
    def evaluate_fn(model_params: NDArray, model_name: str, n_classes: int, test_loader: DataLoader, device: torch.device) -> Tuple[float, Dict[str, float]]:
        test_res = test_model(model_name, n_classes,
                              model_params, test_loader, device)
        test_loss = test_res['test_loss']
        test_acc = test_res['test_acc']
        return test_loss, {'server_test_acc': test_acc}
