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
from flwr.server.strategy import Strategy, aggregate
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from models import init_model, get_parameters, set_parameters

from flwr.common.logger import log
from logging import WARNING

from torch.utils.data import DataLoader
import torch
import torch.nn as nn


class FedDF_strategy(Strategy):
    def __init__(self,
                 distillation_dataloader:DataLoader,
                 model_type:str,
                 model_n_classes:int,
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
                 
                 ) -> None:
        super().__init__()

        # DataLoader for distillation set
        self.distillation_dataloader = distillation_dataloader

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
        
        config = {
            "epochs": 1,
            "lr": 0.001,
            "momentum": 0.9,
            "temperature": 1,
        }
        
        if(self.on_fit_config_fn_server is not None):
            config = self.on_fit_config_fn_server(server_round)

        logits_results = [
            (bytes_to_ndarray(fit_res.metrics['preds']),fit_res.num_examples)
            for _, fit_res in results
        ]
        
        logits_aggregated = aggregate(logits_results)
        
        #Distilling student model using Average Logits

        parameters_aggregated = fed_df_fn.fuse_models(global_parameters=results[0][1]['global_parameters'],preds=logits_results,config=config, dataloader=self.distillation_dataloader, model_type=self.model_type, model_n_classes=self.model_n_classes)
        
        # Aggregate custom metrics if aggregation fn was provided

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(self, server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]], failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]) -> Tuple[Optional[float], Dict[str, Scalar]]:
        return super().aggregate_evaluate(server_round, results, failures)

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return super().evaluate(server_round, parameters)


# IMPLEMENT THISSSSSSSS

class fed_df_fn:
    @staticmethod
    def fuse_models(global_parameters: Parameters, preds: NDArray, config: Dict[str, float], dataloader: DataLoader, model_type:str, model_n_classes:int, DEVICE: torch.device) -> Parameters:
        
        raise NotImplementedError()
        net = init_model(model_name=model_type, n_classes=model_n_classes)
        set_parameters(net, global_parameters)
        criterion = nn.KLDivLoss(reduction='sum')
        temperature = config['temperature']
        optimizer = torch.optim.SGD(params=net.parameters(),lr=config['lr'],momentum=config['momentum'])
        net.train()
        net.to(DEVICE)

        total_epoch_loss = []
        total_epoch_acc = []

        for epoch in range(config['epochs']):
            correct, total, epoch_loss = 0, 0, 0.0

            for images, labels in dataloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                epoch_loss += loss.item()
                total += labels.size(0)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

            epoch_loss /= len(train_loader.dataset)
            epoch_acc = correct/total
            total_epoch_loss.append(epoch_loss)
            total_epoch_acc.append(epoch_acc)
            print(f'Epoch {epoch+1} : loss {epoch_loss}, acc {epoch_acc}')

        new_parameters = get_parameters(model)
        train_res = {'train_loss': np.mean(
            total_epoch_loss), 'train_acc': np.mean(total_epoch_acc)}
        return (new_parameters, train_res)
        
        

        return
