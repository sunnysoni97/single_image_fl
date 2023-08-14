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
    NDArray,
    NDArrays
)

from typing import Optional, Tuple, List, Union, Dict, Callable
from flwr.server.strategy import Strategy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from models import init_model, get_parameters, set_parameters
from fed_df_data_loader.common import make_data_loader
from common import test_model
import strategy.tools.clustering as clustering
from strategy.tools.confidence_select import prune_confident_crops
from strategy.tools.clipping import clip_logits
from strategy.common import cosine_annealing_round

from flwr.common.logger import log
from logging import WARNING, INFO

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pathlib
import random


class FedDF_strategy(Strategy):
    def __init__(self,
                 num_rounds: int,
                 distillation_dataloader: DataLoader,
                 evaluation_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 model_type: str,
                 dataset_name: str,
                 num_classes: int,
                 device: torch.device,
                 kmeans_output_folder: str,
                 evaluation_labels: list = None,
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
                 warm_start_rounds: int = 30,
                 warm_start_interval: int = 30,
                 num_total_images: int = 100000,
                 kmeans_n_crops: int = 2250,
                 kmeans_n_clusters: int = 10,
                 kmeans_heuristics: str = "mixed",
                 kmeans_mixed_factor: str = "50-50",
                 kmeans_balancing: float = 0.5,
                 confidence_threshold: float = 0.1,
                 confidence_strategy: str = "top",
                 confidence_adaptive: bool = False,
                 confidence_max_thresh: float = 0.5,
                 fedprox_factor: float = 1.0,
                 fedprox_adaptive: bool = False,
                 batch_size: int = 512,
                 num_cpu_workers: int = 4,
                 debug: bool = False,
                 use_kmeans: bool = True,
                 use_entropy: bool = True,
                 use_fedprox: bool = False,
                 seed: int = None,
                 cuda_deterministic: bool = False,
                 ) -> None:
        super().__init__()

        self.num_rounds = num_rounds

        # DataLoader for distillation, valuation and evaluation set
        self.distillation_dataloader = distillation_dataloader
        self.evaluation_dataloader = evaluation_dataloader
        self.val_dataloader = val_dataloader

        # Neural Net specifications
        self.model_type = model_type
        self.dataset_name = dataset_name
        self.num_classes = num_classes

        # fraction of participating clients
        self.min_available_clients = min_available_clients
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients

        # initial parameters
        self.initial_parameters = initial_parameters

        # last round parameters
        self.last_parameters = initial_parameters

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

        # handling parameter initialisation for fusion
        self.warm_start_rounds = warm_start_rounds
        self.warm_start_interval = warm_start_interval

        # logging for debugging
        self.debug = debug

        # configuration for kmean selection

        self.kmeans_output_folder = kmeans_output_folder
        self.evaluation_labels = evaluation_labels
        self.num_total_images = num_total_images
        self.kmeans_n_crops = kmeans_n_crops
        self.kmeans_n_clusters = kmeans_n_clusters
        self.kmeans_heuristics = kmeans_heuristics
        self.kmeans_mixed_factor = kmeans_mixed_factor
        self.kmeans_balancing = kmeans_balancing
        self.kmeans_random_seed = 1
        if (seed is not None):
            self.kmeans_random_seed = seed

        self.batch_size = batch_size
        self.num_cpu_workers = num_cpu_workers

        # configuration for conf_threshold selection

        self.confidence_threshold = confidence_threshold
        self.confidence_strategy = confidence_strategy
        self.confidence_adaptive = confidence_adaptive
        self.confidence_max_thresh = confidence_max_thresh

        # configuration for fedprox enable/disable

        self.fedprox_factor = fedprox_factor
        self.fedprox_adaptive = fedprox_adaptive

        # storage of past results

        self.hist_server_acc = []

        # storage for kmean selection from last round

        self.kmeans_last_crops = None

        # enabling/disabling new features (crop selection, fedprox)

        self.use_kmeans = use_kmeans
        self.use_entropy = use_entropy
        self.use_fedprox = use_fedprox

        # seeding operations

        if (seed is not None):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        if (cuda_deterministic):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

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
            if (self.fedprox_adaptive and len(self.hist_server_acc) > 3 and ((server_round-1) % 3 == 0)):
                delta = self.hist_server_acc[-1] - self.hist_server_acc[-3]
                if (delta < 0):
                    self.fedprox_factor += 0.1
                else:
                    self.fedprox_factor -= 0.1

            if (self.use_fedprox and self.debug):
                log(INFO,
                    f"Fedprox factor for next round : {self.fedprox_factor}")

            config = self.on_fit_config_fn_client(
                server_round=server_round, use_fedprox=self.use_fedprox, fedprox_factor=self.fedprox_factor)

        # performing kmeans on crops

        net = init_model(self.dataset_name, self.model_type)
        set_parameters(net, parameters_to_ndarrays(parameters))

        clusters, cluster_score = clustering.cluster_embeddings(
            dataloader=self.distillation_dataloader, model=net, device=self.device, n_clusters=self.kmeans_n_clusters, seed=self.kmeans_random_seed)

        pruned_clusters = clusters

        confidence_threshold = self.confidence_threshold

        if (self.confidence_adaptive):
            confidence_threshold = 1 - cosine_annealing_round(max_lr=(1-self.confidence_threshold), min_lr=(
                1-self.confidence_max_thresh), max_rounds=self.num_rounds, curr_round=server_round)

        kmeans_n_crops = self.kmeans_n_crops

        if (self.use_kmeans and self.use_entropy):
            kmeans_factor = self.kmeans_n_crops / \
                ((1-confidence_threshold)*self.num_total_images)
            kmeans_n_crops = int(kmeans_factor*self.num_total_images)

        if (self.use_kmeans):
            if (self.debug):
                log(INFO,
                    f'Cluster score for round {server_round} = {cluster_score}')
                log(INFO,
                    f'Number of KMeans Crops for this round : {kmeans_n_crops}')
            pruned_clusters = clustering.prune_clusters(
                raw_dataframe=clusters, n_crops=kmeans_n_crops, heuristic=self.kmeans_heuristics, heuristic_percentage=self.kmeans_mixed_factor, kmeans_balancing=self.kmeans_balancing)

        # doing selection on the basis of confidence

        if (self.use_entropy):
            if (self.debug):
                log(INFO,
                    f'Current entropy removal threshold : {confidence_threshold}')
            pruned_clusters = prune_confident_crops(
                cluster_df=pruned_clusters, confidence_threshold=confidence_threshold, confidence_strategy=self.confidence_strategy)

        if (self.debug):
            log(INFO, f'Number of selected crops : {len(pruned_clusters)}')

        # preparing for transport

        self.kmeans_last_crops = clustering.prepare_for_transport(
            pruned_clusters)

        # calculating tsne

        tsne_clusters = clustering.calculate_tsne(
            cluster_df=pruned_clusters, device=self.device, n_cpu=self.num_cpu_workers)

        # outputting visualisations : clusters and tsne scatter plot in same folder

        img_file = pathlib.Path(self.kmeans_output_folder,
                                f'round_no_{server_round-1}.png')
        with open(img_file, 'wb') as f:
            clustering.visualise_clusters(
                cluster_df=tsne_clusters, file=f, device=self.device, grid_size=20)

        img_file_tsne = pathlib.Path(
            self.kmeans_output_folder, f'tsne_round_no_{server_round-1}.png')

        with open(img_file_tsne, 'wb') as f:
            clustering.visualise_tsne(
                tsne_df=tsne_clusters, out_file=f, round_no=(server_round-1), label_metadata=self.evaluation_labels, n_classes=self.num_classes)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # adding crops to clients
        config_list = [{**config, 'distill_crops': clustering.prepare_for_transport(
            pruned_clusters)} for client in clients]
        fit_ins_list = [FitIns(parameters, config) for config in config_list]

        # Return client/config pairs
        return [(client, fit_ins) for (client, fit_ins) in zip(clients, fit_ins_list)]

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
        warm_start = False

        if (self.on_fit_config_fn_server is not None):
            config = self.on_fit_config_fn_server()
            warm_start = config['warm_start']

        # Aggregating logits using averaging

        logits_results = [
            (bytes_to_ndarray(
                fit_res.metrics['preds']), fit_res.num_examples)
            for _, fit_res in results
        ]

        logits_aggregated = np.array(aggregate(logits_results))

        # Aggregating new parameters for warm start using averaging

        if (warm_start and ((server_round <= self.warm_start_rounds) or (server_round % self.warm_start_interval == 0))):
            log(INFO, f'Warm start at round {server_round}')
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            old_parameters = aggregate(weights_results)

        else:
            old_parameters = parameters_to_ndarrays(self.last_parameters)

        # Distilling student model using Average Logits

        distill_dataloader = clustering.extract_from_transport(
            img_bytes=self.kmeans_last_crops, batch_size=self.batch_size, n_workers=self.num_cpu_workers)

        parameters_aggregated, fusion_metrics = self.__fuse_models(global_parameters=old_parameters, preds=logits_aggregated,
                                                                   config=config, dataloader=distill_dataloader, val_dataloader=self.val_dataloader, model_type=self.model_type, dataset_name=self.dataset_name, DEVICE=self.device, enable_step_logging=self.debug)

        # Aggregate custom metrics if aggregation fn was provided

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics)
                           for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        if (self.logger_fn is not None):
            self.logger_fn(server_round, metrics_aggregated, "fit", "client")
            self.logger_fn(server_round, fusion_metrics, "fit", "server")

        parameters_aggregated = ndarrays_to_parameters(parameters_aggregated)

        # storing the parameters in the server for next round of fusion
        self.last_parameters = parameters_aggregated

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

        if (self.logger_fn is not None):
            self.logger_fn(server_round, metrics_aggregated,
                           "evaluate", "client")

        return loss_aggregated, metrics_aggregated

    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            self.hist_server_acc.append(0.0)
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(model_params=parameters_ndarrays, model_name=self.model_type,
                                    dataset_name=self.dataset_name, test_loader=self.evaluation_dataloader, device=self.device)

        val_res = self.evaluate_fn(model_params=parameters_ndarrays, model_name=self.model_type,
                                   dataset_name=self.dataset_name, test_loader=self.val_dataloader, device=self.device)

        if val_res is None:
            self.hist_server_acc.append(0.0)

        _, val_met = val_res
        self.hist_server_acc.append(val_met['server_test_acc'])

        if eval_res is None:
            return None

        loss, metrics = eval_res
        if (self.logger_fn is not None):
            self.logger_fn(server_round, metrics, "evaluate", "server")

        return loss, metrics

    @staticmethod
    def __fuse_models(global_parameters: NDArrays, preds: NDArray, config: Dict[str, float], dataloader: DataLoader, val_dataloader: DataLoader, model_type: str, dataset_name: str, DEVICE: torch.device, enable_step_logging: bool = False) -> Parameters:

        log(INFO, f'Performing server side distillation training...')
        net = init_model(dataset_name, model_type)
        set_parameters(net, global_parameters)
        criterion = nn.KLDivLoss(reduction='batchmean')
        temperature = config['temperature']
        optimizer = torch.optim.Adam(params=net.parameters(), lr=config['lr'])
        if (config['use_adaptive_lr']):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, T_max=config['steps'])
        net.train()
        net.to(DEVICE)

        train_loader = make_data_loader(
            dataloader, preds=preds, batch_size=dataloader.batch_size, n_workers=dataloader.num_workers)

        cur_step = 0
        plateau_step = 0
        best_val_acc = 0.0
        best_val_param = None
        total_step_acc = []
        total_step_loss = []

        log_interval = 100
        val_interval = 50
        # val_interval = log_interval / \
        #     5 if config["use_early_stopping"] else log_interval

        while (cur_step < config['steps'] and (plateau_step < config['early_stopping_steps'] or not config["use_early_stopping"])):
            for images, labels in train_loader:
                if (cur_step == config['steps'] or (plateau_step == config['early_stopping_steps'] and config["use_early_stopping"])):
                    break

                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = net(images)
                if (config['use_clipping']):
                    outputs = clip_logits(
                        outputs=outputs, scaling_factor=config['clipping_factor'])
                outputs = F.log_softmax(outputs/temperature, dim=1)
                labels = F.softmax(labels/temperature, dim=1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if (config['use_adaptive_lr']):
                    scheduler.step()

                plateau_step += 1

                if ((cur_step+1) % val_interval == 0):
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

                    if (step_val_acc >= best_val_acc):
                        plateau_step = 0
                        best_val_acc = step_val_acc
                        best_val_param = get_parameters(net)

                    total_step_acc.append(step_val_acc)

                total_step_loss.append(loss.item())

                cur_step += 1
                if (cur_step % log_interval == 0 and enable_step_logging):
                    log(INFO,
                        f"step {cur_step}, val_acc : {total_step_acc[-1]}")

        log(INFO, f'Distillation training stopped at step number : {cur_step}')

        new_parameters = best_val_param

        train_res = {'fusion_loss': np.mean(
            total_step_loss), 'fusion_acc': np.mean(total_step_acc)}

        log(INFO,
            f'Average fusion training loss : {train_res["fusion_loss"]}, val accuracy : {train_res["fusion_acc"]}, best val accuracy : {best_val_acc}')

        return (new_parameters, train_res)


# callback functions for fed df

class fed_df_fn:
    @staticmethod
    def get_on_fit_config_fn_client(max_rounds: int = 30, client_epochs: int = 20, client_lr: float = 0.1, clipping_factor: float = 1.0, use_clipping: bool = True, use_adaptive_lr: bool = True, use_adaptive_lr_round: bool = False) -> Callable:
        def on_fit_config_fn_client(server_round: int, use_fedprox: bool, fedprox_factor: float) -> Dict[str, float]:
            lr = client_lr
            if (use_adaptive_lr_round):
                lr = cosine_annealing_round(
                    max_lr=client_lr, min_lr=1e-3, max_rounds=max_rounds, curr_round=server_round)
                log(INFO, f"Current Round Client LR : {lr}")

            config = {
                'lr': lr,
                'epochs': client_epochs,
                'clipping_factor': clipping_factor,
                'use_clipping': use_clipping,
                'use_fedprox': use_fedprox,
                'fedprox_factor': fedprox_factor,
                'use_adaptive_lr': use_adaptive_lr
            }
            return config
        return on_fit_config_fn_client

    @staticmethod
    def get_on_fit_config_fn_server(distill_steps: int, use_early_stopping: bool, early_stop_steps: int, use_adaptive_lr: bool = True, server_lr: float = 1e-3, warm_start: bool = False, clipping_factor: float = 1.0, use_clipping: bool = True) -> Callable:
        def on_fit_config_fn_server() -> Dict[str, float]:
            config = {
                "steps": distill_steps,
                "early_stopping_steps": early_stop_steps,
                "use_early_stopping": use_early_stopping,
                "use_adaptive_lr": use_adaptive_lr,
                "lr": server_lr,
                "temperature": 1,
                "warm_start": warm_start,
                "clipping_factor": clipping_factor,
                "use_clipping": use_clipping,
            }
            return config
        return on_fit_config_fn_server

    @staticmethod
    def evaluate_fn(model_params: NDArray, model_name: str, dataset_name: str, test_loader: DataLoader, device: torch.device) -> Tuple[float, Dict[str, float]]:
        test_res = test_model(model_name, dataset_name,
                              model_params, test_loader, device)
        test_loss = test_res['test_loss']
        test_acc = test_res['test_acc']
        return test_loss, {'server_test_acc': test_acc}
