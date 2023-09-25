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
    GetPropertiesIns,
    Code
)

from typing import Optional, Tuple, List, Union, Dict, Callable
from flwr.server.strategy import Strategy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from models import init_model, get_parameters, set_parameters
from fed_df_data_loader.common import make_data_loader
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
from joblib import Parallel, delayed, parallel_backend


class FedDF_hetero_strategy(Strategy):
    def __init__(self,
                 num_rounds: int,
                 distillation_dataloader: DataLoader,
                 evaluation_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 model_list: list,
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
        self.model_list = model_list
        self.dataset_name = dataset_name
        self.num_classes = num_classes

        # fraction of participating clients
        self.min_available_clients = min_available_clients
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients

        # last round parameters
        self.last_parameters = None

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
            torch.use_deterministic_algorithms(True)

        # initialising server side models

        log(INFO, f'Models in the server : {self.model_list}')
        self.server_models = {}
        for model_name in self.model_list:
            self.server_models[model_name] = {'model': init_model(
                dataset_name=self.dataset_name, model_name=model_name), 'acc': 0.0}
        log(INFO, f'Server Models initialised')

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
        return None

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        # Sample clients

        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        client_model_pairs = []

        for client in clients:
            properties = client.get_properties(
                GetPropertiesIns(config={'cid': 0, 'model_name': 0}), timeout=None)
            if (properties.status == 200):
                client_model_pairs.append(
                    (properties.properties["cid"], properties.properties["model_name"]))

            else:
                raise ValueError(
                    f'Incorrect key for fetching client property!')

        # initialising config settings for training

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

        # selecting best model for selection mechanism

        best_model_name = max(self.server_models,
                              key=lambda x: self.server_models[x]['acc'])
        log(INFO, f'Best Server Model Name : {best_model_name}')

        # performing kmeans on crops

        net = self.server_models[best_model_name]['model']

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

        net.to('cpu')

        # adding crops to clients and appropriate parameters
        config = {**config, 'distill_crops': clustering.prepare_for_transport(
            pruned_clusters)}
        fit_ins_list = []

        for _, model_name in client_model_pairs:
            parameters = ndarrays_to_parameters(
                get_parameters(self.server_models[model_name]['model']))
            fit_ins_list.append(FitIns(parameters, config))

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
            config = self.on_fit_config_fn_server(
                total_rounds=self.num_rounds, current_round=server_round)
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
            client_model_idx = {}
            for key in self.server_models.keys():
                client_model_idx[key] = []

            i = 0
            for client, res in results:
                if (res.status.code == Code.OK):
                    client_properties = client.get_properties(
                        GetPropertiesIns(config={'model_name': 0}), timeout=None).properties
                    model_name = client_properties['model_name']
                    client_model_idx[model_name].append(i)

                i += 1

            for model_name in client_model_idx.keys():
                if (len(client_model_idx[model_name]) > 0):
                    trimmed_results = []
                    for idx in client_model_idx[model_name]:
                        trimmed_results.append(results[idx])

                    weights_results = [
                        (parameters_to_ndarrays(
                            fit_res.parameters), fit_res.num_examples)
                        for _, fit_res in trimmed_results
                    ]

                    fusion_parameters = aggregate(weights_results)
                    set_parameters(
                        model=self.server_models[model_name]['model'], parameters=fusion_parameters)
                    log(INFO, f'{model_name} server model initialised for fusion')

        # Distilling all student model using Average Logits

        log(INFO, f'Performing server side distillation training in parallel')

        with parallel_backend(backend="threading", n_jobs=4):
            fusion_metrics_list = Parallel()((delayed(self.__fuse_models)(
                self.server_models[model_name]['model'], self.kmeans_last_crops, self.batch_size, logits_aggregated, config, self.val_dataloader, self.device, self.debug) for model_name in self.server_models.keys()))

        log(INFO, f'Fusion results for all models : {fusion_metrics_list}')

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
            self.logger_fn(server_round, fusion_metrics_list, "fit", "server")

        return None, metrics_aggregated

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

        eval_res_list = []

        for model_name in self.server_models.keys():
            new_res = self.evaluate_fn(**{'model_params': get_parameters(
                self.server_models[model_name]['model']), 'model_name': model_name, 'dataset_name': self.dataset_name, 'test_loader': self.evaluation_dataloader, 'device': self.device})
            eval_res_list.append(new_res)

        eval_loss = np.mean(a=[x for x, _ in eval_res_list])
        eval_metrics = {'server_test_acc': np.mean(
            a=[x['server_test_acc'] for _, x in eval_res_list])}
        eval_res = (eval_loss, eval_metrics)

        for model_name, eval_res in zip(self.server_models.keys(), eval_res_list):
            self.server_models[model_name]['acc'] = eval_res[1]['server_test_acc']

        best_model_name = max(self.server_models,
                              key=lambda x: self.server_models[x]['acc'])

        val_res = self.evaluate_fn(model_params=get_parameters(self.server_models[best_model_name]['model']), model_name=best_model_name,
                                   dataset_name=self.dataset_name, test_loader=self.val_dataloader, device=self.device)

        if val_res is None:
            self.hist_server_acc.append(0.0)
        else:
            _, val_met = val_res
            self.hist_server_acc.append(val_met['server_test_acc'])

        if eval_res is None:
            return None

        loss, metrics = eval_res
        if (self.logger_fn is not None):
            self.logger_fn(server_round, metrics, "evaluate", "server")

        return loss, metrics

    @staticmethod
    def __fuse_models(net, img_bytes: bytes, distill_batch_size: int, preds: NDArray, config: Dict[str, float], val_dataloader: DataLoader, DEVICE: torch.device, enable_step_logging: bool = False) -> Parameters:

        dataloader = clustering.extract_from_transport(
            img_bytes=img_bytes, batch_size=distill_batch_size, n_workers=1)
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
                if (config['use_clipping']):
                    outputs_temp = net(images)
                    outputs = clip_logits(
                        outputs=outputs_temp, scaling_factor=config['clipping_factor'])
                else:
                    outputs = net(images)

                outputs = F.log_softmax(outputs/temperature, dim=1)
                labels = F.softmax(labels/temperature, dim=1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                if (config['use_adaptive_lr']):
                    scheduler.step()

                plateau_step += 1

                if ((cur_step+1) % val_interval == 0 or cur_step == (config['steps']-1) or cur_step == 0):
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

        if (best_val_param != None):
            set_parameters(net, parameters=best_val_param)

        train_res = {'fusion_loss': np.mean(
            total_step_loss), 'fusion_acc': np.mean(total_step_acc)}

        net.to('cpu')

        return train_res
