import os
from typing import Type, Dict, Optional, Tuple
from einops import rearrange
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch import optim
from ts_benchmark.baselines.pre_train.model_plugin.Plugin import Plugin
from ts_benchmark.baselines.time_series_library.utils.tools import (
    EarlyStopping,
    adjust_learning_rate,
)
from torch.nn.functional import interpolate
import torch.nn.functional as F
import itertools
from ts_benchmark.baselines.utils_few import (
    forecasting_data_provider,
    train_val_split,
    anomaly_detection_data_provider,
    get_time_mark,
)
from ts_benchmark.models.model_base import ModelBase, BatchMaker
from ts_benchmark.utils.data_processing import split_before
import time
from ts_benchmark.baselines.pre_train.layers.fre_rec_loss import frequency_criterion
DEFAULT_PreTrain_BASED_HYPER_PARAMS = {
    "freq": "h",
    "num_samples": 100,
    "quantiles_num": 20,
    "batch_size": 64,
    "test_batch_size": 1,
    "num_workers": 0,
    "ckpt_path":"",
    "dataset":"ETTh1",
    'target_dim': 1, 
    'label_len':96,
    "lr": 0.0001,
    "patience": 3,
    "loss": "MSE",
    "itr": 1,
    "num_epochs": 20,
    "sampling_rate": 0.05,
    "sampling_strategy": "uniform",
    "sampling_basis": "sample",
    "is_train": 1,
    "get_train": 0,
    "ending": 0,
    "lradj": "type1",
    "use_p": 1,
    "patch_size": 64,
    "mask_ratio": 0.3,
    "patch_len": 96,
    "horizon": 0,
    "anomaly_ratio": [0.1, 0.5, 1.0, 2, 3, 5.0, 10.0, 15, 20, 25],
    "num_experts":20,
    "K":5,
    "rank":16,
    "alpha":8,
    "mode":'plugin',
    "mask": False,
    "inference_patch_stride": 1,
    "inference_patch_size": 32,
    "auxi_loss": "MAE",
    "auxi_type": "complex",
    "auxi_mode": "fft",
    "auxi_lambda": 0.005,
    # "score_lambda": 0.5,
    "score_lambda": 0.05,
    "regular_lambda": 0.5,
    "module_first": True,
    "dc_lambda": 0.005,
    "mask": False,
    "noise_level":0.1,
    "discrete_noise":True,
    "eve":"normal",
    "score_alpha":0.05,
    "backbone":"DADA",
}
   
class PreTrainConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_PreTrain_BASED_HYPER_PARAMS.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def pred_len(self):
        return self.horizon


class PreTrainAdapter(ModelBase):
    def __init__(self, model_name, model_class, **kwargs):
        super(PreTrainAdapter, self).__init__()
        self.config = PreTrainConfig(**kwargs)
        self._model_name = model_name
        self.model_class = model_class
        self.scaler = StandardScaler()
        self.seq_len = self.config.seq_len
        self.win_size = self.config.seq_len

    @staticmethod
    def required_hyper_params() -> dict:
        """
        Return the hyperparameters required by model.

        :return: An empty dictionary indicating that model does not require additional hyperparameters.
        """
        return {}

    @property
    def model_name(self):
        """
        Returns the name of the model.
        """

        return self._model_name
    
    def detect_hyper_param_tune(self, train_data: pd.DataFrame):
        try:
            freq = pd.infer_freq(train_data.index)
        except Exception as ignore:
            freq = 'S'
        if freq == None:
            raise ValueError("Irregular time intervals")
        elif freq[0].lower() not in ["m", "w", "b", "d", "h", "t", "s"]:
            self.config.freq = "s"
        else:
            self.config.freq = freq[0].lower()

        column_num = train_data.shape[1]
        self.config.enc_in = column_num
        self.config.dec_in = column_num
        self.config.c_out = column_num
        self.config.label_len = 48
    
    
    def detect_validate(self, valid_data_loader, criterion, criterion_cls=None):
        total_moe,total_cls,total_mse,total_con = [], [], [], []
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       
        for continuous, discrete, _ in valid_data_loader:
            continuous = continuous.to(device)
            discrete = discrete.to(device)
            b,t,n = discrete.shape           

            # contrastive loss
            if self.config.mode=='plugin':
                result_dict = self.model(continuous, discrete)
                # 对比学习
                con_loss = self.con_loss(result_dict['emb_c'], result_dict['emb_d'])
                # reconstruct loss
                mse_loss = self.mse_loss(result_dict['rec_c'],continuous)
                # MoELoss
                L_importance = result_dict['L_importance']

                total_mse.append(mse_loss.detach().cpu().numpy())
                total_con.append(con_loss.detach().cpu().numpy())
                total_moe.append(L_importance.detach().cpu().numpy())

            elif self.config.mode=='backbone' or self.config.mode=='LoRA' or self.config.mode=='LoRA':
                result_dict = self.model(continuous, discrete)
                mse_loss = self.mse_loss(result_dict['rec_c'],continuous)
                total_mse.append(mse_loss.detach().cpu().numpy())
            
        if self.config.mode=='plugin':
            total_con = np.mean(total_con)
            total_moe = np.mean(total_moe)
            total_mse = np.mean(total_mse)
            print(f'mse: {total_mse} cls: {total_cls} con: {total_con} moe: {total_moe}')
        elif self.config.mode=='backbone' or self.config.mode=='LoRA':
            total_mse = np.mean(total_mse)
            total_con = 0

        self.model.train()
        return total_mse  + self.config.score_alpha * total_con
       
    def contrastive_loss(self, emb_d, emb_c, temperature=0.07):

        emb_d = emb_d.mean(2)
        emb_c = emb_c.mean(2)
        b, p, d = emb_c.shape  
        
        queries = F.normalize(emb_d, p=2, dim=-1)
        keys = F.normalize(emb_c, p=2, dim=-1)
        logits = torch.matmul(queries, keys.transpose(-2, -1))
        logits = logits / temperature
        labels = torch.arange(p, device=logits.device).unsqueeze(0).expand(b, -1)
        return F.cross_entropy(logits.reshape(-1, p), labels.reshape(-1))

    
    def detect_fit(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        """
        训练模型。

        :param train_data: 用于训练的时间序列数据。
        """

        self.detect_hyper_param_tune(train_data)
        setattr(self.config, "task_name", "anomaly_detection")
        config = self.config
        train_data_value, valid_data = train_val_split(train_data, 0.8, None)
        self.scaler.fit(train_data_value.values)

        train_data_value = pd.DataFrame(
            self.scaler.transform(train_data_value.values),
            columns=train_data_value.columns,
            index=train_data_value.index,
        )

        valid_data = pd.DataFrame(
            self.scaler.transform(valid_data.values),
            columns=valid_data.columns,
            index=valid_data.index,
        )



        self.train_data_loader = anomaly_detection_data_provider(
            train_data_value,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="train",
            sampling_rate=config.sampling_rate,
            discrete = True,
        )

        self.valid_data_loader = anomaly_detection_data_provider(
            valid_data,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="val",
            discrete = True,
            discrete_cols =  self.train_data_loader.dataset.discrete_cols,
            continuous_cols = self.train_data_loader.dataset.continuous_cols
           
        )

        discrete_nums = self.train_data_loader.dataset.discrete_nums
        prefix_sum_iterator = itertools.accumulate(discrete_nums)
        prefix_sum_list = ([0] + list(prefix_sum_iterator))[:len(discrete_nums)]
        
        self.config.n_discrete = self.train_data_loader.dataset.n_discrete
        self.config.n_continuous = self.train_data_loader.dataset.n_continuous
        
        self.config.discrete_nums = discrete_nums
        self.model = self.model_class(self.config)

        self.model = Plugin(self.config,  self.model)
        total_params = sum(
            p.numel() for p in self.model.parameters()
        )
        print(f"Total parameters: {total_params}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prefix_sum = torch.tensor(prefix_sum_list).unsqueeze(0).unsqueeze(0).to(self.device)
        
        params_backbone = self.model.backbone.parameters()
        optimizer_backbone = optim.AdamW(params_backbone, lr=config.lr)

        ids_a = {id(p) for p in params_backbone}
        params_rest = [p for p in self.model.parameters() if id(p) not in ids_a]
        
        if self.config.mode=='plugin':
            optimizer_plugin = optim.Adam(params_rest, lr=config.plugin_lr)

        if config.is_train:

            # Define the loss function and optimizer
            criterion = nn.MSELoss()
            self.mse_loss = nn.MSELoss()
            self.con_loss = self.contrastive_loss
            self.early_stopping = EarlyStopping(patience=config.patience)
            self.model.to(self.device)
            total_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            print(f"Total trainable parameters: {total_params}")            

            for epoch in range(config.num_epochs):
                print(f"Starting epoch {epoch + 1}/{config.num_epochs}")
                self.model.train()
                total_moe,total_cls,total_loss,total_con = [], [], [], []
                for i, (continuous, discrete, label) in enumerate(self.train_data_loader):
                    optimizer_backbone.zero_grad()
                    continuous = continuous.float().to(self.device)
                    discrete = discrete.float().to(self.device).long()
                    
                    if self.config.mode=='plugin':
                        optimizer_plugin.zero_grad()
                        result_dict = self.model(continuous, discrete)

                        # 对比学习
                        con_loss = self.con_loss(result_dict['emb_c'], result_dict['emb_d'])

                        mse_loss = self.mse_loss(result_dict['rec_c'],continuous)

                        # MoELoss
                        L_importance = result_dict['L_importance']

                        loss =   mse_loss + L_importance  + con_loss * self.config.score_alpha
                        loss.backward()
                        optimizer_backbone.step()
                        optimizer_plugin.step()
                        total_loss.append(mse_loss.detach().cpu().numpy())
                        total_con.append(con_loss.detach().cpu().numpy())
                        total_moe.append(L_importance.detach().cpu().numpy())

                    elif self.config.mode=='backbone' or self.config.mode=='LoRA':
                        result_dict = self.model(continuous, discrete)
                        loss =  self.mse_loss(result_dict['rec_c'], continuous)
                        loss.backward()
                        optimizer_backbone.step()

                if self.config.mode=='plugin':
                    total_loss = np.mean(total_loss)
                    total_cls = np.mean(total_cls)
                    total_con = np.mean(total_con)
                    total_moe = np.mean(total_moe)
                    print(f'mse: {total_loss} cls: {total_cls} con: {total_con} moe: {total_moe}')
                valid_loss = self.detect_validate(self.valid_data_loader, criterion)
                self.early_stopping(valid_loss, self.model)
                if self.early_stopping.early_stop:
                    print("Early stopping triggered")
                    break

                adjust_learning_rate(optimizer_backbone, epoch + 1, config, config.lr)

                if self.config.mode=='plugin':
                    adjust_learning_rate(optimizer_plugin, epoch + 1, config, config.plugin_lr)
    
    def detect_score(self, test: pd.DataFrame) -> np.ndarray:
        test = pd.DataFrame(
            self.scaler.transform(test.values), columns=test.columns, index=test.index
        )

        if self.config.is_train:
            self.model.load_state_dict(self.early_stopping.check_point)
  
        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")
        self.model.backbone.phase = 'test'
        config = self.config

        self.thre_loader = anomaly_detection_data_provider(
            test,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="thre",
            discrete = True,
            discrete_cols =  self.train_data_loader.dataset.discrete_cols,
            continuous_cols = self.train_data_loader.dataset.continuous_cols,
        )

        self.model.to(self.device)
        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)
        self.freq_anomaly_criterion = frequency_criterion(config)
        attens_energy = []
        attens_rec = []
        attens_sim = []
        for i, (continuous, discrete, label) in enumerate(self.thre_loader):
            continuous = continuous.float().to(self.device)
            discrete = discrete.float().to(self.device)
            result_dict  = self.model(continuous, discrete)
            if self.config.mode=='plugin':
                reconstruct_scores =  torch.mean(self.anomaly_criterion(continuous,result_dict['rec_c']),dim=-1)
                reconstruct_scores+=  torch.mean(self.freq_anomaly_criterion(continuous ,result_dict['rec_c']), dim=-1) * self.config.score_lambda
                similarity_scores = F.cosine_similarity(result_dict['emb_c'], result_dict['emb_d'], dim=-1).squeeze(-1)
                similarity_scores = interpolate(similarity_scores.unsqueeze(1), size=reconstruct_scores.size(1), mode='linear', align_corners=False).squeeze(1)
                attens_rec.append(reconstruct_scores.detach().cpu())
                attens_sim.append(-1*similarity_scores.detach().cpu())

            elif self.config.mode=='backbone' or self.config.mode=='LoRA':
                score =  torch.mean(self.anomaly_criterion(continuous,result_dict['rec_c']),dim=-1)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

        if self.config.mode=='plugin':
            attens_rec = torch.cat(attens_rec, dim=0).reshape([-1])
            attens_sim = torch.cat(attens_sim, dim=0).reshape([-1])
            attens_energy = attens_rec * torch.softmax(attens_sim,dim=0).numpy()
            test_energy = np.array(attens_energy)
        
        elif self.config.mode=='backbone' or self.config.mode=='LoRA':
            if attens_energy and all(arr.ndim > 0 for arr in attens_energy):
                attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
            test_energy = np.array(attens_energy)

        return test_energy, test_energy

    def detect_label(self, test: pd.DataFrame) -> np.ndarray:
        test = pd.DataFrame(
            self.scaler.transform(test.values), columns=test.columns, index=test.index
        )
        # save_name = f'checkpoints/score_alpha:{self.config.score_alpha},K:{self.config.K}, num_shared_experts:{self.config.num_shared_experts}, num_experts:{self.config.num_experts},d_model:{self.config.d_model},plugin_lr:{self.config.plugin_lr}'
        # save_name = 'test_units'
        if self.config.is_train:
            self.model.load_state_dict(self.early_stopping.check_point)
            import copy
            check_point = copy.deepcopy( self.model.state_dict())
            # torch.save(check_point,f'{save_name}.pt')
            # print(f'Save================>{save_name}.pt')
        
        # self.model.load_state_dict(torch.load(f"{save_name}.pt"))
        # print(f'load================> {save_name}.pt')
        
        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")
        self.model.backbone.phase = 'test'
        config = self.config

        self.test_data_loader = anomaly_detection_data_provider(
            test,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="test",
            discrete = True,
            discrete_cols =  self.train_data_loader.dataset.discrete_cols,
            continuous_cols = self.train_data_loader.dataset.continuous_cols,
        )

        self.thre_loader = anomaly_detection_data_provider(
            test,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="thre",
            discrete = True,
            discrete_cols =  self.train_data_loader.dataset.discrete_cols,
            continuous_cols = self.train_data_loader.dataset.continuous_cols,
        )

        attens_energy = []

        self.model.to(self.device)
        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)
        self.freq_anomaly_criterion = frequency_criterion(config)
        
        with torch.no_grad():
            attens_energy = []
            test_labels = []
            attens_rec = []
            attens_sim = []
            for i, (continuous, discrete, label) in enumerate(self.train_data_loader):
                continuous = continuous.float().to(self.device)
                discrete = discrete.float().to(self.device)
                result_dict = self.model(continuous, discrete)

                if self.config.mode=='backbone':
                    reconstruct_scores = self.anomaly_criterion(continuous,result_dict['rec_c']).mean(-1)
                    attens_rec.append(reconstruct_scores.detach().cpu())
                elif self.config.mode=='plugin':
                    reconstruct_scores = self.anomaly_criterion(continuous,result_dict['rec_c']).mean(-1) + torch.mean(self.freq_anomaly_criterion(continuous ,result_dict['rec_c']), dim=-1) * self.config.score_lambda
                    similarity_scores = F.cosine_similarity(result_dict['emb_c'], result_dict['emb_d'], dim=-1).squeeze(-1)
                    similarity_scores = interpolate(similarity_scores.unsqueeze(1), size=reconstruct_scores.size(1), mode='linear', align_corners=False).squeeze(1)
                    attens_rec.append(reconstruct_scores.detach().cpu())
                    attens_sim.append(-1*similarity_scores.detach().cpu())

            attens_rec = torch.cat(attens_rec, dim=0).reshape([-1])
            train_energy_rec = attens_rec.clone()
            if self.config.mode=='plugin':
                attens_sim = torch.cat(attens_sim, dim=0).reshape([-1])
                train_energy_sim = attens_sim.clone()
                

            attens_energy = []
            test_labels = []
            attens_rec = []
            attens_sim = []
            
            for i, (continuous, discrete, label) in enumerate(self.test_data_loader):
                continuous = continuous.float().to(self.device)
                discrete = discrete.float().to(self.device)
                result_dict = self.model(continuous, discrete)
                if self.config.mode=='backbone':
                    reconstruct_scores = self.anomaly_criterion(continuous,result_dict['rec_c']).mean(-1)
                    attens_rec.append(reconstruct_scores.detach().cpu())
                elif self.config.mode=='plugin':
                    reconstruct_scores = self.anomaly_criterion(continuous,result_dict['rec_c']).mean(-1) + torch.mean(self.freq_anomaly_criterion(continuous ,result_dict['rec_c']), dim=-1) * self.config.score_lambda
                    similarity_scores = F.cosine_similarity(result_dict['emb_c'], result_dict['emb_d'], dim=-1).squeeze(-1)
                    similarity_scores = interpolate(similarity_scores.unsqueeze(1), size=reconstruct_scores.size(1), mode='linear', align_corners=False).squeeze(1)
                    attens_rec.append(reconstruct_scores.detach().cpu())
                    attens_sim.append(-1*similarity_scores.detach().cpu())

            attens_rec = torch.cat(attens_rec, dim=0).reshape([-1])
            test_energy_rec = attens_rec.clone()

            if self.config.mode=='plugin':
                attens_sim = torch.cat(attens_sim, dim=0).reshape([-1])
                test_energy_sim = attens_sim.clone()
            
            if self.config.mode=='backbone':
                combined_energy_rec = torch.cat((train_energy_rec, test_energy_rec),dim=0)
                combined_energy = combined_energy_rec.numpy()
                
            elif self.config.mode=='plugin':
                combined_energy_sim = torch.cat((train_energy_sim, test_energy_sim),dim=0)
                combined_energy_rec = torch.cat((train_energy_rec, test_energy_rec),dim=0)
                # combined_energy = combined_energy_rec.numpy()
                # combined_energy = (combined_energy_sim * combined_energy_rec).numpy()

            test_labels = []
            attens_rec = []
            attens_sim = []

            for i, (continuous, discrete, label) in enumerate(self.thre_loader):
                continuous = continuous.float().to(self.device)
                discrete = discrete.float().to(self.device)
                result_dict = self.model(continuous, discrete)
                if self.config.mode=='backbone':
                    reconstruct_scores = self.anomaly_criterion(continuous, result_dict['rec_c']).mean(-1)
                    attens_rec.append(reconstruct_scores.detach().cpu())

                elif self.config.mode=='plugin':
                    reconstruct_scores = self.anomaly_criterion(continuous, result_dict['rec_c']).mean(-1) + torch.mean(self.freq_anomaly_criterion(continuous ,result_dict['rec_c']), dim=-1) * self.config.score_lambda
                    similarity_scores = F.cosine_similarity(result_dict['emb_c'], result_dict['emb_d'], dim=-1).squeeze(-1)
                    similarity_scores = interpolate(similarity_scores.unsqueeze(1), size=reconstruct_scores.size(1), mode='linear', align_corners=False).squeeze(1)
                    attens_rec.append(reconstruct_scores.detach().cpu())
                    attens_sim.append(-1*similarity_scores.detach().cpu())
                test_labels.append(label)
  

            attens_rec = torch.cat(attens_rec, dim=0).reshape([-1])
            
            if self.config.mode=='plugin':
                attens_sim = torch.cat(attens_sim, dim=0).reshape([-1])
            
            if self.config.mode=='backbone':
                thre_energy = attens_rec.numpy()
            elif self.config.mode=='plugin':
                thre_energy_sim = attens_sim  
                thre_energy_rec = attens_rec  #.numpy() #(torch.softmax(attens_sim,dim=0) * attens_rec).numpy()
        

        if not isinstance(self.config.anomaly_ratio, list):
            self.config.anomaly_ratio = [self.config.anomaly_ratio]

        if self.config.mode=='plugin':
            thre_len = thre_energy_sim.size(0)
            all_energy_sim = torch.softmax(torch.cat((combined_energy_sim,thre_energy_sim),dim=0),dim=0)
            thre_energy_sim = all_energy_sim[-thre_len:]
            combined_energy_sim = all_energy_sim[:-thre_len]
            combined_energy = (combined_energy_sim * combined_energy_rec).numpy()
            thre_energy = (thre_energy_sim * thre_energy_rec).numpy()

        preds = {}
        for ratio in self.config.anomaly_ratio:
            threshold = np.percentile(combined_energy, 100 - ratio)
            preds[ratio] = (thre_energy>threshold).astype(int)

            # preds[ratio] = ((test_energy_sim > threshold_sim)|(test_energy_rec  > threshold_rec)).astype(int)
        return preds, thre_energy
    
    def _padding_time_stamp_mark(
        self, time_stamps_list: np.ndarray, padding_len: int
    ) -> np.ndarray:
        return None
    
    def forecast(self, horizon: int, series: pd.DataFrame, **kwargs) -> np.ndarray:
        return super().forecast(horizon, series, **kwargs) 

    def validate(self, valid_data_loader, criterion):
        return None

    def forecast_fit(
        self, train_valid_data: pd.DataFrame, train_ratio_in_tv: float
    ) -> "ModelBase":
        return None
       
    def batch_forecast(
        self, horizon: int, batch_maker: BatchMaker, **kwargs
    ) -> np.ndarray:
        return None

    def _perform_rolling_predictions(
        self,
        horizon: int,
        input_np: np.ndarray,
        all_mark: np.ndarray,
        all_time_stamp: np.ndarray,
        device: torch.device,
    ) -> list:
        return None

    def _get_rolling_data(
        self,
        input_np: np.ndarray,
        output: Optional[np.ndarray],
        all_mark: np.ndarray,
        all_time_stamp: np.ndarray,
        rolling_time: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return None
def generate_model_factory(
    model_name: str, model_class: type, required_args: dict
) -> Dict:
    """
    Generate model factory information for creating Transformer Adapters model adapters.

    :param model_name: Model name.
    :param model_class: Model class.
    :param required_args: The required parameters for model initialization.
    :return: A dictionary containing model factories and required parameters.
    """

    def model_factory(**kwargs) -> PreTrainAdapter:
        """
        Model factory, used to create TransformerAdapter model adapter objects.

        :param kwargs: Model initialization parameters.
        :return:  Model adapter object.
        """
        return PreTrainAdapter(model_name, model_class, **kwargs)

    return {
        "model_factory": model_factory,
        "required_hyper_params": required_args,
    }

def PreTrain_adapter(model_info: Type[object]) -> object:
    if not isinstance(model_info, type):
        raise ValueError("the model_info does not exist")

    return generate_model_factory(
        model_name=model_info.__name__,
        model_class=model_info,
        required_args={
            "seq_len": "input_chunk_length",
            "horizon": "output_chunk_length",
            "norm": "norm",
        },
    )

