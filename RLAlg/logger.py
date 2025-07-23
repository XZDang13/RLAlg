from typing import Optional
import wandb

class WandbLogger:
    @staticmethod
    def init_project(project_name:str, name:Optional[str]=None, config:Optional[dict]=None):
        wandb.init(project=project_name, name=name, config=config)
        
    @staticmethod
    def log_metrics(metrics:dict, step:int):
        wandb.log(metrics, step=step)