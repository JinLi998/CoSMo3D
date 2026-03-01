import os
import argparse
from dataclasses import dataclass, field, asdict, fields
from datetime import datetime
from typing import Any, Optional, Union, List, Dict

import release_module.decoder
# from craftsman.utils.typing import *


# ============ 替代 OmegaConf 解析器的工具函数 ============= #
def calc_exp_lr_decay_rate(factor: float, n: int) -> float:
    return factor ** (1.0 / n)

def add(a: float, b: float) -> float:
    return a + b

def sub(a: float, b: float) -> float:
    return a - b

def mul(a: float, b: float) -> float:
    return a * b

def div(a: float, b: float) -> float:
    return a / b

def idiv(a: int, b: int) -> int:
    return a // b

def basename(p: str) -> str:
    return os.path.basename(p)

def rmspace(s: str, sub: str) -> str:
    return str(s).replace(" ", sub)

def tuple2(s: float) -> List[float]:
    return [s, s]

def gt0(s: float) -> bool:
    return s > 0

def cmaxgt0(s: Any) -> bool:
    return C_max(s) > 0

def not_(s: bool) -> bool:
    return not s

def cmaxgt0orcmaxgt0(a: Any, b: Any) -> bool:
    return C_max(a) > 0 or C_max(b) > 0

# ======================================================= #


def C_max(value: Any) -> float:
    """原 OmegaConf 解析器依赖的 C_max 函数，逻辑不变"""
    if isinstance(value, int) or isinstance(value, float):
        pass
    else:
        value = config_to_primitive(value)
        if not isinstance(value, list):
            raise TypeError("Scalar specification only supports list, got", type(value))
        if len(value) >= 6:
            max_value = value[2]
            for i in range(4, len(value), 2):
                max_value = max(max_value, value[i])
            value = [value[0], value[1], max_value, value[3]]
        if len(value) == 3:
            value = [0] + value
        assert len(value) == 4
        start_step, start_value, end_value, end_step = value
        value = max(start_value, end_value)
    return value


@dataclass
class ExperimentConfig:
    """配置类，保持原 dataclass 结构不变"""
    name: str = "default"
    description: str = ""
    tag: str = ""
    seed: int = 0
    use_timestamp: bool = True
    timestamp: Optional[str] = None
    exp_root_dir: str = "outputs"

    ### 自动生成的路径，不建议手动设置
    exp_dir: str = "outputs/default"
    trial_name: str = "exp"
    trial_dir: str = "outputs/default/exp"
    n_gpus: int = 1
    ###

    resume: Optional[str] = None

    data_type: str = ""
    data: dict = field(default_factory=dict)

    system_type: str = ""
    system: dict = field(default_factory=dict)

    # pytorch-lightning trainer 参数
    trainer: dict = field(default_factory=dict)

    # 模型 checkpoint 参数
    checkpoint: dict = field(default_factory=dict)

    def __post_init__(self):
        """初始化后自动生成路径，逻辑不变"""
        if not self.tag and not self.use_timestamp:
            raise ValueError("Either tag is specified or use_timestamp is True.")
        self.trial_name = self.tag
        if self.timestamp is None:
            self.timestamp = ""
            if self.use_timestamp:
                if self.n_gpus > 1:
                    craftsman.warn(
                        "Timestamp is disabled when using multiple GPUs, please make sure you have a unique tag."
                    )
                else:
                    self.timestamp = datetime.now().strftime("@%Y%m%d-%H%M%S")
        self.trial_name += self.timestamp
        self.exp_dir = os.path.join(self.exp_root_dir, self.name)
        self.trial_dir = os.path.join(self.exp_dir, self.trial_name)
        os.makedirs(self.trial_dir, exist_ok=True)


# ============ 替代 OmegaConf 的核心功能函数 ============= #
def _merge_dicts(a: Dict, b: Dict) -> Dict:
    """递归合并两个字典（替代 OmegaConf.merge）"""
    merged = a.copy()
    for k, v in b.items():
        if isinstance(v, dict) and k in merged and isinstance(merged[k], dict):
            merged[k] = _merge_dicts(merged[k], v)
        else:
            merged[k] = v
    return merged

def _parse_cli_args(cli_args: List[str]) -> Dict:
    """解析 CLI 参数为字典（替代 OmegaConf.from_cli）"""
    parser = argparse.ArgumentParser()
    for arg in cli_args:
        if "=" in arg:
            key, value = arg.split("=", 1)
            # 简单类型转换（根据需要扩展）
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            elif value.isdigit():
                value = int(value)
            elif "." in value and all(part.isdigit() for part in value.split(".")):
                value = float(value)
            parser.add_argument(f"--{key}", type=type(value), default=value)
    args = parser.parse_args([])  # 仅解析格式，不读取命令行
    return vars(args)

def load_config(*yamls: str, cli_args: list = [], from_string=False, **kwargs) -> ExperimentConfig:
    """加载并合并配置（替代原 OmegaConf 逻辑）"""
    # 加载 YAML 配置（此处用字典模拟，实际需用 yaml 库加载）
    yaml_confs = []
    if from_string:
        # 从字符串加载（需用 yaml.safe_load）
        import yaml
        yaml_confs = [yaml.safe_load(s) for s in yamls]
    else:
        # 从文件加载（需用 yaml.safe_load）
        import yaml
        for f in yamls:
            with open(f, "r") as fp:
                yaml_confs.append(yaml.safe_load(fp))
    
    # 解析 CLI 参数
    cli_conf = _parse_cli_args(cli_args)
    
    # 合并所有配置（YAMLs -> CLI -> 关键字参数）
    merged = {}
    for conf in yaml_confs + [cli_conf, kwargs]:
        if conf:
            merged = _merge_dicts(merged, conf)
    
    # 转换为 ExperimentConfig 实例
    return parse_structured(ExperimentConfig, merged)

def config_to_primitive(config: Any, resolve: bool = True) -> Any:
    """将配置转换为原生 Python 类型（替代 OmegaConf.to_container）"""
    if isinstance(config, ExperimentConfig):
        return asdict(config)
    elif isinstance(config, dict):
        return {k: config_to_primitive(v, resolve) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_to_primitive(v, resolve) for v in config]
    else:
        return config

def dump_config(path: str, config: ExperimentConfig) -> None:
    """保存配置到文件（替代 OmegaConf.save）"""
    import yaml
    with open(path, "w") as fp:
        yaml.safe_dump(asdict(config), fp, sort_keys=False)

def parse_structured(cls: Any, cfg: Optional[Union[dict, Any]] = None) -> Any:
    """将字典转换为 dataclass 实例（替代 OmegaConf.structured）"""
    if cfg is None:
        return cls()
    # 过滤 cfg 中不存在于 dataclass 的字段
    valid_fields = {f.name for f in fields(cls)}
    filtered_cfg = {k: v for k, v in cfg.items() if k in valid_fields}
    return cls(** filtered_cfg)

'''import os
from dataclasses import dataclass, field
from datetime import datetime

from omegaconf import OmegaConf

import craftsman
from craftsman.utils.typing import *

# ============ Register OmegaConf Recolvers ============= #
OmegaConf.register_new_resolver(
    "calc_exp_lr_decay_rate", lambda factor, n: factor ** (1.0 / n)
)
OmegaConf.register_new_resolver("add", lambda a, b: a + b)
OmegaConf.register_new_resolver("sub", lambda a, b: a - b)
OmegaConf.register_new_resolver("mul", lambda a, b: a * b)
OmegaConf.register_new_resolver("div", lambda a, b: a / b)
OmegaConf.register_new_resolver("idiv", lambda a, b: a // b)
OmegaConf.register_new_resolver("basename", lambda p: os.path.basename(p))
OmegaConf.register_new_resolver("rmspace", lambda s, sub: str(s).replace(" ", sub))
OmegaConf.register_new_resolver("tuple2", lambda s: [float(s), float(s)])
OmegaConf.register_new_resolver("gt0", lambda s: s > 0)
OmegaConf.register_new_resolver("cmaxgt0", lambda s: C_max(s) > 0)
OmegaConf.register_new_resolver("not", lambda s: not s)
OmegaConf.register_new_resolver(
    "cmaxgt0orcmaxgt0", lambda a, b: C_max(a) > 0 or C_max(b) > 0
)
# ======================================================= #


def C_max(value: Any) -> float:
    if isinstance(value, int) or isinstance(value, float):
        pass
    else:
        value = config_to_primitive(value)
        if not isinstance(value, list):
            raise TypeError("Scalar specification only supports list, got", type(value))
        if len(value) >= 6:
            max_value = value[2]
            for i in range(4, len(value), 2):
                max_value = max(max_value, value[i])
            value = [value[0], value[1], max_value, value[3]]
        if len(value) == 3:
            value = [0] + value
        assert len(value) == 4
        start_step, start_value, end_value, end_step = value
        value = max(start_value, end_value)
    return value


@dataclass
class ExperimentConfig:
    name: str = "default"
    description: str = ""
    tag: str = ""
    seed: int = 0
    use_timestamp: bool = True
    timestamp: Optional[str] = None
    exp_root_dir: str = "outputs"

    ### these shouldn't be set manually
    exp_dir: str = "outputs/default"
    trial_name: str = "exp"
    trial_dir: str = "outputs/default/exp"
    n_gpus: int = 1
    ###

    resume: Optional[str] = None

    data_type: str = ""
    data: dict = field(default_factory=dict)

    system_type: str = ""
    system: dict = field(default_factory=dict)

    # accept pytorch-lightning trainer parameters
    # see https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-class-api
    trainer: dict = field(default_factory=dict)

    # accept pytorch-lightning checkpoint callback parameters
    # see https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html#modelcheckpoint
    checkpoint: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.tag and not self.use_timestamp:
            raise ValueError("Either tag is specified or use_timestamp is True.")
        self.trial_name = self.tag
        # if resume from an existing config, self.timestamp should not be None
        if self.timestamp is None:
            self.timestamp = ""
            if self.use_timestamp:
                if self.n_gpus > 1:
                    craftsman.warn(
                        "Timestamp is disabled when using multiple GPUs, please make sure you have a unique tag."
                    )
                else:
                    self.timestamp = datetime.now().strftime("@%Y%m%d-%H%M%S")
        self.trial_name += self.timestamp
        self.exp_dir = os.path.join(self.exp_root_dir, self.name)
        self.trial_dir = os.path.join(self.exp_dir, self.trial_name)
        # os.makedirs(self.trial_dir, exist_ok=True)


def load_config(*yamls: str, cli_args: list = [], from_string=False, **kwargs) -> Any:
    if from_string:
        yaml_confs = [OmegaConf.create(s) for s in yamls]
    else:
        yaml_confs = [OmegaConf.load(f) for f in yamls]
    cli_conf = OmegaConf.from_cli(cli_args)
    cfg = OmegaConf.merge(*yaml_confs, cli_conf, kwargs)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    scfg = parse_structured(ExperimentConfig, cfg)
    return scfg


def config_to_primitive(config, resolve: bool = True) -> Any:
    return OmegaConf.to_container(config, resolve=resolve)


def dump_config(path: str, config) -> None:
    with open(path, "w") as fp:
        OmegaConf.save(config=config, f=fp)


def parse_structured(fields: Any, cfg: Optional[Union[dict, DictConfig]] = None) -> Any:
    scfg = OmegaConf.structured(fields(**cfg))
    return scfg'''
