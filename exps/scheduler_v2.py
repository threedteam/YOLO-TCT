import subprocess
import os
import time
import json
import yaml
from pathlib import Path
import logging
import shutil
from datetime import datetime

def setup_logging(log_dir: str, name: str, main_log_file: str) -> logging.Logger:
    """Set up logging for the experiment and main log"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.join(log_dir, f'{name}.log') for h in logger.handlers):
        # Experiment-specific file handler
        exp_file_handler = logging.FileHandler(os.path.join(log_dir, f'{name}.log'))
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        exp_file_handler.setFormatter(formatter)
        logger.addHandler(exp_file_handler)

    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == main_log_file for h in logger.handlers):
        # Main log file handler
        main_file_handler = logging.FileHandler(main_log_file)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        main_file_handler.setFormatter(formatter)
        logger.addHandler(main_file_handler)

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        # Console handler
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

def load_yaml(cfg_path: str) -> dict:
    """Load configuration file"""
    with open(cfg_path, 'r') as file:
        return yaml.safe_load(file)

def load_json(json_path: str) -> list:
    """Load JSON file"""
    with open(json_path, "r") as f:
        return json.load(f)

def format_dict(dictionary: dict) -> str:
    """Format dictionary for pretty printing"""
    return '\n'.join(f'{key:<20}: {value}' for key, value in dictionary.items())

def ensure_directory(path: str):
    """Ensure directory exists, create if it doesn't"""
    Path(path).mkdir(parents=True, exist_ok=True)

def run_experiment(cfg: dict, logger: logging.Logger):
    """Run a single experiment"""
    model_cfg, model_pt = cfg['model_cfg'], cfg['model_pt']
    dataset_cfg = cfg['dataset_cfg']
    work_dir = cfg['work_dir']
    result_dir = cfg['result_dir']
    name = cfg['name']

    train_params = f"--epochs {cfg['epochs']} --imgsz {cfg['imgsz']}"
    model_params = f"--model_cfg {model_cfg} --model_pt {model_pt} --dataset_cfg {dataset_cfg}"
    overrides = {**cfg['override_cfg'], 'project': work_dir, 'name': name, 'batch': -1 if 'yolov6' not in model_cfg else 16}
    override_params = f"--override \"{overrides}\""
    result_save_path_param = f"--result_save_path {result_dir}/{name}.json"

    command = f"{cfg['interpreter']} yolo_entrance.py {train_params} {model_params} {override_params} {result_save_path_param}"

    logger.info(f'Starting training for {name}')
    logger.info(f'Command: {command}')

    start_time = time.time()
    with open(os.path.join(cfg['log_dir'], f'{name}_subprocess.log'), 'w') as subprocess_log:
        train_process = subprocess.Popen(command, shell=True, stdout=subprocess_log, stderr=subprocess.STDOUT)
        train_process.wait()

    train_time = time.time() - start_time
    logger.info(f"Training completed for {name}. Time taken: {train_time:.2f}s")

    predictions_path = os.path.join(work_dir, name, 'predictions.json')
    assert os.path.exists(predictions_path), f"Training for {name} did not complete successfully."

    test_result = load_json(f"{result_dir}/{name}.json")
    logger.info(f"{name} test results:\n{format_dict(test_result[0])}")
    logger.info(f"{name} speed results:\n{format_dict(test_result[1])}")

def main(model_cfgs, dataset_cfgs, max_epochs=200):
    
    proj_root = '/data/yolov10' # mod this
    work_root = f'{proj_root}/exps/exp_dirs' # mod this
    log_root = f'{proj_root}/exps/scheduler_runs' # mod this
    interpreter = '/data/conda_envs/yolov10/bin/python' # mod this

    # Create main log file with current date
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M")
    main_log_file = os.path.join(log_root, f'{current_date}.log')

    override_cfg = load_yaml(f'{work_root}/cfg.yaml')

    # Setup main logger
    main_logger = logging.getLogger('main')
    main_logger.setLevel(logging.INFO)
    main_file_handler = logging.FileHandler(main_log_file)
    main_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    main_file_handler.setFormatter(main_formatter)
    main_logger.addHandler(main_file_handler)

    main_logger.info(f"Starting experiments on {current_date}")

    for dataset_cfg in dataset_cfgs:
        for description, cfgs in model_cfgs.items():
            # 清空 {work_root}/{description} 目录
            description_dir = os.path.join(work_root, description)
            if os.path.exists(description_dir):
                for item in os.listdir(description_dir):
                    item_path = os.path.join(description_dir, item)
                    if os.path.isfile(item_path):
                        os.unlink(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                main_logger.info(f"已清空目录: {description_dir}")
            else:
                main_logger.info(f"目录不存在，无需清空: {description_dir}")

            run_index = 1
            for cfg in cfgs:
                model_cfg, model_pt = cfg
                name = model_cfg.split('.')[0]

                base_log_root = f'{work_root}/{description}/{name}_{dataset_cfg.split(".")[0]}'
                log_root = base_log_root

                while os.path.exists(log_root):
                    log_root = f"{base_log_root}_{run_index}"
                    run_index += 1

                ensure_directory(log_root)

                for folder in ['records', 'workdirs', 'results']:
                    ensure_directory(f'{log_root}/{folder}')

                logger = setup_logging(f'{log_root}/records', name, main_log_file)

                experiment_config = {
                    'model_cfg': model_cfg,
                    'model_pt': model_pt,
                    'dataset_cfg': dataset_cfg,
                    'work_dir': f'{log_root}/workdirs',
                    'result_dir': f'{log_root}/results',
                    'log_dir': f'{log_root}/records',
                    'name': name,
                    'epochs': max_epochs,
                    'imgsz': 640,
                    'override_cfg': override_cfg,
                    'interpreter': interpreter
                }

                main_logger.info(f"Starting experiment: {name}")
                logger.info(f"Starting experiment: {name}")
                logger.info(f"Config:\n{format_dict(experiment_config)}")

                run_experiment(experiment_config, logger)

                logger.info(f"Experiment {name} completed\n")
                main_logger.info(f"Experiment {name} completed\n")

    main_logger.info("All experiments completed")

if __name__ == "__main__":
    model_cfgs = {
        "baseline_cbqfocal": [
            ["yolov3u.yaml", "yolov3u.pt"],
            
            ["yolov5n.yaml", "yolov5nu.pt"],
            ["yolov5s.yaml", "yolov5su.pt"],
            ["yolov5m.yaml", "yolov5mu.pt"],
            ["yolov5l.yaml", "yolov5lu.pt"],
            ["yolov5x.yaml", "yolov5xu.pt"],
            
            ["yolov6n.yaml", "yolov6n.pt"],
            ["yolov6s.yaml", "yolov6s.pt"],
            ["yolov6m.yaml", "yolov6m.pt"],
            ["yolov6l.yaml", "yolov6l.pt"],
            
            ["yolov8n.yaml", "yolov8n.pt"],
            ["yolov8s.yaml", "yolov8s.pt"],
            ["yolov8m.yaml", "yolov8m.pt"],
            ["yolov8l.yaml", "yolov8l.pt"],
            ["yolov8x.yaml", "yolov8x.pt"],
            
            ["yolov9c.yaml",'yolov9c.pt'],
            ["yolov9e.yaml", "yolov9e.pt"],
            
            ["yolov10n.yaml",'yolov10n.pt'],
            ["yolov10s.yaml",'yolov10s.pt'],
            ["yolov10m.yaml",'yolov10m.pt'],
            ["yolov10b.yaml",'yolov10b.pt'],
            ["yolov10l.yaml",'yolov10l.pt'],
            ["yolov10x.yaml",'yolov10x.pt'],
            
            ["yolov9c-LRRDetectHead.yaml",'yolov9c.pt'], # ours
        ],
    }
    
    
    dataset_cfgs = [
        'cquchv4_aug.yaml',
        # "VOC.yaml",
    ]
    main(model_cfgs, dataset_cfgs, 200)
