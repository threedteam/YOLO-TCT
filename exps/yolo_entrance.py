import argparse, time, json
from ultralytics import YOLO, RTDETR

def format_dict(dictionary):
    formatted_str = ''
    max_key_length = max(len(str(key)) for key in dictionary.keys())
    
    for key, value in dictionary.items():
        formatted_str += f'{str(key):<{max_key_length}} : {str(value)}\n'
    
    return formatted_str

def save_dict_list_to_json(dict_list, file_path):
    with open(file_path, 'w') as file:
        json.dump(dict_list, file)

def parse_dict(string):
    try:
        return eval(string)
    except ValueError:
        raise argparse.ArgumentTypeError('Invalid dictionary format')

def parse_arguments():
    """parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='YOLO training entrance'
    )

    parser.add_argument(
        '--epochs', type=int
    )
    parser.add_argument(
        '--imgsz', type=int
    )

    parser.add_argument(
        '--model_cfg', type=str
    )
    parser.add_argument(
        '--model_pt', type=str
    )
    parser.add_argument(
        '--dataset_cfg', type=str
    )

    parser.add_argument(
        '--override', type=parse_dict
    )  
    parser.add_argument(
        '--result_save_path', type=str
    )
    return parser.parse_args()

args = parse_arguments()

ultralytics_func = YOLO if 'yolo' in args.model_cfg else RTDETR

model = ultralytics_func(args.model_cfg).load(args.model_pt)
override_cfg = args.override

for key, value in override_cfg.items():
    model.overrides[key] = value

start_time = time.time()
results = model.train(data=args.dataset_cfg, epochs=args.epochs, imgsz=args.imgsz)
train_time = time.time() - start_time

save_dict_list_to_json([
    results.results_dict, 
    results.speed, 
    {'total_training_time':train_time}
    ], 
    args.result_save_path
)