import os
import sys
import json
from collections import Counter
from statistics import mean
import numpy as np

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(parent_dir)

from options import get_experiment_config
from set_up import setup_experiment
from transforms import image_transform_factory
from data import create_dataloaders
from models import create_qure_models
import torch


def _compute_recall_at_k(model, query_features, target_features, target_names, index_names):
    if not target_names:
        raise ValueError("FashionIQ validation split contains no queries")

    index_name_counts = Counter(index_names)
    invalid_targets = [name for name in target_names if index_name_counts[name] != 1]
    if invalid_targets:
        raise ValueError(
            f"{len(invalid_targets)} FashionIQ targets do not occur exactly once in the gallery"
        )

    recall_ks = (5, 10, 50)
    recall_hits = {k: 0 for k in recall_ks}
    index_names = np.asarray(index_names)
    max_k = min(max(recall_ks), len(index_names))

    for start in range(0, len(target_names), 64):
        end = min(start + 64, len(target_names))
        scores = model.score(query_features[start:end], target_features)
        sorted_indices = torch.argsort(scores, dim=-1, descending=True)[:, :max_k].cpu()
        retrieved_names = index_names[sorted_indices]
        batch_targets = np.asarray(target_names[start:end])[:, None]
        matches = retrieved_names == batch_targets

        for k in recall_ks:
            recall_hits[k] += np.any(matches[:, :min(k, max_k)], axis=1).sum()

    return {
        k: recall_hits[k] / len(target_names) * 100
        for k in recall_ks
    }


def main():
    configs = get_experiment_config()
    export_root, configs = setup_experiment(configs)
    device = torch.device(f"cuda:{configs['device_idx']}") if torch.cuda.is_available() else "cpu"
    print(f"Experiment: {configs['experiment_description']}")

    image_transform = image_transform_factory(config=configs)
    train_dataloader, test_dataloaders = create_dataloaders(image_transform, None, configs)

    print(len(train_dataloader), len(test_dataloaders))

    MS_pretrained_path = configs["pretrained_path"]
    print(f"Pretrained Model Path : {MS_pretrained_path}")

    model, txt_processors = create_qure_models(configs, device)
    msg = model.load_state_dict(torch.load(f'{MS_pretrained_path}/model.pth', map_location=device), strict=False)
    model.to(device)
    model.eval()

    print(f"Loaded Finetuned QuRe models : {msg}")

    recalls_at5 = []
    recalls_at10 = []
    recalls_at50 = []
    results_dict = dict()

    for cloth_type, cur_test_dataloader in test_dataloaders.items():
        cur_test_samples_dataloader = cur_test_dataloader['samples']
        cur_test_query_dataloader = cur_test_dataloader['query']

        index_features, index_names = model.extract_target_features(cur_test_samples_dataloader, configs['use_temp'], device)
        predicted_features, target_names = model.extract_query_features_fiq(
            cur_test_query_dataloader, configs['use_temp'], txt_processors, device)

        recalls = _compute_recall_at_k(
            model, predicted_features, index_features, target_names, index_names
        )
        recall_at5 = recalls[5]
        recall_at10 = recalls[10]
        recall_at50 = recalls[50]

        recalls_at5.append(recall_at5)
        recalls_at10.append(recall_at10)
        recalls_at50.append(recall_at50)

        results_dict[f"{cloth_type}_recall@5"] = recall_at5
        results_dict[f"{cloth_type}_recall@10"] = recall_at10
        results_dict[f"{cloth_type}_recall@50"] = recall_at50

    results_dict.update({
        f'average_recall_at5': mean(recalls_at5),
        f'average_recall_at10': mean(recalls_at10),
        f'average_recall_at50': mean(recalls_at50),
        f'average_recall': (mean(recalls_at50) + mean(recalls_at10)) / 2
    })

    print(json.dumps(results_dict, indent=4))
    save_path = f"./cir_eval/fiq"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pretrained_note = configs['pretrained_path'].split('/')[-1]
    with open(f'{save_path}/{pretrained_note}.json', 'w') as json_file:
        json.dump(results_dict, json_file, indent=4)


if __name__ == '__main__':
    main()