from options import get_experiment_config
from set_up import setup_experiment
from transforms import image_transform_factory
from data import create_dataloaders
from models import create_qure_models
import wandb
wandb.init(mode="disabled")

import re
import math
from tqdm import tqdm
import torch
import torch.nn.functional as F
import os

from evaluate_qure.eval_utils import eval_fiq, eval_cirr


def main():
    configs = get_experiment_config()
    export_root, configs = setup_experiment(configs)
    device = torch.device(f"cuda:{configs['device_idx']}") if torch.cuda.is_available() else "cpu"

    model, txt_processors = create_qure_models(configs, device)

    start_epoch = 0
    checkpoint_path = configs.get("checkpoint_path", "")
    if checkpoint_path:
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint, strict=True)
        checkpoint_log = os.path.join(os.path.dirname(checkpoint_path), "log.txt")
        if not os.path.isfile(checkpoint_log):
            raise FileNotFoundError(f"Checkpoint log not found: {checkpoint_log}")
        with open(checkpoint_log, encoding="utf-8") as file:
            completed_epochs = [
                int(match.group(1))
                for line in file
                if (match := re.search(r"^Epoch\s*:\s*(\d+)\s*$", line.strip()))
            ]
        if not completed_epochs:
            raise ValueError(f"No completed epoch found in checkpoint log: {checkpoint_log}")
        start_epoch = max(completed_epochs)
        print(f"Resuming model weights from {checkpoint_path} at epoch {start_epoch}")

    image_transform = image_transform_factory(config=configs)
    train_dataloader, test_dataloaders = create_dataloaders(image_transform, None, configs)

    optimizer_name = configs.get("optimizer", "SGD")
    if optimizer_name != "SGD":
        raise ValueError(
            f"Experiment requires optimizer=SGD, but got {optimizer_name!r}"
        )
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=configs["init_lr"],
        momentum=configs["momentum"],
        weight_decay=configs["weight_decay"],
    )

    max_epochs = configs['epoch']

    model.train()
    scaler =  torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, max_epochs):
        model.train()
        if configs["use_rerank"]:
            if (epoch % (max_epochs // configs["negative_definition_epoch_num"])) == 0:
                if not configs['rerank_warmup'] or (epoch != 0 and configs['rerank_warmup']):
                    train_dataloader.dataset.rerank_score(model, device, txt_processors, configs, epoch)


        epoch_running_loss = 0.0
        epoch_running_dq = 0.0
        epoch_running_dz = 0.0
        dq_count = 0
        dz_count = 0
        multistep_with_warmup(
            optimizer,
            epoch,
            configs["init_lr"],
            configs["warmup_iters"],
            (configs["decay_step"], configs["decay_step_second"]),
            configs["gamma"],
        )
        train_dataloader_tqdm = tqdm(train_dataloader, desc="Epoch {}".format(epoch+1))
        for batch_idx, (ref_images, tar_images, negative_images, sentences) in enumerate(train_dataloader_tqdm):
            optimizer.zero_grad()

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                ref_images, tar_images, negative_images = ref_images.to(device), tar_images.to(
                    device), negative_images.to(device)

                sentences = [txt_processors["eval"](caption) for caption in sentences]
                scores = model(ref_images, tar_images, sentences, negative_images, configs['use_temp'])

                log_softmax_scores = F.log_softmax(scores.float(), dim=1)  

                target_probs = torch.zeros_like(log_softmax_scores)
                target_probs[:, 0] = 1.0  # p0 = 1 for each data point

                loss = F.kl_div(log_softmax_scores, target_probs, reduction='batchmean')

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"[Warning] NaN/Inf loss at batch {batch_idx}, skipping step")
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                max_norm=1.0
            )
            scaler.step(optimizer)
            scaler.update()

            loss_val = loss.item()
            if not math.isnan(loss_val):
                epoch_running_loss += loss_val
            cur_loss = epoch_running_loss / (batch_idx + 1)
            postfix_dict = {'loss': f"{cur_loss:.4f}"}
            if hasattr(model, 'last_query_diversity') and model.last_query_diversity is not None:
                epoch_running_dq += model.last_query_diversity
                dq_count += 1
                postfix_dict['D_Q'] = f"{model.last_query_diversity:.4f}"
            if hasattr(model, 'last_feature_diversity') and model.last_feature_diversity is not None:
                epoch_running_dz += model.last_feature_diversity
                dz_count += 1
                postfix_dict['D_Z'] = f"{model.last_feature_diversity:.4f}"
            train_dataloader_tqdm.set_postfix(postfix_dict)

        avg_dq = (epoch_running_dq / dq_count) if dq_count > 0 else None
        avg_dz = (epoch_running_dz / dz_count) if dz_count > 0 else None
        train_results = {'loss' : cur_loss, 'lr' : get_current_lr(optimizer)}
        print(f"[EPOCH {epoch+1}/{max_epochs}] Loss : {train_results['loss']:.4f} lr : {train_results['lr']}")
        if avg_dq is not None:
            print(f"[EPOCH {epoch+1}/{max_epochs}] Query Diversity D_Q (avg): {avg_dq:.6f}")
        if avg_dz is not None:
            print(f"[EPOCH {epoch+1}/{max_epochs}] Feature Diversity D_Z (avg): {avg_dz:.6f}")
        if configs["experiment_description"] != 'debug':
            wandb.log({"loss" : cur_loss, "epoch" : epoch})
            wandb.log({'lr' : get_current_lr(optimizer), "epoch": epoch})
            if avg_dq is not None:
                wandb.log({'diversity_q': avg_dq, "epoch": epoch})
            if avg_dz is not None:
                wandb.log({'diversity_z': avg_dz, "epoch": epoch})
            if hasattr(model, 'temp'):
                wandb.log({'temp': model.temp.item(), "epoch": epoch})
            if hasattr(model, 'logit_scale'):
                wandb.log({'temp': model.logit_scale.item(), "epoch": epoch})

        # Save the model for every 10 epochs
        # todo: adding evaluation
        # Save checkpoint every epoch into the experiment folder created by setup_experiment
        if (epoch + 1) % 1 == 0:
            saving_path = export_root
            os.makedirs(saving_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(saving_path, "model.pth"))
            append_log(os.path.join(saving_path, "log.txt"), f"Epoch : {epoch + 1}\n")
            print(f"Saved checkpoint to {os.path.join(saving_path, 'model.pth')}")

        eval_frequency = configs.get('eval_frequency', 1)
        if eval_frequency > 0 and (epoch + 1) % eval_frequency == 0:
            if configs['dataset'] == 'fashionIQ':
                eval_fiq(model, txt_processors, test_dataloaders, epoch, configs)

            elif configs['dataset'] == 'cirr':
                eval_cirr(model, txt_processors, test_dataloaders, epoch, configs)

def multistep_with_warmup(optimizer, epoch, init_lr, warmup_iters, milestones, gamma):
    """Apply epoch-based linear warmup followed by MultiStep decay."""
    if warmup_iters > 0 and epoch < warmup_iters:
        lr = init_lr * (epoch + 1) / warmup_iters
    else:
        lr = init_lr * gamma ** sum(epoch >= milestone for milestone in milestones)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def append_log(file_path, log_message):
    with open(file_path, 'a') as file:
        file.write(log_message + '\n')

if __name__ == '__main__':
    main()