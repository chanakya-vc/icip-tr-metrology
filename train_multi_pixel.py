from model import *
from dataloader import *

import config as cfg
from utils import create_train_val_split, count_parameters, plot_losses, conv_estimator, ctml_estimator
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import os
import json

plt.switch_backend('agg')  # for servers not supporting display
import torch.optim as optim
from torch.optim import lr_scheduler


def set_seed(seed):
    """Ensures absolute reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('device: ', device)

    root_dir = cfg.root_dir
    save_root = cfg.save_root
    validation_frac = cfg.validation_frac
    batch_size = cfg.batch_size
    resume = cfg.resume

    # Assuming normalizer parameters are consistent across all datasets
    normalizer = TargetNormalizer(cfg.target_mins, cfg.target_maxs, device=device)

    # Use cfg.datasets (a list) if available, otherwise fallback to [cfg.dataset]
    dataset_list = cfg.datasets if hasattr(cfg, 'datasets') else [cfg.dataset]
    seeds = [42, 123, 2026]

    # --- Create and Save Dataset Mapping ---
    os.makedirs(save_root, exist_ok=True)

    # Add model identifier to the dictionary structure
    mapping_data = {
        "model_type": cfg.model,
        "mapping": {f"dataset-{i + 1}": name for i, name in enumerate(dataset_list)}
    }

    # Add model identifier to the filename
    mapping_file = os.path.join(save_root, f"dataset_mapping_{cfg.model}.json")

    with open(mapping_file, "w") as f:
        json.dump(mapping_data, f, indent=4)
    print(f"\nSaved dataset mapping to: {mapping_file}")

    # --- OUTER LOOP: Iterate over datasets ---
    for num_dataset, dataset_name in enumerate(dataset_list):
        dataset_alias = f"dataset-{num_dataset + 1}"

        print(f"\n{'#' * 60}")
        print(f"### STARTING PIPELINE FOR: {dataset_alias} (Actual: {dataset_name}) ###")
        print(f"{'#' * 60}")

        # Update paths for the current dataset (Read from actual name)
        params_csv_path = os.path.join(root_dir, "params_csv", dataset_name, "nebula_train_params.csv")
        numpy_path = os.path.join(root_dir, "cube", dataset_name, "train")

        # Instantiate dataset ONCE per dataset to save overhead
        if cfg.model == "qm_sam_var":
            full_train_dataset = SEMSimulatorDataset_multipixel(
                csv_file=params_csv_path,
                npy_dir=numpy_path,
                dose=cfg.dose,
                var_fn="sample_var",
            )
        elif cfg.model == "conv_eta":
            full_train_dataset = SEMSimulatorDataset_multipixel(
                csv_file=params_csv_path,
                npy_dir=numpy_path,
                aggregation_fn=conv_estimator,
                dose=cfg.dose,
                var_fn=None,
            )
        elif cfg.model == "qm":
            full_train_dataset = SEMSimulatorDataset_multipixel(
                csv_file=params_csv_path,
                npy_dir=numpy_path,
                aggregation_fn=np.nanmean,
                dose=cfg.dose,
                var_fn=None,
            )

        # --- INNER LOOP: Iterate over seeds ---
        for seed in seeds:
            print(f"\n{'=' * 40}")
            print(f"=== DATASET: {dataset_alias} | SEED: {seed} ===")
            print(f"{'=' * 40}")

            set_seed(seed)

            # 1. Create DataLoaders (Inside loop to ensure split respects the seed)
            train_loader, val_loader = create_train_val_split(full_train_dataset, val_fraction=validation_frac,
                                                              batch_size=batch_size)

            # 2. Create Model, Loss, Optimizer
            if cfg.model == "qm_sam_var":
                model = SEMResNet2D().double().to(device)
                model_name = r"RESNET_QMEta_sampleVar_b_" + str(batch_size) + "_LR_" + str(cfg.lr) + "_" + \
                             "WD" + str(cfg.weight_decay) + "_" + "min_lr_" + str(cfg.min_lr)
            else:
                model = SEMResNet1D().double().to(device)
                if cfg.model == "conv_eta":
                    model_name = r"RESNET_convETA_b_" + str(batch_size) + "_LR_" + str(cfg.lr) + "_" + \
                                 "WD" + str(cfg.weight_decay) + "_" + "min_lr_" + str(cfg.min_lr)
                elif cfg.model == "qm":
                    model_name = r"RESNET_QMEta_b_" + str(batch_size) + "_LR_" + str(cfg.lr) + "_" + \
                                 "WD" + str(cfg.weight_decay) + "_" + "min_lr_" + str(cfg.min_lr)

            lr = cfg.lr
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                                         weight_decay=cfg.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr)
            loss = nn.MSELoss()

            # Directories for this specific dataset and seed (Save to Alias)
            models_dir_seed = os.path.join(save_root, dataset_alias, "weights", f"seed_{seed}", model_name)
            loss_dir_seed = os.path.join(save_root, dataset_alias, "loss_curves", f"seed_{seed}", model_name)
            os.makedirs(models_dir_seed, exist_ok=True)
            os.makedirs(loss_dir_seed, exist_ok=True)

            best_val_loss = float('inf')
            ckpt_path = os.path.join(models_dir_seed, cfg.ckpt)

            # 3. Train Loop Setup (with safe resume logic)
            if resume and os.path.exists(ckpt_path):
                print(f'\nResuming from checkpoint: {ckpt_path}')
                ckpt = torch.load(ckpt_path)
                model.load_state_dict(ckpt['model_state_dict'])
                model.to(device)

                lr = ckpt['lr']
                optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                                             weight_decay=cfg.weight_decay)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr)

                losses = ckpt['losses']
                running_train_loss = losses['running_train_loss']
                running_val_loss = losses['running_val_loss']
                train_epoch_loss = losses['train_epoch_loss']
                val_epoch_loss = losses['val_epoch_loss']
                epochs_till_now = ckpt['epochs_till_now']

                best_val_loss = ckpt.get('best_val_loss', min(val_epoch_loss) if val_epoch_loss else float('inf'))
            else:
                if resume:
                    print(f'\nResume flag is True, but no checkpoint found at {ckpt_path}. Starting from scratch.')
                else:
                    print('\nStarting from scratch.')

                train_epoch_loss = []
                val_epoch_loss = []
                running_train_loss = []
                running_val_loss = []
                epochs_till_now = 0

            epochs = cfg.epochs

            print('\nmodel has {} parameters'.format(count_parameters(model)))
            print(f'loss_fn        : {loss}')
            print(f'lr             : {lr}')
            print(f'epochs_till_now: {epochs_till_now}')
            print(f'epochs from now: {epochs}')

            # 4. Main Training Iteration
            for epoch in tqdm(range(epochs_till_now, epochs_till_now + epochs)):
                print('\n===== EPOCH {}/{} ====='.format(epoch + 1, epochs_till_now + epochs))
                print('TRAINING...')
                current_epoch_train_loss = []
                current_epoch_val_loss = []
                model.train()

                for batch_idx, batch in enumerate(train_loader):
                    if cfg.model == "qm_sam_var":
                        x = batch['inputs']
                    else:
                        x = batch['inputs'].unsqueeze(1)

                    y = batch['targets']
                    x = x.to(device)
                    y = y.to(device)

                    norm_targets = normalizer.normalize(y)
                    optimizer.zero_grad()
                    out = model(x)
                    batch_loss = loss(out, norm_targets)
                    batch_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    optimizer.step()

                    running_train_loss.append(batch_loss.item())
                    current_epoch_train_loss.append(batch_loss.item())

                epoch_t_loss = np.mean(current_epoch_train_loss)
                print('Epoch_Train_Loss {}'.format(epoch_t_loss))
                train_epoch_loss.append(epoch_t_loss)

                print('\nVALIDATION...')
                model.eval()
                with torch.no_grad():
                    for batch_idx, batch in enumerate(val_loader):
                        if cfg.model == "qm_sam_var":
                            x = batch['inputs']
                        else:
                            x = batch['inputs'].unsqueeze(1)

                        y = batch['targets']
                        x = x.to(device)
                        y = y.to(device)

                        norm_targets = normalizer.normalize(y)
                        out = model(x)
                        batch_loss = loss(out, norm_targets)
                        running_val_loss.append(batch_loss.item())
                        current_epoch_val_loss.append(batch_loss.item())

                epoch_v_loss = np.mean(current_epoch_val_loss)
                val_epoch_loss.append(epoch_v_loss)
                scheduler.step()
                print('Epoch_Val_Loss {}'.format(epoch_v_loss))

                # 5. Save the Best Validation Model
                if epoch_v_loss < best_val_loss:
                    best_val_loss = epoch_v_loss
                    print(f'\n*** New best validation loss: {best_val_loss:.6f}. Saving weight_best.pt ***')
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'best_val_loss': best_val_loss,
                        'lr': scheduler.get_last_lr()[0],
                        'seed': seed,
                        'dataset': dataset_name,
                        'dataset_alias': dataset_alias
                    }, os.path.join(models_dir_seed, 'weight_best.pt'))

                # 6. Regular Checkpoint Saving
                if epoch % cfg.save_interval == 0 or epoch + 1 == (epochs_till_now + epochs):
                    print(f'\nsaving model checkpoint for epoch {epoch + 1}...')

                    torch.save({'model_state_dict': model.state_dict(),
                                'losses': {'running_train_loss': running_train_loss,
                                           'running_val_loss': running_val_loss,
                                           'train_epoch_loss': train_epoch_loss,
                                           'val_epoch_loss': val_epoch_loss},
                                'epochs_till_now': epoch + 1,
                                'best_val_loss': best_val_loss,
                                'lr': scheduler.get_last_lr()[0]},
                               os.path.join(models_dir_seed, 'model{}.pth'.format(str(epoch + 1).zfill(2))))

                    plot_losses(loss_dir_seed, running_train_loss, running_val_loss, train_epoch_loss, val_epoch_loss,
                                epoch)