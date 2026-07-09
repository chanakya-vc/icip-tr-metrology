from model import *
from dataloader import *
import config as cfg
from utils import create_train_val_split, count_parameters, plot_losses, conv_estimator, ctml_estimator
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import json

plt.switch_backend('agg')  # for servers not supporting display


def format_seeds(vals):
    """Helper to format list of floats to a readable string with 4 decimals."""
    return f"[{', '.join([f'{v:.4f}' for v in vals])}]"


def format_mean_std(vals):
    """Helper to compute and format mean ± std."""
    return f"{np.mean(vals):.4f} \u00B1 {np.std(vals):.4f}"


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('device: ', device)

    root_dir = cfg.root_dir
    save_root = cfg.save_root
    batch_size = cfg.batch_size
    lr = cfg.lr

    normalizer = TargetNormalizer(cfg.target_mins, cfg.target_maxs, device=device)

    # Use cfg.datasets if available, fallback to single dataset list
    dataset_list = cfg.datasets if hasattr(cfg, 'datasets') else [cfg.dataset]
    seeds = [42, 123, 2026]

    # Metrics
    criterion_mse = nn.MSELoss()
    criterion_mae = nn.L1Loss()

    # List to hold results for the CSV
    csv_data = []

    # --- OUTER LOOP: Iterate over datasets ---
    for num_dataset, dataset_name in enumerate(dataset_list):
        # Generate the alias to find where the weights were saved
        dataset_alias = f"dataset-{num_dataset + 1}"

        print(f"\n{'#' * 60}")
        print(f"### EVALUATING: {dataset_alias} (Actual Dataset: {dataset_name}) ###")
        print(f"{'#' * 60}")

        # Data paths still use the ACTUAL dataset name
        params_csv_path = os.path.join(root_dir, "params_csv", dataset_name, "nebula_test_params.csv")
        numpy_path = os.path.join(root_dir, "cube", dataset_name, "test")

        # Dictionaries to accumulate metrics across seeds for this dataset
        seed_results = {
            'MAE': {'a': [], 'h': [], 'w': []},
            'RMSE': {'a': [], 'h': [], 'w': []}
        }

        # --- INNER LOOP: Iterate over seeds ---
        for seed in seeds:
            print(f"\n--- Testing Seed: {seed} ---")

            # Determine model name
            if cfg.model == "qm_sam_var":
                model_name = r"RESNET_QMEta_sampleVar_b_" + str(batch_size) + "_LR_" + str(lr) + "_" + \
                             "WD" + str(cfg.weight_decay) + "_" + "min_lr_" + str(cfg.min_lr)
            elif cfg.model == "conv_eta":
                model_name = r"RESNET_convETA_b_" + str(batch_size) + "_LR_" + str(lr) + "_" + \
                             "WD" + str(cfg.weight_decay) + "_" + "min_lr_" + str(cfg.min_lr)
            elif cfg.model == "qm":
                model_name = r"RESNET_QMEta_b_" + str(batch_size) + "_LR_" + str(lr) + "_" + \
                             "WD" + str(cfg.weight_decay) + "_" + "min_lr_" + str(cfg.min_lr)

            # Load weights from the ALIAS directory
            models_dir_seed = os.path.join(save_root, dataset_alias, "weights", f"seed_{seed}", model_name)
            ckpt_path = os.path.join(models_dir_seed, cfg.ckpt)

            # Load Model
            if cfg.model == "qm_sam_var":
                model = SEMResNet2D().double().to(device)
            else:
                model = SEMResNet1D().double().to(device)

            if not os.path.exists(ckpt_path):
                print(f"Warning: Checkpoint not found at {ckpt_path}. Skipping this seed.")
                continue

            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt['model_state_dict'])
            model.to(device)
            model.eval()

            # Load Dataset (without pixel_ranges, using actual dataset_name path)
            if cfg.model == "qm_sam_var":
                test_dataset = SEMSimulatorDataset_multipixel(
                    csv_file=params_csv_path,
                    npy_dir=numpy_path,
                    dose=cfg.dose,
                    var_fn="sample_var",
                )
            elif cfg.model == "conv_eta":
                test_dataset = SEMSimulatorDataset_multipixel(
                    csv_file=params_csv_path,
                    npy_dir=numpy_path,
                    aggregation_fn=conv_estimator,
                    dose=cfg.dose,
                    var_fn=None,
                )
            elif cfg.model == "qm":
                test_dataset = SEMSimulatorDataset_multipixel(
                    csv_file=params_csv_path,
                    npy_dir=numpy_path,
                    aggregation_fn=np.nanmean,
                    dose=cfg.dose,
                    var_fn=None,
                )

            test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=6,
                                     pin_memory=True)

            # Track loss accumulations accurately for varying final batch sizes
            total_samples = 0
            accum_mae = {'a': 0.0, 'h': 0.0, 'w': 0.0}
            accum_se = {'a': 0.0, 'h': 0.0, 'w': 0.0} # Track sum of squared errors

            with torch.no_grad():
                for batch in test_loader:
                    if cfg.model == "qm_sam_var":
                        inputs = batch['inputs'].to(device)
                    else:
                        inputs = batch['inputs'].to(device).unsqueeze(1)

                    targets = batch['targets'].to(device)
                    batch_len = targets.size(0)
                    total_samples += batch_len

                    predictions = model(inputs)
                    physical_predictions = normalizer.denormalize(predictions)

                    # Compute MAE for a, h, w (multiply by batch_len for exact epoch averaging)
                    accum_mae['a'] += criterion_mae(physical_predictions[:, 0], targets[:, 0]).item() * batch_len
                    accum_mae['h'] += criterion_mae(physical_predictions[:, 1], targets[:, 1]).item() * batch_len
                    accum_mae['w'] += criterion_mae(physical_predictions[:, 2], targets[:, 2]).item() * batch_len

                    # Compute Squared Error for a, h, w
                    accum_se['a'] += criterion_mse(physical_predictions[:, 0], targets[:, 0]).item() * batch_len
                    accum_se['h'] += criterion_mse(physical_predictions[:, 1], targets[:, 1]).item() * batch_len
                    accum_se['w'] += criterion_mse(physical_predictions[:, 2], targets[:, 2]).item() * batch_len

            # Calculate final averages for this specific seed
            if total_samples > 0:
                seed_results['MAE']['a'].append(accum_mae['a'] / total_samples)
                seed_results['MAE']['h'].append(accum_mae['h'] / total_samples)
                seed_results['MAE']['w'].append(accum_mae['w'] / total_samples)

                # Divide by total samples to get MSE, then apply sqrt to get RMSE
                seed_results['RMSE']['a'].append(np.sqrt(accum_se['a'] / total_samples))
                seed_results['RMSE']['h'].append(np.sqrt(accum_se['h'] / total_samples))
                seed_results['RMSE']['w'].append(np.sqrt(accum_se['w'] / total_samples))

        # Ensure we have data before appending to CSV to avoid errors if a dataset is completely missing
        if len(seed_results['MAE']['a']) > 0:
            # Construct the row for the CSV using the ACTUAL NAME
            row_data = {
                'Dataset': dataset_name,

                # MAE Columns
                'MAE_a_Seeds': format_seeds(seed_results['MAE']['a']),
                'MAE_a_Mean_Std': format_mean_std(seed_results['MAE']['a']),

                'MAE_h_Seeds': format_seeds(seed_results['MAE']['h']),
                'MAE_h_Mean_Std': format_mean_std(seed_results['MAE']['h']),

                'MAE_w_Seeds': format_seeds(seed_results['MAE']['w']),
                'MAE_w_Mean_Std': format_mean_std(seed_results['MAE']['w']),

                # RMSE Columns
                'RMSE_a_Seeds': format_seeds(seed_results['RMSE']['a']),
                'RMSE_a_Mean_Std': format_mean_std(seed_results['RMSE']['a']),

                'RMSE_h_Seeds': format_seeds(seed_results['RMSE']['h']),
                'RMSE_h_Mean_Std': format_mean_std(seed_results['RMSE']['h']),

                'RMSE_w_Seeds': format_seeds(seed_results['RMSE']['w']),
                'RMSE_w_Mean_Std': format_mean_std(seed_results['RMSE']['w'])
            }
            csv_data.append(row_data)

    # --- SAVE TO CSV ---
    if csv_data:
        df = pd.DataFrame(csv_data)

        # Save to the root of the save directory
        csv_save_path = os.path.join(save_root, f"{cfg.model}_evaluation_results.csv")
        df.to_csv(csv_save_path, index=False ,encoding='utf-8-sig')

        print(f"\n{'=' * 60}")
        print(f"EVALUATION COMPLETE. Results saved to:")
        print(f"{csv_save_path}")
        print(f"{'=' * 60}")
        print("\nSummary of Mean \u00B1 Std (MAE):")
        print(df[['Dataset', 'MAE_a_Mean_Std', 'MAE_h_Mean_Std', 'MAE_w_Mean_Std']].to_string(index=False))
    else:
        print("\nNo evaluation results generated (check if model checkpoints exist).")