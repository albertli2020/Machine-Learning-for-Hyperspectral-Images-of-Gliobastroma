import torch

import time
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
import csv

import HSI_Globals
from HSI_Dataset import HyperspectralDataset

class HyperspectralWorkorderMLP():
    def __init__(self, trainer_class=None, tester_class=None, log_file_prefix=None, attemp_gpu=False):
        self.TrainerClass = trainer_class
        self.TesterClass = tester_class
        self.trainer_log_file_prefix=log_file_prefix, 
        self.gpu_device = None
        if attemp_gpu:
            if torch.backends.mps.is_available():
                self.gpu_device = torch.device("mps")
                print ("Model and data will be moved to Metal Performance backend")
            elif torch.cuda.is_available():
                self.gpu_device = "cuda"
                print("Model and data will be moved to nVidia GPUs")
            else:
                print("Model and data will stay on CPUs")

    def run_train_and_validate(self, mlp_steps, global_specifier, input_spectral_bands, min_accuracy, mlp_log_prefix=None):
        if self.TrainerClass is None:
            print(f"Warning: Trainer Class not specified for this work order.")
            return
        
        if mlp_steps is None:
            print(f"Warning: MLP training and validation steps not specified in this work order.")
            return

        data_loading_start_time = time.time()
        if self.trainer_log_file_prefix is not None:
            # File path for the log
            mlp_log_file_path = f"{self.trainer_log_file_prefix}_{global_specifier['nn_arch_name']}_{global_specifier['data_fold_name']}_training_log_file.txt"
        else:
            mlp_log_file_path = None

        print(f"patch_size={HSI_Globals.patch_size}; num_spectral_bands_to_use={len(input_spectral_bands)}")
        training_set_read_file = HSI_Globals.tvt_data_folds[global_specifier['tvt_data_fold_idx']][0]
        val_set_read_file = HSI_Globals.tvt_data_folds[global_specifier['tvt_data_fold_idx']][1]
        training_identifiers = HSI_Globals.tvt_data_identifiers[global_specifier['tvt_data_fold_idx']][0]
        val_identifiers = HSI_Globals.tvt_data_identifiers[global_specifier['tvt_data_fold_idx']][1]
        # TBF: if we need to split mixed validation and test data...
        #print(training_set_read_file)
        #print(val_set_read_file)
        #print(training_identifiers)
        #print(val_identifiers)

        print(f"Training on {global_specifier['data_fold_type']}s:", training_identifiers)
        print(f"Validation with {global_specifier['data_fold_type']}s:", val_identifiers)
    
        train_dataset = HyperspectralDataset(training_set_read_file, global_specifier['batch_size'], shuffle_files=True, duplicatePos=True,
                                          randomize_pos_data=True, bands=input_spectral_bands,
                                          patch_size=HSI_Globals.patch_size, gpu_device=self.gpu_device)
        val_dataset = HyperspectralDataset(val_set_read_file, global_specifier['batch_size'], shuffle_files=True, duplicatePos=False,
                                          bands=input_spectral_bands,
                                          patch_size=HSI_Globals.patch_size, gpu_device=self.gpu_device)

        train_class_counts = Counter(train_dataset.labels)
        print(f"[Training vs Validation] Patch Image Distribution: " + str(len(train_dataset.image_paths)) + " " + str(len(val_dataset.image_paths)))
        print(f"Training True label distribution [0: Non-Tumor; 1: Tumor]:", train_class_counts, Counter(val_dataset.labels))

        data_loading_end_time = time.time()
        print(f"Elapsed seconds for data loading: {(data_loading_end_time - data_loading_start_time):.2f}")

        for s in mlp_steps:
            # Convert current time to local time and format it
            local_time = datetime.fromtimestamp(time.time()).strftime('%m-%d-%Y %H:%M:%S')
            print(f"{s['desc']} starts at Pacific Time: {local_time}")
            model_num_layers = s['nol_new']
            from_model_nl = s['nol_from']
            training_lr = s['lr']
            num_epoches = s['noe']
            print(f"Num_layers={model_num_layers}; Training_LR={training_lr:.6f}; Batch_Size={global_specifier['batch_size']}; Num_Epoches={num_epoches}")
            trainer = self.TrainerClass(model_num_layers,  input_spectral_bands, global_specifier, class_weight_ratio = train_class_counts[0] / train_class_counts[1],
                                            gpu_device=self.gpu_device, learning_rate=training_lr, num_layers_of_inherited_model=from_model_nl, min_accuracy=min_accuracy)
            is_trained_well = trainer.train_with_validation(train_dataset, val_dataset, mlp_log_file_path,
                                          loss_th_to_stop=0.025, accrucy_th_to_stop=0.98, epochs=num_epoches)
            if not is_trained_well:
                print("The most recent training and validation process didn't converge enough to run the test process.")

            print("----------------------")
            print(" ")

    def run_testonly(self, mlp_steps, global_specifier, input_spectral_bands):    
        if self.TesterClass is None:
            print(f"Warning: Tester Class not specified for this work order.")
            return

        if mlp_steps is None:
            print(f"Warning: MLP test-only steps not specified for this work order.")
            return

        print(f"Test-only: patch_size={HSI_Globals.patch_size}; num_spectral_bands_to_use={len(input_spectral_bands)}")
        for mlp_testonly in mlp_steps:
            test_only_num_of_layers = mlp_testonly['nol']
            tester = self.TesterClass(test_only_num_of_layers, input_spectral_bands, global_specifier, gpu_device=self.gpu_device)
            output_csv_path = f"test_output_{global_specifier['nn_arch_name']}_{global_specifier['data_fold_name']}_{test_only_num_of_layers}L.csv"
            print("Test output will be saved to: ", output_csv_path)

            tdf_index = global_specifier['tvt_data_fold_idx']
            test_identifiers = HSI_Globals.tvt_data_identifiers[tdf_index][2]
            print(f"Testing {global_specifier['data_fold_type']}s:", test_identifiers) 
            with open(output_csv_path, mode="w", newline="") as file:
                writer = csv.writer(file)
                # Write header
                writer.writerow(["Patch Path", "True Label", "Predicted Label", "Probabilities"])
                test_dataset = HyperspectralDataset(HSI_Globals.tvt_data_folds[tdf_index][2], global_specifier['batch_size'],
                                                     bands=input_spectral_bands,
                                                     patch_size=HSI_Globals.patch_size, gpu_device=self.gpu_device)
                local_time = datetime.fromtimestamp(time.time()).strftime('%m-%d-%Y %H:%M:%S')
                print(f"{mlp_testonly['desc']} for {global_specifier['data_fold_name']} starts at Pacific Time: {local_time}")
                print("Test patches are listed in ", HSI_Globals.tvt_data_folds[tdf_index][2])
                true_labels, predicted_labels, probabilities = tester.test_model(test_dataset)
                 # Write rows
                for path, true, pred, prob in zip(test_dataset.image_paths, true_labels, predicted_labels, probabilities):
                    writer.writerow([path, true, pred, prob])
                print("----------------------")
                print(" ")                        

    def fill_orders(self, spec_of_training_val_steps, spec_of_test_only_steps, workorders):  
        for workorder in workorders:
            if workorder['train_or_test'] == 0:
                self.run_train_and_validate(spec_of_training_val_steps, workorder['global_specifier'], workorder['bands_to_use'], workorder['min_accuracy'])
            elif workorder['train_or_test'] == 1:
                self.run_testonly(spec_of_test_only_steps, workorder['global_specifier'], workorder['bands_to_use'])
            else:
                pass
