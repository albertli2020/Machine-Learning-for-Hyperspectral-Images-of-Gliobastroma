import math
import os
import numpy as np
import matplotlib.pyplot as plt

def get_patch_paths(folder_path, roi):
    tumor = []
    nontumor = []
    for folder in os.listdir(folder_path):
        if not folder.startswith(roi) or folder.endswith("G") or folder.startswith("P6"):
            continue
        path = os.path.join(folder_path, folder)
        if folder.endswith("NT"):
            nontumor.append(path)
        else:
            tumor.append(path)

    return tumor, nontumor    

def process_patches(folder_path):
    spectral_averages = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        patch = np.load(file_path)
        patch = patch[:87, :87, :275]

        spectral_averages.append(np.mean(patch, axis=(0, 1)))

    spectral_averages_array = np.stack(spectral_averages)
    avg = np.mean(spectral_averages_array, axis=0)
    std = np.std(spectral_averages_array, axis=0)

    return avg, std

def save_averages(output_dir, patient, avg, std):
    os.makedirs(f'{output_dir}', exist_ok=True)
    avg_filename = f"{output_dir}/{patient}_avg.npy"
    np.save(avg_filename, avg)
    std_filename = f"{output_dir}/{patient}_std.npy"
    np.save(std_filename, std)

path = "ntp_90_90_275/"
output  = "patient_spectral_bands/"
band_array = np.arange(400.482, 1000.73, 2.1828)
read_from_file = True
show_bars = False

plt.ion()
fig, axes = plt.subplots(2, 4, figsize=(18, 10))
t_total_avg = []
t_total_std = []
nt_total_avg = []
nt_total_std = []


for patient_idx in range(1, 9):
    if(patient_idx == 6):
        continue
    row = math.floor((patient_idx - 1) / 4)
    if read_from_file:
        t_avg = np.load(os.path.join(output, f"Patient_{patient_idx}_T_avg.npy"))
        t_std = np.load(os.path.join(output, f"Patient_{patient_idx}_T_std.npy"))
        nt_avg = np.load(os.path.join(output, f"Patient_{patient_idx}_NT_avg.npy"))
        nt_std = np.load(os.path.join(output, f"Patient_{patient_idx}_NT_std.npy"))

        if(show_bars):
            axes[row, patient_idx - 1 - row * 4].errorbar(band_array, t_avg, yerr=t_std, marker = ".", elinewidth = 5, alpha = 0.05, label="Tumor")
            axes[row, patient_idx - 1 - row * 4].errorbar(band_array, nt_avg, yerr=nt_std, marker = ".", elinewidth = 5, alpha = 0.05, label="Non-Tumor")
        else:
            axes[row, patient_idx - 1 - row * 4].plot(band_array, t_avg, marker = ".", alpha = 0.35, label = "Tumor")
            axes[row, patient_idx - 1 - row * 4].plot(band_array, nt_avg, marker = ".", alpha = 0.35, label = "Non-Tumor")
        axes[row, patient_idx - 1 - row * 4].set_xlabel("Wavelength (nm)")
        axes[row, patient_idx - 1 - row * 4].set_ylabel("Average Intensity")
        axes[row, patient_idx - 1 - row * 4].set_title(f"Average Intensity for Patient {patient_idx}", fontsize = 12)
        axes[row, patient_idx - 1 - row * 4].legend()
        plt.pause(0.1)


    else:
        patient = f"P{patient_idx}_"
        tumor, nontumor = get_patch_paths(path, patient)
        t_average_spectral = []
        t_error_spectral = []
        nt_average_spectral = []
        nt_error_spectral = []
        plt.pause(0.1)
        print(f"--------------Processing Patient {patient_idx}--------------")
        
        for idx, roi_path in enumerate(tumor):
            print(f"Processing Tumor: {roi_path}")
            avg, std = process_patches(roi_path)
            t_average_spectral.append(avg)
            t_error_spectral.append(std)
            #axes[0].errorbar(band_array, avg, yerr = std, marker = ".", elinewidth=1, alpha=0.5, capsize=5)

        for idx, roi_path in enumerate(nontumor):
            print(f"Processing Non-tumor: {roi_path}")
            process_patches(roi_path)
            nt_average_spectral.append(avg)
            nt_error_spectral.append(std)
            #axes[1].errorbar(band_array, avg, yerr = std, marker = ".", elinewidth=1, alpha=0.5, capsize=5)

        t_average_spectral = np.array(t_average_spectral)
        t_average = np.mean(t_average_spectral, axis=0)
        t_error_spectral = np.array(t_error_spectral)
        t_average_error = np.mean(t_error_spectral, axis=0)
        nt_average_spectral = np.array(nt_average_spectral)
        nt_average = np.mean(nt_average_spectral, axis=0)
        nt_error_spectral = np.array(nt_error_spectral)
        nt_average_error = np.mean(nt_error_spectral, axis=0)
        
        print(f"Saving Patient {patient_idx} data")
        
        save_averages(output, f"Patient_{patient_idx}_T", t_average, t_average_error)
        save_averages(output, f"Patient_{patient_idx}_NT", nt_average, nt_average_error)

        '''
        axes[0].set_ylabel("Mean")
        axes[0].set_xlabel("Band Index")
        axes[0].set_title(f"Tumor Mean")
        axes[0].grid()

        axes[1].set_ylabel("Mean")
        axes[1].set_xlabel("Band Index")
        axes[1].set_title(f"Non-tumor Mean")
        axes[1].grid()
        '''

        axes[row, patient_idx - 1 - row * 5].errorbar(band_array, t_average, yerr=t_average_error, marker = ".", elinewidth = 5, alpha = 0.15, label="Tumor")
        axes[row, patient_idx - 1 - row * 5].errorbar(band_array, nt_average, yerr=nt_average_error, marker = ".", elinewidth = 5, alpha = 0.15, label="Non-Tumor")
        axes[row, patient_idx - 1 - row * 5].set_xlabel("Wavelength (nm)")
        axes[row, patient_idx - 1 - row * 5].set_ylabel("Average Intensity")
        axes[row, patient_idx - 1 - row * 5].set_title(f"Average Intensity for Patient {patient_idx}", fontsize = 12)
        axes[row, patient_idx - 1 - row * 5].legend()

        t_total_avg.append(t_average)
        t_total_std.append(t_average_error)
        nt_total_avg.append(nt_average)
        nt_total_std.append(nt_average_error)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.ioff()
plt.tight_layout()
plt.show()


if(read_from_file):
    t_avg = np.load(os.path.join(output, f"Total_T_avg.npy"))
    t_std = np.load(os.path.join(output, f"Total_T_std.npy"))
    nt_avg = np.load(os.path.join(output, f"Total_NT_avg.npy"))
    nt_std = np.load(os.path.join(output, f"Total_NT_std.npy"))

    t_combined_avg = []
    nt_combined_avg = []
    t_combined_std = []
    nt_combined_std = []

    bands = np.arange(400.482, 1000.73, 10.914)
    for idx in range(55):
        t_combined_avg.append(np.mean(t_avg[idx*5:(idx+1)*5]))
        nt_combined_avg.append(np.mean(nt_avg[idx*5:(idx+1)*5]))
        t_combined_std.append(np.mean(t_std[idx*5:(idx+1)*5]))
        nt_combined_std.append(np.mean(nt_std[idx*5:(idx+1)*5]))
    if(show_bars):
        plt.errorbar(bands, t_combined_avg, yerr=t_combined_std, marker = ".", elinewidth = 8, alpha = 0.15, label="Tumor", color = "red")
        plt.errorbar(bands, nt_combined_avg, yerr=nt_combined_std, marker = ".", elinewidth = 8, alpha = 0.15, label="Non-Tumor", color = "green")
    else:
        plt.plot(bands, t_combined_avg, marker=".", alpha=0.35, label="Tumor", color = "red")
        plt.plot(bands, nt_combined_avg, marker=".", alpha=0.35, label="Non-Tumor", color = "green")

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Average Intensity")
    plt.title(f"Total Average Intensity", fontsize = 12)
    plt.legend()


else:
    t_total_avg = np.array(t_total_avg)
    t_avg = np.mean(t_total_avg, axis=0)
    t_total_std = np.array(t_total_std)
    t_std = np.mean(t_total_std, axis=0)
    nt_total_avg = np.array(nt_total_avg)
    nt_avg = np.mean(nt_total_avg, axis=0)
    nt_total_std = np.array(nt_total_std)
    nt_std = np.mean(nt_total_std, axis=0)

    save_averages(output, f"Total_T", t_avg, t_std)
    save_averages(output, f"Total_NT", nt_avg, nt_std)

    axes[1, 3].errorbar(band_array, t_average, yerr=t_average_error, marker = ".", elinewidth = 5, alpha = 0.05, label="Tumor")
    axes[1, 3].errorbar(band_array, nt_average, yerr=nt_average_error, marker = ".", elinewidth = 5, alpha = 0.05, label="Non-Tumor")
    axes[1, 3].set_xlabel("Wavelength (nm)")
    axes[1, 3].set_ylabel("Average Intensity")
    axes[1, 3].set_title(f"Total Average Intensity", fontsize = 12)
    axes[1, 3].legend()



plt.grid()
plt.show()