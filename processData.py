import os
import glob
import random


def generatePatchLists(path):

    t_patch_list = []
    nt_patch_list = []

    for entry in os.listdir(path):
        next_level_path = os.path.join(path, entry)
        if os.path.isdir(next_level_path):
            paths = glob.glob(os.path.join(next_level_path, "*.npy"), recursive=False)
            if entry.endswith("_T") or entry.endswith("_T_50G"):            
                t_patch_list.extend(paths)
            elif entry.endswith("_NT") or entry.endswith("_NT_50G"):
                nt_patch_list.extend(paths)
    
    return t_patch_list, nt_patch_list

def savePatchLists(path):

    t_list_file = open("tumor_patch_files.txt", "w")
    nt_list_file = open("non_tumor_patch_files.txt", "w")

    t_patch_list, nt_patch_list = generatePatchLists(path)

    for patch in t_patch_list:
        t_list_file.write(patch + "\n")

    t_list_file.close()
    
    for patch in nt_patch_list:
        nt_list_file.write(patch + "\n")

    nt_list_file.close()


def generatePatchListsByPatient(path):

    patient_patch_dict = {}
    for i in range(13):
        k = "P" + str(i+1)
        patient_patch_dict[k] = []
    

    for entry in os.listdir(path):
        next_level_path = os.path.join(path, entry)
        if os.path.isdir(next_level_path):
            paths = glob.glob(os.path.join(next_level_path, "*.npy"), recursive=False)
            i = entry.find("_")
            p = entry[:i]
            patient_patch_dict[p].extend(paths)

    # print(patient_patch_dict)            
    
    return patient_patch_dict


def savePatchListsByPatient(path):

    pfList = {}
    for i in range(13):
        k = "P" + str(i+1)
        pfList[k] = open( k + "_patch_files.txt", "w")
    

    patient_patch_dict = generatePatchListsByPatient(path)

    for patient in patient_patch_dict.keys():
        patch_files = patient_patch_dict[patient]
        for pf in patch_files:
            pfList[patient].write(pf + "\n")

    for i in range(13):
        k = "P" + str(i+1)
        pfList[k].close()

def splitFilesByRatio(files, validation_ratio, test_ratio):

        
        test_count = int(test_ratio * len(files))
        validation_count = int(validation_ratio * len(files))
        # training_count = len(files) - validation_count - test_count

        random.shuffle(files)

        test_set = files[:test_count] 
        validation_set = files[test_count:(test_count+validation_count)]
        training_set = files[(test_count+validation_count):]

        #test_set = random.sample(files, k = test_count)

        # remove already chosen files from original list
        #files_wo_test_set = [f for f in files if f not in test_set]

        # randomly chose validation_count number of remaining files to validation set
        # validation_set = random.sample(files_wo_test_set, k = validation_count)

        # the remaining files going into the training set
        # training_set = [f for f in files_wo_test_set if f not in validation_set]

        return training_set, validation_set, test_set

def labelHSIDataSet(data_files):

    labels = []

    for file in data_files:
        p = os.path.dirname(file)
        # print(p)
        if p.endswith("_T") or p.endswith("_T_50G"):  
            labels.append(1)
        else:
            labels.append(0)

    return labels

def dataSetSample(precentage, set1, set2=[]):

    data_set = []

    if precentage < 1:
        count1 = int(len(set1) * precentage)
        count2 = int(len(set2) * precentage)
        data_set = random.sample(set1, k = count1) +  random.sample(set2, k = count2)
    else:
        data_set = set1 + set2
    
    random.shuffle(data_set)
    return data_set

def HSIDataSetSplit(tumor_files, non_tumor_files, validation_ratio=0.1, test_ratio=0.3, data_precentage=1):

    t_training_set, t_validation_set, t_test_set = splitFilesByRatio(tumor_files, validation_ratio, test_ratio)
    nt_training_set, nt_validation_set, nt_test_set = splitFilesByRatio(non_tumor_files, validation_ratio, test_ratio)

    training_set = dataSetSample(data_precentage, t_training_set, nt_training_set)
    training_labels = labelHSIDataSet(training_set)

    validation_set = dataSetSample(data_precenrage, t_validation_set, nt_validation_set)
    validation_labels = labelHSIDataSet(validation_set)

    test_set = dataSetSample(data_precentage, t_test_set, nt_test_set)
    test_labels = labelHSIDataSet(test_set)

    return training_set, training_labels, validation_set, validation_labels, test_set, test_labels

def HSIDataSetSplitByPatient(patient_file_dict, validation_patient_count=1, test_patient_count=3, data_precentage=1):

    patients1 = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]
    patients2 = ["P9", "P10", "P11", "P12", "P13"]
    validation_set = []
    test_set = []
    training_set = []

    random.shuffle(patients1)
    random.shuffle(patients2)
    test_patient2_count = test_patient_count//2
    test_patient1_count = test_patient_count - test_patient2_count
    validation_patients = patients1[:validation_patient_count]
    for patient in validation_patients:
        validation_set.extend(patient_file_dict[patient])
    validation_set = dataSetSample(data_precentage, validation_set)
    validation_labels = labelHSIDataSet(validation_set)

    test_patients1 = patients1[validation_patient_count:(validation_patient_count+test_patient1_count)]
    test_patients2 = patients2[:test_patient2_count]
    test_patients = test_patients1 + test_patients2
    for patient in test_patients:
        test_set.extend(patient_file_dict[patient])
    test_set = dataSetSample(data_precentage, test_set)
    test_labels = labelHSIDataSet(test_set)

    training_patients1 = patients1[(validation_patient_count+test_patient1_count):]
    training_patients2 = patients2[test_patient2_count:]
    training_patients = training_patients1 + training_patients2
    for patient in training_patients:
        training_set.extend(patient_file_dict[patient])
    training_set = dataSetSample(data_precentage, training_set)
    training_labels = labelHSIDataSet(training_set)

    return training_set, training_labels, validation_set, validation_labels, test_set, test_labels


root_dir = "C:/projects/ntp_90_90_275"
#savePatchLists(root_dir)
#savePatchListsByPatient(root_dir)
#t_patch_files, nt_patch_files = generatePatchLists(root_dir)
patient_file_dict = generatePatchListsByPatient(root_dir)
# training_set, training_labels, validation_set, validation_labels, test_set, test_labels = HSIDataSetSplit(t_patch_files, nt_patch_files)
training_set, training_labels, validation_set, validation_labels, test_set, test_labels = HSIDataSetSplitByPatient(patient_file_dict)
print(training_set)
print(training_labels)
print(validation_set)
print(validation_labels)
print(test_set)
print(test_labels)