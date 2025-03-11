from itertools import chain

patch_data_root_dir = "ntp_90_90_275/"
patch_size = 87

val_test_split_factor = 0.25 #if splitting val dataset out of a combined val_test dataset

r_band, g_band, b_band = 425//3, 192//3, 109//3
input_spectral_bands_all_275 = range(0, 826//3) #275
input_spectral_bands_110 = list(chain(range(0, 30), range(45, 100), range(115, 130), range(265, 275))) # 30+55+15+10
input_spectral_bands_56 = list(chain(range(0, 30, 2), range(45, 101, 2), range(115, 131, 2), range(265, 275, 2))) #15+28+8+5
input_spectral_bands_27 = list(chain(range(0, 4), range(90, 96), range(128, 141), range(145, 149))) #4+6+13+4
input_spectral_bands_32 = list(chain(range(0, 4), range(90, 96), range(128, 150))) #4+6+22
#input_spectral_bands_32R = list(chain(range(0, 4), range(128, 150), range(269, 275))) #4+22+6
input_spectral_bands_32R = list(chain(range(0, 3), range(84, 92), range(128, 143), range(145, 151))) #3+8+15+6=35
input_spectral_bands_bgr = [b_band, g_band, r_band]
input_spectral_bands_1red = [r_band]

tvt_data_folds = [  ["train_set_Patient_F1.txt", "val_set_Patient_F1.txt", "test_set_Patient_F1.txt"],
                    ["train_set_Patient_F2.txt", "val_set_Patient_F2.txt", "test_set_Patient_F2.txt"],
                    ["train_set_Patient_F3.txt", "val_set_Patient_F3.txt", "test_set_Patient_F3.txt"],
                    ["train_set_Patient_F4.txt", "val_set_Patient_F4.txt", "test_set_Patient_F4.txt"],
                    ["train_set_ROI_F1.txt", "val_test_set_ROI_F1.txt", "val_test_set_ROI_F1.txt"],     # this is programmed partitioning ROI fold1, with val and test set mixed
                    ["train_set_ROI_F2.txt", "val_test_set_ROI_F2.txt", "val_test_set_ROI_F2.txt"],     # this is programmed partitioning ROI fold2, with val and test set mixed
                    ["train_set_ROI_F3.txt", "val_test_set_ROI_F3.txt", "val_test_set_ROI_F3.txt"],     # this is programmed partitioning ROI fold3, with val and test set mixed
                    ["train_set_ROI_F4.txt", "val_test_set_ROI_F4.txt", "val_test_set_ROI_F4.txt"],     # this is programmed partitioning ROI fold4, with val and test set mixed
                    ["train_set_ROI_F5.txt", "val_test_set_ROI_F5.txt", "val_test_set_ROI_F5.txt"],     # this is programmed partitioning ROI fold5, with val and test set mixed
                    ["train_set_ROI_F6.txt", "val_test_set_ROI_F6.txt", "val_test_set_ROI_F6.txt"],     # this is manual ROI fold1, with val and test sets mixed
                    ["train_set_ROI_F7.txt", "val_test_set_ROI_F7.txt", "val_test_set_ROI_F7.txt"],     # this is manual ROI fold2, with val and test sets mixed
                    ["train_set_ROI_F8.txt", "val_test_set_ROI_F8.txt", "val_test_set_ROI_F8.txt"],     # this is manual ROI fold3, with val and test sets mixed
                    ["train_set_ROI_F9.txt", "val_test_set_ROI_F9.txt", "val_test_set_ROI_F9.txt"],     # this is manual ROI fold4, with val and test sets mixed
                    ["train_set_ROI_F10.txt", "val_test_set_ROI_F10.txt", "val_test_set_ROI_F10.txt"],  # this is manual ROI fold5, with val and test sets mixed
                    ["train_set_ROI_F11.txt", "val_set_ROI_F11.txt", "test_set_ROI_F11.txt"],  # this is manual ROI fold1, with val and test sets separated
                    ["train_set_ROI_F12.txt", "val_set_ROI_F12.txt", "test_set_ROI_F12.txt"],  # this is manual ROI fold2, with val and test sets separated
                    ["train_set_ROI_F13.txt", "val_set_ROI_F13.txt", "test_set_ROI_F13.txt"],  # this is manual ROI fold3, with val and test sets separated
                    ["train_set_ROI_F14.txt", "val_set_ROI_F14.txt", "test_set_ROI_F14.txt"],  # this is manual ROI fold4, with val and test sets separated
                    ["train_set_ROI_F15.txt", "val_set_ROI_F15.txt", "test_set_ROI_F15.txt"],  # this is manual ROI fold5, with val and test sets separated
                    ["", "", "test_set_ROI_F16.txt"]   # this is special ROI fold6, with test sets only for P6
                ]

tvt_data_identifiers = [[["P2", "P3", "P4", "P5", "P8", "P9", "P10", "P12", "P13"], ["P1"], ["P7", "P11"]], 
                        [["P2", "P1", "P7", "P5", "P8", "P9", "P10", "P12", "P11"], ["P3"], ["P4", "P13"]],
                        [["P1", "P3", "P4", "P11", "P8", "P10", "P12", "P13"], ["P7"], ["P2", "P5", "P9"]],
                        [["P2", "P7", "P4", "P5", "P9", "P11", "P13"], ["P1"], ["P3", "P8", "P10", "P12"]],
                        [], [], [], [], [],
                        [], [], [], [], [],
                        [ [ 'P4_ROI_01', 'P7_ROI_03', 'P12_ROI_01', 'P3_ROI_01', 'P8_ROI_03', 'P9_ROI_02', 'P10_ROI_01', 'P11_ROI_01', 'P2_ROI_01', 'P13_ROI_01',
                            'P1_ROI_04', 'P8_ROI_02', 'P7_ROI_01', 'P8_ROI_01', 'P1_ROI_03', 'P5_ROI_02', 'P2_ROI_03', 'P5_ROI_04', 'P2_ROI_02', 'P4_ROI_02'],
                            ['P1_ROI_01','P5_ROI_03'], ['P1_ROI_02', 'P3_ROI_02', 'P7_ROI_02', 'P5_ROI_01', 'P9_ROI_01'] ],
                        [ [ 'P4_ROI_01', 'P9_ROI_01', 'P5_ROI_01', 'P7_ROI_03', 'P12_ROI_01', 'P3_ROI_01', 'P9_ROI_02', 'P10_ROI_01', 'P1_ROI_01', 'P13_ROI_01',
                            'P1_ROI_04', 'P7_ROI_01', 'P8_ROI_01', 'P3_ROI_02', 'P5_ROI_03', 'P2_ROI_03', 'P5_ROI_04', 'P1_ROI_02', 'P7_ROI_02', 'P4_ROI_02'],
                            ['P2_ROI_01','P8_ROI_02'], ['P1_ROI_03', 'P2_ROI_02', 'P5_ROI_02', 'P8_ROI_03', 'P11_ROI_01']],
                        [ [ 'P4_ROI_01', 'P9_ROI_01', 'P3_ROI_01', 'P8_ROI_03', 'P9_ROI_02', 'P10_ROI_01', 'P11_ROI_01', 'P2_ROI_01', 'P1_ROI_01', 'P13_ROI_01',
                            'P8_ROI_02', 'P7_ROI_01', 'P8_ROI_01', 'P3_ROI_02', 'P5_ROI_02', 'P5_ROI_03', 'P1_ROI_02', 'P7_ROI_02', 'P2_ROI_02', 'P4_ROI_02'],
                            ['P5_ROI_01','P1_ROI_03'], ['P1_ROI_04', 'P2_ROI_03', 'P5_ROI_04', 'P7_ROI_03', 'P12_ROI_01']],
                        [ [ 'P4_ROI_01', 'P5_ROI_01', 'P7_ROI_03', 'P12_ROI_01', 'P8_ROI_03', 'P9_ROI_02', 'P10_ROI_01', 'P11_ROI_01', 'P2_ROI_01', 'P1_ROI_04',
                            'P8_ROI_02', 'P7_ROI_01', 'P3_ROI_02', 'P1_ROI_03', 'P5_ROI_02', 'P5_ROI_04', 'P1_ROI_02', 'P7_ROI_02', 'P4_ROI_02'],
                            ['P9_ROI_01','P2_ROI_02','P2_ROI_03'], ['P5_ROI_03', 'P8_ROI_01', 'P1_ROI_01', 'P3_ROI_01', 'P13_ROI_01']],
                        [ [ 'P4_ROI_01', 'P9_ROI_01', 'P5_ROI_01', 'P12_ROI_01', 'P3_ROI_01', 'P8_ROI_03', 'P11_ROI_01', 'P1_ROI_01', 'P13_ROI_01', 'P1_ROI_04',
                            'P8_ROI_01', 'P3_ROI_02', 'P5_ROI_02', 'P5_ROI_03', 'P2_ROI_03', 'P5_ROI_04', 'P1_ROI_02', 'P7_ROI_02', 'P2_ROI_02', 'P4_ROI_02'],
                            ['P7_ROI_03','P1_ROI_03'], ['P7_ROI_01', 'P8_ROI_02', 'P2_ROI_01', 'P9_ROI_02', 'P10_ROI_01'] ],
                        [ [], [], ['P6_ROI_01', 'P6_ROI_02', 'P6_ROI_03'] ],    
                        ]


global_specifier_2D_F1 = {'nn_arch_name': '2D', 'data_fold_type': 'Patient', 'data_fold_name': 'F1', 'tvt_data_fold_idx': 0, 'batch_size':24, '1st_hl_size':256}

global_specifier_2D_ROI_F1 = {'nn_arch_name': '2D', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F1', 'tvt_data_fold_idx': 4+0, 'batch_size':24, '1st_hl_size':256}
global_specifier_2D_ROI_F6 = {'nn_arch_name': '2D', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F6', 'tvt_data_fold_idx': 4+5+0, 'batch_size':24, '1st_hl_size':256}
global_specifier_2D_ROI_F7 = {'nn_arch_name': '2D', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F7', 'tvt_data_fold_idx': 4+5+1, 'batch_size':24, '1st_hl_size':256}
global_specifier_2D_ROI_F8 = {'nn_arch_name': '2D', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F8', 'tvt_data_fold_idx': 4+5+2, 'batch_size':24, '1st_hl_size':256}
global_specifier_2D_ROI_F9 = {'nn_arch_name': '2D', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F9', 'tvt_data_fold_idx': 4+5+3, 'batch_size':24, '1st_hl_size':256}
global_specifier_2D_ROI_F10 = {'nn_arch_name': '2D', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F10', 'tvt_data_fold_idx': 4+5+4, 'batch_size':24, '1st_hl_size':256}

global_specifier_2D_ROI_F11 = {'nn_arch_name': '2D', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F11', 'tvt_data_fold_idx': 4+5+5, 'batch_size':24, '1st_hl_size':256}
global_specifier_2D_ROI_F12 = {'nn_arch_name': '2D', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F12', 'tvt_data_fold_idx': 4+5+6, 'batch_size':24, '1st_hl_size':256}
global_specifier_2D_ROI_F13 = {'nn_arch_name': '2D', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F13', 'tvt_data_fold_idx': 4+5+7, 'batch_size':24, '1st_hl_size':256}
global_specifier_2D_ROI_F14 = {'nn_arch_name': '2D', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F14', 'tvt_data_fold_idx': 4+5+8, 'batch_size':24, '1st_hl_size':256}
global_specifier_2D_ROI_F15 = {'nn_arch_name': '2D', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F15', 'tvt_data_fold_idx': 4+5+9, 'batch_size':24, '1st_hl_size':256}
global_specifier_2D_ROI_F16 = {'nn_arch_name': '2D', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F16', 'tvt_data_fold_idx': 4+5+10, 'batch_size':24, '1st_hl_size':256}

global_specifier_2L2D_ROI_F11 = {'nn_arch_name': '2L2D', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F11', 'tvt_data_fold_idx': 4+5+5, 'batch_size':24, '1st_hl_size':64}
global_specifier_2L2D_ROI_F12 = {'nn_arch_name': '2L2D', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F12', 'tvt_data_fold_idx': 4+5+6, 'batch_size':24, '1st_hl_size':64}
global_specifier_2L2D_ROI_F13 = {'nn_arch_name': '2L2D', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F13', 'tvt_data_fold_idx': 4+5+7, 'batch_size':24, '1st_hl_size':64}
global_specifier_2L2D_ROI_F14 = {'nn_arch_name': '2L2D', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F14', 'tvt_data_fold_idx': 4+5+8, 'batch_size':24, '1st_hl_size':64}
global_specifier_2L2D_ROI_F15 = {'nn_arch_name': '2L2D', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F15', 'tvt_data_fold_idx': 4+5+9, 'batch_size':24, '1st_hl_size':64}
global_specifier_2L2D_ROI_F16 = {'nn_arch_name': '2L2D', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F16', 'tvt_data_fold_idx': 4+5+10, 'batch_size':24, '1st_hl_size':64}

global_specifier_2D_110B_ROI_F10 = {'nn_arch_name': '2D_110B', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F10', 'tvt_data_fold_idx': 4+5+4, 'batch_size':24, '1st_hl_size':128}
global_specifier_2D_110B_ROI_F9 = {'nn_arch_name': '2D_110B', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F9', 'tvt_data_fold_idx': 4+5+3, 'batch_size':24, '1st_hl_size':128}
global_specifier_2D_110B_ROI_F8 = {'nn_arch_name': '2D_110B', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F8', 'tvt_data_fold_idx': 4+5+2, 'batch_size':24, '1st_hl_size':128}
global_specifier_2D_110B_ROI_F7 = {'nn_arch_name': '2D_110B', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F7', 'tvt_data_fold_idx': 4+5+1, 'batch_size':24, '1st_hl_size':128}
global_specifier_2D_110B_ROI_F6 = {'nn_arch_name': '2D_110B', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F6', 'tvt_data_fold_idx': 4+5+0, 'batch_size':24, '1st_hl_size':128}

global_specifier_2D_110BS_ROI_F10 = {'nn_arch_name': '2D_RED', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F10', 'tvt_data_fold_idx': 4+5+4, 'batch_size':24, '1st_hl_size':64}
global_specifier_2D_110BS_ROI_F9 = {'nn_arch_name': '2D_110BS', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F9', 'tvt_data_fold_idx': 4+5+3, 'batch_size':24, '1st_hl_size':64}
global_specifier_2D_110BS_ROI_F8 = {'nn_arch_name': '2D_110BS', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F8', 'tvt_data_fold_idx': 4+5+2, 'batch_size':24, '1st_hl_size':64}
global_specifier_2D_110BS_ROI_F7 = {'nn_arch_name': '2D_110BS', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F7', 'tvt_data_fold_idx': 4+5+1, 'batch_size':24, '1st_hl_size':64}
global_specifier_2D_110BS_ROI_F6 = {'nn_arch_name': '2D_110BS', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F6', 'tvt_data_fold_idx': 4+5+0, 'batch_size':24, '1st_hl_size':64}

global_specifier_2D_56B_ROI_F10 = {'nn_arch_name': '2D_56B', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F10', 'tvt_data_fold_idx': 4+5+4, 'batch_size':32, '1st_hl_size':64}
global_specifier_2D_56B_ROI_F9 = {'nn_arch_name': '2D_56B', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F9', 'tvt_data_fold_idx': 4+5+3, 'batch_size':32, '1st_hl_size':64}
global_specifier_2D_56B_ROI_F8 = {'nn_arch_name': '2D_56B', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F8', 'tvt_data_fold_idx': 4+5+2, 'batch_size':32, '1st_hl_size':64}
global_specifier_2D_56B_ROI_F7 = {'nn_arch_name': '2D_56B', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F7', 'tvt_data_fold_idx': 4+5+1, 'batch_size':32, '1st_hl_size':64}
global_specifier_2D_56B_ROI_F6 = {'nn_arch_name': '2D_56B', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F6', 'tvt_data_fold_idx': 4+5+0, 'batch_size':32, '1st_hl_size':64}

global_specifier_2D_32B_ROI_F11 = {'nn_arch_name': '2D_32B', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F11', 'tvt_data_fold_idx': 4+5+5, 'batch_size':32, '1st_hl_size':64}
global_specifier_2D_32B_ROI_F12 = {'nn_arch_name': '2D_32B', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F12', 'tvt_data_fold_idx': 4+5+6, 'batch_size':32, '1st_hl_size':64}
global_specifier_2D_32B_ROI_F13 = {'nn_arch_name': '2D_32B', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F13', 'tvt_data_fold_idx': 4+5+7, 'batch_size':32, '1st_hl_size':64}
global_specifier_2D_32B_ROI_F14 = {'nn_arch_name': '2D_32B', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F14', 'tvt_data_fold_idx': 4+5+8, 'batch_size':32, '1st_hl_size':64}
global_specifier_2D_32B_ROI_F15 = {'nn_arch_name': '2D_32B', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F15', 'tvt_data_fold_idx': 4+5+9, 'batch_size':32, '1st_hl_size':64}
global_specifier_2D_32B_ROI_F16 = {'nn_arch_name': '2D_32B', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F16', 'tvt_data_fold_idx': 4+5+10, 'batch_size':32, '1st_hl_size':64}

global_specifier_2L2D_32B_ROI_F11 = {'nn_arch_name': '2L2D_32B', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F11', 'tvt_data_fold_idx': 4+5+5, 'batch_size':32, '1st_hl_size':64}
global_specifier_2L2D_32B_ROI_F12 = {'nn_arch_name': '2L2D_32B', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F12', 'tvt_data_fold_idx': 4+5+6, 'batch_size':32, '1st_hl_size':64}
global_specifier_2L2D_32B_ROI_F13 = {'nn_arch_name': '2L2D_32B', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F13', 'tvt_data_fold_idx': 4+5+7, 'batch_size':32, '1st_hl_size':64}
global_specifier_2L2D_32B_ROI_F14 = {'nn_arch_name': '2L2D_32B', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F14', 'tvt_data_fold_idx': 4+5+8, 'batch_size':32, '1st_hl_size':64}
global_specifier_2L2D_32B_ROI_F15 = {'nn_arch_name': '2L2D_32B', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F15', 'tvt_data_fold_idx': 4+5+9, 'batch_size':32, '1st_hl_size':64}
global_specifier_2L2D_32B_ROI_F16 = {'nn_arch_name': '2L2D_32B', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F16', 'tvt_data_fold_idx': 4+5+10, 'batch_size':32, '1st_hl_size':64}

global_specifier_2D_32RB_ROI_F11 = {'nn_arch_name': '2D_32RB', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F11', 'tvt_data_fold_idx': 4+5+5, 'batch_size':32, '1st_hl_size':64}
global_specifier_2D_32RB_ROI_F12 = {'nn_arch_name': '2D_32RB', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F12', 'tvt_data_fold_idx': 4+5+6, 'batch_size':32, '1st_hl_size':64}
global_specifier_2D_32RB_ROI_F13 = {'nn_arch_name': '2D_32RB', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F13', 'tvt_data_fold_idx': 4+5+7, 'batch_size':32, '1st_hl_size':64}
global_specifier_2D_32RB_ROI_F14 = {'nn_arch_name': '2D_32RB', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F14', 'tvt_data_fold_idx': 4+5+8, 'batch_size':32, '1st_hl_size':64}
global_specifier_2D_32RB_ROI_F15 = {'nn_arch_name': '2D_32RB', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F15', 'tvt_data_fold_idx': 4+5+9, 'batch_size':32, '1st_hl_size':64}
global_specifier_2D_32RB_ROI_F16 = {'nn_arch_name': '2D_32RB', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F16', 'tvt_data_fold_idx': 4+5+10, 'batch_size':32, '1st_hl_size':64}

global_specifier_2D_27B_ROI_F11 = {'nn_arch_name': '2D_27B', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F11', 'tvt_data_fold_idx': 4+5+5, 'batch_size':32, '1st_hl_size':64}
global_specifier_2D_27B_ROI_F12 = {'nn_arch_name': '2D_27B', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F12', 'tvt_data_fold_idx': 4+5+6, 'batch_size':32, '1st_hl_size':64}
global_specifier_2D_27B_ROI_F13 = {'nn_arch_name': '2D_27B', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F13', 'tvt_data_fold_idx': 4+5+7, 'batch_size':32, '1st_hl_size':64}
global_specifier_2D_27B_ROI_F14 = {'nn_arch_name': '2D_27B', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F14', 'tvt_data_fold_idx': 4+5+8, 'batch_size':32, '1st_hl_size':64}
global_specifier_2D_27B_ROI_F15 = {'nn_arch_name': '2D_27B', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F15', 'tvt_data_fold_idx': 4+5+9, 'batch_size':32, '1st_hl_size':64}

global_specifier_2D_BGR_ROI_F10 = {'nn_arch_name': '2D_BGR', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F10', 'tvt_data_fold_idx': 4+5+4, 'batch_size':24, '1st_hl_size':256}
global_specifier_2D_BGR_ROI_F9 = {'nn_arch_name': '2D_BGR', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F9', 'tvt_data_fold_idx': 4+5+3, 'batch_size':24, '1st_hl_size':256}
global_specifier_2D_BGR_ROI_F8 = {'nn_arch_name': '2D_BGR', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F8', 'tvt_data_fold_idx': 4+5+2, 'batch_size':24, '1st_hl_size':256}
global_specifier_2D_BGR_ROI_F7 = {'nn_arch_name': '2D_BGR', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F7', 'tvt_data_fold_idx': 4+5+1, 'batch_size':24, '1st_hl_size':256}
global_specifier_2D_BGR_ROI_F6 = {'nn_arch_name': '2D_BGR', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F6', 'tvt_data_fold_idx': 4+5+0, 'batch_size':24, '1st_hl_size':256}

global_specifier_2D_1RED_ROI_F10 = {'nn_arch_name': '2D_1RED', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F10', 'tvt_data_fold_idx': 4+5+4, 'batch_size':24, '1st_hl_size':32}
global_specifier_2D_1RED_ROI_F9 = {'nn_arch_name': '2D_1RED', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F9', 'tvt_data_fold_idx': 4+5+3, 'batch_size':24, '1st_hl_size':32}
global_specifier_2D_1RED_ROI_F8 = {'nn_arch_name': '2D_1RED', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F8', 'tvt_data_fold_idx': 4+5+2, 'batch_size':24, '1st_hl_size':32}
global_specifier_2D_1RED_ROI_F7 = {'nn_arch_name': '2D_1RED', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F7', 'tvt_data_fold_idx': 4+5+1, 'batch_size':24, '1st_hl_size':32}
global_specifier_2D_1RED_ROI_F6 = {'nn_arch_name': '2D_1RED', 'data_fold_type': 'ROI', 'data_fold_name': 'ROI_F6', 'tvt_data_fold_idx': 4+5+0, 'batch_size':24, '1st_hl_size':32}

work_orders_275B_train_and_test = [
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_all_275, 'global_specifier': global_specifier_2D_ROI_F11, 'min_accuracy':0.40}, 
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_all_275, 'global_specifier': global_specifier_2D_ROI_F11, 'min_accuracy':0.40},
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_all_275, 'global_specifier': global_specifier_2D_ROI_F12, 'min_accuracy':0.40},
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_all_275, 'global_specifier': global_specifier_2D_ROI_F12, 'min_accuracy':0.40},
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_all_275, 'global_specifier': global_specifier_2D_ROI_F13, 'min_accuracy':0.40}, 
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_all_275, 'global_specifier': global_specifier_2D_ROI_F13, 'min_accuracy':0.40},
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_all_275, 'global_specifier': global_specifier_2D_ROI_F14, 'min_accuracy':0.40},
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_all_275, 'global_specifier': global_specifier_2D_ROI_F14, 'min_accuracy':0.40},
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_all_275, 'global_specifier': global_specifier_2D_ROI_F15, 'min_accuracy':0.40},
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_all_275, 'global_specifier': global_specifier_2D_ROI_F15, 'min_accuracy':0.40},
    ]

work_orders_275B_test_only = [
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_all_275, 'global_specifier': global_specifier_2D_ROI_F11, 'min_accuracy':0.40},    
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_all_275, 'global_specifier': global_specifier_2D_ROI_F12, 'min_accuracy':0.40},
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_all_275, 'global_specifier': global_specifier_2D_ROI_F13, 'min_accuracy':0.40},    
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_all_275, 'global_specifier': global_specifier_2D_ROI_F14, 'min_accuracy':0.40},
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_all_275, 'global_specifier': global_specifier_2D_ROI_F15, 'min_accuracy':0.40},
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_all_275, 'global_specifier': global_specifier_2D_ROI_F16, 'min_accuracy':0.40},
    ]

work_orders_275B_s2l2d_test_only = [
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_all_275, 'global_specifier': global_specifier_2L2D_ROI_F11, 'min_accuracy':0.40}, 
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_all_275, 'global_specifier': global_specifier_2L2D_ROI_F12, 'min_accuracy':0.40},
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_all_275, 'global_specifier': global_specifier_2L2D_ROI_F13, 'min_accuracy':0.40}, 
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_all_275, 'global_specifier': global_specifier_2L2D_ROI_F14, 'min_accuracy':0.40},
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_all_275, 'global_specifier': global_specifier_2L2D_ROI_F15, 'min_accuracy':0.40},
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_all_275, 'global_specifier': global_specifier_2L2D_ROI_F16, 'min_accuracy':0.40},
    ]

work_orders_275B_train_and_val = [
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_all_275, 'global_specifier': global_specifier_2D_ROI_F11, 'min_accuracy':0.40},    
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_all_275, 'global_specifier': global_specifier_2D_ROI_F12, 'min_accuracy':0.40},
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_all_275, 'global_specifier': global_specifier_2D_ROI_F13, 'min_accuracy':0.40},    
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_all_275, 'global_specifier': global_specifier_2D_ROI_F14, 'min_accuracy':0.40},
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_all_275, 'global_specifier': global_specifier_2D_ROI_F15, 'min_accuracy':0.40},
    ]

work_orders_275B_s2l2d_train_and_val = [
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_all_275, 'global_specifier': global_specifier_2L2D_ROI_F11, 'min_accuracy':0.40},    
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_all_275, 'global_specifier': global_specifier_2L2D_ROI_F12, 'min_accuracy':0.40},
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_all_275, 'global_specifier': global_specifier_2L2D_ROI_F13, 'min_accuracy':0.40},    
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_all_275, 'global_specifier': global_specifier_2L2D_ROI_F14, 'min_accuracy':0.40},
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_all_275, 'global_specifier': global_specifier_2L2D_ROI_F15, 'min_accuracy':0.40},
    ]


work_orders_110B_train_and_test = [
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_110, 'global_specifier': global_specifier_2D_110BS_ROI_F10, 'min_accuracy':0.60}, 
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_110, 'global_specifier': global_specifier_2D_110BS_ROI_F10, 'min_accuracy':0.60},
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_110, 'global_specifier': global_specifier_2D_110BS_ROI_F10, 'min_accuracy':0.60},
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_110, 'global_specifier': global_specifier_2D_110BS_ROI_F10, 'min_accuracy':0.60},
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_110, 'global_specifier': global_specifier_2D_110BS_ROI_F10, 'min_accuracy':0.60}, 
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_110, 'global_specifier': global_specifier_2D_110BS_ROI_F10, 'min_accuracy':0.60},
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_110, 'global_specifier': global_specifier_2D_110BS_ROI_F10, 'min_accuracy':0.60},
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_110, 'global_specifier': global_specifier_2D_110BS_ROI_F10, 'min_accuracy':0.60},
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_110, 'global_specifier': global_specifier_2D_110BS_ROI_F10, 'min_accuracy':0.60},
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_110, 'global_specifier': global_specifier_2D_110BS_ROI_F10, 'min_accuracy':0.60},
    ]

work_orders_56B_train_and_test = [
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_56, 'global_specifier': global_specifier_2D_56B_ROI_F10, 'min_accuracy':0.60}, 
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_56, 'global_specifier': global_specifier_2D_56B_ROI_F10, 'min_accuracy':0.60},
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_56, 'global_specifier': global_specifier_2D_56B_ROI_F9, 'min_accuracy':0.60},
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_56, 'global_specifier': global_specifier_2D_56B_ROI_F9, 'min_accuracy':0.60},
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_56, 'global_specifier': global_specifier_2D_56B_ROI_F8, 'min_accuracy':0.60}, 
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_56, 'global_specifier': global_specifier_2D_56B_ROI_F8, 'min_accuracy':0.60},
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_56, 'global_specifier': global_specifier_2D_56B_ROI_F7, 'min_accuracy':0.60},
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_56, 'global_specifier': global_specifier_2D_56B_ROI_F7, 'min_accuracy':0.60},
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_56, 'global_specifier': global_specifier_2D_56B_ROI_F6, 'min_accuracy':0.60},
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_56, 'global_specifier': global_specifier_2D_56B_ROI_F6, 'min_accuracy':0.60},
    ]

work_orders_32B_train_and_test = [
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_32, 'global_specifier': global_specifier_2D_32B_ROI_F11, 'min_accuracy':0.40}, 
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_32, 'global_specifier': global_specifier_2D_32B_ROI_F11, 'min_accuracy':0.40},
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_32, 'global_specifier': global_specifier_2D_32B_ROI_F12, 'min_accuracy':0.40},
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_32, 'global_specifier': global_specifier_2D_32B_ROI_F12, 'min_accuracy':0.40},
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_32, 'global_specifier': global_specifier_2D_32B_ROI_F13, 'min_accuracy':0.40}, 
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_32, 'global_specifier': global_specifier_2D_32B_ROI_F13, 'min_accuracy':0.40},
    #{'train_or_test':0, 'bands_to_use':input_spectral_bands_32, 'global_specifier': global_specifier_2D_32B_ROI_F14, 'min_accuracy':0.40},
    #{'train_or_test':1, 'bands_to_use':input_spectral_bands_32, 'global_specifier': global_specifier_2D_32B_ROI_F14, 'min_accuracy':0.40},
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_32, 'global_specifier': global_specifier_2D_32B_ROI_F15, 'min_accuracy':0.40},
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_32, 'global_specifier': global_specifier_2D_32B_ROI_F15, 'min_accuracy':0.40},
    ]

work_orders_32B_test_only = [
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_32, 'global_specifier': global_specifier_2D_32B_ROI_F11, 'min_accuracy':0.40},
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_32, 'global_specifier': global_specifier_2D_32B_ROI_F12, 'min_accuracy':0.40},
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_32, 'global_specifier': global_specifier_2D_32B_ROI_F13, 'min_accuracy':0.40},
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_32, 'global_specifier': global_specifier_2D_32B_ROI_F14, 'min_accuracy':0.40},
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_32, 'global_specifier': global_specifier_2D_32B_ROI_F15, 'min_accuracy':0.40},
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_32, 'global_specifier': global_specifier_2D_32B_ROI_F16, 'min_accuracy':0.40},
    ]

work_orders_32B_s2l2d_test_only = [
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_32, 'global_specifier': global_specifier_2L2D_32B_ROI_F11, 'min_accuracy':0.40},
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_32, 'global_specifier': global_specifier_2L2D_32B_ROI_F12, 'min_accuracy':0.40},
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_32, 'global_specifier': global_specifier_2L2D_32B_ROI_F13, 'min_accuracy':0.40},
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_32, 'global_specifier': global_specifier_2L2D_32B_ROI_F14, 'min_accuracy':0.40},
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_32, 'global_specifier': global_specifier_2L2D_32B_ROI_F15, 'min_accuracy':0.40},
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_32, 'global_specifier': global_specifier_2L2D_32B_ROI_F16, 'min_accuracy':0.40},
    ]

work_orders_32B_train_and_val = [
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_32, 'global_specifier': global_specifier_2D_32B_ROI_F11, 'min_accuracy':0.40},
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_32, 'global_specifier': global_specifier_2D_32B_ROI_F12, 'min_accuracy':0.40},
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_32, 'global_specifier': global_specifier_2D_32B_ROI_F13, 'min_accuracy':0.40},
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_32, 'global_specifier': global_specifier_2D_32B_ROI_F14, 'min_accuracy':0.40},
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_32, 'global_specifier': global_specifier_2D_32B_ROI_F15, 'min_accuracy':0.40},
    ]

work_orders_32RB_train_and_test = [
    #{'train_or_test':0, 'bands_to_use':input_spectral_bands_32R, 'global_specifier': global_specifier_2D_32RB_ROI_F11, 'min_accuracy':0.40}, 
    #{'train_or_test':1, 'bands_to_use':input_spectral_bands_32R, 'global_specifier': global_specifier_2D_32RB_ROI_F11, 'min_accuracy':0.40},
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_32R, 'global_specifier': global_specifier_2D_32RB_ROI_F12, 'min_accuracy':0.40},
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_32R, 'global_specifier': global_specifier_2D_32RB_ROI_F12, 'min_accuracy':0.40},
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_32R, 'global_specifier': global_specifier_2D_32RB_ROI_F13, 'min_accuracy':0.40}, 
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_32R, 'global_specifier': global_specifier_2D_32RB_ROI_F13, 'min_accuracy':0.40},
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_32R, 'global_specifier': global_specifier_2D_32RB_ROI_F14, 'min_accuracy':0.40},
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_32R, 'global_specifier': global_specifier_2D_32RB_ROI_F14, 'min_accuracy':0.40},
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_32R, 'global_specifier': global_specifier_2D_32RB_ROI_F15, 'min_accuracy':0.40},
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_32R, 'global_specifier': global_specifier_2D_32RB_ROI_F15, 'min_accuracy':0.40},
    ]

work_orders_27B_train_and_test = [
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_27, 'global_specifier': global_specifier_2D_27B_ROI_F11, 'min_accuracy':0.40}, 
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_27, 'global_specifier': global_specifier_2D_27B_ROI_F11, 'min_accuracy':0.40},
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_27, 'global_specifier': global_specifier_2D_27B_ROI_F12, 'min_accuracy':0.40},
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_27, 'global_specifier': global_specifier_2D_27B_ROI_F12, 'min_accuracy':0.40},
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_27, 'global_specifier': global_specifier_2D_27B_ROI_F13, 'min_accuracy':0.40}, 
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_27, 'global_specifier': global_specifier_2D_27B_ROI_F13, 'min_accuracy':0.40},
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_27, 'global_specifier': global_specifier_2D_27B_ROI_F14, 'min_accuracy':0.40},
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_27, 'global_specifier': global_specifier_2D_27B_ROI_F14, 'min_accuracy':0.40},
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_27, 'global_specifier': global_specifier_2D_27B_ROI_F15, 'min_accuracy':0.40},
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_27, 'global_specifier': global_specifier_2D_27B_ROI_F15, 'min_accuracy':0.40},
    ]

work_orders_bgr_train_and_test = [
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_bgr, 'global_specifier': global_specifier_2D_BGR_ROI_F10, 'min_accuracy':0.40}, 
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_bgr, 'global_specifier': global_specifier_2D_BGR_ROI_F10, 'min_accuracy':0.40},
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_bgr, 'global_specifier': global_specifier_2D_BGR_ROI_F9, 'min_accuracy':0.40},
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_bgr, 'global_specifier': global_specifier_2D_BGR_ROI_F9, 'min_accuracy':0.40},
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_bgr, 'global_specifier': global_specifier_2D_BGR_ROI_F8, 'min_accuracy':0.40}, 
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_bgr, 'global_specifier': global_specifier_2D_BGR_ROI_F8, 'min_accuracy':0.40},
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_bgr, 'global_specifier': global_specifier_2D_BGR_ROI_F7, 'min_accuracy':0.40},
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_bgr, 'global_specifier': global_specifier_2D_BGR_ROI_F7, 'min_accuracy':0.40},
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_bgr, 'global_specifier': global_specifier_2D_BGR_ROI_F6, 'min_accuracy':0.40},
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_bgr, 'global_specifier': global_specifier_2D_BGR_ROI_F6, 'min_accuracy':0.40},
    ]

work_orders_1red_train_and_test = [
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_1red, 'global_specifier': global_specifier_2D_1RED_ROI_F10, 'min_accuracy':0.40}, 
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_1red, 'global_specifier': global_specifier_2D_1RED_ROI_F10, 'min_accuracy':0.40},
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_1red, 'global_specifier': global_specifier_2D_1RED_ROI_F9, 'min_accuracy':0.40},
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_1red, 'global_specifier': global_specifier_2D_1RED_ROI_F9, 'min_accuracy':0.40},
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_1red, 'global_specifier': global_specifier_2D_1RED_ROI_F8, 'min_accuracy':0.40}, 
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_1red, 'global_specifier': global_specifier_2D_1RED_ROI_F8, 'min_accuracy':0.40},
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_1red, 'global_specifier': global_specifier_2D_1RED_ROI_F7, 'min_accuracy':0.40},
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_1red, 'global_specifier': global_specifier_2D_1RED_ROI_F7, 'min_accuracy':0.40},
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_1red, 'global_specifier': global_specifier_2D_1RED_ROI_F6, 'min_accuracy':0.40},
    {'train_or_test':1, 'bands_to_use':input_spectral_bands_1red, 'global_specifier': global_specifier_2D_1RED_ROI_F6, 'min_accuracy':0.40},
    ]

work_orders_train_mask_filter = [
    #{'train_or_test':0, 'bands_to_use':input_spectral_bands_all_275, 'global_specifier': global_specifier_2D_ROI_F11, 'min_accuracy':0.30}, 
    #{'train_or_test':0, 'bands_to_use':input_spectral_bands_all_275, 'global_specifier': global_specifier_2D_ROI_F12, 'min_accuracy':0.40},
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_all_275, 'global_specifier': global_specifier_2D_ROI_F13, 'min_accuracy':0.40}, 
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_all_275, 'global_specifier': global_specifier_2D_ROI_F14, 'min_accuracy':0.40},
    #{'train_or_test':0, 'bands_to_use':input_spectral_bands_all_275, 'global_specifier': global_specifier_2D_ROI_F15, 'min_accuracy':0.40},
    ]

work_orders_train_mask_filter_s2l2d = [
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_all_275, 'global_specifier': global_specifier_2L2D_ROI_F11, 'min_accuracy':0.40}, 
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_all_275, 'global_specifier': global_specifier_2L2D_ROI_F12, 'min_accuracy':0.40},
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_all_275, 'global_specifier': global_specifier_2L2D_ROI_F13, 'min_accuracy':0.40}, 
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_all_275, 'global_specifier': global_specifier_2L2D_ROI_F14, 'min_accuracy':0.40},
    {'train_or_test':0, 'bands_to_use':input_spectral_bands_all_275, 'global_specifier': global_specifier_2L2D_ROI_F15, 'min_accuracy':0.40},
    ]

# Define the steps of PNN Training and Validation machine learning processing of HSI data patches
mlp_steps_train_and_val_for_pnn_all_layers = [
    {'desc':'1-L from scratch, TVT', 'nol_from':0, 'nol_new':1, 'lr':0.0025, 'noe':4}, #8}, #18},
    {'desc':'1-L to 1-L, TVT', 'nol_from':1, 'nol_new':1, 'lr':0.00125, 'noe':4}, #8},#16},
    {'desc':'1-L to 2-L, TVT', 'nol_from':1, 'nol_new':2, 'lr':0.001, 'noe':4}, #25},
    {'desc':'2-L to 2-L, TVT', 'nol_from':2, 'nol_new':2, 'lr':0.0005, 'noe':4}, #8}, #32}, #'lr':0.0005, 'noe':16}, #25},
    #{'desc':'2-L to 2-L, TVT', 'nol_from':2, 'nol_new':2, 'lr':0.01, 'noe':2},
    #{'desc':'2-L to 2-L, TVT', 'nol_from':2, 'nol_new':2, 'lr':0.005, 'noe':2},
    #{'desc':'2-L to 2-L, TVT', 'nol_from':2, 'nol_new':2, 'lr':0.0025, 'noe':2},
    #{'desc':'2-L to 2-L, TVT', 'nol_from':2, 'nol_new':2, 'lr':0.00125, 'noe':8}, #24
    #{'desc':'2-L to 2-L, TVT', 'nol_from':2, 'nol_new':2, 'lr':0.000625, 'noe':16},
    {'desc':'2-L to 3-L, TVT', 'nol_from':2, 'nol_new':3, 'lr':0.00025, 'noe':4},
    {'desc':'3-L to 3-L, TVT', 'nol_from':3, 'nol_new':3, 'lr':0.0001, 'noe':4}, #8},
    {'desc':'3-L to 3-L, TVT', 'nol_from':3, 'nol_new':3, 'lr':0.00005, 'noe':4}, #8},
    {'desc':'3-L to 3-L, TVT', 'nol_from':3, 'nol_new':3, 'lr':0.000025, 'noe':4}, #8},
    {'desc':'3-L to 3-L, TVT', 'nol_from':3, 'nol_new':3, 'lr':0.00001, 'noe':16}, #8},
    #{'desc':'3-L to 4-L, TVT', 'nol_from':3, 'nol_new':4, 'lr':0.00005, 'noe':2},
    #{'desc':'4-L to 4-L, TVT', 'nol_from':4, 'nol_new':4, 'lr':0.0000375, 'noe':6},
    #{'desc':'4-L to 4-L, TVT', 'nol_from':4, 'nol_new':4, 'lr':0.000025, 'noe':8},
    #{'desc':'4-L to 4-L, TVT', 'nol_from':4, 'nol_new':4, 'lr':0.000025, 'noe':7},
    ]

mlp_steps_train_and_val_for_pnn_3L_only = [
    #{'desc':'3-L to 3-L, TVT', 'nol_from':3, 'nol_new':3, 'lr':0.0005, 'noe':8},
    {'desc':'3-L to 3-L, TVT', 'nol_from':3, 'nol_new':3, 'lr':0.00001, 'noe':16} #0.00002
    ]

mlp_steps_train_and_val_for_pnn_s2l2d = [
    {'desc':'2-L to 2-L, TVT', 'nol_from':2, 'nol_new':2, 'lr':0.00001, 'noe':16} #0.00002
    ]
# Define the steps of PNN Testing machine learning processing of HSI data patches
mlp_pnn_testonly_1L = {'desc':'1-L Test-only', 'nol':1}
mlp_pnn_testonly_2L = {'desc':'2-L Test-only', 'nol':2}
mlp_pnn_testonly_3L = {'desc':'3-L Test-only', 'nol':3}
mlp_pnn_testonly_4L = {'desc':'4-L Test-only', 'nol':4}
mlp_steps_testonly_for_pnn = [mlp_pnn_testonly_3L]
mlp_steps_testonly_for_pnn_s2l2d = [mlp_pnn_testonly_2L]

# Define the steps of Mask Filter Training and Validation machine learning processing of HSI data patches
mlp_steps_train_and_val_for_mf = [
    #{'desc':'3-L to 3-L, TVT', 'nol_from':3, 'nol_new':3, 'lr':0.0025, 'noe':8},
    #{'desc':'3-L to 3-L, TVT', 'nol_from':3, 'nol_new':3, 'lr':0.00125, 'noe':2},
    #{'desc':'3-L to 3-L, TVT', 'nol_from':3, 'nol_new':3, 'lr':0.000625, 'noe':2},
    #{'desc':'3-L to 3-L, TVT', 'nol_from':3, 'nol_new':3, 'lr':0.0002, 'noe':8}, #24
    {'desc':'3-L to 3-L, TVT', 'nol_from':3, 'nol_new':3, 'lr':0.0001, 'noe':8},#16},
    #{'desc':'4-L to 4-L, TVT', 'nol_from':4, 'nol_new':4, 'lr':0.0000375, 'noe':6},
    #{'desc':'4-L to 4-L, TVT', 'nol_from':4, 'nol_new':4, 'lr':0.000025, 'noe':8},
    #{'desc':'4-L to 4-L, TVT', 'nol_from':4, 'nol_new':4, 'lr':0.00001, 'noe':8},
    ]

mlp_steps_train_and_val_for_mf_s2l2d = [
    {'desc':'2-L to 2-L, TVT', 'nol_from':2, 'nol_new':2, 'lr':0.000005, 'noe':16} ]