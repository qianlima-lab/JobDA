import keras
import numpy as np


def readucr(filename):
    data = np.loadtxt(filename, delimiter = ',')
    Y = data[:,0]
    X = data[:,1:]
    return X, Y


def up_sample(data):
    length = data.shape[0]
    insert_data = [(data[i] + data[i + 1]) / 2.0 for i in range(0, length - 1, 2)]
    up_sample_series = []
    k = 0

    for j in range(int(length / 2) * 2):
        up_sample_series.append(data[j])
        if j % 2 == 0:
            up_sample_series.append(insert_data[k])
            k += 1
    if length % 2 != 0:
        up_sample_series.append(data[-1])

    return up_sample_series


def down_sample(data):
    length = data.shape[0]
    if length % 2 == 0:
        return [(data[i] + data[i + 1]) / 2.0 for i in range(0, length - 1, 2)]
    else:
        down_sample_series = [(data[i] + data[i + 1]) / 2.0 for i in range(0, length - 1, 2)]
        down_sample_series.append(data[length - 1])
        return down_sample_series


def generate_series(origin_series):
    series_length = origin_series.shape[0]
    half = series_length / 2
    down_sample_series = down_sample(origin_series[:int(half)])
    if series_length % 2 == 0:
        up_sample_series = up_sample(origin_series[int(half):])
        series = np.append(down_sample_series, up_sample_series)
        return series
    else:
        up_sample_series = up_sample(origin_series[int(half): -1])
        series = np.append(down_sample_series, up_sample_series)
        series = np.append(series, origin_series[-1])
        return series


def get_target_series(slices, origin_data):
    sample_nums = origin_data.shape[0]
    length = origin_data.shape[1]
    part_length = int(length / slices) * 2
    target_series = []

    for j in range(int(slices / 2) - 1):
        target_series.append(np.array([generate_series(origin_data[i][j * part_length:(j + 1) * part_length])
                                       for i in range(sample_nums)]).reshape(sample_nums, -1))

    target_series.append(np.array([generate_series(origin_data[i][part_length * int((slices / 2 - 1)):])
                                   for i in range(sample_nums)]).reshape(sample_nums, -1))
    return np.concatenate(target_series, axis=1)


def TSW(data, label, nb_trans):
    slice_list = [2, 4, 8, 16]
    sample_nums = data.shape[0]

    sample_labels_list = []
    version_labels_list = []
    for i in range(nb_trans):
        sample_labels_list.append(label)
        version_labels_list.append(label*nb_trans+i)

    data_list = [data]
    for i in range(nb_trans-1):
        version_target_series = get_target_series(slices=slice_list[i], origin_data=data)
        data_list.append(version_target_series)

    all_series = np.concatenate(data_list, axis=0)
    sample_labels = np.concatenate(sample_labels_list, axis=0).reshape(sample_nums*nb_trans, 1)
    version_labels = np.concatenate(version_labels_list, axis=0).reshape(sample_nums*nb_trans, 1)

    concat_data = np.concatenate((sample_labels, version_labels, all_series), axis=1)
    return concat_data


def read_dataset_jll(dataset_name, nb_trans):
    x_test, y_test_true = readucr('./datasets/'+dataset_name+'/'+dataset_name+'_TEST')
    
    data_label = np.loadtxt('./datasets/'+dataset_name+'/'+dataset_name+'_TRAIN', delimiter=',')
    data = data_label[:,1:]
    label = data_label[:,0]

    # Convert label [-1,1] to [0,1]
    for name in ['ECG200', 'Lighting2', 'FordA', 'FordB', 'wafer']:
        if dataset_name == name:
            label = (label + 1) / 2

    concat_data = TSW(data, label, nb_trans)
    x_train = concat_data[:,2:]
    y_train_joint = concat_data[:,1]
    
    return_data = {}
    return_data['nb_training'] = len(x_train)
    return_data['nb_testing'] = len(x_test)
    return_data['nb_classes'] = len(np.unique(y_train_joint))
    return_data['len_series'] = len(x_test[0])
    return_data['batch_size'] = min(int(return_data['nb_training']/10), 16)

    x_train_mean = x_train.mean()
    x_train_std = x_train.std()
    x_train = (x_train - x_train_mean)/(x_train_std)
    x_test = (x_test - x_train_mean)/(x_train_std)

    x_train = x_train.reshape(x_train.shape + (1,1,))
    x_test = x_test.reshape(x_test.shape + (1,1,))

    y_train_joint = y_train_joint - y_train_joint.min()
    y_test_true = y_test_true - y_test_true.min()
    Y_train_joint = keras.utils.to_categorical(y_train_joint, return_data['nb_classes'])
    Y_test_true = keras.utils.to_categorical(y_test_true, int(return_data['nb_classes']/nb_trans))

    return_data['training_set'] = [x_train, Y_train_joint]
    return_data['testing_set'] = [x_test, Y_test_true]

    return return_data
