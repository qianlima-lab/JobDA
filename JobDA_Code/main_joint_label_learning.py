import keras
import numpy as np
import tensorflow as tf

from utils import read_dataset_jll
import model

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

np.random.seed(813306)


def main(dataset_name, classifier_name, nb_epochs, nb_trans, mode):
    return_data = read_dataset_jll(dataset_name, nb_trans)

    x_train, Y_train_joint = return_data['training_set']
    x_test, Y_test_true = return_data['testing_set']
    nb_training, nb_testing = return_data['nb_training'], return_data['nb_testing']
    nb_classes, len_series = return_data['nb_classes'], return_data['len_series']
    batch_size = return_data['batch_size']

    ResNet = model.ResNet(x_train.shape[1:], nb_classes)
    classifier = ResNet.build_model()

    if mode == 'train':
        is_exists = os.path.exists('./weights/'+classifier_name+'_JLL')
        if not is_exists:
            os.makedirs('./weights/'+classifier_name+'_JLL')
        
        checkpoint = keras.callbacks.ModelCheckpoint('./weights/'+classifier_name+'_JLL/'+dataset_name+'.h5', monitor='loss', verbose=1, save_best_only=True, mode='min', period=1)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)
        hist = classifier.fit(x_train, Y_train_joint, batch_size=batch_size, nb_epoch=nb_epochs, verbose=1, callbacks = [checkpoint, reduce_lr])
    
    if mode == 'eval':
        classifier.load_weights('./weights/'+classifier_name+'_JLL/'+dataset_name+'.h5')

        if nb_testing > 50:
            nb_batch_test = np.int32(np.floor(nb_testing / 50))
            if nb_testing % 50 == 0:
                nb_total_test = nb_testing
            else:
                nb_total_test = np.int32((nb_batch_test + 1) * 50)
            delta_test = np.int32(nb_total_test - nb_testing)

            test_data_full = np.zeros((nb_total_test, len_series, 1, 1))
            for m in range(nb_testing):
                test_data_full[m, :] = x_test[m, :]
            for m in range(delta_test):
                test_data_full[nb_testing + m, :] = x_test[m, :]

            preds_tmp = []
            for i in range(np.int32(nb_total_test / 50)):
                pred = classifier.predict(test_data_full[i*50: (i+1)*50], batch_size=50, verbose=1)
                if i == 0:
                    preds_tmp = pred
                else:
                    preds_tmp = np.concatenate((preds_tmp, pred))

            preds = preds_tmp[0:nb_testing]
        else:
            preds = classifier.predict(x_test, batch_size=nb_testing, verbose=1)

        new_preds = []
        for i in range(nb_testing):
            new_preds_each = []
            for j in range(int(nb_classes/nb_trans)):
                temp = 0.0
                for k in range(nb_trans):
                    temp += preds[i][j*nb_trans+k]
                new_preds_each.append(temp)
            new_preds.append(new_preds_each)

        new_preds = np.array(new_preds)
        correct_pred = np.equal(np.argmax(new_preds, 1), np.argmax(Y_test_true, 1))
        correct_pred = correct_pred.astype(np.float32)
        accuracy = np.mean(correct_pred)

        print(dataset_name + ': ' + str(accuracy))


if __name__ == '__main__':
	dataset_name = 'ArrowHead'
    classifier_name = 'ResNet'
    nb_epochs = 1500
    nb_trans = 4
    mode = 'eval'	# train or eval
    main(dataset_name, classifier_name, nb_epochs, nb_trans, mode)
