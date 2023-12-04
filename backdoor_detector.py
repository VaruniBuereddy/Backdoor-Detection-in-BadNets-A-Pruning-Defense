import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
from keras.models import Model, load_model
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Add
from eval import data_loader
import warnings

warnings.filterwarnings("ignore")


class G(tf.keras.Model):
    def __init__(self, B, B_prime):
        super(G, self).__init__()
        self.B = B
        self.B_prime = B_prime

    def call(self, data):
        y_pred = self.B(data)
        y = np.argmax(y_pred, axis=1)
        y_prime = np.argmax(self.B_prime(data), axis=1)
        pred = np.zeros((y.shape[0], 1284))
        for i in range(y.shape[0]):
            if y[i] == y_prime[i]:
                pred[i, :-1] = y_pred[i, :]
            else:
                pred[i, 1283] = 1
        return pred


class Backdoor_Detector():
    def __init__(self, B_net, clean_valid, clean_test, bad_valid, bad_test, pruning_thresholds):
        self.clean_valid_x, self.clean_valid_y = data_loader(clean_valid)
        self.clean_test_x, self.clean_test_y = data_loader(clean_test)
        self.bad_valid_x, self.bad_valid_y = data_loader(bad_valid)
        self.bad_test_x, self.bad_test_y = data_loader(bad_test)
        self.bad_model = load_model(B_net)
        self.pruning_thresholds = pruning_thresholds

    def plot_images(self, bdata, cdata):
        figure = plt.figure(figsize=(10,8))
        cols, rows = 2,2
        for i in range(1, cols*rows+1):
            if(i>rows):
                index = np.random.randint(bdata[0].shape[0], size=1)
                img, label = (cdata[0][index], cdata[1][index])
            else:
                index = np.random.randint(bdata[0].shape[0], size=1)
                img, label = (bdata[0][index], bdata[1][index])

            figure.add_subplot(rows, cols, i)
            plt.title("Label: {}".format(label))
            plt.axis("off")
            plt.imshow(img[0]/255)
        plt.show()

    def Attack_success_rate(self, model, bad_x, bad_y):
        ASR = np.mean(np.equal(np.argmax(model(bad_x), axis=1), bad_y)) * 100
        return ASR

    def Clean_Accuracy(self, model, clean_x, clean_y):
        Acc = np.mean(np.equal(np.argmax(model(clean_x), axis=1), clean_y)) * 100
        return Acc

    def get_intermediate_activations(self, B, lastlayer, x_valid):
        B_partial = Model(inputs=B.input, outputs=B.get_layer(lastlayer).output)
        feature_maps_valid = B_partial.predict(x_valid)
        average_activations = np.mean(feature_maps_valid, axis=(0, 1, 2))

        return average_activations

    def prune_last_layer(self, B_net, Dvalid):
        B = load_model(B_net)
        B_prime = load_model(B_net)
        x_valid, y_valid = Dvalid

        average_activations = self.get_intermediate_activations(B, "pool_3", x_valid)
        idx_to_prune = np.argsort(average_activations)

        lastlayerWeights, lastlayerBiases = B.get_layer("conv_3").get_weights()[0], B.get_layer("conv_3").get_weights()[1]
        initial_accuracy = self.Clean_Accuracy(B, x_valid, y_valid)

        threshold = self.pruning_thresholds[0]
        j = 0
        repaired_net_accuracy = []
        repaired_net_asr = []

        for i, ch_idx in enumerate(idx_to_prune):
            # Prune one channel at a time
            lastlayerWeights[:, :, :, ch_idx], lastlayerBiases[ch_idx] = 0, 0

            # Update weights of G
            B_prime.get_layer("conv_3").set_weights([lastlayerWeights, lastlayerBiases])

            # Evaluate the updated model's accuracy on the validation set
            current_accuracy = self.Clean_Accuracy(B_prime, x_valid, y_valid)
            
            # Repair the Network
            repaired_net = G(B, B_prime)

            test_ASR_repaired_net = self.Attack_success_rate(repaired_net, self.bad_test_x, self.bad_test_y)
            test_accuracy_repaired_net = self.Clean_Accuracy(repaired_net, self.clean_test_x, self.clean_test_y)
            print(f"Number of Channels Pruned: {i + 1}, Repaired Test accuracy: {test_accuracy_repaired_net:.5f}%, Repaired ASR: {test_ASR_repaired_net:.5f}%")
            repaired_net_accuracy.append(test_accuracy_repaired_net)
            repaired_net_asr.append(test_ASR_repaired_net)

            # If accuracy drops by at least pruning_threshold, stop pruning
            if initial_accuracy - current_accuracy >= threshold:
                model_filename = os.path.join('models', f"Repaired_net_{threshold}_percent_threshold.h5")
                B_prime.save(model_filename)
                print(f"Saving repaired network for {threshold}% threshold at: {model_filename}")
                j += 1
                threshold = self.pruning_thresholds[j]

        return repaired_net_accuracy, repaired_net_asr


def _parse_arguments():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-m",
        "--model",
        help="Enter path to the model with weights in .h5 format",
        type=str,
        default='models/bd_net.h5',
    )
    argparser.add_argument(
        "-v",
        "--clean_valid",
        help="Enter path to the clean validation data in .h5 format",
        type=str,
        default='data/cl/valid.h5',
    )
    argparser.add_argument(
        "-t",
        "--clean_test",
        help="Enter path to the clean test data in .h5 format",
        type=str,
        default='data/cl/test.h5',
    )
    argparser.add_argument(
        "-b",
        "--bad_valid",
        help="Enter path to the poisoned validation data in .h5 format",
        type=str,
        default='data/bd/bd_valid.h5',
    )
    argparser.add_argument(
        "-bt",
        "--bad_test",
        help="Enter path to the poisoned test data in .h5 format",
        type=str,
        default='data/bd/bd_test.h5',
    )
    argparser.add_argument(
        "-th",
        "--thresholds",
        help="Enter pruning thresholds as a list of comma-separated values",
        type=str,
        default='2,4,10,100',
    )
    return argparser.parse_args()


if __name__ == "__main__":
    args = _parse_arguments()
    model_path = args.model
    clean_valid_path = args.clean_valid
    clean_test_path = args.clean_test
    bad_valid_path = args.bad_valid
    bad_test_path = args.bad_test
    thresholds = [int(th) for th in args.thresholds.split(',')]
    BD = Backdoor_Detector(model_path, clean_valid_path, clean_test_path, bad_valid_path, bad_test_path, thresholds)
    BD.plot_images([BD.bad_valid_x, BD.bad_valid_y], [BD.clean_valid_x, BD.clean_valid_y])
    print("Before Pruning the Model: ")
    print("Clean Accuracy:", BD.Clean_Accuracy(BD.bad_model, BD.clean_valid_x, BD.clean_valid_y))
    print("Attack Success Rate:", BD.Attack_success_rate(BD.bad_model, BD.bad_valid_x, BD.bad_valid_y))

    print("##########################\nRepairing the network: ")
    repaired_net_accuracy, repaired_net_asr = BD.prune_last_layer(model_path, [BD.clean_valid_x, BD.clean_valid_y])
    plt.figure(2)
    plt.plot(np.arange(1,61)/60, repaired_net_accuracy)
    plt.plot(np.arange(1,61)/60, repaired_net_asr)
    plt.legend("Clean Accuracy", "Attack Success Rate")
    plt.xlabel("Fraction of Channels Pruned")
    plt.title("Clean Accuracy and ASR for Repaired Net on Test Dataset")
    plt.show()
    K.clear_session()
