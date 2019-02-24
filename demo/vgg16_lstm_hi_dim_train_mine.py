import numpy as np
from keras import backend as K
import os
import sys


def main():
    K.set_image_dim_ordering('tf')
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    from keras_video_classifier.library.utility.plot_utils import plot_and_save_history
    from keras_video_classifier.library.recurrent_networks import VGG16LSTMVideoClassifier

    for i in range(4):
        data_set_name = 'CV_{}_Train'.format(i + 1)
        input_dir_path = os.path.join(
            os.path.dirname(__file__), 'very_large_data')
        output_dir_path = os.path.join(
            os.path.dirname(__file__), 'models', data_set_name)
        report_dir_path = os.path.join(
            os.path.dirname(__file__), 'reports', data_set_name)

        print("data_set_name: ", data_set_name)
        print("input_dir_path: ", input_dir_path)
        print("output_dir_path: ", output_dir_path)
        print("report_dir_path: ", report_dir_path)

        np.random.seed(42)

        classifier = VGG16LSTMVideoClassifier()

        history = classifier.fit(data_dir_path=input_dir_path, model_dir_path=output_dir_path, vgg16_include_top=False,
                                data_set_name=data_set_name)

        plot_and_save_history(history, VGG16LSTMVideoClassifier.model_name,
                            report_dir_path + '/' + VGG16LSTMVideoClassifier.model_name + '-hi-dim-history.png')
        
        break


if __name__ == '__main__':
    main()
