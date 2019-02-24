import numpy as np
from keras import backend as K
import sys
import os


def main():
    K.set_image_dim_ordering('tf')
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    from keras_video_classifier.library.recurrent_networks import VGG16LSTMVideoClassifier
    from keras_video_classifier.library.utility.ucf.UCF101_loader import scan_dataset_with_labels

    for i in range(4):
        data_set_name = 'CV_{}_Test'.format(i + 1)
        vgg16_include_top = False
        data_dir_path = os.path.join(
            os.path.dirname(__file__), 'very_large_data')
        saved_model_name = 'CV_{}_Train'.format(i + 1)
        model_dir_path = os.path.join(
            os.path.dirname(__file__), 'models', saved_model_name)
        config_file_path = VGG16LSTMVideoClassifier.get_config_file_path(
            model_dir_path, vgg16_include_top=vgg16_include_top)
        weight_file_path = VGG16LSTMVideoClassifier.get_weight_file_path(
            model_dir_path, vgg16_include_top=vgg16_include_top)

        print("data_set_name: ", data_set_name)
        print("saved_model_name: ", saved_model_name)
        print("data_dir_path: ", data_dir_path)
        print("model_dir_path: ", model_dir_path)
        print("config_file_path: ", config_file_path)
        print("weight_file_path: ", weight_file_path)

        np.random.seed(42)

        predictor = VGG16LSTMVideoClassifier()
        predictor.load_model(config_file_path, weight_file_path)

        videos = scan_dataset_with_labels(
            data_dir_path,
            [label for (label, label_index) in predictor.labels.items()], data_set_name)

        print("videos: ", videos)

        video_file_path_list = np.array(
            [file_path for file_path in videos.keys()])

        print("video_file_path_list: ", video_file_path_list)

        np.random.shuffle(video_file_path_list)

        correct_count = 0
        count = 0

        for video_file_path in video_file_path_list:
            label = videos[video_file_path]
            predicted_label = predictor.predict(video_file_path)
            print("'predicted: ' + predicted_label + ' actual: ' + label: ", 'predicted: ' + predicted_label + ' actual: ' + label)
            correct_count = correct_count + 1 if label == predicted_label else correct_count
            count += 1
            accuracy = correct_count / count
            print("'accuracy: ', accuracy: ", 'accuracy: ', accuracy)

        break


if __name__ == '__main__':
    main()
