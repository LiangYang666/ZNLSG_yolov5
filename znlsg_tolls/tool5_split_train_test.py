import os
import random

from tool2_watch_img import *
parent_dir = os.path.dirname(train_a_annotations_file)
all_yolov5_imgs_txts_file = os.path.join(parent_dir, 'a_all_yolov5_imgs_txts.txt')
a_train_txt_file = os.path.join(parent_dir, 'a_train.txt')
a_test_txt_file = os.path.join(parent_dir, 'a_test.txt')

ratio = 0.7


if __name__ == "__main__":
    with open(all_yolov5_imgs_txts_file, 'r') as f:
        all_yolov5_imgs_txts = f.readlines()
        all_yolov5_imgs_txts = [x.strip() for x in all_yolov5_imgs_txts]
    random.shuffle(all_yolov5_imgs_txts)
    total = len(all_yolov5_imgs_txts)
    train_yolov5_imgs_txts = all_yolov5_imgs_txts[:int(ratio*total)]
    test_yolov5_imgs_txts = all_yolov5_imgs_txts[int(ratio*total):]

    print('\tAll available files total', len(all_yolov5_imgs_txts))
    print('\tTrain files total', len(train_yolov5_imgs_txts))
    print('\tTest files total', len(test_yolov5_imgs_txts))

    print(f'\tWriting the txts to {a_train_txt_file} and {a_test_txt_file}.')
    with open(a_train_txt_file, 'w') as f:
        for s in train_yolov5_imgs_txts:
            f.write(s+'\n')
    with open(a_test_txt_file, 'w') as f:
        for s in test_yolov5_imgs_txts:
            f.write(s+'\n')
    print('\tDone!')
