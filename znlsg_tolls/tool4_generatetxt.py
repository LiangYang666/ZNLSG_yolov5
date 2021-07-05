import os

from tool2_watch_img import ZNLSG_COCO

def generate_all_labels_from_json(annotations_file):
    assert ('a_annotations' in annotations_file) or ('b_annotations' in annotations_file)
    if 'a_annotations' in annotations_file:
        ab ='a'
    else:
        ab = 'b'

    coco = ZNLSG_COCO(annotations_file)
    parent_dir = os.path.dirname(annotations_file)
    labels_path = os.path.join(parent_dir, f'{ab}_labels')
    images_path = os.path.join(parent_dir, f'{ab}_images')

    all_yolov5_imgs_txts_file = os.path.join(parent_dir, f'{ab}_all_yolov5_imgs_txts.txt')
    check_create_path(labels_path)

    for imageid in coco.getImgIds():
        imgInfo = coco.loadImgs(imageid)[0]
        height = int(imgInfo['height'])
        width = int(imgInfo['width'])
        file_name = imgInfo['file_name']
        annotationIds = coco.getAnnIds(imageid)
        anns = coco.loadAnns(annotationIds)
        txt_name = file_name.split('.')[0] + '.txt'
        # yolo_labels = []
        with open(os.path.join(labels_path, txt_name), 'w') as f:
            for ann in anns:
                bbox = ann['bbox']
                catogory = int(ann['category_id'])
                bbox = [float(x) for x in bbox]
                bbox[0] = (bbox[0] + bbox[2]/2) / width
                bbox[1] = (bbox[1] + bbox[3]/2) / height
                bbox[2] = bbox[2] / width
                bbox[3] = bbox[3] / height
                x = [catogory] + bbox
                f.write(f'{x[0]} {x[1]} {x[2]} {x[3]} {x[4]}\n')
    with open(all_yolov5_imgs_txts_file, 'w') as f:
        labels_l = sorted(os.listdir(labels_path))
        images_l = sorted(os.listdir(images_path))
        print(f'Total {len(labels_l)} labels!')
        print(f'Total {len(images_l)} images!')
        nums = 0
        for image_path in images_l:
            label_path = image_path.split('.')[0]+'.txt'
            if label_path in labels_l:
                txt = os.path.join(f'{ab}_labels', label_path)
                img = os.path.join(f'{ab}_images', image_path)
                f.write(' '.join([txt, img])+'\n')
                nums += 1
        print(f'Total {nums} available!')





def check_create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print('Create ', path)



if __name__ == '__main__':
    data_dir = "../../../data"
    train_a_annotations_file = os.path.join(data_dir, "cssjj/train/a_annotations.json")
    train_b_annotations_file = os.path.join(data_dir, "cssjj/train/b_annotations.json")

    test_a_annotations_file = os.path.join(data_dir, "cssjj/test/a_annotations.json")
    test_b_annotations_file = os.path.join(data_dir, "cssjj/test/b_annotations.json")

    # train_a_coco = ZNLSG_COCO(train_a_annotations_file)
    generate_all_labels_from_json(test_b_annotations_file)






