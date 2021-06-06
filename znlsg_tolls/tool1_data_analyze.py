import os
import json
if __name__ == "__main__":
    data_dir = "../../../data"
    train_a_annotations_file = os.path.join(data_dir, "cssjj/train/a_annotations.json")
    train_b_annotations_file = os.path.join(data_dir, "cssjj/train/b_annotations.json")

    test_a_annotations_file = os.path.join(data_dir, "cssjj/test/a_annotations.json")
    test_b_annotations_file = os.path.join(data_dir, "cssjj/test/b_annotations.json")
    with open(train_a_annotations_file, 'r') as f:
        train_a_annotations = json.load(f)
    with open(train_b_annotations_file, 'r') as f:
        train_b_annotations = json.load(f)
    with open(test_a_annotations_file, 'r') as f:
        test_a_annotations = json.load(f)
    with open(test_b_annotations_file, 'r') as f:
        test_b_annotations = json.load(f)
    '''
    train data
        a_annotations.keys() ->  dict_keys(['images', 'annotations', 'categories'])
        b_annotations.keys() -> dict_keys(['images', 'annotations', 'categories'])
        a_annotations['images'].__len__() -> 1458
        a_annotations['annotations'].__len__() -> 23617
        a_annotations['categories'].__len__() -> 116
        a_annotations['categories'][:2] -> [{'id': 0, 'name': 'asamu'}, {'id': 1, 'name': 'baishikele'}]
        
        b_annotations['images'].__len__() -> 3964
        b_annotations['annotations'].__len__() -> 3965
        b_annotations['categories'].__len__() -> 116
        b_annotations['categories'][:2] -> [{'id': 0, 'name': 'asamu'}, {'id': 1, 'name': 'baishikele'}]
    '''
    '''
    test_a_annotations.keys() -> dict_keys(['images', 'annotations', 'categories'])
    
    test_a_annotations['images'].__len__() -> 1513
    test_a_annotations['annotations'].__len__() -> 0
    test_a_annotations['categories'].__len__() -> 116
    test_a_annotations['categories'][:2] -> [{'id': 0, 'name': 'asamu'}, {'id': 1, 'name': 'baishikele'}]
    
    test_b_annotations.keys() -> dict_keys(['images', 'annotations', 'categories'])
    test_b_annotations['images'].__len__() -> 3827
    test_b_annotations['annotations'].__len__() -> 3828
    test_b_annotations['categories'].__len__() -> 116
    test_b_annotations['categories'][:2] -> [{'id': 0, 'name': 'asamu'}, {'id': 1, 'name': 'baishikele'}]
    '''

