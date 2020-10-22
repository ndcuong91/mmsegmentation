import os, cv2
import numpy as np
from classifier_crnn.prepare_crnn_data import get_list_file_in_folder


def convert_anno_detection_to_segmentation(img_dir, anno_det_dir, output_anno_segment_dir, extend=-1, format_anno_det='icdar', class_list=dict()):
    list_images = get_list_file_in_folder(img_dir)
    list_images = sorted(list_images)
    for img_name in list_images:
        print(img_name)
        img_path=os.path.join(img_dir,img_name)
        img = cv2.imread(img_path)
        anno_mask =  np.zeros((img.shape[0], img.shape[1]), np.uint8)
        anno_file = os.path.join(anno_det_dir,img_name.replace('.jpg','.txt').replace('.png','.txt'))


        tree = open(anno_file, 'r', encoding='UTF-8')
        root = tree.readlines()
        for i, line in enumerate(root):
            line_str = line.split('\t')[0].replace('\n', '')
            idx = -1
            for i in range(0, 8):
                idx = line_str.find(',', idx + 1)

            coordinates = line_str[:idx]
            val = line_str[idx + 1:]
            left, top, right, _, _, bottom, _, _ = coordinates.split(",")
            cv2.rectangle(anno_mask,(int(left)-extend,int(top)-extend),(int(right)+extend,int(bottom)+extend),1,-1)
        cv2.imwrite(os.path.join(output_anno_segment_dir,img_name),anno_mask)



if __name__=='__main__':
    #img=cv2.imread('/home/cuongnd/PycharmProjects/aicr/source/mmsegmentation/data/ade/ADEChallengeData2016/annotations/validation/ADE_val_00000012.png', cv2.IMREAD_GRAYSCALE)
    img_dir='/data20.04/data/table recognition/from_Korea/201016_132333_obj1_perf_testset/images'
    anno_det_dir='/data20.04/data/table recognition/from_Korea/201016_132333_obj1_perf_testset/gt_refined_16Oct_icdar'
    output_anno_segment_dir='/data20.04/data/table recognition/from_Korea/201016_132333_obj1_perf_testset/gt_refined_16Oct_segment'

    convert_anno_detection_to_segmentation(img_dir, anno_det_dir, output_anno_segment_dir)



