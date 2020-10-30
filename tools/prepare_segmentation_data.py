import os, cv2
import numpy as np
import random, shutil

def get_list_file_in_folder(dir, ext=['jpg', 'png', 'JPG', 'PNG']):
    included_extensions = ext
    file_names = [fn for fn in os.listdir(dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    return file_names


def get_list_dir_in_folder(dir):
    sub_dir = [o for o in os.listdir(dir) if os.path.isdir(os.path.join(dir, o))]
    return sub_dir


def get_list_file_in_dir_and_subdirs(folder, ext=['jpg', 'png', 'JPG', 'PNG']):
    file_names = []
    for path, subdirs, files in os.walk(folder):
        for name in files:
            extension = os.path.splitext(name)[1].replace('.', '')
            if extension in ext:
                file_names.append(os.path.join(path, name).replace(folder, '')[1:])
                # print(os.path.join(path, name).replace(folder,'')[1:])
    return file_names


def get_list_dir_and_subdirs_in_folder(folder):
    list_dir = [x[0].replace(folder, '').lstrip('/') for x in os.walk(folder)]
    return list_dir


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

def convert_anno_objective2_to_segmentation(img_dir, anno_det_dir, output_anno_segment_dir, extend=-1, format_anno_det='icdar', class_list=dict()):
    list_images = get_list_file_in_folder(img_dir)
    list_images = sorted(list_images)
    for idx, img_name in enumerate(list_images):
        print(idx, img_name)
        img_path=os.path.join(img_dir,img_name)
        img = cv2.imread(img_path)
        anno_mask =  np.zeros((img.shape[0], img.shape[1]), np.uint8)
        anno_file = os.path.join(anno_det_dir,img_name.replace('.jpg','.json').replace('.png','.json'))

        import json
        with open(anno_file, "r") as anno:
            anno_str = json.load(anno)

        for i, line in enumerate(anno_str['cellboxes']):
            left, top, right, bottom = line[0], line[1], line[2], line[3]
            cv2.rectangle(anno_mask,(int(left)-extend,int(top)-extend),(int(right)+extend,int(bottom)+extend),1,-1)
        cv2.imwrite(os.path.join(output_anno_segment_dir,img_name),anno_mask)
        print('ok')

def split_dataset(img_dir, ann_dir, img_dst_dir, ann_dst_dir, ratio=0.5):
    list_images = get_list_file_in_folder(img_dir)
    random.shuffle(list_images)
    num_file=int(len(list_images)*ratio)
    print('split_dataset. Copy',num_file,'files')
    for idx, img_name in enumerate(list_images):
        if idx>num_file:
            continue
        print(idx, img_name)
        ann_name=img_name.replace('.jpg','.png').replace('.JPG','.png')
        shutil.copy(os.path.join(img_dir,img_name),os.path.join(img_dst_dir,img_name))
        shutil.copy(os.path.join(ann_dir,ann_name),os.path.join(ann_dst_dir,ann_name))
    print('Done')

def del_dataset(img_dir, ann_dir):
    list_images = get_list_file_in_folder(img_dir)
    list_images = sorted(list_images)
    for idx, img_name in enumerate(list_images):
        print(idx, img_name)
        ann_path=os.path.join(ann_dir,img_name.replace('.jpg','.png'))
        if not os.path.exists(ann_path):
            os.remove(os.path.join(img_dir,img_name))
    print('Done')


def refine_dataset(img_dir, ann_dir):
    list_images = get_list_file_in_folder(img_dir)
    parent_dir=os.path.dirname(img_dir)
    output_dir=os.path.join(parent_dir,'img_wo_anno')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for idx, file in enumerate(list_images):
        img_path=os.path.join(img_dir,file)
        anno_file= os.path.join(ann_dir, file.replace('.jpg','.png'))
        if not os.path.exists(anno_file):
            print(idx, file)
            print('----------------------------------------------------------------------------------')
            shutil.move(img_path,os.path.join(output_dir,file))
            kk=1


if __name__=='__main__':
    #img=cv2.imread('/home/cuongnd/PycharmProjects/aicr/source/mmsegmentation/data/ade/ADEChallengeData2016/annotations/validation/ADE_val_00000012.png', cv2.IMREAD_GRAYSCALE)

    data_dir='/data20.04/data/table recognition/from_Korea/201012_172754_pubtabnet_valid_sample_objective#2'
    img_dir= data_dir + '/images'
    anno_det_dir=data_dir + '/annots'
    output_anno_segment_dir=data_dir + '/annot_seg'

    #convert_anno_objective2_to_segmentation(img_dir, anno_det_dir, output_anno_segment_dir)

    # split_dataset(img_dir='/data4T/ntanh/publaynet/train',
    #               ann_dir='/data4T/ntanh/publaynet_gen_gt_oct2.1/train/label'  ,
    #               img_dst_dir='/data20.04/data/doc_structure/publaynet/img_dir/train',
    #               ann_dst_dir='/data20.04/data/doc_structure/publaynet/ann_dir/train',
    #               ratio=0.5)

    # del_dataset(img_dir='/data20.04/data/doc_structure/publaynet/img_dir/train',
    #             ann_dir='/data20.04/data/doc_structure/publaynet/ann_dir/train')

    refine_dataset(img_dir='/data4T/ntanh/publaynet/train',
                ann_dir='/data4T/ntanh/publaynet_gen_gt_oct2.1/train/label')




