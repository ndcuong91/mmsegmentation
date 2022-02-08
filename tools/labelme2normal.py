import os, cv2
import numpy as np
import json

def get_list_file_in_folder(dir, ext=['jpg', 'png', 'JPG', 'PNG']):
    included_extensions = ext
    file_names = [fn for fn in os.listdir(dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    return file_names

def convert_labelme_label_to_normal_format(src_anno_dir, src_img_dir, dst_anno_dir, label_list, debug=False):
    print('convert_voc_label_to_normal_format')
    print('src_anno_dir',src_anno_dir)
    print('dst_anno_dir',dst_anno_dir)

    list_imgs = get_list_file_in_folder(src_img_dir)
    list_imgs = sorted(list_imgs)

    count_samples ={}
    for label in label_list:
        count_samples[label]=0

    for idx, img_name in enumerate(list_imgs):
        base_name = img_name.split('.')[0]
        if idx < 0:
            continue
        print(idx, 'labelme2normal. Convert', base_name)

        json_path = os.path.join(src_anno_dir, base_name+'.json')
        img = cv2.imread(os.path.join(src_img_dir, img_name))

        segment_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        with open(json_path) as json_file:
            data = json.load(json_file)
            shapes = data["shapes"]
            for shape in shapes:
                point = shape["points"]
                label = shape["label"]
                pts = np.asarray(point,np.int32)
                # label_idx = label_list.index(label)
                label_idx = 1 # chỉ có vùng giấy tờ với idx  =1
                if label not in count_samples.keys():
                    count_samples[label] = 0
                count_samples[label] +=1

                # color = int(20*label_idx)
                color = label_idx

                cv2.fillPoly(segment_img, pts=[pts], color=color)
        if debug:
            cv2.imshow('origin' ,img)
            cv2.imshow('mask' ,segment_img)
            cv2.waitKey(0)


        output_anno_path = os.path.join(dst_anno_dir, base_name+'.png')
        cv2.imwrite(output_anno_path, segment_img)

    print('Number of samples', count_samples)


if __name__ == "__main__":
    # test = cv2.imread('/home/duycuong/PycharmProjects/ocr/others/conversion_tools/segmentation/00004.png')

    src_anno_dir ='/data_backup/cuongnd/mmseg/doc_seg/anno/bhyt'
    src_img_dir = '/data_backup/cuongnd/Viettel_freeform/MAFC/BHYT_origin/imgs/clean'
    dst_anno_dir ='/data_backup/cuongnd/mmseg/doc_seg/anno/bhyt_imgs'
    label_list = ['background','cccd','cccd_back','cmnd_new','cmnd_old','cmnd_old_back',
                    'driverlicense_new','driverlicense_new_back','driverlicense_old','driverlicense_old_back']
    convert_labelme_label_to_normal_format(src_anno_dir,
                                           src_img_dir,
                                           dst_anno_dir,
                                           label_list,
                                           debug = False)