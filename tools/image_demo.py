from argparse import ArgumentParser
import os


from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from mmsegmentation.tools.prepare_segmentation_data import get_list_file_in_folder

img_dir='/home/cuongnd/PycharmProjects/open-mmlab/mmsegmentation/data/table_structure1/img_dir/val'
img_path='/data20.04/data/SEVT/SEVT_img_1022/smd_005.jpg'
#img_path=''
config='../configs/pspnet/pspnet_r50-d8_512x512_320k_publaynet_split1.py'
ckpt='../work_dirs/pspnet_r50-d8_512x512_320k_publaynet_split1/iter_256000.pth'

def main():
    parser = ArgumentParser()
    parser.add_argument('--img', help='Image file', default=img_path)
    parser.add_argument('--config', help='Config file', default=config)
    parser.add_argument('--checkpoint', help='Checkpoint file', default=ckpt)
    parser.add_argument(
        '--device', default='cuda:1', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default=None,
        help='Color palette used for segmentation map')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)

    if args.img=='':
        list_img=get_list_file_in_folder(img_dir)
        list_img=sorted(list_img)
        for img_ in list_img:
            img=os.path.join(img_dir,img_)
            print(img)
            result = inference_segmentor(model, img)
            show_result_pyplot(model, img, result, get_palette(args.palette))
    else:
        result = inference_segmentor(model, args.img)
        show_result_pyplot(model, args.img, result, get_palette(args.palette))


if __name__ == '__main__':
    main()
