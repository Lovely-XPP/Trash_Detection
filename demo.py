import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import time

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode
from predictor import VisualizationDemo

# constants
WINDOW_NAME = "detections"

# inference
INPUT_IMG_PATH = './input_img/'
OUTPUT_IMG_PATH = './ouput_img/'
OUTPUT_VIDEO_PATH = './out_video/'

# 数据集路径
ROOT_DIR = os.getcwd()
DATASET_ROOT = ROOT_DIR
ANN_ROOT = os.path.join(DATASET_ROOT, 'coco/annotations/')
TRAIN_PATH = os.path.join(DATASET_ROOT, 'coco/images/train2019/')
VAL_PATH = os.path.join(DATASET_ROOT, 'coco/images/val2019/')
TRAIN_JSON = os.path.join(ANN_ROOT, 'instances_train2019.json')
VAL_JSON = os.path.join(ANN_ROOT, 'instances_val2019.json')

# 数据集类别元数据
DATASET_CATEGORIES = [
    {"name": "foodbox", "id": 1, "isthing": 1, "color": [0, 255, 255]},
    {"name": "paper box", "id": 2, "isthing": 1, "color": [255, 255, 0]},
    {"name": "battery cell", "id": 3, "isthing": 1, "color": [255, 165, 0]},
    {"name": "cup", "id": 4, "isthing": 1, "color": [255, 0, 0]},
    {"name": "battery", "id": 5, "isthing": 1, "color": [47, 79, 79]},
    {"name": "bottle", "id": 6, "isthing": 1, "color": [65, 105, 225]},
    {"name": "can", "id": 7, "isthing": 1, "color": [148, 0, 211]},
]


# 声明数据集的子集
PREDEFINED_SPLITS_DATASET = {
    "trash_train": (TRAIN_PATH, TRAIN_JSON),
    "trash_val": (VAL_PATH, VAL_JSON),
}

def register_dataset():
    """
    purpose: register all splits of dataset with PREDEFINED_SPLITS_DATASET
    """
    for key, (image_root, json_file) in PREDEFINED_SPLITS_DATASET.items():
        register_dataset_instances(name=key, 
                                   metadate=get_dataset_instances_meta(), 
                                   json_file=json_file, 
                                   image_root=image_root)


def get_dataset_instances_meta():
    """
    purpose: get metadata of dataset from DATASET_CATEGORIES
    return: dict[metadata]
    """
    thing_ids = [k["id"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    # assert len(thing_ids) == 2, len(thing_ids)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def register_dataset_instances(name, metadate, json_file, image_root):
    """
    purpose: register dataset to DatasetCatalog,
             register metadata to MetadataCatalog and set attribute
    """
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(json_file=json_file, 
                                  image_root=image_root, 
                                  evaluator_type="coco", 
                                  **metadate)


# 注册数据集和元数据
def plain_register_dataset():
    DatasetCatalog.register("trash_train", lambda: load_coco_json(TRAIN_JSON, TRAIN_PATH, "trash_train"))
    MetadataCatalog.get("trash_train").set(thing_classes=["pos", "neg"],
                                                    json_file=TRAIN_JSON,
                                                    image_root=TRAIN_PATH)
    DatasetCatalog.register("trash_val", lambda: load_coco_json(VAL_JSON, VAL_PATH, "trash_val"))
    MetadataCatalog.get("trash_val").set(thing_classes=["pos", "neg"],
                                                json_file=VAL_JSON,
                                                image_root=VAL_PATH)


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    args.config_file = "/home/XPP/桌面/trash/source/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    # 更改配置参数
    cfg.DATASETS.TRAIN = ("trash_train",)
    cfg.DATASETS.TEST = ("trash_val",)
    cfg.DATALOADER.NUM_WORKERS = 2  # 单线程
    cfg.INPUT.MAX_SIZE_TRAIN = 400
    cfg.INPUT.MAX_SIZE_TEST = 400
    cfg.INPUT.MIN_SIZE_TRAIN = (160,)
    cfg.INPUT.MIN_SIZE_TEST = 160
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8 # 类别数
    # cfg.MODEL.WEIGHTS = "/home/XPP/桌面/trash/source/R-50.pkl"  # 预训练模型权重
    cfg.MODEL.WEIGHTS = "./output/model_final.pth"   # 最终权重
    cfg.SOLVER.IMS_PER_BATCH = 2  # batch_size=2; iters_in_one_epoch = dataset_imgs/batch_size  
    ITERS_IN_ONE_EPOCH = int(1434 / cfg.SOLVER.IMS_PER_BATCH)
    cfg.SOLVER.MAX_ITER = (ITERS_IN_ONE_EPOCH * 12) - 1 # 12 epochs
    cfg.SOLVER.BASE_LR = 0.002
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.WEIGHT_DECAY_NORM = 0.0
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.STEPS = (30000,)
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    cfg.SOLVER.WARMUP_ITERS = 20
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.CHECKPOINT_PERIOD = ITERS_IN_ONE_EPOCH - 1
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold

    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    # 是否读取摄像头
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    
    # 是否读取视频
    parser.add_argument("--video-input", help="Path to video file.")

    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/e2e_mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    
    # 注册数据集
    register_dataset()
    
    demo = VisualizationDemo(cfg, instance_mode=ColorMode.SEGMENTATION)

    if args.webcam:
        assert args.input is None, "不可以同时输入 --input 和 --webcam!"
        time_now = time.strftime("%Y%m%d%H%M", time.localtime()) # 获取当前时间
        cap = cv2.VideoCapture(0)
        fps = cap.get(cv2.CAP_PROP_FPS)  # 返回视频的fps--帧率
        path = OUTPUT_VIDEO_PATH + time_now + "/" #获取当前路径
        vpath =  path + 'output.mp4'#视频目录
        if not(os.path.exists(path)):
            #print('n')  #没有就建一个
            os.makedirs(path)
        output_fname = time_now + '.mp4'# 以当前时间命名文件
        output_video = cv2.VideoWriter(
                filename=vpath,
                fourcc=cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                fps=20.0,
                frameSize=(640, 480),
            )
        """
            参数1 即将保存的文件路径
            参数2 VideoWriter_fourcc为视频编解码器
            fourcc意为四字符代码（Four-Character Codes），顾名思义，该编码由四个字符组成,下面是VideoWriter_fourcc对象一些常用的参数,注意：字符顺序不能弄混
            cv2.VideoWriter_fourcc('I', '4', '2', '0'),该参数是YUV编码类型，文件名后缀为.avi 
            cv2.VideoWriter_fourcc('P', 'I', 'M', 'I'),该参数是MPEG-1编码类型，文件名后缀为.avi 
            cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'),该参数是MPEG-4编码类型，文件名后缀为.avi 
            cv2.VideoWriter_fourcc('T', 'H', 'E', 'O'),该参数是Ogg Vorbis,文件名后缀为.ogv 
            cv2.VideoWriter_fourcc('F', 'L', 'V', '1'),该参数是Flash视频，文件名后缀为.flv
            cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),文件名后缀为.mp4
            参数3 为帧播放速率
            参数4 (width,height)为视频帧大小
        """
        for vis in tqdm.tqdm(demo.run_on_video(cap)):
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, vis)
            output_video.write(vis)
            if cv2.waitKey(1) == 27:
                break  # 按Esc键结束
        cap.release()
        output_video.release()
        cv2.destroyAllWindows()

    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        output_fname = os.path.join(OUTPUT_VIDEO_PATH, basename)
        output_fname = os.path.splitext(output_fname)[0] + ".mp4"
        output_file = cv2.VideoWriter(
            filename=output_fname,
            fourcc=cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
            fps=float(frames_per_second),
            frameSize=(width, height),
            isColor=True,
        )
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
            cv2.imshow(basename, vis_frame)
            output_file.write(vis_frame)
            if cv2.waitKey(1) == 27:
                break  # 按Esc键结束
        video.release()
        
    else:
        for imgfile in os.listdir(INPUT_IMG_PATH):

            # use PIL, to be consistent with evaluation
            img_fullName = os.path.join(INPUT_IMG_PATH, imgfile)
            img = read_image(img_fullName, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            logger.info(
                "{}: detected {} instances in {:.2f}s".format(
                    imgfile, len(predictions["instances"]), time.time() - start_time
                )
            )

            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            img = visualized_output.get_image()[:, :, ::-1]
            cv2.imshow(WINDOW_NAME, img)
            img_name = OUPUT_IMG_PATH + os.path.basename(imgfile)
            cv2.imwrite(img_name , img)
            if cv2.waitKey(0) == 27:
                continue  # 按Esc键继续下一个图片
