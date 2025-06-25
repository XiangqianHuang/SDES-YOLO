import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# 验证参数官方详解链接：https://docs.ultralytics.com/modes/val/#usage-examples:~:text=of%20each%20category-,Arguments%20for%20YOLO%20Model%20Validation,-When%20validating%20YOLO

if __name__ == '__main__':
    model = YOLO('/home/hxq/yolov8/ultralytics-main0729/runs/train/SDES-YOLO (Ours)/weights/best.pt')
    model.val(data='/home/hxq/yolov8/ultralytics-main0729/dataset/data1.yaml',
              split='test',
              imgsz=640,
              batch=64,
              # iou=0.7,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/test',
              name='exp',
              )