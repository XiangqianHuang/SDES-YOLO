import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# 训练参数官方详解链接：https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings:~:text=a%20training%20run.-,Train%20Settings,-The%20training%20settings

if __name__ == '__main__':
    model = YOLO('/home/hxq/yolov8/ultralytics-main0729/hxq/yolov8-EIEStem-FPS-SEAM.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='/home/hxq/yolov8/ultralytics-main0729/dataset/data1.yaml',
                cache=False,
                imgsz=640,
                epochs=250,
                batch=64,
                # close_mosaic=0,
                # workers=4,
                 device='0',
                optimizer='SGD', # using SGD
                patience=0, # close earlystop
                # resume=True, # 断点续训,YOLO初始化时选择last.pt
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )