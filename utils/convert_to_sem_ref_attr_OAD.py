import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import os
# import some common libraries
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer2 import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

classes = np.load('labels_OAD.npy', allow_pickle=True) # add the OAD labels here
m_add =  '/media/matt/Samsung_T5/OAD/OAD/data/' # add the path for OAD data here

def _create_text_labels(classes, scores, class_names):
    """
    Args:
        classes (list[int] or None):
        scores (list[float] or None):
        class_names (list[str] or None):

    Returns:
        list[str] or None
    """
    labels = None
    if classes is not None and class_names is not None and len(class_names) > 0:
        labels = [class_names[i] for i in classes]
    if scores is not None:
        if labels is None:
            labels = ['Uknkown']
            scores = ["{:.0f}".format(s * 100) for s in scores]
        else:
            #labels = ["{} {:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
            labels = ["{}".format(l) for l in labels]
            scores = ["{:.0f}".format(s * 100) for s in scores]
    return labels, scores



cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])


predictor = DefaultPredictor(cfg)

def run_img(img):
    predictions = predictor(img)
    predictions = predictions["instances"].to("cpu")
    
    boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    scores = predictions.scores if predictions.has("scores") else None
    classes = predictions.pred_classes if predictions.has("pred_classes") else None
    labels, scores = _create_text_labels(classes, scores, metadata.get("thing_classes", None))
    return labels, scores, boxes.get_centers().data.numpy()


def runOnVideo(video, keyFrames):
    """ Runs the predictor on every frame in the video (unless maxFrames is given),
    and returns the frame with the predictions drawn.
    """
    count = 0
    labels_s = []
    scores_s = []
    centers_s = []
    #readFrames = 0
    while True:
        
        hasFrame, frame = video.read()
        if not hasFrame:
            break

        # Get prediction results for this frame

        # Make sure the frame is colored
        if (count % keyFrames == 0):

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Draw a visualization of the predictions using the video visualizer
            labels, scores, centers = run_img(frame)
    
            # Convert Matplotlib RGB format to OpenCV BGR format
            # visualization = cv2.cvtColor(visualization.get_image(), cv2.COLOR_RGB2BGR)
    
            # yield visualization
            labels_s.append(labels)
            scores_s.append(scores)
            centers_s.append(centers)
        count = count + 1
    return labels_s, scores_s, centers_s


dir_folders = sorted(os.listdir(m_add), key=lambda x: int(os.path.splitext(x)[0]))
labels_total = []
scores_total = []
centers_total = []
for i in range(0, len(dir_folders)):
    folder = m_add + dir_folders[i] + '/color'
    dir_files =  sorted(os.listdir(folder), key=lambda x: int(os.path.splitext(x)[0]))
    labels_class = []
    scores_class = []
    centers_class = []    
    for j in range(0, len(dir_files)):

        print (dir_files[j])
        image = cv2.imread(folder + '/' + dir_files[j])
        # width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # frames_per_second = video.get(cv2.CAP_PROP_FPS)
        # num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        
        # key_frame =  int(num_frames / 20) 
            
        l, s, c = run_img(image)
        
        labels_class.append(l)
        scores_class.append(s)
        centers_class.append(c)

        #video.release()
    labels_total.append(labels_class)
    scores_total.append(scores_class)
    centers_total.append(centers_class)
    
labels = labels_total
scores = scores_total
centers = centers_total

dict_objects_video = []
for i in range(0, len(labels)):
    print(i)
    diff = len(classes[i]) - len(labels[i])
    classes[i] = classes[i][diff:len(classes[i])]

    dict_objects_class = []
    for k in range(0, len(labels[i])):
        if (k == 0):
            dict_objects = np.array(['object', 'repetitions', 'total movement',
                                     'average confidence', 'current_center', 'current_i_k_n', 'class', 'diff'])
        if (classes[i][k] != classes[i][k - 1]):
            dict_objects = np.array(['object', 'repetitions', 'total movement',
                                     'average confidence', 'current_center', 'current_i_k_n', 'class', 'diff'])

        for n in range(0, len(labels[i][k])):

            if (labels[i][k][n] != 'person'):
                if labels[i][k][n] in dict_objects:
                    result = np.where(dict_objects[:, 0] == labels[i][k][n])
                    result = result[0][0]
                    dict_objects[result, 1] = dict_objects[result, 1] + 1
                    dict_objects[result, 3] = (float(dict_objects[result, 3]) * (dict_objects[result, 1] - 1)
                                               + float(scores[i][k][n])) / float(dict_objects[result, 1])

                    if (i == dict_objects[result, 5][0] and
                            k != dict_objects[result, 5][1]):
                        dist = np.linalg.norm(dict_objects[result, 4] - centers[i][k][n])
                        dict_objects[result, 2] = dict_objects[result, 2] + dist

                    dict_objects[result, 5] = np.array([i, k, n])
                    dict_objects[result, 6] = classes[i][k]
                    dict_objects[result, 7] = diff
                    # classes[i][k+diff]


                else:
                    dict_objects = np.append(dict_objects.reshape([int(dict_objects.size / 8), 8]),
                                             np.array([labels[i][k][n], 1, 0,
                                                       scores[i][k][n], centers[i][k][n],
                                                       np.array([i, k, n]), classes[i][k], diff]).reshape([1, 8]),
                                             axis=0)

        if (classes[i][k] != classes[i][k - 1]):
            dict_objects_class.append(dict_objects)

    dict_objects_video.append(dict_objects_class)

np.save('off_sem_ref_attr_OAD', dict_objects_video)
