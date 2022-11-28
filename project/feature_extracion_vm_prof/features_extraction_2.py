import os, json, cv2, random, sys, math, io, base64
import numpy as np
from PIL import Image
from tqdm import tqdm
tqdm.pandas()

from imantics import Mask

from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from tensorflow.keras.preprocessing import image

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import torch

from transformers import DetrFeatureExtractor, DetrForSegmentation, DetrModel

feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-101-panoptic")
model_detr = DetrForSegmentation.from_pretrained("facebook/detr-resnet-101-panoptic")

cfg = get_cfg()
cfg.MODEL.DEVICE='cpu'
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml"))

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml")

model_maskrcnn = DefaultPredictor(cfg)

from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.applications.resnet import ResNet101

resnet = ResNet101(weights='imagenet', include_top=False, pooling='max')

import pandas as pd
from scipy import spatial

def get_size(image_size):
    min_size = 600
    max_size = 1000
    if not isinstance(min_size, (list, tuple)):
        min_size = (min_size,)
    w, h = image_size
    size = random.choice(min_size)
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))

    if (w <= h and w == size) or (h <= w and h == size):
        return (h, w)

    if w < h:
        ow = size
        oh = int(size * h / w)
    else:
        oh = size
        ow = int(size * w / h)

    return (oh, ow)

def generate_additional_features(rect,h,w):
    
    mask = np.array([w,h,w,h],dtype=np.float32)
    rect = np.clip(rect/mask,0,1)
    res = np.hstack((rect,[rect[3]-rect[1], rect[2]-rect[0]]))
    return res.astype(np.float32)


def compute_features(ROI, features_extractor):
  if (ROI.shape[0] * ROI.shape[1]> 0):
    ROI = cv2.resize(np.float32(ROI), (224, 224), interpolation = cv2.INTER_LINEAR)
    img_data = image.img_to_array(ROI)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    feature = features_extractor.predict(img_data)
    return feature
  else:
    return []

def compute_results(seg_outputs, model_name, classes, im, features_extractor, height, width, classes_lvis = [], bbox_id = 0, clean_image = []):
  result = []
  if model_name == "LVIS":
    lvis_im = im
    for class_name, bbox, mask, score in zip(seg_outputs["instances"].pred_classes.cpu().numpy(), seg_outputs["instances"].pred_boxes.tensor.cpu().numpy(), seg_outputs["instances"].pred_masks.cpu().numpy(), seg_outputs["instances"].scores.cpu().numpy()):
      class_name = classes[str(class_name)].capitalize()
      x,y,w,h = tuple(bbox)
      mask = cv2.resize(np.float32(mask), (np.array(im).shape[1], np.array(im).shape[0]), interpolation = cv2.INTER_AREA)
      ROI = np.where(mask[...,None]!=0, im, [255,255,255])[int(y):int(h), int(x):int(w)]
      feature = compute_features(ROI, features_extractor)
      feature = np.array(feature)
      pos_feat = generate_additional_features(bbox, height, width)
      feature_seg = feature.flatten() # le uso per filtrare gli oggetti
      feature = np.hstack((feature_seg, pos_feat)).astype(np.float32)
      ROI_region = im[int(y):int(h), int(x):int(w)] # usato per filtrare gli oggetti
      result.append({"rect": bbox.tolist(), "bbox_id": bbox_id, "class": class_name, "conf": score,  "feature_seg": feature_seg, "feature": feature, "feature_base64": base64.b64encode(feature).decode("utf-8"), "features_bbox": np.array(compute_features(ROI_region, features_extractor)).flatten()})
      bbox_id += 1
      classes_lvis.append(class_name)
      lvis_im = np.where(mask[...,None]==0, lvis_im,[255,255,255])
    return result, classes_lvis, lvis_im, bbox_id
  
  elif model_name == "COCO":
    for i in range(len(seg_outputs["labels"])):
      coco_class = classes[str(seg_outputs["labels"][i].numpy())].capitalize()
      mask = seg_outputs["masks"][i]
      mask = cv2.resize(np.float32(mask), (np.array(im).shape[1], np.array(im).shape[0]), interpolation = cv2.INTER_AREA)
      bbox = Mask(mask).bbox()
      x,y,w,h = tuple(bbox)
      area = w*h
      if area >0:
        
        ROI_region = im[int(y):int(h), int(x):int(w)]
        if coco_class not in classes_lvis:
          ROI_clean = np.where(mask[...,None]!=0, clean_image, [255,255,255])[int(y):int(h), int(x):int(w)]
          number_of_white_pix_clean = np.sum(ROI_clean == 255)
          number_of_not_white_pix_clean = np.sum(ROI_clean != 255)
          if number_of_white_pix_clean  > number_of_not_white_pix_clean: # se l'immagine è troppo bianca (troppo segmentata) la nuova segmentazione è fatta su quella di partenza
            ROI = np.where(mask[...,None]!=0, im, [255,255,255])[int(y):int(h), int(x):int(w)]
            feature = compute_features(ROI, features_extractor)
          else:
            ROI = ROI_clean
            feature = compute_features(ROI, features_extractor)
        elif coco_class in classes_lvis:
            ROI = np.where(mask[...,None]!=0, im, [255,255,255])[int(y):int(h), int(x):int(w)]

            feature = compute_features(ROI, features_extractor)
        feature = np.array(feature)
        if len(feature) > 0:
          clean_image = np.where(mask[...,None]==0, clean_image, [255,255,255]) # contiene la parte non segmentata da aggiungere eventualmente con classe vuota ""
          bbox = list(bbox)
          feature_seg = feature.flatten() # usato per filtrare gli oggetti
          pos_feat = generate_additional_features(bbox, height, width)
          feature = np.hstack((feature_seg, pos_feat)).astype(np.float32)
          ROI_region = im[int(y):int(h), int(x):int(w)] # usato per filtrare gli oggetti
          result.append({"rect": bbox, "bbox_id": bbox_id, "class": coco_class, "conf": seg_outputs["scores"][i].item(), "feature_seg": feature_seg, "feature": feature, "feature_base64": base64.b64encode(feature).decode("utf-8"), "features_bbox": np.array(compute_features(ROI_region, features_extractor)).flatten()})
          bbox_id += 1
        else:
          print("ROI shape: ", ROI.shape)
    return result, clean_image, bbox_id

def filter_segmentation_results(df): # elimina oggetti estratti simili
  eliminated_index = [] # lista di appoggio per non eliminare elementi non più presenti
  for index1, row1 in df.iterrows():
    for index2, row2 in df.loc[(index1+1):].iterrows(): # salto i primi index1 elementi per ridurre il numero di iterazioni
      featurs_similarity_bbox = 1 - spatial.distance.cosine(row1["features_bbox"], row2["features_bbox"])
      featurs_similarity = 1 - spatial.distance.cosine(row1["feature_seg"], row2["feature_seg"])
      bbox_similarity = 1 - spatial.distance.cosine(row1["rect"], row2["rect"])
      if featurs_similarity > 0.7 and featurs_similarity_bbox > 0.7 and bbox_similarity > 0.98: # soglie di similarità
        if row1["conf"] > row2["conf"] and index2 not in eliminated_index:
          df = df.drop(index2)
          eliminated_index.append(index2)
        elif index1 not in eliminated_index:
          df = df.drop(index1)
          eliminated_index.append(index1)
  return df

def compute_extraction(im_cv2, maskrcnn, feature_extractor, detr, resnet, height, width):
  
  outputs = maskrcnn(im_cv2)
  result, classes_lvis, lvis_im, bbox_id = compute_results(outputs, "LVIS", lvis, im_cv2, resnet, height, width)

  im_pil =  Image.fromarray(im_cv2[:, :, ::-1]) # convert image in PIL format
  encoding = feature_extractor(im_pil, return_tensors="pt")
  outputs = detr(**encoding)
  processed_sizes = torch.as_tensor(encoding['pixel_values'].shape[-2:]).unsqueeze(0)
  result_seg = feature_extractor.post_process_segmentation(outputs, processed_sizes, threshold = 0.6)[0]
  result_lvis, clean_image, bbox_id = compute_results(result_seg, "COCO", coco, im_cv2, resnet, height, width, classes_lvis, bbox_id, lvis_im)
  result = result + result_lvis
  if np.sum(clean_image != 255) > np.sum(clean_image == 255)* 90/100: # aggiungo la parte non segmentata solo se più del 60% circa dei pixels non sono bianchi
    feature_seg = np.array(compute_features(clean_image, resnet)).flatten()
    print("(height, width): ",(height, width))
    bbox = [0, 0, clean_image.shape[1], clean_image.shape[0]]
    print("clean_image.shape[1], clean_image.shape[0]: ",(clean_image.shape[1], clean_image.shape[0]))
    pos_feat = generate_additional_features(bbox, height, width)
    feature = np.hstack((feature_seg, pos_feat)).astype(np.float32)
    bbox_id += 1
    result.append({"rect": bbox, "bbox_id": bbox_id, "class": "", "conf": 0, "feature_seg": feature_seg, "feature": feature, "feature_base64": base64.b64encode(feature).decode("utf-8"), "features_bbox": feature_seg})
  df = pd.DataFrame(result)
  return filter_segmentation_results(df)
  
def compute_row(img):
  img_path = "./validation/"+img
  print(img_path)
  im_cv2 = cv2.imread(img_path)
  new_size = get_size((im_cv2.shape[1],im_cv2.shape[0]))
  im_cv2 = cv2.resize(im_cv2, (new_size[1], new_size[0]), interpolation = cv2.INTER_AREA)
  height = im_cv2.shape[0]
  width = im_cv2.shape[1]
  global model_maskrcnn
  global feature_extractor
  global model_detr
  global resnet
  df_extraction = compute_extraction(im_cv2, model_maskrcnn, feature_extractor, model_detr, resnet, height, width)
  features_arr = df_extraction["feature"].values 
  del df_extraction["feature"]
  df_extraction = df_extraction.rename(columns={"feature_base64": "feature"})
  features = np.vstack(tuple(features_arr))
  features = base64.b64encode(features).decode("utf-8")
  #predictions_column.append(json.dumps({"objects": df_extraction.to_dict("records")})) # full predictions
  series = pd.Series([json.dumps({"features":features, "num_boxes":len(features_arr)}), json.dumps(df_extraction[["class", "conf", "rect"]].to_dict("records"))])

  series.to_csv('./features_val2/{}.csv'.format(img.replace(".jpg","")))

  return series 

from maskrcnn_benchmark.structures.tsv_file_ops import tsv_reader, tsv_writer
import yaml
import os.path as op
from shutil import copyfile

if __name__ == '__main__':

  f = open('./coco_categories_detr.json')
  coco = json.load(f)
  
  f = open('./lvis_categories_maskrcnn.json')
  lvis = json.load(f)
  
  sg_tsv = './data/val.hw.tsv'
  df_train_hw = pd.read_csv(sg_tsv,sep='\t',header = None,converters={1:json.loads})
  print("Numero di immagini: {}".format(len(df_train_hw)))
  
  df_predictions = pd.DataFrame({"image_id": df_train_hw[0]})
  df_predictions["img"] = df_predictions["image_id"].apply(lambda x: str(x)+".jpg")
  #df_predictions = df_predictions.iloc[3000:14000]
  #df_predictions = df_predictions[~df_predictions["img"].isin([sub.replace(".csv", ".jpg") for sub in os.listdir("./features_test2/")])]
  df_predictions[["features", "label"]] = df_predictions['img'].progress_apply(compute_row)
  print("Features estratte")
  
  OUTPUT_DIR = 'Oscar/inference_test_segmentation_val/'
  LABEL_FILE = os.path.join(OUTPUT_DIR,'label.tsv')
  FEATURE_FILE = os.path.join(OUTPUT_DIR,'features.tsv')
  if not os.path.exists(OUTPUT_DIR):
      os.makedirs(OUTPUT_DIR)
      print(f"path to {OUTPUT_DIR} created")
  
  tsv_writer(df_predictions[['image_id','label']].values.tolist(),LABEL_FILE)
  tsv_writer(df_predictions[['image_id','features']].values.tolist(),FEATURE_FILE)
  
  #yaml_dict = {"label": "label.tsv", "feature": "features.tsv", "img": "train.tsv", "hw": "train.hw.tsv"}
  
  #with open(op.join(OUTPUT_DIR, 'test.yaml'), 'w') as file:
  #        yaml.dump(yaml_dict, file)
          
  #copyfile("./data/train.hw.tsv", os.path.join(OUTPUT_DIR,'train.hw.tsv'))
  #copyfile("./data/train.tsv", os.path.join(OUTPUT_DIR,'train.tsv'))
  print("File .tsv creati")