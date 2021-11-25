import os 
import re
import sys
sys.path.append('../')
from os.path import isfile, join
from os import listdir
import tensorflow 
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from keras import backend as K
from models.pelee import pelee
from losses.keras_ssd_loss import SSDLoss
from utils.object_detection_2d_data_generator import DataGenerator
from utils.object_detection_2d_geometric_ops import Resize
from utils.object_detection_2d_photometric_ops import ConvertTo3Channels
from utils.data_augmentation_chain_original_ssd import SSDDataAugmentation
from utils.coco import get_coco_category_maps
from utils.ssd_input_encoder import SSDInputEncoder
from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
import json
from tqdm import trange
from math import ceil


from utils.object_detection_2d_geometric_ops import Resize
from utils.object_detection_2d_patch_sampling_ops import RandomPadFixedAR
from utils.object_detection_2d_photometric_ops import ConvertTo3Channels
from utils.ssd_output_decoder import decode_detections
from utils.object_detection_2d_misc import apply_inverse_transforms


def predict_all_to_json_coco(out_file,
                        model,
                        img_height,
                        img_width,
                        classes_to_cats,
                        data_generator,
                        batch_size,
                        results_in_coco,
                        data_generator_mode='resize',
                        model_mode='training',
                        confidence_thresh=0.35,
                        iou_threshold=0.45,
                        top_k=200,
                        pred_coords='centroids',
                        normalize_coords=True):

    convert_to_3_channels = ConvertTo3Channels()
    resize = Resize(height=img_height, width=img_width)
    if data_generator_mode == 'resize':
        transformations = [convert_to_3_channels,
                           resize]
    elif data_generator_mode == 'pad':
        random_pad = RandomPadFixedAR(patch_aspect_ratio=img_width / img_height, clip_boxes=False)
        transformations = [convert_to_3_channels,
                           random_pad,
                           resize]
    else:
        raise ValueError(
            "Unexpected argument value: `data_generator_mode` can be either of 'resize' or 'pad', \
            but received '{}'.".format(
                data_generator_mode))

    # Set the generator parameters.
    generator = data_generator.generate(batch_size=batch_size,
                                        shuffle=False,
                                        transformations=transformations,
                                        label_encoder=None,
                                        returns={'processed_images',
                                                 'image_ids',
                                                 'inverse_transform'},
                                        keep_images_without_gt=True)
    # Put the results in this list.
    results = []
    # Compute the number of batches to iterate over the entire dataset.
    n_images = data_generator.get_dataset_size()
    print("Number of images in the evaluation dataset: {}".format(n_images))
    n_batches = int(ceil(n_images / batch_size))
    # Loop over all batches.
    tr = trange(n_batches, file=sys.stdout)
    tr.set_description('Producing results file')
    cnt=0
    for i in tr:
        # Generate batch.
        batch_X, batch_image_ids, batch_inverse_transforms = next(generator)
        # Predict.
        y_pred = model.predict(batch_X)
        
        # If the model was created in 'training' mode, the raw predictions need to
        # be decoded and filtered, otherwise that's already taken care of.
        if model_mode == 'training':
            # Decode.
            y_pred = decode_detections(y_pred,
                                       confidence_thresh=confidence_thresh,
                                       iou_threshold=iou_threshold,
                                       top_k=top_k,
                                       input_coords=pred_coords,
                                       normalize_coords=normalize_coords,
                                       img_height=img_height,
                                       img_width=img_width)
        else:
            # Filter out the all-zeros dummy elements of `y_pred`.
            y_pred_filtered = []
            for i in range(len(y_pred)):
                y_pred_filtered.append(y_pred[i][y_pred[i, :, 0] != 0])
            y_pred = y_pred_filtered
        # Convert the predicted box coordinates for the original images.
        y_pred = apply_inverse_transforms(y_pred, batch_inverse_transforms)
        # Convert each predicted box into the results format.

        for k, batch_item in enumerate(y_pred):
            for box in batch_item:
                class_id = box[0]
                # Transform the consecutive class IDs back to the original COCO category IDs.
                
                cat_id = classes_to_cats[class_id]
               
                # Round the box coordinates to reduce the JSON file size.
                xmin = float(round(box[2], 1))
                ymin = float(round(box[3], 1))
                xmax = float(round(box[4], 1))
                ymax = float(round(box[5], 1))
                width = xmax - xmin
                height = ymax - ymin
                bbox = [xmin, ymin, width, height]
                result = {}
                result['id'] = cnt
                cnt=cnt+1
                result['image_id'] = batch_image_ids[k]
                result['category_id'] = cat_id
                result['score'] = float(round(box[1], 3))
                result['bbox'] = bbox
                result['iscrowd']=0
                result['segmentation']:[]
                result['area'] = float(width * height)
                results.append(result)


    results_in_coco['annotations'] = results

    with open(out_file, 'w') as f:
        json.dump(results_in_coco, f)

    print("Prediction results saved in '{}'".format(out_file))
batch_size = 16
image_size = (300, 300, 3)
n_classes = 80
mode = 'training'
l2_regularization = 0.0005
min_scale = 0.1
max_scale = 0.9
scales = None
aspect_ratios_global = None
aspect_ratios_per_layer = [[1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                           [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                           [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                           [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                           [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0]]
two_boxes_for_ar1 = True
steps = None
offsets = None
clip_boxes = False
variances = [0.1, 0.1, 0.2, 0.2]
coords = 'centroids'
normalize_coords = True
subtract_mean = [123, 117, 104]
divide_by_stddev = 128
swap_channels = None
confidence_thresh = 0.01
iou_threshold = 0.45
top_k = 200
nms_max_output_size = 400
return_predictor_sizes = False

K.clear_session()
ssd_data_augmentation = SSDDataAugmentation(img_height=image_size[0],
                                            img_width=image_size[1],
                                            background=subtract_mean)

# For the validation generator:
convert_to_3_channels = ConvertTo3Channels()
resize = Resize(height=image_size[0], width=image_size[1])
model = pelee(image_size, n_classes, mode, l2_regularization, min_scale, max_scale, scales,
                         aspect_ratios_global, aspect_ratios_per_layer, two_boxes_for_ar1, steps,
                         offsets, clip_boxes, variances, coords, normalize_coords, subtract_mean,
                         divide_by_stddev, swap_channels, confidence_thresh, iou_threshold, top_k,
                         nms_max_output_size, return_predictor_sizes)

weights_path = '../Pelee_keras_logs/pelee_coco_30_loss-4.2146_val_loss-3.6678.h5'

model.load_weights(weights_path, by_name=True)

adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
# instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.

# The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.
predictor_sizes = [model.get_layer('ssd_cls1conv2_bn').output_shape[1:3],
                   model.get_layer('ssd_cls2conv2_bn').output_shape[1:3],
                   model.get_layer('ssd_cls3conv2_bn').output_shape[1:3],
                   model.get_layer('ssd_cls4conv2_bn').output_shape[1:3],
                   model.get_layer('ssd_cls5conv2_bn').output_shape[1:3]]

ssd_input_encoder = SSDInputEncoder(img_height=image_size[0],
                                    img_width=image_size[1],
                                    n_classes=n_classes,
                                    predictor_sizes=predictor_sizes,
                                    scales=scales,
                                    aspect_ratios_per_layer=aspect_ratios_per_layer,
                                    two_boxes_for_ar1=two_boxes_for_ar1,
                                    steps=steps,
                                    offsets=offsets,
                                    clip_boxes=clip_boxes,
                                    variances=variances,
                                    matching_type='multi',
                                    pos_iou_threshold=0.5,
                                    neg_iou_limit=0.3,
                                    normalize_coords=normalize_coords)
test_images_dir = '../dataset/5-Dir'

coco_val_annotation = '../dataset/new-training/instances_val2017.json'
cats_to_classes, classes_to_cats, cats_to_names, classes_to_names = get_coco_category_maps(coco_val_annotation)

imageslist = [f for f in listdir(test_images_dir) if isfile(join(test_images_dir, f))]
images = []
for idx, name in enumerate(imageslist):
            item = {}
            item['id'] = idx
            item['license'] = 1
            item['file_name'] = name
            item['height'] = 960
            item['width'] = 1280
            item['date_captured'] = "na"
            images.append(item)

cats = []
cnt=0
for idx, name in enumerate(classes_to_names):
            categories= {}
            categories['id'] = cnt
            cnt=cnt+1
            categories['name'] = name
            categories['supercategory'] = 'none'
            if cnt<=4 :cats.append(categories)
res = {}
res['categories'] = cats
res['images'] = images
res['annotations'] = []

with open('./test_img_filenames.json', 'w') as f:
  json.dump(res, f)
test_annotations_filename = 'test_img_filenames.json'

test_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
imgs, filenames, labels, img_ids =test_dataset.parse_json(images_dirs=[test_images_dir], annotations_filenames=[test_annotations_filename],
                         ground_truth_available=False, include_classes='all', ret=True)

out_file = './5-dir-ann-0.3-2.json'

predict_all_to_json_coco(out_file, model, 300,300, classes_to_cats, test_dataset, len(filenames), res)
