import os
import re
import sys
sys.path.append('../')
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

# model config
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

# train_images_dir = '../dataset/cocoformatwithimg/images'
# train_annotations_filename = '../dataset/cocoformatwithimg/annotations/instances_default.json'
# val_images_dir = '../dataset/cocoformatwithimg/images'
# val_annotations_filename = '../dataset/cocoformatwithimg/annotations/instances_default.json'
# log_dir = '../Pelee_keras_logs'
train_images_dir = '../dataset/4-Dir'
train_annotations_filename = '../dataset/4-dir-ann/train.json'
val_images_dir = '../dataset/new-training/train'
val_annotations_filename = '../dataset/new-training/train.json'
log_dir = '../Pelee_keras_logs'
# learning rate schedule
def lr_schedule(epoch):
    if epoch < 20:
        return 0.0001
    elif epoch < 50:
        return 0.00001
    else:
        return 0.000001


# set trainable layers
def set_trainable(layer_regex, keras_model=None, indent=0, verbose=1):
    # In multi-GPU training, we wrap the model. Get layers
    # of the inner model because they have the weights.
    layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") \
        else keras_model.layers

    for layer in layers:
        # Is the layer a model?
        if layer.__class__.__name__ == 'Model':
            print("In model: ", layer.name)
            set_trainable(
                layer_regex, keras_model=layer)
            continue

        if not layer.weights:
            continue
        # Is it trainable?
        trainable = bool(re.fullmatch(layer_regex, layer.name))
        # Update layer. If layer is a container, update inner layer.
        if layer.__class__.__name__ == 'TimeDistributed':
            layer.layer.trainable = trainable
        else:
            layer.trainable = trainable
        # Print trainable layer names
        if trainable and verbose > 0:
            print("{}{:20}   ({})".format(" " * indent, layer.name, layer.__class__.__name__))


# build model
model = pelee(image_size, n_classes, mode, l2_regularization, min_scale, max_scale, scales,
                         aspect_ratios_global, aspect_ratios_per_layer, two_boxes_for_ar1, steps,
                         offsets, clip_boxes, variances, coords, normalize_coords, subtract_mean,
                         divide_by_stddev, swap_channels, confidence_thresh, iou_threshold, top_k,
                         nms_max_output_size, return_predictor_sizes)

# load weights
weights_path = '../Pelee_keras_logs/pelee_coco_30_loss-4.2146_val_loss-3.6678.h5'

model.load_weights(weights_path, by_name=True)

# compile the model
adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
# set_trainable(r"(ssd\_.*)", model)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

#print(model.summary())

# load data
train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)

train_dataset.parse_json(images_dirs=[train_images_dir], annotations_filenames=[train_annotations_filename],
                         ground_truth_available=True, include_classes='all', ret=False)
val_dataset.parse_json(images_dirs=[val_images_dir], annotations_filenames=[val_annotations_filename],
                       ground_truth_available=True, include_classes='all', ret=False)

# We need the `classes_to_cats` dictionary. Read the documentation of this function to understand why.
cats_to_classes, classes_to_cats, cats_to_names, classes_to_names = get_coco_category_maps(train_annotations_filename)

# set the image transformations for pre-processing and data augmentation options.
# For the training generator:
ssd_data_augmentation = SSDDataAugmentation(img_height=image_size[0],
                                            img_width=image_size[1],
                                            background=subtract_mean)

# For the validation generator:
convert_to_3_channels = ConvertTo3Channels()
resize = Resize(height=image_size[0], width=image_size[1])

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

# create the generator handles that will be passed to Keras' `fit_generator()` function.

train_generator = train_dataset.generate(batch_size=batch_size,
                                         shuffle=True,
                                         transformations=[ssd_data_augmentation],
                                         label_encoder=ssd_input_encoder,
                                         returns={'processed_images',
                                                  'encoded_labels'},
                                         keep_images_without_gt=False)

val_generator = val_dataset.generate(batch_size=batch_size,
                                     shuffle=False,
                                     transformations=[convert_to_3_channels,
                                                      resize],
                                     label_encoder=ssd_input_encoder,
                                     returns={'processed_images',
                                              'encoded_labels'},
                                     keep_images_without_gt=False)

# Get the number of samples in the training and validations datasets.
train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size = val_dataset.get_dataset_size()

print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))

callbacks = [LearningRateScheduler(schedule=lr_schedule, verbose=1),
             TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=False),
             ModelCheckpoint(
                 os.path.join(log_dir, "pelee_coco_{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5"),
                 monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)]

model.fit(train_generator, epochs=30, steps_per_epoch=100,
                    callbacks=callbacks, validation_data=val_generator,
                    validation_steps=100, initial_epoch=0)
