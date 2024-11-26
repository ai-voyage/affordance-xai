import json
import cv2
import numpy as np
import torch
import pickle
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
import os
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, ClassifierOutputSoftmaxTarget

from data_train_pipeline import num_epochs
from helper_functions import heatmap_generator as h
import gzip


methods = {"gradcam": GradCAM,
           "scorecam": ScoreCAM,
           "gradcam++": GradCAMPlusPlus,
           "ablationcam": AblationCAM,
           "xgradcam": XGradCAM,
           "eigencam": EigenCAM,
           "eigengradcam": EigenGradCAM,
           "layercam": LayerCAM,
           "fullgrad": FullGrad}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "swin_base_patch4_window7_224"
epochs = "10"

with open(f'data/all_possible_labels_list_{model_name}_epoch{epochs}.pkl', 'rb') as f:
    label_list_saved = pickle.load(f)

num_labels = len(label_list_saved)

print(label_list_saved)

# Load model parameters


model = torch.load(f'models/model_{model_name}_affordance_{num_epochs}.pth')
model = model.to(device)  # Ensure you load to the correct device

model.eval()
print(model)

target_layers = [model.layers[-1].blocks[-1].norm2, model.layers[-1].blocks[-2].norm2]
print(target_layers)

print(model.norm)

# Get the CAM mask and the original image
target_category = None  # Let Grad-CAM decide the most influential class

list_of_method = ['eigencam', 'scorecam']

root_directory = f"data/Test Images-{model_name}-tiny_{epochs}"
json_directory = "data/RGBD_affordance_dataset/JSON_format"



test_folder_list = os.listdir(root_directory)
test_folder_list_name = [i.split('.')[0] for i in test_folder_list]

iou_threshold = 0.05

for fold in os.listdir(json_directory):
    print(f"------'{fold}'--------- ")
    class_path = os.path.join(json_directory, fold)
    for class_name in os.listdir(class_path):
        print(f"++++++{class_name}+++++++")
        json_path = os.path.join(class_path, class_name)
        # json_files = [f for f in os.listdir(json_path) if f.endswith('.json')]

        # Iterate over each JSON file in the directory

        for json_filename in os.listdir(json_path):
            filename = json_filename.split('.')[0]
            if json_filename.endswith('.json') and filename in test_folder_list_name:

                image_name_index = test_folder_list_name.index(filename)
                json_file_path = os.path.join(json_path, json_filename)
                for method in list_of_method:
                    iou_scores = []
                    destination_root = f'images/vit_results/Model_{model_name}_epoch{epochs}_testDataset_iouscore_/{method}_2Labels_lastLayer_smooth'
                    cam_file_name = test_folder_list[image_name_index].split('.')[0] + f'_{method}_' + '.' + \
                                    test_folder_list[image_name_index].split('.')[1]
                    destination_image_path = os.path.join(destination_root, cam_file_name)
                    if (os.path.exists(destination_image_path)):
                        print("exists skipping")
                        continue
                    print(method)


                    if method == "ablationcam":
                        cam = methods[method](model=model,
                                              target_layers=target_layers,
                                              # use_cuda=True,
                                              reshape_transform=h.reshape_transform_swin,
                                              ablation_layer=AblationLayerVit())
                    else:
                        cam = methods[method](model=model,
                                              target_layers=target_layers,
                                              # use_cuda=use_cuda,

                                              reshape_transform=h.reshape_transform_swin)

                    print("############################")
                    targets = None
                    cam.batch_size = 32

                    eigen_smooth = True
                    aug_smooth = True
                    image_path = os.path.join(root_directory, test_folder_list[image_name_index])
                    shape = cv2.imread(image_path, 1).shape
                    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]

                    rgb_img = cv2.resize(rgb_img, (224, 224))
                    rgb_img = np.float32(rgb_img) / 255
                    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]).to("cuda")

                    # find the most predicted labels:
                    predictions = h.predict_single_image_tensor(input_tensor, model, label_list_saved)
                    # print(predictions)
                    sorted_predictions = dict(sorted(predictions.items(), key=lambda x: x[1]))
                    # print(sorted_predictions)
                    if not list(predictions):
                        print("Empty predictions")
                        continue

                    category_label_list = []
                    label_logit_list = []
                    if len(list(predictions)) < 2:
                        category_label = list(sorted_predictions.keys())[
                            -1]  # select the second last else select the only one or something like that.
                        category = label_list_saved.index(category_label)
                        targets = [ClassifierOutputTarget(category)]
                        label_logit = list(sorted_predictions.values())[-1]


                        label_logit_list.append(label_logit)
                        category_label_list.append(category_label)
                    else:
                        category_label = list(sorted_predictions.keys())[-2]
                        category = label_list_saved.index(category_label)


                        category_label_1 = list(sorted_predictions.keys())[-1]
                        category_1 = label_list_saved.index(category_label_1)


                        targets = [ClassifierOutputTarget(category),
                                   ClassifierOutputTarget(category_1)]
                        category_label_list.append(category_label)
                        category_label_list.append(category_label_1)
                        label_logit_list.append(list(sorted_predictions.values())[-2])
                        label_logit_list.append(list(sorted_predictions.values())[-1])

                    heatmaps = cam(input_tensor=input_tensor,
                                   targets=targets,
                                   eigen_smooth=eigen_smooth,
                                   aug_smooth=aug_smooth
                                   )
                    heatmap_ = heatmaps[0, :]
                    mask = h.json_to_mask(json_file_path, shape, heatmap_.shape)
                    binarized_heatmap = h.binarize_map(heatmap_, iou_threshold)
                    iou_score = h.compute_iou(binarized_heatmap, mask)
                    iou_scores.append(iou_score)
                    print(f"IoU Score for {filename}: {iou_score:.4f}")
                    # Store heatmap path and IoU score for the current method
                    cam_image = show_cam_on_image(rgb_img, heatmap_)


                    entry = {"filename": test_folder_list[image_name_index],
                             "threshold": iou_threshold,
                             "iou_score": iou_score,
                             "heatmap": heatmap_.tolist(),
                             "method": method,
                             "predicted_labels": np.array(category_label_list).tolist(),
                             "predicted_logits": np.array(label_logit_list).tolist()

                             }

                    # If None, returns the map for the highest scoring category.
                    # Otherwise, targets the requested category.
                    if not os.path.exists(destination_root):
                        print(f"making directory for {destination_root} ")
                        os.makedirs(destination_root)
                    print(f"saving '{destination_image_path}'")
                    cv2.imwrite(destination_image_path, cam_image)
                    print("saved image")


                    # Save results to a JSON file
                    # Serialize and compress JSON data with gzip
                    gzip_name = f"{json_filename.split('.')[0]}_{method}" + "." + f"{json_filename.split('.')[1]}" + ".gz"
                    print(gzip_name)
                    gzip_path = os.path.join(destination_root, gzip_name)
                    print(gzip_name)
                    if os.path.exists(gzip_path):
                        print("json exists for \n", gzip_path)
                        continue
                    with gzip.open(gzip_path, 'wt', encoding='utf-8') as outfile:
                        json.dump(entry, outfile)
                        print("saved gzip")

print("done")
