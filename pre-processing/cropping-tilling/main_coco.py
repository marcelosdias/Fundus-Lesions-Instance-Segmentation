import asyncio
import glob
import json
import random
from itertools import chain

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm

from util import sanitize_pixels

import os

random.seed(42)

MAIN_PATH = '../../datasets/datasets-cropping-tilling'

DATASET = '../../datasets/{DATASET_NAME}'
DATASET_OUT = '../../datasets/datasets-cropping-tilling/{DATASET_NAME}/'
FOLDER_PATH = '{DATASET}/{DATASET_TYPE}/*.json'

pbar = tqdm()

image_data = {}

def coco2albumentations(segmentation):
    return [(segmentation[i], segmentation[i + 1]) for i in range(0, len(segmentation), 2)]

def albumentations2coco(keypoints):
    return list(chain.from_iterable(keypoints))

def coco2cv2(keypoints):
    return np.array([(keypoints[i], keypoints[i + 1]) for i in range(0, len(keypoints), 2)], dtype='int')

def hough_crop_coco(image, annotations, image_id, anno_id):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 31)

    circles = cv2.HoughCircles(image=gray, method=cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=70, param2=50)

    if circles is None:
        return image, annotations

    circles = np.uint16(np.around(circles))
    x, y, r = circles[0][0]

    height, width, channels = image.shape
    x_min, y_min = int(x - r), int(y - r)
    x_max, y_max = int(x_min + 2 * r), int(y_min + 2 * r)

    x_min, y_min, x_max, y_max = sanitize_pixels(
        x_min=x_min, y_min=y_min,
        x_max=x_max, y_max=y_max,
        width=width, height=height,
    )

    return albumentations_crop(image, annotations, image_id, anno_id, x_min, y_min, x_max, y_max)

def crop_tiles_coco(image, annotations, anno_id, image_id, num_rows=2, num_cols=2, percent=0.15):
    original_height, original_width, _ = image.shape
    height = int(original_height / num_rows)
    width = int(original_width / num_cols)
    image_tiles, labels_tiles = [], []
    index = 0

    image_data[image_id] = {}

    for row in range(num_rows):
        for col in range(num_cols):
            y_min = row * height
            y_max = y_min + height
            x_min = col * width
            x_max = x_min + width
            x_min, y_min, x_max, y_max = sanitize_pixels(
                x_min=round(x_min * (1 - percent)), y_min=round(y_min * (1 - percent)),
                x_max=round(x_max * (1 + percent)), y_max=round(y_max * (1 + percent)),
                width=original_width, height=original_height,
            )

            letra = 'a'
            if index == 1:
                letra = 'b'
            elif index == 2:
                letra = 'c'
            elif index == 3:
                letra = 'd'

            image_tile, labels_tile = albumentations_crop(
                image, annotations, str(image_id) + letra, anno_id, x_min, y_min, x_max, y_max
            )
            index += 1
            image_data[image_id][letra] = {'height': image_tile.shape[0], 'width': image_tile.shape[1]}

            image_tiles.append(image_tile)
            labels_tiles.append(labels_tile)
    return image_tiles, labels_tiles

def albumentations_crop(image, annotations, image_id, anno_id, x_min, y_min, x_max, y_max):
    height, width, _ = image.shape
    transforms = [A.Crop(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, always_apply=True)]
    return apply_albumentations(image, annotations, image_id, anno_id, transforms)

def apply_albumentations(image, annotations, image_id, annotation_id, transforms):
    keypoint_kwargs = {
        f'keypoints_{index}': coco2albumentations(anno['segmentation'][0]) for index, anno in enumerate(annotations)
    }

    transform = A.Compose(
        transforms=transforms,
        keypoint_params=A.KeypointParams(format='xy'),
        additional_targets={f'keypoints_{index}': 'keypoints' for index, anno in enumerate(annotations)},
    )
    transformed = transform(image=image, keypoints=[], **keypoint_kwargs)

    new_annos = []
    for anno, keypoint_name in zip(annotations, keypoint_kwargs):
        keypoints = albumentations2coco(transformed[keypoint_name])
        cv2_keypoints = coco2cv2(keypoints)
        if not keypoints or len(keypoints) <= 4:
            continue
        new_annos.append({
            'id': annotation_id,
            'image_id': image_id,
            'iscrowd': anno['iscrowd'],
            'segmentation': [keypoints],
            'category_id': anno['category_id'],
            'area': cv2.contourArea(cv2_keypoints),
            'bbox': list(cv2.boundingRect(cv2_keypoints)),
        })

    return transformed['image'], new_annos

def process(image, dataset_type, anno, anno_id, dataset, dataset_out):
    input_image_path = f'{dataset}/{dataset_type}/images/{image["file_name"]}'
    output_image_path = f'{dataset_out}/{dataset_type}/{image["file_name"]}'
    cv2_image = cv2.imread(filename=input_image_path)
    img_annotations = list(filter(lambda x: x['image_id'] == image['id'], anno['annotations']))

    cv2_image, img_annotations = hough_crop_coco(cv2_image, img_annotations, None, anno_id)

    cv2_image_tiles, img_annotations_tiles = crop_tiles_coco(cv2_image, img_annotations, anno_id, image['id'])

    for index, cv2_image in enumerate(cv2_image_tiles):
        new_file_name = output_image_path.replace('.jpg', f'_{index}.jpg')
        cv2.imwrite(filename=new_file_name, img=cv2_image)

    if pbar.desc != dataset_type:
        pbar.reset()
        pbar.total = len(anno['images'])
        pbar.desc = dataset_type

    pbar.update(1)

    return list(chain.from_iterable(img_annotations_tiles))

def parse_ids(anno, new_annos):
    new_images = []
    new_image_id = 0
    for dict_image in anno['images']:
        for letra, index in zip(['a', 'b', 'c', 'd'], [0, 1, 2, 3]):
            new_image_id += 1
            current = image_data[dict_image['id']][letra]
            new_images.append({
                'id': new_image_id,
                'width': current['width'],
                'height': current['height'],
                'file_name': dict_image['file_name'].replace('.jpg', f'_{index}.jpg'),
            })

            for index, new_anno in enumerate(new_annos):
                if new_anno['image_id'] == f'{dict_image["id"]}{letra}':
                    new_annos[index]['image_id'] = new_image_id

    for index, _ in enumerate(new_annos):
        new_annos[index]['id'] = index + 1

    return new_images, new_annos

async def main():
    os.makedirs(MAIN_PATH, exist_ok=True)

    for selected_dataset in ['ddr', 'idrid']:
        out_path = DATASET_OUT.format(DATASET_NAME=selected_dataset)

        os.makedirs(out_path, exist_ok=True)

        for dataset_type in ['valid', 'train', 'test']:   
            os.makedirs(f'{out_path}/{dataset_type}',exist_ok=True)

            path = DATASET.format(DATASET_NAME = selected_dataset)

            for anno_path in glob.iglob(FOLDER_PATH.format(DATASET = path, DATASET_TYPE = dataset_type)):
                with open(anno_path, 'r') as f_in:
                    anno = json.load(f_in)

                    tasks = [
                        asyncio.to_thread(
                            process, 
                            image=image, 
                            dataset_type=dataset_type, 
                            anno=anno, 
                            anno_id='x', 
                            dataset=DATASET.format(DATASET_NAME = selected_dataset),
                            dataset_out=out_path

                        )
                        for image in anno['images']
                    ]

                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    new_annos = list(chain.from_iterable(results))

                    with open(anno_path.replace(DATASET.format(DATASET_NAME = selected_dataset), out_path), 'w') as f_out:
                        anno['images'], anno['annotations'] = parse_ids(anno, new_annos)
                        json.dump(anno, f_out)

if __name__ == '__main__':
    asyncio.run(main())
