import glob
import json
import os

import cv2

from src.create_annotations import (create_image_annotation, create_annotation_format, find_contours,
                                    get_coco_json_format, create_category_annotation)

MAIN_PATH = '../../datasets'

category_ids = {
    'EX': 1,
    'HE': 2,
    'SE': 3,
    'MA': 4
}

MASK_TYPE = 'tif'
IMAGE_TYPE = 'jpg'

def images_annotations_info(maskpath, dataset_name):
    annotation_id = 0
    annotations = []
    images = []

    for category in category_ids.keys():
        for mask_image in glob.glob(os.path.join(maskpath, category, f'*.{MASK_TYPE}')):
            original_file_name = (f'{os.path.basename(mask_image).split(".")[0]}').split('_') 

            if dataset_name == 'idrid':
                original_file_name = f'{original_file_name[0]}_{original_file_name[1]}.{IMAGE_TYPE}' 
            else:
                original_file_name = f'{original_file_name[0]}.{IMAGE_TYPE}' 

            mask_image_open = cv2.imread(mask_image)
            height, width, c = mask_image_open.shape

            if original_file_name not in map(lambda img: img['file_name'], images):
                image = create_image_annotation(file_name=original_file_name, width=width, height=height)
                images.append(image)
            else:
                image = [element for element in images if element['file_name'] == original_file_name][0]

            contours = find_contours(mask_image_open)

            for contour in contours:
                annotation = create_annotation_format(contour, image['id'], category_ids[category], annotation_id)
                if annotation['area'] > 0:
                    annotations.append(annotation)
                    annotation_id += 1

    return images, annotations, annotation_id

if __name__ == '__main__':
    for dataset in ['ddr', 'idrid']:
        coco_format = get_coco_json_format() 

        for type in ['valid', 'test', 'train']:
            mask_path = f'../../datasets/{dataset}/{type}/label'

            coco_format['categories'] = create_category_annotation(category_ids)

            coco_format['images'], coco_format['annotations'], annotation_cnt = images_annotations_info(mask_path, dataset)

            with open(f'{MAIN_PATH}/{dataset}/{type}/_annotations.coco.json', 'w') as outfile:
                json.dump(coco_format, outfile, sort_keys=True, indent=4)

            print('Created %d annotations for images in folder: %s' % (annotation_cnt, mask_path))
