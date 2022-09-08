#!/usr/bin/env python3

import xml.etree.ElementTree as ET
from PIL import Image
import os
import json


def createAnnotationPascalVocTree(folder, basename, path, width, height):
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = folder
    ET.SubElement(annotation, 'filename').text = basename
    ET.SubElement(annotation, 'path').text = path

    source = ET.SubElement(annotation, 'source')
    ET.SubElement(source, 'database').text = 'Unknown'

    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = width
    ET.SubElement(size, 'height').text = height
    ET.SubElement(size, 'depth').text = '3'

    ET.SubElement(annotation, 'segmented').text = '0'

    return ET.ElementTree(annotation)


def createObjectPascalVocTree(xmin, ymin, xmax, ymax):
    obj = ET.Element('object')
    ET.SubElement(obj, 'name').text = 'face'
    ET.SubElement(obj, 'pose').text = 'Unspecified'
    ET.SubElement(obj, 'truncated').text = '0'
    ET.SubElement(obj, 'difficult').text = '0'

    bndbox = ET.SubElement(obj, 'bndbox')
    ET.SubElement(bndbox, 'xmin').text = xmin
    ET.SubElement(bndbox, 'ymin').text = ymin
    ET.SubElement(bndbox, 'xmax').text = xmax
    ET.SubElement(bndbox, 'ymax').text = ymax

    return ET.ElementTree(obj)


def createAnnotationPascalVocTreeWithAgeGender(folder, basename, path, width, height, age_class, gender_class):
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = folder
    ET.SubElement(annotation, 'filename').text = basename
    ET.SubElement(annotation, 'path').text = path
    ET.SubElement(annotation, 'age_class').text = json.dumps(age_class)
    ET.SubElement(annotation, 'gender_class').text = json.dumps(gender_class, ensure_ascii=False)

    source = ET.SubElement(annotation, 'source')
    ET.SubElement(source, 'database').text = 'xintu'

    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = width
    ET.SubElement(size, 'height').text = height
    ET.SubElement(size, 'depth').text = '3'

    ET.SubElement(annotation, 'segmented').text = '0'

    return ET.ElementTree(annotation)


def createObjectPascalVocTreeWithAgeGender(xmin, ymin, xmax, ymax, age_seg, gender, age):
    obj = ET.Element('object')
    ET.SubElement(obj, 'name').text = 'face'
    ET.SubElement(obj, 'pose').text = 'Unspecified'
    ET.SubElement(obj, 'truncated').text = '0'
    ET.SubElement(obj, 'difficult').text = '0'

    bndbox = ET.SubElement(obj, 'bndbox')
    ET.SubElement(bndbox, 'xmin').text = xmin
    ET.SubElement(bndbox, 'ymin').text = ymin
    ET.SubElement(bndbox, 'xmax').text = xmax
    ET.SubElement(bndbox, 'ymax').text = ymax
    ET.SubElement(bndbox, 'age_seg').text = age_seg
    ET.SubElement(bndbox, 'gender').text = gender
    ET.SubElement(bndbox, 'age').text = age

    return ET.ElementTree(obj)


def parseImFilename(imFilename, imPath):
    im = Image.open(os.path.join(imPath, imFilename))

    folder, basename = imFilename.split('/')
    width, height = im.size

    return folder, basename, imFilename, str(width), str(height)


def convertWFAnnotations(annotationsPath, targetPath, imPath):
    ann = None
    basename = ''
    with open(annotationsPath) as f:
        while True:
            imFilename = f.readline().strip()
            if imFilename:
                nbBndboxes = f.readline()
                if int(nbBndboxes) == 0:
                    x1, y1, w, h, _, _, _, _, _, _ = [int(i) for i in f.readline().split()]
                    print('the {} has {} box, x={} y={} w={} h={}\n'.format(basename, nbBndboxes, x1, y1, w, h))
                    continue

                folder, basename, path, width, height = parseImFilename(imFilename, imPath)
                ann = createAnnotationPascalVocTree(folder, basename, os.path.join(imPath, path), width, height)

                i = 0
                while i < int(nbBndboxes):
                    i = i + 1
                    x1, y1, w, h, _, _, _, _, _, _ = [int(i) for i in f.readline().split()]

                    ann.getroot().append(createObjectPascalVocTree(str(x1), str(y1), str(x1 + w), str(y1 + h)).getroot())

                if not os.path.exists(targetPath):
                    os.makedirs(targetPath)
                annFilename = os.path.join(targetPath, basename.replace('.jpg', '.xml'))
                ann.write(annFilename)
                # print('{} => {}'.format(basename, annFilename))
            else:
                break
    f.close()


def convertXTAnnotations(root_path):
    ann = None
    # /mnt/sdb/data.set/face_and_age_gender
    # Annotations

    max_path = ''
    img_size = dict()
    for sub_dir in ['part01', 'part02', 'part03']:

        target_path = os.path.join(root_path, 'Annotations', sub_dir)
        os.makedirs(target_path, exist_ok=True)

        with open(os.path.join(root_path, '{}.json'.format(sub_dir)), mode='r') as r:
            anno = json.load(r)

        with open(os.path.join(root_path, '{}.err.txt'.format(sub_dir)), mode='r') as r:
            err_img = r.readlines()

        err_name = [name.strip('\n').split('#')[1] for name in err_img]

        with open(os.path.join(root_path, '{}.img.err.txt'.format(sub_dir)), mode='r') as r:
            err_img = r.readlines()

        err_name.extend([name.strip('\n').split('#')[1] for name in err_img])

        img_metadata = anno['_via_img_metadata']
        image_id_list = anno['_via_image_id_list']
        attributes = anno['_via_attributes']
        female_class = attributes['region']['性别']['options']
        new_female_class = dict()
        for k, v in female_class.items():
            if v == '男':
                new_female_class[k] = 'man'
            elif v == '女':
                new_female_class[k] = 'woman'

        age_class = attributes['region']['年龄']['options']
        for img_sub_name, metadata in anno['_via_img_metadata'].items():
            if img_sub_name in err_name:
                continue

            folder, basename, path, width, height = parseImFilename(img_sub_name, root_path)

            ann = createAnnotationPascalVocTreeWithAgeGender(folder, basename, img_sub_name, width, height, age_class, new_female_class)
            for region in metadata['regions']:
                shape = region['shape_attributes']
                female = region['region_attributes']
                x = shape['x']
                y = shape['y']
                w = shape['width']
                h = shape['height']

                import math
                size = int(math.sqrt(w * h)) // 100
                if img_size.get(size, None) is None:
                    img_size[size] = list()
                img_size[size].append((w, h))

                continue

                age_seg = female['年龄']
                age = female['识别年龄']
                gender = female['性别']

                ann.getroot().append(createObjectPascalVocTreeWithAgeGender(str(x), str(y), str(x + w), str(y + h), age_seg, gender, age).getroot())

            # if basename.find('JPG') != -1:
            #     annFilename = os.path.join(target_path, basename.replace('.JPG', '.xml'))
            # else:
            #     annFilename = os.path.join(target_path, basename.replace('.jpg', '.xml'))
            # ann.write(annFilename)
    import math
    img_size_sorted = sorted(img_size.items(), key=lambda kv:(len(kv[1])), reverse=True)
    with open('/mnt/sdb/data.set/face_and_age_gender/face.size.txt', mode='w') as wh:
        for k, size in img_size_sorted:
            wh.write('size = {}*100 cnt={}\n'.format(k, len(size)))
            idx = 0
            wh.write('  ')
            for s in size:
                wh.write('{} '.format(s))
                idx += 1
                if idx % 10 == 0:
                    wh.write('\n  ')
            wh.write('\n')






if __name__ == '__main__':
    import argparse

    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('-ap', '--annotations-path', help='the annotations file path. ie:"./wider_face_split/wider_face_train_bbx_gt.txt".', default='')
    PARSER.add_argument('-tp', '--target-path', help='the target directory path where XML files will be copied.', default='')
    PARSER.add_argument('-ip', '--images-path', help='the images directory path. ie:"./WIDER_train/images"', default='')

    ARGS = vars(PARSER.parse_args())

    # convertWFAnnotations(ARGS['annotations_path'], ARGS['target_path'], ARGS['images_path'])
    convertXTAnnotations(ARGS['images_path'])

