import cv2
import os
import json
import random
import shutil
import tqdm


def test():
    root_path = '/mnt/sdb/data.set/face_and_age_gender'
    out_path = '/mnt/sdb/data.set/face_and_age_gender/test.img'
    os.makedirs(out_path, exist_ok=True)
    for sub_dir in ['part01', 'part02', 'part03']:
        if sub_dir != 'part02':
            continue

        with open(os.path.join(root_path, '{}.json'.format(sub_dir)), mode='r') as r:
            anno = json.load(r)

        # img_metadata = dict()
        # for k, v in anno['_via_img_metadata'].items():
        #     # assert k[:-2] == v['filename'].split('/')[-1]
        #     v['filename'] = v['filename'].split('/')[-1]
        #     assert k[:-2] == v['filename']
        #     img_metadata['{}/{}'.format(sub_dir, k[:-2])] = v
        # image_id_list = ['{}/{}'.format(sub_dir, id[:-2]) for id in anno['_via_image_id_list']]
        #
        # anno['_via_img_metadata'] = img_metadata
        # anno['_via_image_id_list'] = image_id_list
        # with open(os.path.join(root_path, '{}.json'.format(sub_dir)), mode='w') as w:
        #     json.dump(anno, w)

        img_metadata = anno['_via_img_metadata']
        image_id_list = anno['_via_image_id_list']
        attributes = anno['_via_attributes']
        female_class = attributes['region']['性别']['options']
        age_class = attributes['region']['年龄']['options']
        with open(os.path.join(root_path, '{}.err.txt'.format(sub_dir)), mode='w') as we:
            for img_name in os.listdir(os.path.join(root_path, sub_dir)):
                img_path_sub = os.path.join(sub_dir, img_name)
                metadata = img_metadata.get(img_path_sub, None)
                if metadata is None:
                    we.write('not found#{}\n'.format(img_path_sub))
                    continue

                if metadata['regions'] == 0:
                    we.write('region#{}#{}\n'.format(img_path_sub, json.dumps(metadata['regions'], ensure_ascii=False)))
                    continue

                try:
                    region = metadata['regions'][0]
                except Exception as err:
                    we.write('region#{}#{}\n'.format(img_path_sub, json.dumps(metadata['regions'], ensure_ascii=False)))
                    continue

                for region in metadata['regions']:
                    shape = region['shape_attributes']
                    female = region['region_attributes']

                    left_top = (shape['x'], shape['y'])
                    right_bottom = (left_top[0] + shape['width'], left_top[1] + shape['height'])
                    age_seg = female.get('年龄', '')
                    if age_seg == '':
                        we.write('age seg#{}#{}\n'.format(img_path_sub, json.dumps(female, ensure_ascii=False)))
                        print('age empty')
                        continue
                    age_range = [int(age) for age in age_class[age_seg].split('-')]
                    age = female.get('识别年龄', None)
                    if age == '' or age is None:
                        we.write('age#{}#{}\n'.format(img_path_sub, json.dumps(female, ensure_ascii=False)))
                        continue

                    gender = female.get('性别', None)
                    if gender is None:
                        we.write('gender#{}#{}\n'.format(img_path_sub, json.dumps(female, ensure_ascii=False)))
                        continue

                    if int(age) < age_range[0] or int(age) >= age_range[1]:
                        we.write('age#{}#{}\n'.format(img_path_sub, json.dumps(female, ensure_ascii=False)))
    return


def draw():
    root_path = '/mnt/sdb/data.set/face_and_age_gender'
    out_path = '/mnt/sdb/data.set/face_and_age_gender/test.img'
    os.makedirs(out_path, exist_ok=True)
    for sub_dir in ['part01', 'part02', 'part03']:
        if sub_dir == 'part01':
            continue
        with open(os.path.join(root_path, '{}.json'.format(sub_dir)), mode='r') as r:
            anno = json.load(r)

        with open(os.path.join(root_path, '{}.err.txt'.format(sub_dir)), mode='r') as r:
            empty = r.readlines()

        empty = [name.strip('\n').split('#')[1] for name in empty]

        img_metadata = anno['_via_img_metadata']
        image_id_list = anno['_via_image_id_list']
        attributes = anno['_via_attributes']
        female_class = attributes['region']['性别']['options']
        age_class = attributes['region']['年龄']['options']

        with open(os.path.join(root_path, '{}.img.err.txt'.format(sub_dir)), mode='w') as wh:
            img_names = os.listdir(os.path.join(root_path, sub_dir))
            random.shuffle(img_names)
            draw_names = random.choices(img_names, k=int(len(img_names) * 0.04))

            for img_name in tqdm.tqdm(img_names):
                img_path_sub = os.path.join(sub_dir, img_name)

                if img_path_sub in empty:
                    continue

                metadata = img_metadata[img_path_sub]
                region = metadata['regions'][0]

                img = cv2.imread(os.path.join(root_path, img_path_sub))
                h, w, c = img.shape
                if h < region['shape_attributes']['y'] + region['shape_attributes']['width'] or w < region['shape_attributes']['x'] + region['shape_attributes']['height']:
                    wh.write('shape#{}#{}\n'.format(img_path_sub, json.dumps(metadata['regions'], ensure_ascii=False)))
                    continue

                if img_name not in draw_names:
                    continue

                print('start draw')
                for region in metadata['regions']:
                    shape = region['shape_attributes']
                    female = region['region_attributes']

                    left_top = (shape['x'], shape['y'])
                    right_bottom = (left_top[0] + shape['width'], left_top[1] + shape['height'])
                    cv2.rectangle(img, left_top, right_bottom, color=[0, 255, 0], thickness=2)
                    age_seg = female.get('年龄', '')
                    # if age_seg == '':
                    #     print('age empty')
                    #     continue
                    age_range = [int(age) for age in age_class[age_seg].split('-')]
                    age = female['识别年龄']
                    # gender = female.get('性别', None)

                    # if age == '' or age is None:
                    #     continue
                    if int(age) < age_range[0] or int(age) >= age_range[1]:
                        raise RuntimeError(json.dumps(female))
                    label_text = '{}/{}'.format(female['性别'], '{}-{}'.format(age, age_seg))
                    cv2.putText(img, label_text, (left_top[0], left_top[1] - 2), cv2.FONT_HERSHEY_COMPLEX, 4, [0, 0, 255], thickness=2) # FONT_HERSHEY_SIMPLEX

                cv2.imwrite(os.path.join(out_path, '{}-{}'.format(sub_dir, img_name)), img)

    return


def empty():
    root_path = '/mnt/sdb/data.set/face_and_age_gender'
    out_path = '/mnt/sdb/data.set/face_and_age_gender/test.img'
    os.makedirs(out_path, exist_ok=True)
    for sub_dir in ['part01', 'part02', 'part03']:
        with open(os.path.join(root_path, '{}.empty.txt'.format(sub_dir)), mode='r') as r:
            lines = r.readlines()

        for line in lines:
            arr = line.split('#')
            tmp = arr[1].split('/')
            shutil.copy(os.path.join(root_path, arr[1]), os.path.join(out_path, '{}-{}-{}'.format(tmp[0], arr[0], tmp[1])))
    return


def modify():
    root_path = '/mnt/sdb/data.set/face_and_age_gender'
    out_path = '/mnt/sdb/data.set/face_and_age_gender/test.img'
    os.makedirs(out_path, exist_ok=True)
    for sub_dir in ['part01', 'part02', 'part03']:
        # if sub_dir != 'part03':
        #     print(sub_dir)
        #     continue
        with open(os.path.join(root_path, 'old.{}.json'.format(sub_dir)), mode='r') as r:
            anno = json.load(r)

        img_metadata = anno['_via_img_metadata']
        image_id_list = anno['_via_image_id_list']
        attributes = anno['_via_attributes']
        female_class = attributes['region']['性别']['options']
        age_class = attributes['region']['年龄']['options']

        for img_name in os.listdir(os.path.join(root_path, sub_dir)):
            img_path_sub = os.path.join(sub_dir, img_name)
            metadata = anno['_via_img_metadata'].get(img_path_sub, None)
            if metadata is None:
                continue

            if metadata['regions'] == 0:
                continue

            try:
                region = metadata['regions'][0]
            except Exception as err:
                continue

            for region in anno['_via_img_metadata'][img_path_sub]['regions']:
                shape = region['shape_attributes']
                female = region['region_attributes']

                age_seg = female.get('年龄', '')
                if age_seg == '':
                    age_range = [-1, -1]
                else:
                    age_range = [int(age) for age in age_class[age_seg].split('-')]
                age = female.get('识别年龄', None)
                if age == '' or age is None:
                    continue
                age = int(age)
                if int(age) < age_range[0] or int(age) >= age_range[1]:
                    if age < 3:
                        age_seg = '0'
                    elif age < 6:
                        age_seg = '1'
                    elif age < 12:
                        age_seg = '2'
                    elif age < 18:
                        age_seg = '3'
                    elif age < 30:
                        age_seg = '4'
                    elif age < 50:
                        age_seg = '5'
                    else:
                        age_seg = '6'
                    female['年龄'] = age_seg
                print('')
                # label_text = '{}/{}'.format(female_class[female['性别']], age)
                # cv2.putText(img, label_text, (left_top[0], left_top[1] - 2), cv2.FONT_HERSHEY_COMPLEX, 4, [0, 0, 255], thickness=2) # FONT_HERSHEY_SIMPLEX

        with open(os.path.join(root_path, '{}.json'.format(sub_dir)), mode='w') as w:
            json.dump(anno, w)
    return


if __name__ == '__main__':
    draw()
