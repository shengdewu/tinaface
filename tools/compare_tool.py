import cv2
import numpy as np
import os
import tqdm


class CompareBase:
    def __init__(self):
        return

    def compare(self, base_path, compare_paths, out_path, skip=False, special_name=None):
        if special_name is not None:
            assert (isinstance(special_name, list) or isinstance(special_name, tuple)) and len(special_name) > 0

        os.makedirs(out_path, exist_ok=True)
        skip_name = list()
        if skip:
            skip_name = os.listdir(out_path)

        base_names = [name for name in os.listdir(base_path) if name.find('lut') == -1]
        for name in tqdm.tqdm(base_names):
            if special_name is not None and name not in special_name:
                continue

            if name in skip_name:
                continue

            compare_names = [os.path.join(compare_path, name) for compare_path in compare_paths]

            is_exist = [not os.path.exists(compare_name) for compare_name in compare_names]
            if np.any(is_exist):
                continue
            base_img = cv2.imread(os.path.join(base_path, name), cv2.IMREAD_UNCHANGED)
            compare_imgs = [cv2.imread(compare_name, cv2.IMREAD_UNCHANGED) for compare_name in compare_names]

            shapes = [compare_img.shape for compare_img in compare_imgs]

            concat = self.execute(shapes, base_img, compare_imgs)

            cv2.imwrite(os.path.join(out_path, name), concat)
        return

    def execute(self, shapes, base_img, compare_imgs):
        '''
        基础输入图片时 input|gt, 其他是 input|gt
        :param shapes:
        :param base_img:
        :param compare_imgs:
        :return: input|gt
                 input|gt
                 ...
                 input|gt
        '''
        h, w, c = base_img.shape
        ch = [shape[0] for shape in shapes]
        cw = [shape[1] for shape in shapes]
        ch = sum(ch)
        cw = max(cw)
        concat = np.zeros(shape=(ch + h, max(w, cw), c), dtype=base_img.dtype)
        concat[:h, :w, :] = base_img
        start_h = h
        for i in range(len(compare_imgs)):
            h, w, c = shapes[i]
            assert shapes[i] == compare_imgs[i].shape
            concat[start_h:h + start_h, :w, :] = compare_imgs[i]
            start_h += h
        return concat


class CompareRow(CompareBase):
    def __init__(self):
        super(CompareRow, self).__init__()

    def execute(self, shapes, base_img, compare_imgs):
        '''
        基础输入图片时 input|gt, 其他是 input|gt
        :param shapes:
        :param base_img:
        :param compare_imgs:
        :return: input|gt|gt|...|gt
        '''
        h, w, c = base_img.shape
        ch = [shape[0] for shape in shapes]
        cw = [shape[1]//2 for shape in shapes]
        ch = max(ch)
        cw = sum(cw)
        concat = np.zeros(shape=(max(ch, h), w + cw, c), dtype=base_img.dtype)
        concat[:h, :w, :] = base_img
        start_w = w
        for i in range(len(compare_imgs)):
            h, w, c = shapes[i]
            assert shapes[i] == compare_imgs[i].shape
            concat[:h, start_w:start_w+w//2, :] = compare_imgs[i][:, w//2:, :]
            start_w += w//2
        return concat


class CompareCol(CompareBase):
    def execute(self, shapes, base_img, compare_imgs):
        '''
        基础输入图片时 input|gt|gt, 其他是 input|gt
        :param shapes:
        :param base_img:
        :param compare_imgs:
        :return: input|gt|gt|
                  gt|gt|gt
        '''
        h, w, c = base_img.shape
        ch = [shape[0] for shape in shapes]
        cw = [shape[1]//2 for shape in shapes]
        assert len(cw) == 3
        concat = np.zeros(shape=(max(ch)+h, max(w, sum(cw)), c), dtype=base_img.dtype)
        concat[:h, :w, :] = base_img
        start_h = h
        start_w = 0
        for i in range(len(compare_imgs)):
            h = ch[i]
            w = cw[i]
            assert shapes[i] == compare_imgs[i].shape
            concat[start_h:h+start_h, start_w:w+start_w, :] = compare_imgs[i][:, w:, :]
            start_w += w
        return concat


class CompareOne(CompareBase):
    def execute(self, shapes, base_img, compare_imgs):
        '''
        基础输入图片时 input, 其他是 input
        :param shapes:
        :param base_img:
        :param compare_imgs:
        :return: input|input
        '''
        h, w, c = base_img.shape
        ch = [shape[0] for shape in shapes]
        cw = [shape[1] for shape in shapes]

        concat = np.zeros(shape=(max(h, max(ch)), (w + sum(cw)), c), dtype=base_img.dtype)
        concat[:h, :w, :] = base_img
        start_w = w
        for i in range(len(compare_imgs)):
            h = ch[i]
            w = cw[i]
            assert shapes[i] == compare_imgs[i].shape
            concat[:h, start_w:w+start_w, :] = compare_imgs[i]
            start_w += w
        return concat


if __name__ == '__main__':
    base_path = '/mnt/sdb/error.collection/identify.evaluate/face'
    compare_path = ['/mnt/sda1/face.test/tinaface.base', '/mnt/sda1/face.test/tinaface']
    out_path = '/mnt/sda1/face.test/tinaface.compare'
    compare_cls = CompareOne()
    compare_cls.compare(base_path=base_path, compare_paths=compare_path, out_path=out_path)
