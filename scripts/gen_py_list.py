import os

# img_dir = '/export/home/dxx/FGVC/Compcars_CUHK/data/image/'
anno_dir = '/export/home/dxx/FGVC/Compcars_CUHK/data/train_test_split/classification/'
levels = {'make', 'model'} # 厂家 车型
phase = {'train', 'val', 'test'}

def gen_list(phase='train', level='model'):
    labelset = set()
    in_file = '{}/{}.txt'.format(anno_dir, phase)
    out_file = '{}/{}/{}list.txt'.format(anno_dir, level, phase)
    with open(in_file) as f, open(out_file, 'w') as f1:
        for line in f.readlines():
            target = line.split('/')
            if level == 'make':
                target = target[0]
            else:
                target = target[0] + '_' + target[1]
            f1.write('{} {}\n'.format(line.strip(), target))
            labelset.add(target)
    print('There are {} classes in {}set.'.format(len(labelset), phase))

if __name__ == '__main__':
    level = 'model'
    try:
        os.mkdir(anno_dir + level)
    except:
        pass
    gen_list('train', level)
    gen_list('test', level)
