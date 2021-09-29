import os
import json
import shutil
from PIL import Image
import time
import random
from tqdm import tqdm
import datetime
import pandas as pd

width = 1280
height = 720
#drop_old = False

def prep_dirs(source_dir, ds_name='train_data'):

    #create train_data path in source directory

    if os.path.exists(os.path.join(source_dir, ds_name)):
        shutil.rmtree(os.path.join(source_dir, ds_name))
        time.sleep(1)

    #create images paths in source directory
    os.mkdir(os.path.join(source_dir, ds_name))

    img_path = os.path.join(source_dir, os.path.join(ds_name, 'images'))
    os.makedirs(img_path, exist_ok=True)

    train_img_path = os.path.join(img_path, 'train')
    os.makedirs(train_img_path, exist_ok=True)
    val_img_path = os.path.join(img_path, 'val')
    os.makedirs(val_img_path, exist_ok=True)

    #create labels paths 
    lab_path = os.path.join(source_dir, os.path.join(ds_name, 'labels'))
    os.makedirs(lab_path, exist_ok=True)

    train_lab_path = os.path.join(lab_path, 'train')
    os.makedirs(train_lab_path, exist_ok=True)
    val_lab_path = os.path.join(lab_path, 'val')
    os.makedirs(val_lab_path, exist_ok=True)

def fromYolaToVia(path, imgs, yoloAnnotations, labels, project_name='fromYolaToVia'):
    template = json.load(open('template.json', 'r'))
    
    path = os.path.normpath(path) + os.sep

    def metadataGenerator(path, imgs, yoloAnnotations):
        objects = dict()
        for i in range(len(imgs)):
            file = imgs[i]
            marks = yoloAnnotations[i]
            filePath = os.path.join(path, file)
            size = os.path.getsize(filePath)
            regions = []
            for reg in marks.values:
                if reg[-1] in labels.keys():
                    regions.append({
                        'shape_attributes': {
                            'name': 'rect',
                            'x': reg[0],
                            'y': reg[1],
                            'width': reg[2] - reg[0],
                            'height': reg[3] - reg[1]
                        },
                        "region_attributes": {"class": reg[-1]}
                    })

            objects[file + str(size)] = {
                'filename': file,
                'size': size,
                'regions': regions,
                'file_attributes': {}
            }
        return objects

    template['_via_settings']['project']['name'] = project_name
    template['_via_settings']['core']['filepath'] = {}
    template['_via_settings']['core']['default_filepath'] = path

    template['_via_attributes']['region']['class']['options'] = labels
    template['_via_attributes']['region']['class']['default_options'] = {}

    template['_via_img_metadata'] = metadataGenerator(path, imgs, yoloAnnotations)

    with open(os.path.normpath(path) + '.json', 'w') as f:
        # print('save to', os.path.normpath(path) + '.json')
        f.writelines(json.dumps(template, indent=5))

    return template

def viaLabels(project_file):
    if project_file.endswith('json'):
        try:
            with open(project_file, 'r') as file:
                info = json.load(file)
                data = info['_via_attributes']['region']['class']
                try:                   
                    source_path_name = info['_via_settings']['core']['default_filepath'].split('\\')[-2]
                except:
                    source_path_name = os.path.basename(project_file)[:-5]
                type = data['type']
                classes = [x for x in data['options']]
                return (source_path_name, {classes[x]: str(x) for x in range(len(classes))})
        except:
            print('error parsing project file', project_file)
            return None

def via2yola(annotation_file, path2files, labels):
    annotation_file = os.path.normpath(annotation_file)
    path2files = os.path.normpath(path2files)

    if annotation_file.endswith('json'):
        try:
            with open(annotation_file, 'r') as file:
                data = json.load(file)['_via_img_metadata'].items()
                data = {key: value for key, value in data}
        except:
            print('error parsing', annotation_file)
            return None
        out = dict()
        for key, value in tqdm(data.items(), desc='convert via annotate for ' + os.path.basename(path2files)):
            m = set()
            if key == '_via_settings':
                break
            try:
                width, height = Image.open(os.path.join(path2files, value['filename'])).size
            except:
                print('no such file in dir', os.path.join(path2files, value['filename']))
                continue
            for v in value['regions']:
                try:
                    if v['shape_attributes']['name'] == 'polygon':
                        v['shape_attributes']['x'] = min(v['shape_attributes']['all_points_x'])
                        v['shape_attributes']['y'] = min(v['shape_attributes']['all_points_y'])
                        v['shape_attributes']['width'] = max(v['shape_attributes']['all_points_x'])
                        v['shape_attributes']['width'] -= v['shape_attributes']['x']
                        v['shape_attributes']['height'] = max(v['shape_attributes']['all_points_y'])
                        v['shape_attributes']['height'] -= v['shape_attributes']['y']
                    string = labels[v['region_attributes']['class']] + ' '
                    string += str((v['shape_attributes']['x'] + 0.5 * v['shape_attributes']['width']) / width) + ' '
                    string += str((v['shape_attributes']['y'] + 0.5 * v['shape_attributes']['height']) / height) + ' '
                    string += str(v['shape_attributes']['width'] / width) + ' '
                    string += str(v['shape_attributes']['height'] / height)
                    m.add(string)
                except:
                    cl = ""
                    if 'class' in v['region_attributes'].keys():
                        cl = v['region_attributes']['class']
                    # print(value['filename'], 'skip class:', cl)
            out.update({value['filename'].replace('jpg', 'txt'): m})
        return out
    else:
        print('not json', annotation_file)
        return None


def createYolaTxt(annotation_file, path2files, labels, drop_old=False):
    if drop_old:
        print('cleaning', path2files)
        for root, dirs, files in os.walk(path2files):
            for f in files:
                if f.endswith('txt'):
                    os.remove(os.path.join(root, f))

    annotats = via2yola(annotation_file, path2files, labels)
    for annotate in tqdm(annotats['annotate'], desc='creating txt for yola'):
        with open(os.path.join(annotats['path2files'], annotate), 'w') as file:
            file.writelines('\n'.join(annotats['annotate'][annotate]))

def get_data(path, except_dir=[]):
    ff = set()
    for root, dirs, files in os.walk(path):
        if os.path.basename(root) not in except_dir:
            for f in files:
                if f.endswith('txt'):
                    if os.path.exists(os.path.join(root, f[:-4] + '.jpg')):
                        ff.add(os.path.join(root, f[:-4]))
    return ff

def create_datasets(path, val_size_prc=10):
    train_data = 'train_data'
    if os.path.exists(os.path.join(path, train_data)):
        shutil.rmtree(os.path.join(path, train_data))
        time.sleep(1)
    os.mkdir(os.path.join(path, train_data))
    os.mkdir(os.path.join(path, os.path.join(train_data, 'images')))
    os.mkdir(os.path.join(path, os.path.join(train_data, 'images\\train')))
    os.mkdir(os.path.join(path, os.path.join(train_data, 'images\\val')))
    os.mkdir(os.path.join(path, os.path.join(train_data, 'labels')))
    os.mkdir(os.path.join(path, os.path.join(train_data, 'labels\\train')))
    os.mkdir(os.path.join(path, os.path.join(train_data, 'labels\\val')))
    data = get_data(path)
    data_by_class = dict()
    for f in data:
        with open(f + '.txt', 'r') as cl:
            cl = cl.readlines()
            total_cl = set()
            for kl in cl:
                total_cl.add(kl.split()[0])
            for kl in total_cl:
                d = data_by_class.get(kl, set())
                d.update({f})
                data_by_class[kl] = d
    val = set()
    for kl in data_by_class.keys():
        all_class_files = list(data_by_class[kl])
        l = int(len(all_class_files) / 100 * val_size + 1)
        random.shuffle(all_class_files)
        val.update(set(all_class_files[:l]))

    train = list(data.difference(val))
    val = list(val)

    for i in range(len(train)):
        p = 'train'
        bn = os.path.basename(train[i])
        shutil.copy(train[i] + '.txt', os.path.join(path, 'train_val\labels\\' + p + '\\' + bn + '.txt'))
        shutil.copy(train[i] + '.jpg', os.path.join(path, 'train_val\images\\' + p + '\\' + bn + '.jpg'))
        print(p, train[i])

    for i in range(len(val)):
        p = 'val'
        bn = os.path.basename(val[i])
        shutil.copy(val[i] + '.txt', os.path.join(path, 'train_val\labels\\' + p + '\\' + bn + '.txt'))
        shutil.copy(val[i] + '.jpg', os.path.join(path, 'train_val\images\\' + p + '\\' + bn + '.jpg'))
        print(p, val[i])