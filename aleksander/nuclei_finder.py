from optparse import OptionParser
import numpy as np
import cv2
import pylab as py
import os
import re
from tqdm import tqdm
import imageio
from sklearn import model_selection, svm
import multiprocessing
import pickle
from detection_frame import DetectionFrame
import time #debug

def smoothen(img):
###non-linear filter preventing averaging across image edges
    return cv2.bilateralFilter(img, 9, 75, 75) #TODO: hardcode

def norm_image(img):
    maximum = np.max(img)
    minimum = np.min(img)
    max_min_diff = maximum - minimum

    if max_min_diff == 0:
        return np.ones(img.size)

    return (img - minimum) / max_min_diff


def grads(image):
    #TODO: I don't have border with zeros here as it was in Aleksander's example. Do I need it?
    h, w = image.shape
    result = np.zeros((h, w, 2))
    x = np.copy(image)
    y = np.copy(image)
    x[:,1:-1] = image[:,2:]-image[:,:-2]
    y[1:-1,:] = image[2:,:]-image[:-2,:]
    l = np.sqrt((x**2 + y**2))
    
    a = np.zeros((h,w))
    a[np.logical_and(x==0, y == 0)] = np.pi/2
    a[np.logical_and(x==0, y > 0)] = np.pi
    a[x!=0] = np.rad2deg(np.arctan(y[x!=0]/x[x!=0]) + np.pi/2)

    return np.dstack((l,a))

def cell_hist(gradients):
    h, w, d  = gradients.shape
    window_size = 8
    res_h = int(h / window_size)
    res_w = int(w / window_size)
    bins_number = 9
    bin_width = (180 / bins_number)
    bins = [bin_width * x for x in range(bins_number)]
    result = np.zeros((res_h, res_w, bins_number))
    
    for a in range(res_h):
        for b in range(res_w):
            fragment = gradients[a*window_size:(a+1)*window_size,b*window_size:(b+1)*window_size]
            hist = np.zeros(bins_number)
        
            for row in fragment:
                for pixel in row:
                    nbin = int((pixel[1] // bin_width) % 9)
                    hist[nbin] += ((pixel[1] - bins[nbin]) / bin_width) * pixel[0]
                    if nbin != bins_number - 1:
                        hist[nbin+1] += ((bins[nbin+1] - pixel[1]) / bin_width) * pixel[0]
                    else:
                        hist[0] += ((180 - pixel[1]) / bin_width) * pixel[0]
            result[a,b] = hist
    return result

def norm_block(block):
    norm = np.linalg.norm(block)
    if norm == 0:
        norm = 1
    return block / norm
    
def features(hist):
    window_size = 8
    h, w, d = hist.shape
    result = np.zeros((h-1, w-1, d*4))
    for a in range(h-1):
        for b in range(w-1):
            result[a,b] = norm_block(np.ndarray.flatten(hist[a:a+2, b:b+2]))
    return result.flatten()

def hog(image):
    img = norm_image(image)
    img = smoothen(img.astype(np.float32))
    img = norm_image(img)
    gradients = grads(img)
    hists = cell_hist(gradients)
    feats = features(hists)
    return feats

def prep_image(img_orig, size=(112,112), rotate=False, gamma=1):
    img = np.copy(img_orig)
    img = norm_image(img) ** (1/gamma)
    img = cv2.resize(img, size)
    if rotate:
        angle = np.random.choice([0, 90, 180, 270])
        M = cv2.getRotationMatrix2D((size[1]/2, size[0]/2), angle, 1)
        img = cv2.warpAffine(img, M, size)
    return img

def join_boxes(img, boxes_orig, thresh):
    i = 0
    boxes = boxes_orig.copy()
    while i < len(boxes):
        j = 0
        
        while j < len(boxes): 
            if i == j:
                j += 1
                continue
                    
            odl = boxes[i].centers_distance(boxes[j])
            if odl < thresh:
                boxes[i] = boxes[i].merge_frames(boxes[j], img)
                
                boxes.pop(j)
                if j < i:
                    i-= 1
            else:
                j += 1
        i += 1
    return boxes

def find_and_draw(image, classifier, filename, output_directory, frame_count, draw):
    minimum, maximum = get_min_max_nuclei_size(image)
    radius = np.int((maximum + minimum) / 6)

    where = search(image, classifier, minimum, maximum)
    joined = join_boxes(image, where, radius)
    
    output_file = open(os.path.join(output_directory, 'output.csv'), 'a+')
    for frame in joined:
        output_file.write('\"{}\",{},{},{},{},{}\n'.format(filename, frame_count, frame.x1, frame.y1, frame.x2, frame.y2))
    output_file.close()

    if draw:
        img_copy = norm_image(np.copy(image)) * 255
        for frame in joined:
            cv2.rectangle(img_copy, (frame.y1, frame.x1), (frame.y2, frame.x2), 255, 2)
        imageio.imwrite(output_directory + '/{}_{}.jpg'.format(filename, frame_count), img_copy)

def get_min_max_nuclei_size(image):
    # TODO: zastąpić wykorzystaniem wstępnego badania wielkości
    h, w = image.shape
    minimum = np.min([int(w/4), int(h/4)])
    maximum = np.min([int(w/2), int(h/2)])
    return minimum, maximum

def search_rows(img, clf, i, rows, results, s, scale, h, w): # te nazwy cols i rows to chyba odwrotnie, ale mniejsza z tym
    for j in rows:
        h_to = np.int(i + s)
        w_to = np.int(j + s)

        if i != h_to and j != w_to:
            fragment_hog = hog(prep_image(img[i:h_to, j:w_to], gamma=2, size=(s,s)))
            if clf.predict([fragment_hog])[0] == 1:
                i_to_save = int(i / scale)
                j_to_save = int(j / scale)
                h_to_save = int(h_to / scale)
                w_to_save = int(w_to / scale)
                if h_to_save > h - 1:
                    h_to_save = h - 1
                if w_to_save > w - 1:
                    w_to_save = w - 1
                results.put(DetectionFrame((i_to_save , j_to_save),(h_to_save, w_to_save)))

def search(image, clf, min_cell, max_cell, s=32):
    results = multiprocessing.Queue()
    
    cell_sizes = np.linspace(min_cell, max_cell, num=5, endpoint=True)
    
    for cell_size in tqdm(cell_sizes, 'steps'):
        scale = s / cell_size
        
        img = cv2.resize(image, None, fx=scale, fy=scale)

        h, w = image.shape
        h_curr, w_curr = img.shape
        
        w_t = int((w_curr - s) / (s / 4)) + 1
        h_t = int((h_curr - s) / (s / 4)) + 1
        
        columns = np.linspace(0, h_curr - s, num = h_t, endpoint = True, dtype=int)
        rows = np.linspace(0, w_curr - s, num = w_t, endpoint = True, dtype=int)
        
        jobs = []
        
        for column in columns:
            p = multiprocessing.Process(target = search_rows, args = (img, clf, column, rows, results, s, scale, h, w,))
            jobs.append(p)
            p.start()
    
    for proc in jobs:
        proc.join()
        
    final_results = []
    
    while not results.empty():
        res = results.get()
        final_results.append(res)
                    
    return final_results

def main():
    parser = OptionParser()

    parser.add_option("-o", "--output-directory", dest="output_directory",
                    help="save output to ./DIRECTORY/\n./output/ by default", metavar="DIRECTORY/")

    parser.add_option("-i", "--input-directory", dest="input_directory",
                    help="load input from ./DIRECTORY/\n./input/ by default", metavar="DIRECTORY/")

    parser.add_option("-d", "--draw-frames", dest="draw", action="store_true",
                    help="save images with drawn frames")

    yes = ['yes', 'y']
    no = ['no', 'n']
    possible_answer = yes + no

    (options, args) = parser.parse_args()

    cwd = os.getcwd()

    input_directory = os.path.join(cwd, 'input/')
    if options.input_directory:
        input_directory = os.path.join(cwd, options.input_directory)
    if not os.path.isdir(input_directory):
        print('Input directory does not exist.')
        exit()

    output_directory = os.path.join(cwd, 'output/')
    if options.output_directory:
        output_directory = os.path.join(cwd, options.output_directory)
    if not os.path.isdir(output_directory):
        answer = ''
        while not answer.lower() in possible_answer:
            answer = input('Output directory does not exist. Do you want to create new directory {} ? [y/n]'.format(output_directory))
        if answer in no:
            exit()
        else:
            os.mkdir(output_directory)

    output_file = open(os.path.join(output_directory, 'output.csv'), 'w+')
    output_file.write('')
    output_file.close()

    model_file_name = 'model.sav'
    if not os.path.isfile(os.path.join(cwd, model_file_name)):
        print('model.sav not found in current working directory.')
        exit()

    clf = pickle.load(open(model_file_name, 'rb'))

    single_img_extensions = ['.jpg', '.jpeg']

    for dir_path, dir_names, file_names in os.walk(input_directory):
        for f in tqdm(file_names, desc='files'):
            extension = os.path.splitext(f)[1].lower()
            
            if extension in single_img_extensions:
                img = imageio.imread(os.path.join(input_directory, f))
                if len(img.shape) > 2:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

                find_and_draw(img, clf, f, output_directory, 1, options.draw)
            
            elif extension == '.tif':
                images = imageio.mimread(os.path.join(input_directory, f))
                i = 0
                for image in tqdm(images, desc='frames'):
                    find_and_draw(image, clf, f, output_directory, i+1, options.draw)
                    i += 1

if __name__ == '__main__':
    np.set_printoptions(linewidth=np.inf)
    # main()
    img = py.imread('/home/maciek/github/FociAnalyzer/16.png')
    # img = np.random.randint(5, size=(10,10))
    # grads(img)
