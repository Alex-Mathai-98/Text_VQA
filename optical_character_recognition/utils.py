import numpy as np, math, os, cv2, matplotlib.pyplot as plt, collections
import torch, torch.nn as nn, torchvision.transforms as transforms
from skimage import io
from torch.autograd import Variable
from PIL import Image
from collections import OrderedDict
import imutils

def warpCoord(Minv, pt):
    out = np.matmul(Minv, (pt[0], pt[1], 1))
    return np.array([out[0]/out[2], out[1]/out[2]])

def crop_polygon(image, points):
    mask = np.zeros(image.shape, dtype=np.uint8)
    channel_count = image.shape[2]
    ignore_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, [points], ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text):
    linkmap = linkmap.copy()
    textmap = textmap.copy()
    img_h, img_w = textmap.shape

    ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
    ret, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)
    text_score_comb = np.clip(text_score + link_score, 0, 1)
    
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(text_score_comb.astype(np.uint8), connectivity=4)
    det, mapper = [], []
    
    for k in range(1,nLabels):
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10: continue
        if np.max(textmap[labels==k]) < text_threshold: continue

        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels==k] = 255
        segmap[np.logical_and(link_score==1, text_score==0)] = 0   # remove link area
        
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        
        # boundary check
        if sx < 0 : sx = 0
        if sy < 0 : sy = 0
        if ex >= img_w: ex = img_w
        if ey >= img_h: ey = img_h
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)
        
        # make box
        np_contours = np.roll(np.array(np.where(segmap!=0)),1,axis=0).transpose().reshape(-1,2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)
        
        # align diamond-shape
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:,0]), max(np_contours[:,0])
            t, b = min(np_contours[:,1]), max(np_contours[:,1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)
        
        # make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4-startidx, 0)
        box = np.array(box)
        det.append(box), mapper.append(k)

    return det, labels, mapper

def getPoly_core(boxes, labels, mapper, linkmap):
    num_cp, max_len_ratio, expand_ratio, max_r, step_r, polys = 5, 0.7, 1.45, 2.0, 0.2, []
    
    for k, box in enumerate(boxes):
        w, h = int(np.linalg.norm(box[0] - box[1]) + 1), int(np.linalg.norm(box[1] - box[2]) + 1)
        if w < 10 or h < 10:
            polys.append(None); continue

        # warp image
        tar = np.float32([[0,0],[w,0],[w,h],[0,h]])
        M = cv2.getPerspectiveTransform(box, tar)
        word_label = cv2.warpPerspective(labels, M, (w, h), flags=cv2.INTER_NEAREST)
        
        try: Minv = np.linalg.inv(M)
        except: polys.append(None); continue

        # binarization for selected label
        cur_label = mapper[k]
        word_label[word_label != cur_label] = 0
        word_label[word_label > 0] = 1
        
        # find top/bottom contours
        max_len, cp = -1, []
        for i in range(w):
            region = np.where(word_label[:,i] != 0)[0]
            if len(region) < 2 : continue
            cp.append((i, region[0], region[-1]))
            length = region[-1] - region[0] + 1
            if length > max_len: max_len = length
        
        # pass if max_len is similar to h
        if h * max_len_ratio < max_len:
            polys.append(None); continue
        
        # get pivot points with fixed length
        tot_seg = num_cp * 2 + 1
        seg_w = w / tot_seg     
        pp = [None] * num_cp    
        cp_section = [[0, 0]] * tot_seg
        seg_height = [0] * num_cp
        seg_num = 0
        num_sec = 0
        prev_h = -1
        
        for i in range(0,len(cp)):
            (x, sy, ey) = cp[i]
            
            if (seg_num + 1) * seg_w <= x and seg_num <= tot_seg:
                
                # average previous segment
                if num_sec == 0: break
                cp_section[seg_num] = [cp_section[seg_num][0] / num_sec, cp_section[seg_num][1] / num_sec]
                num_sec = 0

                # reset variables
                seg_num += 1
                prev_h = -1

            # accumulate center points
            cy = (sy + ey) * 0.5
            cur_h = ey - sy + 1
            cp_section[seg_num] = [cp_section[seg_num][0] + x, cp_section[seg_num][1] + cy]
            num_sec += 1
            
            if seg_num % 2 == 0: continue 
            if prev_h < cur_h:
                pp[int((seg_num - 1)/2)] = (x, cy)
                seg_height[int((seg_num - 1)/2)] = cur_h
                prev_h = cur_h

        # processing last segment
        if num_sec != 0:
            cp_section[-1] = [cp_section[-1][0] / num_sec, cp_section[-1][1] / num_sec]

        # pass if num of pivots is not sufficient or segment widh is smaller than character height 
        if None in pp or seg_w < np.max(seg_height) * 0.25:
            polys.append(None); continue

        # calc median maximum of pivot points
        half_char_h = np.median(seg_height) * expand_ratio / 2

        # calc gradiant and apply to make horizontal pivots
        new_pp = []
        for i, (x, cy) in enumerate(pp):
            dx = cp_section[i * 2 + 2][0] - cp_section[i * 2][0]
            dy = cp_section[i * 2 + 2][1] - cp_section[i * 2][1]
            
            if dx == 0: 
                new_pp.append([x, cy - half_char_h, x, cy + half_char_h])
                continue
            
            rad = - math.atan2(dy, dx)
            c, s = half_char_h * math.cos(rad), half_char_h * math.sin(rad)
            new_pp.append([x - s, cy - c, x + s, cy + c])

        # get edge points to cover character heatmaps
        isSppFound, isEppFound = False, False
        grad_s = (pp[1][1] - pp[0][1]) / (pp[1][0] - pp[0][0]) + (pp[2][1] - pp[1][1]) / (pp[2][0] - pp[1][0])
        grad_e = (pp[-2][1] - pp[-1][1]) / (pp[-2][0] - pp[-1][0]) + (pp[-3][1] - pp[-2][1]) / (pp[-3][0] - pp[-2][0])
        
        for r in np.arange(0.5, max_r, step_r):
            dx = 2 * half_char_h * r
            
            if not isSppFound:
                line_img = np.zeros(word_label.shape, dtype=np.uint8)
                dy = grad_s * dx
                p = np.array(new_pp[0]) - np.array([dx, dy, dx, dy])
                cv2.line(line_img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), 1, thickness=1)
                
                if np.sum(np.logical_and(word_label, line_img)) == 0 or r + 2 * step_r >= max_r:
                    spp = p
                    isSppFound = True
            
            if not isEppFound:
                line_img = np.zeros(word_label.shape, dtype=np.uint8)
                dy = grad_e * dx
                p = np.array(new_pp[-1]) + np.array([dx, dy, dx, dy])
                cv2.line(line_img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), 1, thickness=1)
                
                if np.sum(np.logical_and(word_label, line_img)) == 0 or r + 2 * step_r >= max_r:
                    epp = p
                    isEppFound = True
            
            if isSppFound and isEppFound:
                break

        # pass if boundary of polygon is not found
        if not (isSppFound and isEppFound):
            polys.append(None); continue

        # make final polygon
        poly = []
        poly.append(warpCoord(Minv, (spp[0], spp[1])))
        
        for p in new_pp:
            poly.append(warpCoord(Minv, (p[0], p[1])))
        
        poly.append(warpCoord(Minv, (epp[0], epp[1])))
        poly.append(warpCoord(Minv, (epp[2], epp[3])))
        
        for p in reversed(new_pp):
            poly.append(warpCoord(Minv, (p[2], p[3])))
        
        poly.append(warpCoord(Minv, (spp[2], spp[3])))

        # add to final result
        polys.append(np.array(poly))

    return polys

def getDetBoxes(textmap, linkmap, text_threshold, link_threshold, low_text, poly=False):
    boxes, labels, mapper = getDetBoxes_core(textmap, linkmap, text_threshold, link_threshold, low_text)
    if poly: polys = getPoly_core(boxes, labels, mapper, linkmap)
    else: polys = [None] * len(boxes)
    return boxes, polys

def adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net = 2):
    
    if len(polys) > 0:
        polys = np.array(polys)
        
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
    
    return polys

def get_files(img_dir):
    imgs, masks, xmls = list_files(img_dir)
    return imgs, masks, xmls

def list_files(in_path):
    img_files, mask_files, gt_files = [], [], []
    
    for (dirpath, dirnames, filenames) in os.walk(in_path):
        
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
            
            elif ext == '.bmp':
                 mask_files.append(os.path.join(dirpath, file))
            
            elif ext == '.xml' or ext == '.gt' or ext == '.txt':
                gt_files.append(os.path.join(dirpath, file))
            
            elif ext == '.zip':
                continue
    
    return img_files, mask_files, gt_files

def get_character_level_boxes(image: np.ndarray, region_score_map: np.ndarray):
    """
    Given the region score map and the image itself, generate character level boxes 
    Make sure Region Score map is in range 0 -> 1 
    """
    _, thresh = cv2.threshold(np.uint8(region_score_map * 255), 128, 255, cv2.THRESH_BINARY) 
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN, kernel, iterations = 2)
    ret, markers = cv2.connectedComponents(opening)
    letter_boxes = []

    for i in np.unique(markers)[1:]:
        mask = np.zeros((image.shape[0], image.shape[1]), dtype = np.uint8)
        mask[markers == i] = 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations = 12)
        
        top, bottom = np.min(np.where(np.max(mask,axis=1)==255)), np.max(np.where(np.max(mask,axis=1)==255))
        left, right  = np.min(np.where(np.max(mask,axis=0)==255)), np.max(np.where(np.max(mask,axis=0)==255))
        
        letter_box = ((left, top), (right, bottom))
        letter_boxes.append(np.array([(top, left), (right, bottom)]))
        vis = cv2.rectangle(image, letter_box[1], letter_box[0], color = (0, 255, 0))

    return letter_boxes, image

def get_characters_top_down(image: np.ndarray, region_score_map: np.ndarray, text_recognizer):
    """
    Given the region score map and the word image, generate letter level boxes and return the 
    predictions made on these letter level boxes. Make sure Region Score map is in range 0 -> 1 
    """
    _, thresh = cv2.threshold(np.uint8(region_score_map * 255), 128, 255, cv2.THRESH_BINARY) 
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN, kernel, iterations = 6)
    ret, markers = cv2.connectedComponents(opening)
    letter_boxes, current_str = [], ""

    for i in np.unique(markers)[1:]:
        mask = np.zeros((image.shape[0], image.shape[1]), dtype = np.uint8)
        mask[markers == i] = 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations = 15)
        
        top, bottom = np.min(np.where(np.max(mask,axis=1)==255)), np.max(np.where(np.max(mask,axis=1)==255))
        left, right = 0, mask.shape[1]
        
        letter_box = ((left, top), (right, bottom))
        letter_crop = image[top: bottom, left: right, :]
        
        _, pred_letter = text_recognizer(letter_crop)
        pred_letter = str(pred_letter)[0]
        current_str += pred_letter
        
        letter_boxes.append(np.array([(top, left), (right, bottom)]))

    return letter_boxes, image, current_str

def get_straightened_boxes(image, region_map, boxes):
    """
    Take all the slant boxes and return a list of images that are straightened. Make sure box co-ordinates makes sense, 
    and is within the image all images must be cv2 read (They use BGR for some reason); just use utils.loadImage
    """
    words, region_scores = [], []
    
    for cnt in boxes:
        rect = cv2.minAreaRect(cnt)
        box = np.int0(cv2.boxPoints(rect))
        box = np.array([[list(b)] for b in box])

        width = int(rect[1][0])
        height = int(rect[1][1])
        src_pts = box.astype("float32")

        # coordinate of the points in box points after the rectangle has been straightened
        dst_pts = np.array([[0, height-1],
                            [0, 0],
                            [width-1, 0],
                            [width-1, height-1]], dtype="float32")

        # the perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # directly warp the rotated rectangle to get the straightened rectangle
        warped_region_cut = cv2.warpPerspective(region_map, M, (width, height)) 
        warped = cv2.warpPerspective(image, M, (width, height))
        
        words.append(warped)
        region_scores.append(warped_region_cut)
        
        warped_rotate = warped.transpose((1, 0, 2))[::-1]
        region_rot = warped_region_cut.transpose((1, 0))[::-1]
        
        words.append(warped_rotate)
        region_scores.append(region_rot)
    
    return words, region_scores

def get_boxes(image, region_map, boxes):
    """
    Get the word crops given the image and the word boxes; and do the same transformations on the word heatmap
    """
    words, region_scores = [], []
    top_lefts, bottom_rights = [], []
    
    for cnt in boxes:
        rect = cv2.minAreaRect(cnt)
        box = np.absolute(np.int0(cv2.boxPoints(rect)))
        
        top_left, bottom_right = tuple(box[1]), tuple(box[3])
        top_lefts.append(top_left)
        bottom_rights.append(bottom_right)
        
        word_image = image[min(top_left[1], bottom_right[1]): max(top_left[1], bottom_right[1]), min(top_left[0], bottom_right[0]): max(top_left[0], bottom_right[0]), :]
        word_map = region_map[min(top_left[1], bottom_right[1]): max(top_left[1], bottom_right[1]), min(top_left[0], bottom_right[0]): max(top_left[0], bottom_right[0])]

        region_scores.append(word_map)
        words.append(word_image)
    
    return words, region_scores, top_lefts, bottom_rights

def word_level_breakdown(largest_box_cuts, text_detector, boxes):
    """
    Break a phrase into its constituent words; sort them left to right or top to bottom based on height and width 
    Keep track of all the orientations for further use ; Keep track of all the locations for further use ; Keep 
    track of the heatmaps and apply the same transformations to the heatmaps as you would to the image 
    """
    refined_words, refined_word_heatmaps = [], []
    cleaned_largest_box_cuts, final_coordinates = [], []
    word_coords, final_boxes, ffb = [], [], []
    orientations, num = [], 0

    for box in boxes:
        final_boxes.append(box)
        final_boxes.append(box)
        
    for box_cut, fb in zip(largest_box_cuts, final_boxes):
        left_to_right = box_cut.shape[1] >= box_cut.shape[0]        
        words_coordinates, _, temp_region_score, _ = text_detector(box_cut, refine = False)
        word_cuts, word_heatmap, top_lefts, bottom_rights = get_boxes(box_cut, temp_region_score, words_coordinates)
        top_lefts_l2r = [x[0] for x in top_lefts]
        top_lefts_t2b = [x[1] for x in top_lefts]

        if left_to_right and len(word_cuts) > 1:
            word_cuts = [x for _, x in sorted(zip(top_lefts_l2r, word_cuts), key=lambda pair: pair[0])]
            word_heatmap = [x for _, x in sorted(zip(top_lefts_l2r, word_heatmap), key=lambda pair: pair[0])]
            #words_coordinates = [x for _, x in sorted(zip(top_lefts_l2r, words_coordinates), key=lambda pair: pair[0])]
        
        if not left_to_right and len(word_cuts) > 1:
            word_cuts = [x for _, x in sorted(zip(top_lefts_t2b, word_cuts), key=lambda pair: pair[0])]
            word_heatmap = [x for _, x in sorted(zip(top_lefts_t2b, word_heatmap), key=lambda pair: pair[0])]
            #words_coordinates = [x for _, x in sorted(zip(top_lefts_l2r, words_coordinates), key=lambda pair: pair[0])]

        if len(word_cuts) > 0:
            refined_words.append(word_cuts)
            refined_word_heatmaps.append(word_heatmap)
            cleaned_largest_box_cuts.append(box_cut)
            final_coordinates.append(words_coordinates)
            orientations.append(left_to_right)
            ffb.append(fb)

    return cleaned_largest_box_cuts, refined_words, refined_word_heatmaps, orientations, final_coordinates, ffb
    
def display(img_file, img, boxes, dirname='./inference_examples/', show = True):
        img = np.array(img)
        filename, _ = os.path.splitext(os.path.basename(img_file))
        result = os.path.join(dirname, filename +'.jpg')
        if not os.path.isdir(dirname): os.mkdir(dirname)

        for i, box in enumerate(boxes):
            poly = np.array(box).astype(np.int32).reshape((-1))
            poly = poly.reshape(-1, 2)
            cv2.polylines(img, [poly.reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=3)

        if show == True:
            plt.figure(figsize = (20, 30))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.xticks([]), plt.yticks([]), plt.show()
        else: cv2.imwrite(result, img)
        
        return img

def loadImage(img_file):
    """
    For the sake of consistency of image input types 
    """
    img = cv2.imread(img_file)           
    if img.shape[0] == 2: img = img[0]
    if len(img.shape) == 2 : img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:   img = img[:,:,:3]
    img = np.array(img)
    return img

def normalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    img = in_img.copy().astype(np.float32)
    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img

def denormalizeMeanVariance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    img = in_img.copy()
    img *= variance
    img += mean
    img *= 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def resize_aspect_ratio(img, square_size, interpolation, mag_ratio=1):
    height, width, channel = img.shape

    # magnify image size
    target_size = mag_ratio * max(height, width)

    # set original image size
    if target_size > square_size: target_size = square_size
    ratio = target_size / max(height, width)    
    target_h, target_w = int(height * ratio), int(width * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation = interpolation)

    # make canvas and paste image
    target_h32, target_w32 = target_h, target_w

    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)

    resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
    resized[0:target_h, 0:target_w, :] = proc
    target_h, target_w = target_h32, target_w32
    size_heatmap = (int(target_w/2), int(target_h/2))
    
    return resized, ratio, size_heatmap

def cvt2HeatmapImg(img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img

class strLabelConverter(object):
    def __init__(self, alphabet, ignore_case=True):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()

        self.alphabet = alphabet + '-'  # for `-1` index
        self.dict = {}

        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text):
        if isinstance(text, str):
            text = [self.dict[char.lower() if self._ignore_case else char] for char in text]
            length = [len(text)]

        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)

        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            
            if raw: return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts

class averager(object):
    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res

def oneHot(v, v_length, nc):
    batchSize = v_length.size(0)
    maxLength = v_length.max()
    v_onehot = torch.FloatTensor(batchSize, maxLength, nc).fill_(0)
    acc = 0
    
    for i in range(batchSize):
        length = v_length[i]
        label = v[acc:acc + length].view(-1, 1).long()
        v_onehot[i, :length].scatter_(1, label, 1.0)
        acc += length
    
    return v_onehot

def loadData(v, data):
    v.data.resize_(data.size()).copy_(data)

def prettyPrint(v):
    print('Size {0}, Type: {1}'.format(str(v.size()), v.data.type()))
    print('| Max: %f | Min: %f | Mean: %f' % (v.max().data[0], v.min().data[0], v.mean().data[0]))

def assureRatio(img):
    b, c, h, w = img.size()
    if h > w:
        main = nn.UpsamplingBilinear2d(size=(h, h), scale_factor=None)
        img = main(img)
    return img

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"): start_idx = 1
    else: start_idx = 0
    
    new_state_dict = OrderedDict()    
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    
    return new_state_dict

class resizeNormalize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img
