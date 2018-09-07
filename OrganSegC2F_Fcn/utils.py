import numpy as np
import os
import sys
import math
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral, create_pairwise_gaussian, unary_from_softmax

data_path = "/media/jionie/Disk1"
image_path = os.path.join(data_path, 'images')
image_path_ = {}
for plane in ['X', 'Y', 'Z']:
    image_path_[plane] = os.path.join(data_path, 'images_' + plane)
    if not os.path.exists(image_path_[plane]):
        os.makedirs(image_path_[plane])
label_path = os.path.join(data_path, 'labels')
label_path_ = {}
for plane in ['X', 'Y', 'Z']:
    label_path_[plane] = os.path.join(data_path, 'labels_' + plane)
    if not os.path.exists(label_path_[plane]):
        os.makedirs(label_path_[plane])
list_path = os.path.join(data_path, 'lists')
if not os.path.exists(list_path):
    os.makedirs(list_path)
coarse_list_training = {}
fine_list_training = {}
coarse_list_retraining = {}
fine_list_retraining = {}
for plane in ['X', 'Y', 'Z']:
    coarse_list_training[plane] = os.path.join(list_path, 'coarse_' + plane + '.txt')
    fine_list_training[plane] = os.path.join(list_path, 'fine_' + plane + '.txt')


####################################################################################################
# returning the binary label map by the organ ID (especially useful under overlapping cases)
#   label: the label matrix
#   organ_ID: the organ ID
def is_organ(label, organ_ID):
    return label == organ_ID


####################################################################################################
# determining if a sample belongs to the training set by the fold number
#   total_samples: the total number of samples
#   i: sample ID, an integer in [0, total_samples - 1]
#   folds: the total number of folds
#   current_fold: the current fold ID, an integer in [0, folds - 1]
def in_training_set(total_samples, i, folds, current_fold):
    fold_remainder = folds - total_samples % folds
    fold_size = (total_samples - total_samples % folds) / folds
    start_index = fold_size * current_fold + max(0, current_fold - fold_remainder)
    end_index = fold_size * (current_fold + 1) + max(0, current_fold + 1 - fold_remainder)
    return not (i >= start_index and i < end_index)


####################################################################################################
# returning the filename of the training set according to the current fold ID
def training_set_filename(current_fold):
    return os.path.join(list_path, 'training_' + 'FD' + str(current_fold) + '.txt')


####################################################################################################
# returning the filename of the testing set according to the current fold ID
def testing_set_filename(current_fold):
    return os.path.join(list_path, 'testing_' + 'FD' + str(current_fold) + '.txt')


####################################################################################################
# returning the volume filename as in the testing stage
def volume_filename_testing(result_directory, t, i):
    return os.path.join(result_directory, str(t) + '_' + str(i + 1) + '.npz')


####################################################################################################
# returning the volume filename as in the fusion stage
def volume_filename_fusion(result_directory, code, i):
    return os.path.join(result_directory, code + '_' + str(i + 1) + '.npz')


####################################################################################################
# returning the volume filename as in the coarse-to-fine testing stage
def volume_filename_coarse2fine(result_directory, r, i):
    return os.path.join(result_directory, 'R' + str(r) + '_' + str(i + 1) + '.npz')


def resized_parameter(input_parameter, multiple):

    input_parameter = float(input_parameter)
    output_parameter = math.ceil(input_parameter / multiple) * multiple

    return output_parameter


####################################################################################################
# computing the DSC together with other values based on the label and prediction volumes
def DSC_computation(label, pred):
    pred_sum = pred.sum()
    label_sum = label.sum()
    inter_sum = np.multiply(pred, label)
    inter_sum = inter_sum.sum()
    return 2 * float(inter_sum) / (pred_sum + label_sum), inter_sum, pred_sum, label_sum


####################################################################################################

def post_processing(F, S, threshold, organ_ID):
    if F.sum() == 0:
        return F
    if F.sum() >= np.product(F.shape) / 2:
        return F
    height  = F.shape[0]
    width = F.shape[1]
    depth = F.shape[2]
    ll = np.array(np.nonzero(S))
    marked = np.zeros_like(F, dtype = np.bool)
    queue = np.zeros((F.sum(), 3), dtype = np.int)
    volume = np.zeros(F.sum(), dtype = np.int)
    head = 0
    tail = 0
    bestHead = 0
    bestTail = 0
    bestHead2 = 0
    bestTail2 = 0
    for l in range(ll.shape[1]):
        if not marked[ll[0, l], ll[1, l], ll[2, l]]:
            temp = head
            marked[ll[0, l], ll[1, l], ll[2, l]] = True
            queue[tail, :] = [ll[0, l], ll[1, l], ll[2, l]]
            tail = tail + 1
            while (head < tail):
                t1 = queue[head, 0]
                t2 = queue[head, 1]
                t3 = queue[head, 2]
                if t1 > 0 and F[t1 - 1, t2, t3] and not marked[t1 - 1, t2, t3]:
                    marked[t1 - 1, t2, t3] = True
                    queue[tail, :] = [t1 - 1, t2, t3]
                    tail = tail + 1
                if t1 < height - 1 and F[t1 + 1, t2, t3] and not marked[t1 + 1, t2, t3]:
                    marked[t1 + 1, t2, t3] = True
                    queue[tail, :] = [t1 + 1, t2, t3]
                    tail = tail + 1
                if t2 > 0 and F[t1, t2 - 1, t3] and not marked[t1, t2 - 1, t3]:
                    marked[t1, t2 - 1, t3] = True
                    queue[tail, :] = [t1, t2 - 1, t3]
                    tail = tail + 1
                if t2 < width - 1 and F[t1, t2 + 1, t3] and not marked[t1, t2 + 1, t3]:
                    marked[t1, t2 + 1, t3] = True
                    queue[tail, :] = [t1, t2 + 1, t3]
                    tail = tail + 1
                if t3 > 0 and F[t1, t2, t3 - 1] and not marked[t1, t2, t3 - 1]:
                    marked[t1, t2, t3 - 1] = True
                    queue[tail, :] = [t1, t2, t3 - 1]
                    tail = tail + 1
                if t3 < depth - 1 and F[t1, t2, t3 + 1] and not marked[t1, t2, t3 + 1]:
                    marked[t1, t2, t3 + 1] = True
                    queue[tail, :] = [t1, t2, t3 + 1]
                    tail = tail + 1
                head = head + 1
            if tail - temp > bestTail - bestHead:
                bestHead2 = bestHead
                bestTail2 = bestTail
                bestHead = temp
                bestTail = tail
            elif tail - temp > bestTail2 - bestHead2:
                bestHead2 = temp
                bestTail2 = tail
            volume[temp: tail] = tail - temp
    volume = volume[0: tail]
    target_voxel = np.where(volume >= (bestTail - bestHead) * threshold)
    F0 = np.zeros_like(F, dtype = np.bool)
    F0[tuple(map(tuple, np.transpose(queue[target_voxel, :])))] = True
    return F0

####################################################################################################
# dense CRF
def dense_crf(probs, img=None, n_iters=10, n_classes=2,
              sxy_gaussian=(1,1), compat_gaussian=4,
              kernel_gaussian=dcrf.DIAG_KERNEL,
              normalisation_gaussian=dcrf.NORMALIZE_SYMMETRIC,
              sxy_bilateral=(10, 10), compat_bilateral=5,
              srgb_bilateral=(5, 5, 5),
              kernel_bilateral=dcrf.DIAG_KERNEL,
              normalisation_bilateral=dcrf.NORMALIZE_SYMMETRIC):
    """DenseCRF over unnormalised predictions.
       More details on the arguments at https://github.com/lucasb-eyer/pydensecrf.
    
    Args:
      probs: class probabilities per pixel.
      img: if given, the pairwise bilateral potential on raw RGB values will be computed.
      n_iters: number of iterations of MAP inference.
      sxy_gaussian: standard deviations for the location component of the colour-independent term.
      compat_gaussian: label compatibilities for the colour-independent term (can be a number, a 1D array, or a 2D array).
      kernel_gaussian: kernel precision matrix for the colour-independent term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
      normalisation_gaussian: normalisation for the colour-independent term (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
      sxy_bilateral: standard deviations for the location component of the colour-dependent term.
      compat_bilateral: label compatibilities for the colour-dependent term (can be a number, a 1D array, or a 2D array).
      srgb_bilateral: standard deviations for the colour component of the colour-dependent term.
      kernel_bilateral: kernel precision matrix for the colour-dependent term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
      normalisation_bilateral: normalisation for the colour-dependent term (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
      
    Returns:
      Refined predictions after MAP inference.
    """
    _, h, w, _ = probs.shape
    
    
    probs = probs[0].transpose(2, 0, 1).copy(order='C') # Need a contiguous array.
    
    d = dcrf.DenseCRF2D(w, h, n_classes) # Define DenseCRF model.
    U = unary_from_softmax(probs) # Unary potential.
    U = U.reshape((n_classes, -1)) # Needs to be flat.
    d.setUnaryEnergy(U)
    energy = create_pairwise_gaussian(sxy_gaussian, [w, h])
    d.addPairwiseEnergy(energy, compat=compat_gaussian)

    if img is not None:
        assert(img.shape[1:3] == (h, w)), "The image height and width must coincide with dimensions of the logits."
        energy = create_pairwise_bilateral(sdims=sxy_bilateral, schan=srgb_bilateral[0], img=img, chdim=-1)
        d.addPairwiseEnergy(energy, compat=compat_bilateral)

    Q = d.inference(n_iters)
    preds = np.array(Q, dtype=np.float32).reshape((n_classes, h, w)).transpose(1, 2, 0)
    return np.expand_dims(preds, 0)
