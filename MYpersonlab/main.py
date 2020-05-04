from matplotlib import pyplot as plt
import tensorflow as tf 
from config import config
import model
from data_generator import DataGeneraotr
import numpy as np
from skimage import io
from plot import plot_poses, apply_mask, plot_instance_masks
from post_proc import compute_heatmaps, get_keypoints, group_skeletons, get_instance_masks

from scipy.ndimage.filters import gaussian_filter
import argparse
import os
import cv2

multiscale = [1.,1.5, 2.]

# build the model
batch_size,height,width=1,config.IMAGE_SHAPE[0],config.IMAGE_SHAPE[1]
tf_img = []
outputs = []
for i in range(len(multiscale)):
    scale = multiscale[i]
    tf_img.append(tf.placeholder(tf.float32,shape=[batch_size,int(scale*height),int(scale*width),3]))
    outputs.append(model.model(tf_img[i])) 
sess = tf.Session()

# load the parameters
global_vars = tf.global_variables()
saver = tf.train.Saver(var_list = global_vars)
checkpoint_path = './model/personlab/'+'model.ckpt'
saver.restore(sess,checkpoint_path)
print("Trained Model Restored!")


def get_multi_scale_img(filepath,scale):
    #img_id = give_id
    #filepath = os.path.join(IMG_DIR,self.coco.imgs[img_id]['file_name'])
    img = cv2.imread(filepath)
    cv_shape = (config.IMAGE_SHAPE[1], config.IMAGE_SHAPE[0])
    cv_shape2 = (int(cv_shape[0]*scale),int(cv_shape[1]*scale))
    max_shape = max(img.shape[0],img.shape[1])
    scale2 = cv_shape2[0]/max_shape
    img = cv2.resize(img,None,fx=scale2,fy=scale2)
    img = cv2.copyMakeBorder(img,0,cv_shape2[0]-img.shape[0],0,cv_shape2[1]-img.shape[1],cv2.BORDER_CONSTANT,value=[127,127,127])
    return img


def process_image(filepath, save_dir):
    scale_outputs = []
    for i in range(len(multiscale)):
        scale = multiscale[i]
        scale_img = get_multi_scale_img(filepath=filepath,scale=scale)
        if i==0:
            img = scale_img[:,:,[2,1,0]]
            plt.imsave(os.path.join(save_dir,'input_image.jpg'),img)
        imgs_batch = np.zeros((batch_size,int(scale*height),int(scale*width),3))
        imgs_batch[0] = scale_img

        # make prediction  
        one_scale_output = sess.run(outputs[i],feed_dict={tf_img[i]:imgs_batch})
        scale_outputs.append([o[0] for o in one_scale_output])

    sample_output = scale_outputs[0]
    for i in range(1,len(multiscale)):
        for j in range(len(sample_output)):
            sample_output[j]+=scale_outputs[i][j]
    for j in range(len(sample_output)):
        sample_output[j] /=len(multiscale)

    # visualization
    print('Visualization image has been saved into ',save_dir)

    # Here is the output map for right shoulder
    #Rshoulder_map = sample_output[0][:,:,config.KEYPOINTS.index('Rshoulder')]
    #plt.imsave(save_path+'kp_map.jpg',overlay(img, Rshoulder_map, alpha=0.7))


    # Gaussian filtering helps when there are multiple local maxima for the same keypoint.
    H = compute_heatmaps(kp_maps=sample_output[0], short_offsets=sample_output[1])
    for i in range(17):
        H[:,:,i] = gaussian_filter(H[:,:,i], sigma=2)
    #plt.imsave(save_path+'heatmaps.jpg',H[:,:,config.KEYPOINTS.index('Rshoulder')]*10)


    # The heatmaps are computed using the short offsets predicted by the network
    # Here are the right shoulder offsets
    #visualize_short_offsets(offsets=sample_output[1], heatmaps=H, keypoint_id='Rshoulder', img=img, every=8,save_path=save_path)

    # The connections between keypoints are computed via the mid-range offsets.
    # We can visuzalize them as well; for example right shoulder -> right hip
    #visualize_mid_offsets(offsets= sample_output[2], heatmaps=H, from_kp='Rshoulder', to_kp='Rhip', img=img, every=8,save_path=save_path)

    # And we can see the reverse connection (Rhip -> Rshjoulder) as well
    # visualize_mid_offsets(offsets= sample_output[2], heatmaps=H, to_kp='Rshoulder', from_kp='Rhip', img=img, every=8,save_path=save_path)

    # We can use the heatmaps to compute the skeletons
    pred_kp = get_keypoints(H)
    pred_skels = group_skeletons(keypoints=pred_kp, mid_offsets=sample_output[2])
    pred_skels = [skel for skel in pred_skels if (skel[:,2]>0).sum() > 4]
    print ('Number of detected skeletons: {}'.format(len(pred_skels)))

    plot_poses(img, pred_skels,save_path=save_dir)

    # we can use the predicted skeletons along with the long-range offsets and binary segmentation mask to compute the instance masks. 
    applied_mask = apply_mask(img, sample_output[4][:,:,0]>0.5, color=[255,0,0])
    plt.imsave(os.path.join(save_dir,'segmentation_mask.jpg'), applied_mask)

    #visualize_long_offsets(offsets=sample_output[3], keypoint_id='Rshoulder', seg_mask=sample_output[4], img=img, every=8,save_path=save_path)
    
    instance_masks = get_instance_masks(pred_skels, sample_output[-1][:,:,0], sample_output[-2])
    plot_instance_masks(instance_masks, img,save_path=save_dir)


def overlay(img, over, alpha=0.5):
    out = img.copy()
    if img.max() > 1.:
        out = out / 255.
    out *= 1-alpha
    if len(over.shape)==2:
        out += alpha*over[:,:,np.newaxis]
    else:
        out += alpha*over    
    return out





"""----------------------------- Main options -----------------------------"""
parser = argparse.ArgumentParser(description='AlphaPose Demo')
parser.add_argument('--input', dest='input',help='input', default="")
parser.add_argument('--output', dest='output',help='output', default="./output/")

args = parser.parse_args()

images = []
if os.path.isdir(args.input):
    images = [os.path.join(args.input, f) for f in os.listdir(args.input)]
else:
    images = [args.input]

for img_path in images:
    save_dir = os.mkdir(os.path.join(args.output, img_path.split(".")[0]))
    print(img_path)
    process_image(img_path, save_dir)