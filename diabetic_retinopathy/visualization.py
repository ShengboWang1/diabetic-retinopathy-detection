from gradcam import GradCAM, overlay_gradCAM, get_img_array
from guided_backpropagation import GuidedBackprop, deprocess_image
import cv2
from matplotlib import pyplot as plt
import os
import tensorflow as tf

image_paths = ["./IDRID_dataset/images/train/IDRiD_003.jpg",
               "./IDRID_dataset/images/train/IDRiD_010.jpg",
               "./IDRID_dataset/images/train/IDRiD_087.jpg"]


def visualize(model, layername, save_path):
    """Perform visalization on a series of images"""
    i = 0
    plt.figure()
    for image_path in image_paths:
        i += 1
        image = get_img_array(image_path)

        # Plot original image
        original_image = cv2.imread(image_path)
        plt.matshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title("original_image 0" + str(i))
        plt.axis('off')
        plt.savefig(os.path.join(save_path['path_plt'], 'OriginalImage' + str(i) + '.png'), bbox_inches='tight')
        plt.show()

        # Show the result of GradCAM
        gradcam = GradCAM(model=model, layerName=layername)
        # gradcam = GradCAM(model=model, layerName="conv5_block3_out")
        cam3 = gradcam.compute_heatmap(image=image, classIdx=1, upsample_size=(3426, 3426))
        plt.matshow(cam3)
        plt.title("CAM 0" + str(i))
        plt.axis('off')
        plt.savefig(os.path.join(save_path['path_plt'], 'CAM' + str(i) + '.png'), bbox_inches='tight')
        plt.show()

        # Overlay the result of cam and original image
        gradcam = overlay_gradCAM(original_image, cam3)
        gradcam = cv2.cvtColor(gradcam, cv2.COLOR_BGR2RGB)
        plt.matshow(gradcam)
        plt.title("GradCAM 0" + str(i))
        plt.axis('off')
        plt.savefig(os.path.join(save_path['path_plt'], 'GradCAM' + str(i) + '.png'), bbox_inches='tight')
        plt.show()

        # Show the result of Guided Backpropagation
        GuidedBP = GuidedBackprop(model=model, layerName=layername)
        gb = GuidedBP.guided_backprop(image, upsample_size=(3426, 3426))
        gb = tf.image.crop_to_bounding_box(gb, 289, 0, 2848, 3426)
        gb = tf.image.pad_to_bounding_box(gb, 0, 266, 2848, 4288)
        gb_im = deprocess_image(gb.numpy())
        gb_im = cv2.cvtColor(gb_im, cv2.COLOR_BGR2RGB)
        plt.matshow(gb_im)
        plt.title("Guided Backpropagation 0" + str(i))
        plt.axis('off')
        plt.savefig(os.path.join(save_path['path_plt'], 'GuidedBackprop' + str(i) + '.png'), bbox_inches='tight')
        plt.show()

        # Show the result of Guided GradCAM
        guided_gradcam = deprocess_image((gb * cam3).numpy())
        guided_gradcam = cv2.cvtColor(guided_gradcam, cv2.COLOR_BGR2RGB)
        plt.matshow(guided_gradcam)
        plt.title("GradCAM 0" + str(i))
        plt.axis('off')
        plt.savefig(os.path.join(save_path['path_plt'], 'Guided GradCAM' + str(i) + '.png'), bbox_inches='tight')
        plt.show()
