from gradcam import GradCAM, overlay_gradCAM, get_img_array
from guided_backpropagation import GuidedBackprop, deprocess_image
import cv2
from matplotlib import pyplot as plt
import os
from models.resnet import resnet50
import tensorflow as tf


# set up the model
def visualize(model, layerName, save_path):
    image_path = "./IDRID_dataset/images/train/IDRiD_005.jpg"
    image = get_img_array(image_path, (256, 256))


    # plot original image
    original_image = cv2.imread(image_path)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title("original_image")
    plt.axis('off')
    plt.savefig(os.path.join(save_path['path_plt'], 'OriginalImage.png'))
    plt.show()

    # Show the result of GradCAM
    gradcam = GradCAM(model=model, layerName=layerName)
    # gradcam = GradCAM(model=model, layerName="conv5_block3_out")
    cam3 = gradcam.compute_heatmap(image=image, classIdx=1, upsample_size=(4288, 2848))
    plt.matshow(cam3)
    plt.title("cam3")
    plt.axis('off')
    plt.savefig(os.path.join(save_path['path_plt'], 'Cam3.png'))
    plt.show()

    # overlay the result of
    cam3 = overlay_gradCAM(original_image, cam3)
    gradcam = cv2.cvtColor(cam3, cv2.COLOR_BGR2RGB)
    plt.matshow(gradcam)
    plt.title("GradCAM")
    plt.axis('off')
    plt.savefig(os.path.join(save_path['path_plt'], 'GradCam.png'))
    plt.show()

    # Show the result of Guided Backpropagation
    GuidedBP = GuidedBackprop(model=model, layerName=layerName)
    gb = GuidedBP.guided_backprop(image, upsample_size=(4288, 2848))
    gb_im = deprocess_image(gb)
    gb_im = cv2.cvtColor(gb_im, cv2.COLOR_BGR2RGB)
    plt.matshow(gb_im)
    plt.axis('off')
    plt.title("Guided Backpropagation")
    plt.savefig(os.path.join(save_path['path_plt'], 'GuidedBackpropagation.png'))
    plt.show()

    # Show the result of Guided GradCAM
    guided_gradcam = deprocess_image(gb * cam3)
    guided_gradcam = cv2.cvtColor(guided_gradcam, cv2.COLOR_BGR2RGB)
    plt.matshow(guided_gradcam)
    plt.title("Guided GradCAM")
    plt.show()
