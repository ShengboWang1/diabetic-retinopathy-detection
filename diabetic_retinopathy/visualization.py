from gradcam import GradCAM, overlay_gradCAM, get_img_array
from guided_backpropagation import GuidedBackprop, deprocess_image
import cv2
from matplotlib import pyplot as plt
from models.resnet import resnet50
import tensorflow as tf

image_path = "./IDRID_dataset/images/test/IDRiD_001.jpg"
image = get_img_array(image_path, (256, 256))
model = resnet50(5)
model.build(input_shape=(16, 256, 256, 3))
# #
checkpoint = tf.train.Checkpoint(myModel=model)
checkpoint.restore(tf.train.latest_checkpoint('./checkpoint/train/20201214-161759'))
model.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics='SparseCategoricalAccuracy')
model.summary()


# GradCAM
# gradcam = GradCAM(model=model, layerName="conv5_block3_out")
gradcam = GradCAM(model=model, layerName="resnet50")
cam3 = gradcam.compute_heatmap(image=image, classIdx=1, upsample_size=(4288, 2848))
plt.matshow(cam3)
plt.title("cam3")
plt.show()
original_image = cv2.imread(image_path)
cam3 = overlay_gradCAM(original_image, cam3)
gradcam = cv2.cvtColor(cam3, cv2.COLOR_BGR2RGB)
plt.matshow(gradcam)
plt.title("GradCAM")
plt.show()

# Guided Backpropagation
# GuidedBP = GuidedBackprop(model=model, layerName="conv5_block3_out")
GuidedBP = GuidedBackprop(model=model, layerName="resnet50")
gb = GuidedBP.guided_backprop(image, upsample_size=(4288, 2848))
gb_im = deprocess_image(gb)
gb_im = cv2.cvtColor(gb_im, cv2.COLOR_BGR2RGB)
plt.matshow(gb_im)
plt.title("Guided Backpropagation")
plt.show()

# Guided GradCAM
guided_gradcam = deprocess_image(gb * cam3)
guided_gradcam = cv2.cvtColor(guided_gradcam, cv2.COLOR_BGR2RGB)
plt.matshow(guided_gradcam)
plt.title("Guided GradCAM")
plt.show()
