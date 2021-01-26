import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import Model
from models.resnet import resnet50
from matplotlib import pyplot as plt
# from input_pipeline.preprocessing import preprocess

class GradCAM:
    # Adapted with some modification from https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/
    def __init__(self, model, layerindex):
        """
        model: pre-softmax layer (logit layer)
        """
        self.model = model
        self.layerindex = layerindex

        # if self.layerName == None:
        #     self.layerName = self.find_target_layer()

    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM")

    def compute_heatmap(self, image, classIdx, upsample_size, eps=1e-5):
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(index=self.layerindex).output, self.model.output]
        )
        # record operations for automatic differentiation

        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (convOuts, preds) = gradModel(inputs)  # preds after softmax
            loss = preds[:, classIdx]

        # compute gradients with automatic differentiation
        grads = tape.gradient(loss, convOuts)
        # discard batch
        convOuts = convOuts[0]
        grads = grads[0]
        norm_grads = tf.divide(grads, tf.reduce_mean(tf.square(grads)) + tf.constant(eps))

        # compute weights
        weights = tf.reduce_mean(norm_grads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOuts), axis=-1)

        # Apply reLU
        cam = np.maximum(cam, 0)
        cam = cam / np.max(cam)
        cam = cv2.resize(cam, upsample_size,cv2.INTER_LINEAR)

        # convert to 3D
        cam3 = np.expand_dims(cam, axis=2)
        cam3 = np.tile(cam3, [1, 1, 3])

        return cam3

def overlay_gradCAM(img, cam3):
    cam3 = np.uint8(255 * cam3)
    cam3 = cv2.applyColorMap(cam3, cv2.COLORMAP_JET)

    new_img = 0.3 * cam3 + 0.5 * img
    #new_img = 0.3 * cam3 + img

    return (new_img * 255.0 / new_img.max()).astype("uint8")

def get_img_array(img_path, size):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    array = tf.keras.preprocessing.image.img_to_array(img)
    #array, _ = preprocess(array, 0)
    array = np.expand_dims(array, axis=0)
    return array

# #
# model = resnet50(2)
# model.build(input_shape=(16, 256, 256, 3))
# # #
# # checkpoint = tf.train.Checkpoint(myModel=model)
# # checkpoint.restore(tf.train.latest_checkpoint('./checkpoint/train'))
# # model.compile(optimizer='adam', loss='SparseCategoricalCrossentropy', metrics='SparseCategoricalAccuracy')
# model.summary()
# gradcam = GradCAM(model=model, layerName="conv5_block3_out")
# image = get_img_array("./IDRID_dataset/images/test/IDRiD_002.jpg", (256, 256))
#
# cam3 = gradcam.compute_heatmap(image=image, classIdx=1, upsample_size=(4288, 2848))
# plt.matshow(cam3)
# plt.show()
# original_image = cv2.imread("./IDRID_dataset/images/test/IDRiD_002.jpg")
# cam3 = overlay_gradCAM(original_image, cam3)
# gradcam = cv2.cvtColor(cam3, cv2.COLOR_BGR2RGB)
# plt.matshow(gradcam)
# plt.show()
