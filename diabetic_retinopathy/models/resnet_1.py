import tensorflow as tf
from tensorflow.python.keras.api._v2.keras import layers, Sequential, regularizers
import tensorflow.keras as keras


def regularized_padded_conv(*args, **kwargs):
    """
    Define a 3x3 convolution with l2 regularizer and different kernel_initializers
    kernel_initializer='glorot_normal', 'he_normal', 'lecun_normal'...

    """
    return layers.Conv2D(*args, **kwargs, padding='same', kernel_regularizer=regularizers.l2(5e-5),
                         use_bias=False, kernel_initializer='he_normal')


class BasicBlock(layers.Layer):
    """Define Basic Block for Resnet18 and Resnet34"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        # The first conv+bn
        self.conv1 = regularized_padded_conv(out_channels, kernel_size=3, strides=stride)
        self.bn1 = layers.BatchNormalization()

        # The second conv+bn
        self.conv2 = regularized_padded_conv(out_channels, kernel_size=3, strides=1)
        self.bn2 = layers.BatchNormalization()

        # Define shortcut
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = Sequential([regularized_padded_conv(self.expansion * out_channels, kernel_size=1, strides=stride),
                                        layers.BatchNormalization()])
        else:
            self.shortcut = lambda x, _: x

    def call(self, inputs, training=False):
        out = self.conv1(inputs)
        out = self.bn1(out, training=training)
        out = tf.nn.relu(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)

        out = out + self.shortcut(inputs, training)
        out = tf.nn.relu(out)

        return out


class Bottleneck(tf.keras.Model):
    """Define Bottleneck for Resnet50,Resnet101 and Resnet152"""
    expansion = 4

    def __init__(self, in_channels, out_channels, strides=1):
        super(Bottleneck, self).__init__()

        self.conv1 = layers.Conv2D(out_channels, 1, 1, use_bias=False)
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2D(out_channels, 3, strides, padding="same", use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(out_channels*self.expansion, 1, 1, use_bias=False)
        self.bn3 = layers.BatchNormalization()

        if strides != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = Sequential([layers.Conv2D(self.expansion * out_channels, kernel_size=1, strides=strides, use_bias=False),
                                        layers.BatchNormalization()])
        else:
            self.shortcut = lambda x, _: x

    def call(self, x, training=False):
        out = tf.nn.relu(self.bn1(self.conv1(x), training))
        out = tf.nn.relu(self.bn2(self.conv2(out), training))
        out = self.bn3(self.conv3(out), training)

        out = out + self.shortcut(x, training)
        out = tf.nn.relu(out)

        return out


class ResNet(keras.Model):
    """Defines a ResNet architecture."""

    def __init__(self, blocks, layer_dims, problem_type, initial_filters=64, num_classes=2):
        super(ResNet, self).__init__()
        self.in_channels = initial_filters

        # preprocess the convolutional layers with initializing
        self.stem = Sequential([regularized_padded_conv(initial_filters, kernel_size=3, strides=1),
                                layers.BatchNormalization()])

        # build 4 resblocks
        self.layer1 = self.build_resblock(blocks, initial_filters,    layer_dims[0], stride=1)
        self.layer2 = self.build_resblock(blocks, initial_filters*2,  layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(blocks, initial_filters*4,  layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(blocks, initial_filters*8,  layer_dims[3], stride=2)
        # self.final_bn  = layers.BatchNormalization()

        self.avgpool = layers.GlobalAveragePooling2D()
        if problem_type == 'regression':
            self.fc = layers.Dense(1)
        elif problem_type == 'classification':
            self.fc = layers.Dense(num_classes, activation='softmax')
        else:
            raise ValueError

    # build ResBlock
    def build_resblock(self, blocks, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)                    # [1]*3 = [1, 1, 1]
        res_blocks = Sequential()

        for stride in strides:
            res_blocks.add(blocks(self.in_channels, out_channels, stride))
            self.in_channels = out_channels

        return res_blocks

    def call(self, inputs, training=False):
        out = self.stem(inputs, training)
        out = tf.nn.relu(out)

        out = self.layer1(out, training=training)
        out = self.layer2(out, training=training)
        out = self.layer3(out, training=training)
        out = self.layer4(out, training=training)
        # out = self.final_bn(out, training=training)
        # out = tf.nn.relu(out)


        out = self.avgpool(out)
        out = self.fc(out)

        return out


def ResNet18(problem_type, num_classes):
    """ Resnet18 """
    return ResNet(BasicBlock, [2, 2, 2, 2], problem_type=problem_type, num_classes=num_classes)


def ResNet34(problem_type, num_classes):
    """ ResNet-34"""
    return ResNet(BasicBlock, [3, 4, 6, 3], problem_type=problem_type, num_classes=num_classes)


def ResNet50(problem_type, num_classes):
    """ Resnet50 """
    return ResNet(Bottleneck, [3, 4, 6, 3], problem_type=problem_type, num_classes=num_classes)


def ResNet101(problem_type, num_classes):
    """ Resnet101 """
    return ResNet(Bottleneck, [3, 4, 23, 3], problem_type=problem_type, num_classes=num_classes)


def ResNet152(problem_type, num_classes):
    """ Resnet152 """
    return ResNet(Bottleneck, [3, 8, 36, 3], problem_type=problem_type, num_classes=num_classes)
