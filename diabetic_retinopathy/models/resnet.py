import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras import layers, Sequential

# Basic Block 模块。
class BasicBlock(layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        #上一块如果做Stride就会有一个下采样，在这个里面就不做下采样了。这一块始终保持size一致，把stride固定为1
        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()

        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride))
        else:
            self.downsample = lambda x:x

    def call(self, inputs, training=None):

        # [b, h, w, c]
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.downsample(inputs)

        output = layers.add([out, identity])  #layers下面有一个add，把这2个层添加进来相加。
        output = tf.nn.relu(output)
        return output


class Bottleneck(tf.keras.Model):
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
            self.shortcut = lambda x,_: x

    def call(self, x, training=False):
        out = tf.nn.relu(self.bn1(self.conv1(x), training))
        out = tf.nn.relu(self.bn2(self.conv2(out), training))
        out = self.bn3(self.conv3(out), training)

        out = out + self.shortcut(x, training)
        out = tf.nn.relu(out)

        return out

# Res Block 模块。继承keras.Model或者keras.Layer都可以
class ResNet(k.Model):

    # 第一个参数layer_dims：[2, 2, 2, 2] 4个Res Block，每个包含2个Basic Block
    # 第二个参数num_classes：我们的全连接输出，取决于输出有多少类。
    def __init__(self, layer_dims, num_classes=2):
        super(ResNet, self).__init__()

        # 预处理层；实现起来比较灵活可以加 MAXPool2D，可以没有。
        self.stem = Sequential([layers.Conv2D(64, (3, 3), strides=(1, 1)),
                                layers.BatchNormalization(),
                                layers.Activation('relu'),
                                layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')
                                ])

        # 创建4个Res Block；注意第1项不一定以2倍形式扩张，都是比较随意的，这里都是经验值。
        self.layer1 = self.build_resblock(64, layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)

        self.avgpool = layers.GlobalAveragePooling2D()
        # self.dropout = tf.keras.layers.Dropout(0.2)
        self.out1 = layers.Dense(num_classes, activation='softmax')

    def call(self,inputs, training=None):
        # __init__中准备工作完毕；下面完成前向运算过程。
        x = self.stem(inputs)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 做一个global average pooling，得到之后只会得到一个channel，不需要做reshape操作了。
        # shape为 [batchsize, channel]
        x = self.avgpool(x)
        # x = self.dropout(x)
        # [b, 100]
        x = self.out1(x)
        return x

    # 实现 Res Block； 创建一个Res Block
    def build_resblock(self, filter_num, blocks, stride=1):

        res_blocks = Sequential()
        # may down sample 也许进行下采样。
        # 对于当前Res Block中的Basic Block，我们要求每个Res Block只有一次下采样的能力。
        res_blocks.add(BasicBlock(filter_num, stride))

        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1)) # 这里stride设置为1，只会在第一个Basic Block做一个下采样。

        return res_blocks

def resnet18():
    return ResNet([2, 2, 2, 2])

# 如果我们要使用 ResNet-34 的话，那34是怎样的配置呢？只需要改一下这里就可以了。对于56，152去查一下配置
def resnet34():
    return ResNet([3, 4, 6, 3]) #4个Res Block，第1个包含3个Basic Block,第2为4，第3为6，第4为3

def resnet50():
    ResNet50 = k.applications.ResNet50(include_top=False,
                                              weights='imagenet',
                                              input_shape=(256, 256, 3))
    ResNet50.trainable = False
    model = tf.keras.Sequential()
    model.add(ResNet50)
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(10, activation='relu'))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    return model

def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])

def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])
