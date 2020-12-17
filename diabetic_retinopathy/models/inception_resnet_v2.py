import tensorflow.keras as k


def inception_resnet_v2(num_classes):
    base_model = k.applications.InceptionV3(weights='imagenet', include_top=False)
    base_model.trainable = False
    out = base_model.output
    out = k.layers.GlobalAveragePooling2D()(out)
    out = k.layers.Flatten(name='flatten')(out)
    out = k.layers.Dense(2048, activation='relu', kernel_regularizer=k.regularizers.l2(0.0001))(out)
    out = k.layers.BatchNormalization()(out)
    out = k.layers.Dense(1024, activation='relu', kernel_regularizer=k.regularizers.l2(0.0001))(out)
    out = k.layers.BatchNormalization(name='bn_fc_01')(out)
    predictions = k.layers.Dense(num_classes, activation='softmax')(out)
    model = k.Model(inputs=base_model.input, outputs=predictions)
    return model
