from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Activation
from keras.layers import BatchNormalization,Dropout
from keras.initializers import Initializer
from keras import regularizers
import param

def create_cnn_model(optimizer,input_shape):
    model = Sequential()
    #第１層目
    model.add(Conv2D(filters=param.filter_number[0],
                kernel_size=param.kernel_size[0],
                padding="same",
                #bias_initializer=Initializer"he_normal",
                #kernel_initializer="he_normal",
                #bias_regularizer = regularizers.l2(regularization_parameter),
                #kernel_regularizer = regularizers.l2(param.regularization_parameter),
                #activity_regularizer = regularizers.l2(regularization_parameter),
                input_shape=input_shape))

    #model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=param.pool_size,strides=param.strides))
    model.add(Activation(param.activation_function))

    #第n層目
    for layer in range(1,int(param.layer_number)):
        model.add(Conv2D(filters=int(param.filter_number[layer]),
                kernel_size=int(param.kernel_size[layer]),
                padding="same",
                #bias_initializer="he_normal",
                #kernel_initializer="he_normal",
                #bias_regularizer = regularizers.l2(regularization_parameter),
                #kernel_regularizer = regularizers.l2(param.regularization_parameter),
                #activity_regularizer = regularizers.l2(regularization_parameter),
        ))
        #model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=param.pool_size,strides=param.strides))
        model.add(Activation(param.activation_function))

    model.add(Flatten())
    model.add(Dropout(param.dropout_rate))
    model.add(Dense(units=param.classes,activation="softmax"))

    model.compile(loss = "binary_crossentropy",
                    optimizer = optimizer,
                    metrics = ["accuracy"])

    return model