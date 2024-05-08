from keras import layers,Model

def design_model():
    l1=layers.Input(shape=(128,128,3)) ######## 128*128*3
    l1=layers.Conv2D(filters=32,activation='leaky_relu',kernel_size=(3,3),padding='same')(l1)
    l1=layers.Conv2D(filters=64,activation='leaky_relu',kernel_size=(3,3),padding='same')(l1)
    l1=layers.BatchNormalization(epsilon=.001)(l1)
    l1=layers.Dropout(rate=.3)(l1)

    l1=layers.MaxPool2D(pool_size=(2,2))(l1) ##########  64*64*64 
    # 1
    l2=layers.Conv2D(filters=128,activation='leaky_relu',padding='same',kernel_size=(3,3))(l1)
    l2=layers.Conv2D(filters=256,activation='leaky_relu',padding='same',kernel_size=(3,3))(l2)
    l2=layers.BatchNormalization(epsilon=.001)(l2)
    l2=layers.Dropout(rate=.3)(l2)

    l2=layers.MaxPool2D(pool_size=(2,2))(l2) ##########  32*32*256

    # 2

    l3=layers.Conv2D(filters=256,activation='leaky_relu',padding='same',kernel_size=(3,3))(l2)
    l3=layers.Conv2D(filters=256,activation='leaky_relu',padding='same',kernel_size=(3,3))(l3)
    l3=layers.BatchNormalization(epsilon=.001)(l3)
    l3=layers.Dropout(rate=.3)(l3)

    l3=layers.MaxPool2D(pool_size=(2,2))(l3) ##########  16*16*256

    # 3

    l4=layers.Conv2D(filters=256,activation='leaky_relu',padding='same',kernel_size=(3,3))(l3)
    l4=layers.Conv2D(filters=256,activation='leaky_relu',padding='same',kernel_size=(3,3))(l4)

    l4=layers.MaxPool2D(pool_size=(2,2))(l4) ##########  8*8*256

    l5=layers.Conv2DTranspose(filters=256,strides=(2,2),kernel_size=(3,3),padding='same',activation='leaky_relu')(l4)
    l5=layers.Concatenate()([l3,l5])
    l5=layers.Conv2D(filters=128,kernel_size=(3,3),padding='same',activation='leaky_relu')(l5)
    l5=layers.BatchNormalization(epsilon=.001)(l5)
    l5=layers.Dropout(rate=.3)(l5)      ###### 16 16 128

    l6=layers.Conv2DTranspose(filters=64,strides=(2,2),kernel_size=(3,3),padding='same',activation='leaky_relu')(l5)
    l6=layers.Concatenate()([l2,l6])
    l6=layers.Conv2D(filters=32,kernel_size=(3,3),padding='same',activation='leaky_relu')(l6)
    l6=layers.BatchNormalization(epsilon=.001)(l6)
    l6=layers.Dropout(rate=.3)(l6) ####### 32 32 32

    l7=layers.Conv2DTranspose(filters=16,strides=(2,2),kernel_size=(3,3),padding='same',activation='leaky_relu')(l6)
    l7=layers.Concatenate()([l1,l7])
    l7=layers.Conv2D(filters=16,kernel_size=(3,3),padding='same',activation='leaky_relu')(l7) ### 64 64 16

    l7=layers.Conv2DTranspose(filters=16,strides=(2,2),kernel_size=(3,3),padding='same',activation='leaky_relu')(l7) ## 128 128 16
    l7=layers.Conv2D(filters=8,kernel_size=(3,3),padding='same',activation='leaky_relu')(l7) 
    l7=layers.Conv2D(filters=1,kernel_size=(8,8),padding='same',activation='leaky_relu')(l7)
    # 128 128 1
    model=Model(l1,l7)

    model.summary()
    return model
