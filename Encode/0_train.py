import os
from skimage import io, transform, color
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import random
import numpy as np
import pickle

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"

height, width = 32, 32
w_hei, w_wid = 32, 32

batch_size = 64
dataset_len = 50000 # total images

test_percentage = 0.1
test_len = int(dataset_len * test_percentage)
train_len = dataset_len - test_len

### data
from keras.datasets import cifar10
(C_array, _), (_, _) = cifar10.load_data()
from keras.datasets import mnist
(W_array, _), (_, _) = mnist.load_data()

W_array = W_array[0:min(W_array.shape[0],C_array.shape[0])]
C_array = C_array[0:min(W_array.shape[0],C_array.shape[0])]

def get_w_array(train=1):
    np.random.seed(1313)
    np.random.shuffle(W_array)
    
    if train == 1:
        return W_array[0:train_len,:,:]
    else:
        test_set = W_array[train_len:,:,:]
        np.random.seed(None)
        np.random.shuffle(test_set)
        return test_set

def get_c_array(train=1):
    np.random.seed(1313)
    np.random.shuffle(C_array)
    
    if train == 1:
        return C_array[0:train_len,:,:,:]
    else:
        test_set = C_array[train_len:,:,:,:]
        np.random.seed(None)
        np.random.shuffle(test_set)
        return test_set

def get_batch(train=1, batch_size=64):
    cn = get_c_array(train)
    wn = get_w_array(train)
    
    i_c, i_w = 0, 0
    while True:
        
        ### C
        if i_c+batch_size >= cn.shape[0]:
            i_c = 0
            np.random.seed(None)
            np.random.shuffle(cn)
            c = cn[np.random.randint(0,cn.shape[0], size=batch_size), :, :, :]
        else:
            c = cn[i_c:i_c+batch_size,:,:,:]
        i_c += batch_size
        
        c_batch = []
        for each_c in c:
            img_c = (each_c - each_c.min()) / (each_c.max() - each_c.min())
            img_c = transform.resize(img_c, (height, width, 3), mode='reflect')
            c_batch.append(img_c)
        c_batch = np.array(c_batch)
        c_batch = np.reshape(c_batch, [batch_size, height, width, 3])
        # print('c:',c_batch.shape, c_batch.max(), c_batch.min())
        
        #------------------------------------------------------------------
        
        ### wm
        if i_w+batch_size >= wn.shape[0]:
            i_w = 0
            np.random.seed(None)
            np.random.shuffle(wn)
            w = wn[np.random.randint(0,wn.shape[0], size=batch_size), :, :]
        else:
            w = wn[i_w:i_w+batch_size,:,:]
        i_w += batch_size
        
        w_batch = []
        for each_w in w:
            # img_w = color.rgb2gray(each_w)
            img_w = (each_w - each_w.min()) / (each_w.max() - each_w.min())
            img_w = transform.resize(each_w, (w_hei, w_wid), mode='reflect')
            w_batch.append(img_w)
        w_batch = np.array(w_batch)
        w_batch = np.reshape(w_batch, [batch_size, w_hei, w_wid, 1])
        # print('w:',w_batch.shape, w_batch.max(), w_batch.min())
        
        yield ({'C': c_batch, 'W': w_batch}, \
              {'M': c_batch, 'W_prime': w_batch})


### layer / model
from keras.layers import Input, Conv2D, concatenate, Dense, Dropout, add
# GaussianNoise, GaussianDropout
from keras.models import Model
import keras.backend as K
# from keras.utils import multi_gpu_model

def conv_block(x, scale, prefix):  
    
    d = K.int_shape(x)
    d = d[-1]
    
    filters = 32
    
    ### path #1
    p1 = Conv2D(int(filters * scale), kernel_size=(1, 1), strides=1, activation='relu', \
                padding='same', name=prefix + 'path1_1x1_conv')(x)
    
    ### path #2
    p2 = Conv2D(int(filters * scale), kernel_size=(1, 1), strides=1, activation='relu', \
                padding='same', name=prefix + 'path2_1x1_conv')(x)
    p2 = Conv2D(int(filters * scale), kernel_size=(3, 3), strides=1, activation='relu', \
                padding='same', name=prefix + 'path2_3x3_conv')(p2)
    
    ### path #3
    p3 = Conv2D(int(filters * scale), kernel_size=(1, 1), strides=1, activation='relu', \
                padding='same', name=prefix + 'path3_1x1_conv')(x)
    p3 = Conv2D(int(filters * scale), kernel_size=(3, 3), strides=1, activation='relu', \
                padding='same', name=prefix + 'path3_3x3_conv1')(p3)
    p3 = Conv2D(int(filters * scale), kernel_size=(3, 3), strides=1, activation='relu', \
                padding='same', name=prefix + 'path3_3x3_conv2')(p3)
    
    pc = concatenate([p1, p2, p3], axis=-1, name=prefix + 'path_combine')
    
    ### res
    pr = Conv2D(d, kernel_size=(1, 1), strides=1, activation='relu', \
                padding='same', name=prefix + 'path_combine_conv')(pc)
    out = add([x, pr], name=prefix + 'block_output')

    return out

def main_model(in_w = (w_hei, w_wid, 1), in_c = (height, width, 3), scale=1):
    
    C = Input(shape=in_c, name='C')
    
    W = Input(shape=in_w, name='W')
    W1 = conv_block(W, scale, prefix='w_en_')
    
    G = concatenate([C,W1], axis=-1)
    x = conv_block(G, scale, prefix='em_en_')
    #x = conv_block(x, scale, prefix='em_de_')
    
    #M = Conv2D(3, kernel_size=(3, 3), padding='same', \
    #           strides=1, activation='sigmoid', name='M')(x)
    M = Dense(3,activation='sigmoid', name='M')(x)
    
    x = conv_block(M, scale=int(scale*2), prefix='ex_en_')
    # x = conv_block(x, scale=int(scale*2), prefix='ex_de_')
    
    W_prime = Conv2D(1, kernel_size=(3, 3), padding='same', \
              strides=1, activation='sigmoid', name='W_prime')(x)
              
    train_model = Model(inputs=[C,W], outputs=[M,W_prime])
    train_model.compile(optimizer='adam', \
                        loss='mae')
    train_model.summary()

    return train_model

def train(epoch=1):
    train_model = main_model(scale=1)
    
    history = train_model.fit_generator(generator=get_batch(batch_size=batch_size, train=1), \
                               steps_per_epoch=int(train_len / batch_size), \
                               epochs=epoch, \
                               validation_data=get_batch(batch_size=batch_size, train=0), \
                               validation_steps=int(test_len / batch_size), \
                              )
    
    train_model.save('/home/CVL1/Shaobo/Encode/0_Base_Model.h5')
    
    with open('/home/CVL1/Shaobo/Encode/history_0_whole.pkl', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


if __name__ == "__main__":
    
    train(epoch=5)
    print("===============")
    ### test data
    itr = get_batch(train=0)
    test = next(itr)
    img = test[0]['C']
    msk = test[0]['W']
    model = load_model('/home/CVL1/Shaobo/Encode/0_Base_Model.h5')
    
    [M,W_prime] = model.predict([img,msk])


    plt.figure(figsize=(60, 4))
    n = 32
    for i in range(n):
        ax = plt.subplot(4, n, i+1)
        plt.imshow(img[i].reshape(32, 32, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(4, n, i+1 + n)
        plt.imshow(M[i].reshape(32, 32, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(4, n, i +1 + 2*n)
        plt.imshow(msk[i].reshape(32, 32),cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(4, n, i +1 + 3*n)
        plt.imshow(W_prime[i].reshape(32, 32),cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    plt.savefig('/home/CVL1/Shaobo/Encode/10.jpg')

        
        
        
        
        