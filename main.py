#
# # import keras
# # from keras.datasets import mnist
# # (train_images,train_labels),(test_images,test_labels) = mnist.load_data() #加载数据
# # print('shape of train images is ',train_images.shape)
# # print('shape of train labels is ',train_labels.shape)
# # print('train labels is ',train_labels)
# # print('shape of test images is ',test_images.shape)
# # print('shape of test labels is',test_labels.shape)
# # print('test labels is',test_labels)
import pandas as pd
from matplotlib import pyplot as plt
import dataprocess
import tensorflow as tf
from keras.layers import *
from keras import Sequential, backend, regularizers
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  print(True)
  try:
    # 设置 GPU 显存占用为按需分配，增长式
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    # 异常处理
    print(e)

#导入数据库
path = r"data"
train_X, train_Y, valid_X, valid_Y, test_X, test_Y=dataprocess.prepro(
                                                                d_path=path,
                                                                length=1024,
                                                                number=2000,
                                                                normal=True,
                                                                rate=[0.7, 0.2, 0.1],
                                                                enc=True,
                                                                enc_step=28)
train_X = tf.expand_dims(train_X, axis=2)
#train_Y = tf.expand_dims(train_Y,axis=2)
test_X = tf.expand_dims(test_X, axis=2)
#test_Y = tf.expand_dims(test_Y,axis=2)

#搭建卷积神经网络模型
cnn_network = Sequential()
cnn_network.add(Conv1D(4,kernel_size=15,strides=1,padding='same',activation='relu',input_shape=(1024, 1),kernel_regularizer=regularizers.l2(0.01)))
cnn_network.add(MaxPool1D(pool_size=2,strides=2))
cnn_network.add(Conv1D(4,kernel_size=15,strides=1,padding='same',activation='relu',kernel_regularizer=regularizers.l2(0.01)))
cnn_network.add(MaxPool1D(pool_size=2,strides=2))
cnn_network.add(Conv1D(4,kernel_size=15,strides=1,padding='same',activation='relu',kernel_regularizer=regularizers.l2(0.01)))
cnn_network.add(MaxPool1D(pool_size=2,strides=2))
cnn_network.add(Conv1D(4,kernel_size=15,strides=1,padding='same',activation='relu',kernel_regularizer=regularizers.l2(0.01)))
cnn_network.add(MaxPool1D(pool_size=2,strides=2))
cnn_network.add(Conv1D(4,kernel_size=15,strides=1,padding='same',activation='relu',kernel_regularizer=regularizers.l2(0.01)))
cnn_network.add(MaxPool1D(pool_size=2,strides=2))
cnn_network.add(Flatten())
cnn_network.add(Dropout(0.5))
cnn_network.add(Dense(128,activation='relu',kernel_regularizer=regularizers.l2(0.01)))
cnn_network.add(Dropout(0.5))
cnn_network.add(Dense(64,activation='relu',kernel_regularizer=regularizers.l2(0.01)))
cnn_network.add(Dropout(0.5))
# cnn_network.add(GlobalAveragePooling1D())
cnn_network.add(Dense(10,activation='softmax'))
# cnn_network.cuda()

#优化器和损失函数
opt = Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=0.0)
'''
learning_rate：float> = 0.学习率
beta_1：float，0 <beta <1。一般接近1。一阶矩估计的指数衰减率
beta_2：float，0 <beta <1。一般接近1。二阶矩估计的指数衰减率
epsilon：float> = 0,模糊因子。如果None，默认为K.epsilon()。该参数是非常小的数，其为了防止在实现中除以零
decay：float> = 0,每次更新时学习率下降
'''
cnn_network.compile(
    optimizer=opt,
    #loss=tf.keras.losses.CategoricalCrossentropy,#损失函数
    loss='categorical_crossentropy',
    metrics=['accuracy'])# 评价函数，比较真实标签值和模型预测值
# 设置动态学习率
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',factor=0.1,patience=10,verbose=1,mode='max',epsilon=0.0001)
# 保存最佳模型
filepath = 'weights.best.hdf5'
checkpoint = ModelCheckpoint(filepath,monitor='val_accuracy',verbose=1,save_best_only=True,mode='max')
callbacks_list = [reduce_lr, checkpoint]
#显示网络参数
cnn_network.summary()#打印神经网络结构，统计参数数目
# 训练集和验证集送入模型框架，进行训练。
# fit函数会返回每一个epoch后的训练集准确率、损失和验证集准确率和损失，并保存在history中，具体代码如下
history = cnn_network.fit(train_X,
                          train_Y,
                          batch_size = 64,#批大小
                          epochs=100,#迭代数
                          validation_data=(test_X, test_Y),#用来评估损失，以及在每轮结束时的任何模型度量指标
                          shuffle=True,
                          verbose=1,
                          callbacks=callbacks_list
                          )

#将测试集和训练集准确率和损失放入data字典中。
data = {}
data['accuracy']=history.history['accuracy']
data['val_accuracy']=history.history['val_accuracy']
data['loss']=history.history['loss']
data['val_loss']=history.history['val_loss']



pd.DataFrame(data).plot(figsize=(8, 5))#图片大小（宽，高）
plt.grid(True)#图片是否有网格
plt.axis([0, 100, 0, 1.5])
plt.show()