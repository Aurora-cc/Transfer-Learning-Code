import XJTU_dataprocess
import tensorflow as tf
from keras import models
from keras.layers import *
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint
import pandas as pd
from matplotlib import pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # 设置 GPU 显存占用为按需分配，增长式
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    # 异常处理
    print(e)

train_X, train_Y, valid_X,valid_Y,test_X, test_Y=XJTU_dataprocess.prepro(
                                                                # d_path=path,
                                                                length=1024,
                                                                number=2000,
                                                                normal=True,
                                                                rate=[0.2,0.4,0.4],
                                                                enc=True,
                                                                enc_step=28)
train_X = tf.expand_dims(train_X, axis=2)
test_X = tf.expand_dims(test_X, axis=2)
valid_X = tf.expand_dims(valid_X, axis=2)
model = models.load_model('weights.best.hdf5')
model.summary()
model.pop()
model.add(Dense(5, activation='softmax', name='dense_output'))
model.summary()
# # 测试正确率
# loss,accuracy = model.evaluate(test_X,test_Y)
# print('\ntest loss',loss)
# print('accuracy',accuracy)

#冻结微调
model.trainable = True
fine_tune_at = 6
for layer in model.layers[:fine_tune_at]:
    layer.trainable = False
    pass

opt=Adam(lr=0.01, beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=0.0)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy',factor=0.5,patience=10,verbose=1,mode='max',epsilon=0.0001)
filepath = 'fine_tune_CNN model.h5'
checkpoint = ModelCheckpoint(filepath,monitor='val_accuracy',verbose=1,save_best_only=True,mode='max')
callbacks_list = [reduce_lr,checkpoint]
model.compile(
    optimizer=opt,
    loss='categorical_crossentropy',
    metrics=['accuracy'])# 评价函数，比较真实标签值和模型预测值

history = model.fit(
    train_X,
    train_Y,
    batch_size=32,
    epochs=150,
    validation_data=(valid_X, valid_Y),
    shuffle=True,
    verbose=1,
    callbacks=callbacks_list
)
model.summary()
model.save('fine_tune_CNN model.h5')

data = {}
data['accuracy']=history.history['accuracy']
data['val_accuracy']=history.history['val_accuracy']
data['loss']=history.history['loss']
data['val_loss']=history.history['val_loss']



pd.DataFrame(data).plot(figsize=(8, 5))#图片大小（宽，高）
plt.grid(True)#图片是否有网格
plt.axis([0, 150, 0, 1.5])
plt.show()