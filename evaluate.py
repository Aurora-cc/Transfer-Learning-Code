import XJTU_dataprocess
import tensorflow as tf
from keras import models

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
                                                                rate=[0.05,0.05,0.9],
                                                                enc=True,
                                                                enc_step=28)

train_X = tf.expand_dims(train_X,axis=2)
test_X = tf.expand_dims(test_X,axis=2)
valid_X = tf.expand_dims(valid_X,axis=2)
fine_tune_model = models.load_model('fine_tune_CNN model.h5')
fine_tune_model.summary()
loss,accuracy = fine_tune_model.evaluate(test_X,test_Y)
print('\ntest loss', loss)
print('accuracy', accuracy)