# 对数据进行基本处理
import scipy.io as sio
import numpy as np
import os
import tensorflow as tf
from sklearn import preprocessing  # 0-1编码
from sklearn.model_selection import StratifiedShuffleSplit  # 随机划分，保证每一类比例相同

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def awgn(x, snr, seed=7):
    '''
    加入高斯白噪声 Additive White Gaussian Noise
    :param x: 原始信号
    :param snr: 信噪比
    :return: 加入噪声后的信号
    '''
    np.random.seed(seed)  # 设置随机种子
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / len(x)
    npower = xpower / snr
    noise = np.random.randn(len(x)) * np.sqrt(npower)
    return x + noise


def prepro(d_path, length=864, number=1000, normal=True, rate=[0.5, 0.25, 0.25], enc=True, enc_step=28, nose=False):
    """对数据进行预处理,返回train_X, train_Y, valid_X, valid_Y, test_X, test_Y样本.
    :param d_path: 源数据地址
    :param length: 信号长度，默认2个信号周期，864
    :param number: 每种信号个数,总共10类,默认每个类别1000个数据
    :param normal: 是否标准化.True,Fales.默认True
    :param rate: 训练集/验证集/测试集比例.默认[0.5,0.25,0.25],相加要等于1
    :param enc: 训练集、验证集是否采用数据增强.Bool,默认True
    :param enc_step: 增强数据集采样顺延间隔
    :return: Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y
    ```
    import preprocess.preprocess_nonoise as pre
    train_X, train_Y, valid_X, valid_Y, test_X, test_Y = pre.prepro(d_path=path,
                                                                    length=864,
                                                                    number=1000,
                                                                    normal=False,
                                                                    rate=[0.5, 0.25, 0.25],
                                                                    enc=True,
                                                                    enc_step=28)
    ```
    """

    # 获得该文件夹下所有.mat文件名
    firstfilenames = os.listdir(d_path)

    def capture(original_path):
        """读取mat文件，返回字典
        :param original_path: 读取路径
        :return: 数据字典
        """
        files = {}
        errortype = 0

        # for firstfilename in firstfilenames:
        #     first_path = os.path.join(original_path, firstfilename)
        #     #print("First Path:" + first_path)
        #     secondaryfilenames = os.listdir(first_path)
        #     for secondaryfilename in secondaryfilenames:
        #         second_path = os.path.join(first_path, secondaryfilename)
        #         #print("Second Path:" + second_path)
        #         thirdfilenames = os.listdir(second_path)
        #
        #         if(secondaryfilename=='Outer Race'):
        #             for thirdfilename in thirdfilenames:
        #                 third_path = os.path.join(second_path, thirdfilename)
        #                 #print("Third Path:" + third_path)
        #                 fourthfilenames = os.listdir(third_path)
        #                 #print(fourthfilenames)
        #                 for fourthfilename in fourthfilenames:
        #                     fourth_path = os.path.join(third_path, fourthfilename)
        #                    # print("Fourth Path:" + fourth_path)
        #                     filenames = os.listdir(fourth_path)
        #                     for filename in filenames:
        #                         file_path = os.path.join(fourth_path, filename)
        #                         #print(file_path)
        #                         file = sio.loadmat(file_path)
        #                         file_keys = file.keys()  # 获取每个mat文件中所有变量名
        #                         for key in file_keys:
        #                             if 'DE' in key:  # DE应该是某一测的振动信号，大概率是电机驱动侧的
        #                                 files[errortype] = file[key].ravel()
        #                                 errortype += 1
        #         else:
        #             for thirdfilename in thirdfilenames:
        #                 third_path = os.path.join(second_path, thirdfilename)
        #                 #print("Third Path:" + third_path)
        #                 filenames = os.listdir(third_path)
        #                 for filename in filenames:
        #                     file_path = os.path.join(third_path, filename)
        #                     #print(file_path)
        #                     file = sio.loadmat(file_path)
        #                     file_keys = file.keys()  # 获取每个mat文件中所有变量名
        #                     for key in file_keys:
        #                         if 'DE' in key:  # DE应该是某一测的振动信号，大概率是电机驱动侧的
        #                             files[errortype] = file[key].ravel()
        #                             errortype += 1

        filenames = os.listdir(original_path)
        for filename in filenames:
            file_path = os.path.join(original_path, filename)
            file = sio.loadmat(file_path)
            file_keys = file.keys()  # 获取每个mat文件中所有变量名
            for key in file_keys:
                if 'DE' in key:  # DE应该是某一测的振动信号，大概率是电机驱动侧的
                    files[errortype] = file[key].ravel()
                    errortype += 1

        print(errortype)
        return files

    def slice_enc(data, slice_rate=rate[1] + rate[2]):
        """将数据切分为前面多少比例，后面多少比例.
        :param data: 单挑数据
        :param slice_rate: 验证集以及测试集所占的比例
        :return: 切分好的数据
        """
        keys = data.keys()
        Train_Samples = {}
        Test_Samples = {}
        for i in keys:
            slice_data = data[i]
            if nose:
                slice_data = awgn(slice_data, 5)
            all_lenght = len(slice_data)
            end_index = int(all_lenght * (1 - slice_rate))
            samp_train = int(number * (1 - slice_rate))
            Train_sample = []
            Test_Sample = []
            if enc:
                enc_time = length // enc_step
                samp_step = 0  # 用来计数Train采样次数
                for j in range(samp_train):
                    random_start = np.random.randint(low=0, high=(end_index - 2 * length))
                    label = 0
                    for h in range(enc_time):
                        samp_step += 1
                        random_start += enc_step
                        sample = slice_data[random_start: random_start + length]
                        Train_sample.append(sample)
                        if samp_step == samp_train:
                            label = 1
                            break
                    if label:
                        break
            else:
                for j in range(samp_train):
                    random_start = np.random.randint(low=0, high=(end_index - length))
                    sample = slice_data[random_start:random_start + length]
                    Train_sample.append(sample)

            # 抓取测试数据
            for h in range(number - samp_train):
                random_start = np.random.randint(low=end_index, high=(all_lenght - length))
                sample = slice_data[random_start:random_start + length]
                Test_Sample.append(sample)
            Train_Samples[i] = Train_sample
            Test_Samples[i] = Test_Sample
        return Train_Samples, Test_Samples

    # 仅抽样完成，打标签
    def add_labels(train_test):
        X = []
        Y = []
        label = 0
        for i in data.keys():
            x = train_test[i]
            X += x
            lenx = len(x)
            Y += [label] * lenx
            label += 1
        return X, Y

    # one-hot编码
    def one_hot(Train_Y, Test_Y):
        Train_Y = np.array(Train_Y).reshape([-1, 1])
        Test_Y = np.array(Test_Y).reshape([-1, 1])
        Encoder = preprocessing.OneHotEncoder()
        Encoder.fit(Train_Y)
        Train_Y = Encoder.transform(Train_Y).toarray()
        Test_Y = Encoder.transform(Test_Y).toarray()
        Train_Y = np.asarray(Train_Y, dtype=np.int32)
        Test_Y = np.asarray(Test_Y, dtype=np.int32)
        return Train_Y, Test_Y

    def scalar_stand(Train_X, Test_X):
        # 用训练集标准差标准化训练集以及测试集
        scalar = preprocessing.StandardScaler().fit(Train_X)
        Train_X = scalar.transform(Train_X)
        Test_X = scalar.transform(Test_X)
        return Train_X, Test_X

    def valid_test_slice(Test_X, Test_Y):
        test_size = rate[2] / (rate[1] + rate[2])
        ss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
        for train_index, test_index in ss.split(Test_X, Test_Y):
            X_valid, X_test = Test_X[train_index], Test_X[test_index]
            Y_valid, Y_test = Test_Y[train_index], Test_Y[test_index]
            return X_valid, Y_valid, X_test, Y_test

    # 从所有.mat文件中读取出数据的字典
    data = capture(original_path=d_path)

    # 将数据切分为训练集、测试集
    train, test = slice_enc(data)
    # 为训练集制作标签，返回X，Y
    Train_X, Train_Y = add_labels(train)
    # 为测试集制作标签，返回X，Y
    Test_X, Test_Y = add_labels(test)
    # 为训练集Y/测试集One-hot标签
    Train_Y, Test_Y = one_hot(Train_Y, Test_Y)
    # 训练数据/测试数据 是否标准化.
    if normal:
        Train_X, Test_X = scalar_stand(Train_X, Test_X)
    else:
        # 需要做一个数据转换，转换成np格式.
        Train_X = np.asarray(Train_X)
        Test_X = np.asarray(Test_X)
    # 将测试集切分为验证集合和测试集.
    Valid_X, Valid_Y, Test_X, Test_Y = valid_test_slice(Test_X, Test_Y)
    return Train_X, Train_Y, Test_X, Test_Y, Valid_X, Valid_Y


if __name__ == "__main__":
    path = r'data'

    train_X, train_Y, valid_X, valid_Y, test_X, test_Y = prepro(d_path=path,
                                                                length=864,
                                                                number=1000,
                                                                normal=True,
                                                                rate=[0.5, 0.25, 0.25],
                                                                enc=True,
                                                                enc_step=28)

    train_X = tf.expand_dims(train_X, axis=2)
    #train_Y = tf.expand_dims(train_Y, axis=2)
    test_X = tf.expand_dims(test_X, axis=2)
    #test_Y = tf.expand_dims(test_Y, axis=2)
    valid_X = tf.expand_dims(valid_X,axis=2)
    print(train_X.shape, train_Y.shape)
    print(test_X.shape, test_Y.shape)
    print(valid_X.shape, valid_Y.shape)
