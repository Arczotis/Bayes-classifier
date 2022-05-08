from numpy import *
import numpy as np
import os
import cv2


# 预处理
class pretreatment:
    # 将数据集分为训练集与测试集
    def train_test_split(self, X, y, ratio, seed):
        assert X.shape[0] == y.shape[0]
        assert 0.0 <= ratio <= 1.0

        if seed:
            np.random.seed(seed)

        data = np.hstack((X, y.reshape(-1, 1)))
        np.random.shuffle(data)

        train = np.array(data[:int(len(y) * ratio)])
        test = np.array(data[int(len(y) * ratio):])

        return train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]

    # 将图像二值化
    def normalize(self, X):
        XX = []
        for i in X:
            # 读取图像
            # print i
            image = cv2.imread(i)
            # 图像像素大小一致
            img = cv2.resize(image, (64, 64),
                             interpolation=cv2.INTER_CUBIC)
            # 计算图像直方图并存储至X数组
            # hist = cv2.calcHist([img], [0, 1], None,[256, 256], [0.0, 255.0, 0.0, 255.0])
            hist = img
            XX.append(((hist / 255).flatten()))
        XX = np.array(XX)
        for i in range(XX.shape[0]):
            for j in range(XX.shape[1]):
                if XX[i][j] >= 0.5:
                    XX[i][j] = 1
                else:
                    XX[i][j] = 0
        return XX

    # 计算预测准确率
    def cal_accuracy(self, test_y, predict_y):
        m = test_y.shape[0]
        errorCount = 0.0
        lable_dic = {'0': '八大关', '1': '党史纪念馆', '2': '五月的风', '3': '莱西革命烈士公园'}
        num_dic = {'0': 0, '1': 0, '2': 0, '3': 0}
        error_dict = {'0': 0, '1': 0, '2': 0, '3': 0}
        for i in range(m):
            num_dic[str(test_y[i])] += 1
            if test_y[i] != predict_y[i]:
                errorCount += 1
                error_dict[str(test_y[i])] = error_dict[str(test_y[i])] + 1
        for i in range(4):
            print(lable_dic[str(i)] + '的预测准确率为', 1 - (error_dict[str(i)] / num_dic[str(i)]))
            print('对'+lable_dic[str(i)] + '预测时，发生错误的次数为', error_dict[str(i)])
            print('')
        accuracy = 1.0 - float(errorCount) / m
        return accuracy


# 分类器实现
class bayes_classifier:
    # （1）计算先验概率及条件概率
    def train_model(self, train_x, train_y, classNum):  # classNum是指有几个类别，这里的train_x是已经二值化，
        m = train_x.shape[0]
        n = train_x.shape[1]
        # prior_probability=np.zeros(n)#先验概率
        prior_probability = np.zeros(classNum)  # 先验概率
        conditional_probability = np.zeros((classNum, n, 2))  # 条件概率
        # 计算先验概率和条件概率
        for i in range(m):  # m是图片数量
            img = train_x[i]  # img是第i个图片，是1*n的行向量
            label = train_y[i]  # label是第i个图片对应的label
            prior_probability[label] += 1  # 统计label类的label数量(p(Y=ck)，下标用来存放label,prior_probability[label]除以m就是某个类的先验概率
            for j in range(n):  # n是特征数
                temp = img[j].astype(int)  # img[j]是0.0，放到下标去会显示错误，只能用整数

                conditional_probability[label][j][temp] += 1

                # conditional_probability[label][j][img[j]]+=1#统计的是类为label的，在每个列中为1或者0的行数为多少，img[j]的值要么就是0要么就是1，计算条件概率

        # 将概率归到[1,10001]
        for i in range(classNum):
            for j in range(n):
                # 经过二值化的图像只有0，1两种取值
                pix_0 = conditional_probability[i][j][0]
                pix_1 = conditional_probability[i][j][1]

                # 计算0，1像素点对应的条件概率
                probability_0 = (float(pix_0) / float(pix_0 + pix_1)) * 10000 + 1
                probability_1 = (float(pix_1) / float(pix_0 + pix_1)) * 10000 + 1

                conditional_probability[i][j][0] = probability_0
                conditional_probability[i][j][1] = probability_1
        return prior_probability, conditional_probability

    # （2）对给定的x，计算先验概率和条件概率的乘积
    def cal_probability(self, img, label, prior_probability, conditional_probability):
        probability = int(prior_probability[label])  # 先验概率
        n = img.shape[0]
        # print(n)
        for i in range(n):  # n为特征数
            probability *= int(conditional_probability[label][i][img[i].astype(int)])

        return probability

    # 确定实例x的类，相当于argmax
    def predict(self, test_x, test_y, prior_probability, conditional_probability,
                classNum):  # 传进来的test_x或者是train_x都是二值化后的
        predict_y = []
        m = test_x.shape[0]
        n = test_x.shape[1]
        for i in range(m):
            img = np.array(test_x[i])  # img已经是二值化以后的列向量
            label = test_y[i]
            max_label = 0
            max_probability = bayes_classifier.cal_probability(self, img, 0, prior_probability, conditional_probability)
            for j in range(1, classNum):  # 从下标为1开始，因为初始值是下标为0
                probability = bayes_classifier.cal_probability(self, img, j, prior_probability, conditional_probability)
                if max_probability < probability:
                    max_probability = probability
                    max_label = j
            predict_y.append(max_label)  # 用来记录每行最大概率的label
        return np.array(predict_y)


# 主程序
if __name__ == '__main__':
    classNum = 4  # 类别数
    lable_dic = {'0': '八大关', '1': '党史纪念馆', '2': '五月的风', '3': '莱西革命烈士公园'}
    # 定义类
    p = pretreatment()
    bayes = bayes_classifier()

    # 读取数据
    X = []  # 定义图像名称
    Y = []  # 定义图像分类类标
    for i in range(0, 4):
        # 遍历文件夹，读取图片
        for f in os.listdir("data/%s" % i):
            # 获取图像名称
            X.append("data//" + str(i) + "//" + str(f))
            # 获取图像类标即为文件夹名称
            Y.append(int(i))

    X = np.array(X)
    Y = np.array(Y)
    # 将所有图像二值化
    X = p.normalize(X)
    # 随机率为100% 选取其中的70%作为训练集
    X_train, y_train, X_test, y_test = p.train_test_split(X, Y, 0.7, 1)
    print('训练集长度为',len(X_train),'测试集长度为' , len(X_test))
    X_train = X_train.astype(np.int_)
    X_test = X_test.astype(np.int_)
    y_train = y_train.astype(np.int_)
    y_test = y_test.astype(np.int_)
    '''print(X_train)
    print(y_train)
    print(X_test)
    print(y_test)'''
    # 生成训练模型
    prior_probability, conditional_probability = bayes.train_model(X_train, y_train, classNum)
    for i in range(classNum):
        print('标签为' + lable_dic[str(i)] + '的图片的数量为', int(prior_probability[i]))  # 输出一下每个标签的总共数量
    # 进行预测
    print('')
    y_predict = bayes.predict(X_test, y_test, prior_probability, conditional_probability, classNum)
    # 预测结果
    acc = p.cal_accuracy(y_test, y_predict)
    print("平均正确率", acc)
