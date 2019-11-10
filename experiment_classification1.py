import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import copy
from numpy import genfromtxt

def load_data():
    FILE_NAME = './Data/forestfires_withoutday.csv'

    datas = genfromtxt(FILE_NAME,delimiter=',')

    X_data = datas[:,:-1]
    Y_data = datas[:,-1]
    y_onehot = []
    for y in Y_data:

        if y == 0:
            li = [1,0]
        else:
            li = [0,1]
        y_onehot.append(li)
    return X_data, np.array(y_onehot,dtype=np.float32)

class Deep_neural_network():
    def __init__(self,learning_rate, n_input, hidden_nums,batch_size):
        #하이퍼 파라미터 저장
        self.batch_size=batch_size
        self.X = tf.placeholder(tf.float32, [None, n_input])
        self.Y = tf.placeholder(tf.float32, [None, 2])
        #self.Real_x = tf.placeholder(tf.float32, [None, 1])
        #self.Real_y = tf.placeholder(tf.float32, [None, 1])
        #self.Real_z = tf.placeholder(tf.float32, [None, 1])

        #신경망 변수 생성
        before_layer_num = n_input
        self.w = list()
        self.b = list()
        #init encoder variable
        for n in range(len(hidden_nums)):
            self.w.append(tf.Variable(tf.random_normal([before_layer_num, hidden_nums[n]])));
            self.b.append( tf.Variable(tf.random_normal([hidden_nums[n]])));
            before_layer_num = hidden_nums[n]

        # init hidden layer
        self.hidden = list()
        self.hidden.append(tf.nn.leaky_relu(tf.add(tf.matmul(self.X, self.w[0]), self.b[0])))
        for n in range(1,len(self.w)):
            self.hidden.append( tf.nn.leaky_relu(tf.add(tf.matmul(self.hidden[n-1], self.w[n]), self.b[n])) )

        self.output_w = tf.Variable(tf.random_normal([before_layer_num, 2]))
        #self.output_x_w = tf.Variable(tf.random_normal([before_layer_num, 1], stddev=0.1))
        #self.output_y_w = tf.Variable(tf.random_normal([before_layer_num, 1], stddev=0.1))
        #self.output_z_w = tf.Variable(tf.random_normal([before_layer_num, 1], stddev=0.1))
        self.output_b = tf.Variable(tf.random_normal([2]))
        #self.output_x_b = tf.Variable(tf.random_normal([1], stddev=0.1))
        #self.output_y_b = tf.Variable(tf.random_normal([1], stddev=0.1))
        #self.output_z_b = tf.Variable(tf.random_normal([1], stddev=0.1))
        #self.outputs = [tf.nn.leaky_relu(tf.add(tf.matmul(self.hidden[n], self.output_x_w), self.output_x_b)), tf.nn.leaky_relu(tf.add(tf.matmul(self.hidden[n], self.output_y_w), self.output_y_b))]
        self.outputs = tf.add(tf.matmul(self.hidden[n], self.output_w), self.output_b)
        #self.outputs=[ tf.nn.leaky_relu(tf.add(tf.matmul(self.hidden[n],self.output_x_w),self.output_x_b)) ,  tf.nn.leaky_relu(tf.add(tf.matmul(self.hidden[n],self.output_y_w),self.output_y_b)) , tf.nn.leaky_relu(tf.add(tf.matmul(self.hidden[n],self.output_z_w),self.output_z_b))  ]

        # cost

        #self.cost = tf.losses.mean_squared_error(self.Real_x, self.outputs[0])+ tf.losses.mean_squared_error(self.Real_y, self.outputs[1])
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( logits=self.outputs,labels=self.Y))
        #self.cost = tf.losses.mean_squared_error([self.Real_x,self.Real_y], [self.outputs[0],self.outputs[1]])#tf.reduce_mean((self.outputs[0] - self.Real_x)**2 +(self.outputs[1] - self.Real_y)**2 +(self.outputs[2] - self.Real_z)**2)
        #self.cost = tf.losses.mean_squared_error([self.Real_x,self.Real_y,self.Real_z], [self.outputs[0],self.outputs[1],self.outputs[2]])#tf.reduce_mean((self.outputs[0] - self.Real_x)**2 +(self.outputs[1] - self.Real_y)**2 +(self.outputs[2] - self.Real_z)**2)

        self.correct_pred = tf.equal(tf.argmax(self.outputs, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        # optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)
    #신경망 훈련
    def run_train_sess(self, sess, X_train, Y_train):
        #batch
        input_x = copy.deepcopy(X_train)
        input_y = copy.deepcopy(Y_train)
        inputs = np.c_[input_x.reshape(len(input_x),-1), input_y.reshape(len(input_y),-1)]
        np.random.shuffle(inputs);

        n = int(inputs.shape[0] / self.batch_size)
        cost_sum = 0
        for i in range(n):
            if i==n-1:
                input = inputs[self.batch_size*i:]
            else:
                input = inputs[self.batch_size*i:self.batch_size*i+self.batch_size]
            #noise_xs = self.add_noise_to_data(copy.deepcopy(X_train), self.noise_probability)
            x = input[:,:input_x.shape[1]]
            y = input[:, input_x.shape[1]:]

            #y = input[:,input_x.shape[1]:]
            #y_x = y[:, 0].reshape(len(y[:, 0]), -1)
            #y_y = y[:, 1].reshape(len(y[:, 1]), -1)
            #y_z= y[:,2].reshape(len(y[:,2]),-1)
            #_, cost_val = sess.run([self.optimizer, self.cost], feed_dict={self.X: x, self.Real_x: y_x,self.Real_y:y_y, self.Real_z:y_z})
            sess.run(self.optimizer, feed_dict={self.X: x, self.Y: y})
            # _, cost_val = sess.run([optimizer, cost], feed_dict={X: X_train, output_true: X_train})

        return sess.run([self.cost, self.accuracy],feed_dict={self.X:x,self.Y:y})
        #print("Epoch:", "%05d" % (epoch + 1), 'Avg. cost = ', '{:.4f}'.format(cost_val))

    #성능평가
    def test_accuracy(self,sess,X_data,Y_data):

        #cost_val = sess.run(self.cost, feed_dict={self.X: X_data, self.Real: Y_data[:,0:2]})
        output = sess.run(self.accuracy, feed_dict={self.X: X_data, self.Y:Y_data })
        return output

if __name__ == "__main__":
    X_data, Y_data = load_data()
    print(X_data, Y_data)
    #하이퍼파라미터 설정
    learning_rate = 0.01
    #training_epoch = 50
    batch_size = int(X_data.shape[0]/20)
    #hidden_layer_neural_num = [100]
    final_error = 0
    sess = tf.Session()
    while final_error < 0.7:
        training_epoch = int(int(np.random.randint(low=100, high = 100000) / 10)*10)
        hidden_layer_neural_num = [np.random.randint(low=11,high=50),np.random.randint(low=11,high=50)]
        dae = Deep_neural_network(learning_rate,X_data.shape[1],hidden_layer_neural_num,batch_size)
        print('training_epoch : ', training_epoch, 'hidden num : ', hidden_layer_neural_num)
        # batch
        input_x = copy.deepcopy(X_data)
        input_y = copy.deepcopy(Y_data)
        inputs = np.c_[input_x.reshape(len(input_x), -1), input_y.reshape(len(input_y), -1)]
        np.random.shuffle(inputs);

        train = inputs[:int(inputs.shape[0]*0.95)]
        x_train = train[:,:input_x.shape[1]]
        y_train = train[:, -2:]

        test = inputs[int(inputs.shape[0]*0.95):]
        x_test = test[:,:input_x.shape[1]]
        y_test = test[:, -2:]

        sess.run(tf.global_variables_initializer())
        for epoch in range(training_epoch):
            loss,acu = dae.run_train_sess(sess,x_train,y_train)
            if epoch % 1000 == 0:
                print('Epoch:',epoch)
            #print('Epoch:', "%03d" % (epoch+1),'loss : ' ,loss, 'accuracy : ', acu )
        final_error = dae.test_accuracy(sess,x_test,y_test)
        print('error : ', final_error)