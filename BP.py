import math
import ast
import numpy as np
import matplotlib.pyplot as plt
import random

#produce a num in [a,b]
def random_number(a,b):
    return a+(b-a)*random.random()

#Generate a zero matrix with m rows and n columns
def matrix(m,n,fill=0.0):
    mat=[]
    for i in range(m):
        mat.append([fill]*n)
    return mat

def sigmoid(x):
    return 1.0/(1.0+math.exp(-x))

def derived_sigmoid(x):
    return x*(1-x)

class BPNeuralNetwork():
    def __init__(self,num_input,num_hidden,num_output):
        #the number of nodes in input layer、hidden layer and output layer
        self.input_num = num_input + 1  # include a bias node
        self.hidden_num = num_hidden+1
        self.output_num = num_output

        #all nodes in activate neural netwo
        self.input_nodes=[1.0]*self.input_num
        self.hidden_nodes=[1.0]*self.hidden_num
        self.output_nodes=[1.0]*self.output_num

        #generate a weight matrix
        self.input_weights=matrix(self.input_num,self.hidden_num)
        self.output_weights=matrix(self.hidden_num,self.output_num)

        #random activate
        for i in range(self.input_num):
            for j in range(self.hidden_num):
                self.input_weights[i][j]=random_number(-1.0,1.0)

        for i in range(self.hidden_num):
            for j in range(self.output_num):
                self.output_weights[i][j] = random_number(-1.0, 1.0)

    def predict(self,inputs):
        for i in range(self.input_num-1):
          #  print(type(self.input_nodes[i]),end="**float**")
            self.input_nodes[i]=inputs[i]
            #self.input_nodes[self.input_num]=1.0 初始化赋予的值

        for i in range(self.hidden_num):
            total=0.0 #total=sum(Xi*W1) i=1,2,…… input_num
            for j in range(self.input_num):
                 total+=self.input_nodes[j]*self.input_weights[j][i]
            #fresh the value of hidden layer nodes
            self.hidden_nodes[i]=sigmoid(total)# f(total)

        for i in range(self.output_num):
            total=0.0
            for j in range(self.hidden_num):
                total+=self.hidden_nodes[j]*self.output_weights[j][i]
            self.output_nodes[i]=sigmoid(total)
        return self.output_nodes

    def back_propagation(self,case,label,learn):
        self.predict(case)
        #calculate the error of output layer
        output_error=[0.0]*self.output_num
        for i in range(self.output_num):
            #最后一层，只需计算label与网络输出之间的差异，无需全联接，因此只有一层循环
            error=label[i]-self.output_nodes[i]
            output_error[i]=derived_sigmoid(self.output_nodes[i])*error   #????
        #calculate the error of hidden layer
        hidden_error=[0.0]*self.hidden_num
        for i  in range(self.hidden_num):
            error=0.0
            for j in range(self.output_num):
                #反向传播平摊误差
                error+=output_error[j]*self.output_weights[i][j]
            hidden_error[i]=derived_sigmoid(self.hidden_nodes[i])*error
        #fresh output_weights
        for i in range(self.hidden_num):
            for j in range(self.output_num):
                #累加在与继承上一次权重
                self.output_weights[i][j]+=learn*output_error[j]*self.hidden_nodes[i]
        #fresh input_weights
        for i in range(self.input_num):
            for j in range(self.hidden_num):
                # 累加在与继承上一次权重
                self.input_weights[i][j] += learn * hidden_error[j] * self.input_nodes[i]

        error=0.0
        for i in range(len(label)):
            error+=0.5*(label[i]-self.output_nodes[i])**2
        return error

    def train(self,cases,labels,times=100,learn=0.01):
        for i in range(times):
            if(i%1000==0):
                print("The number of training reaches %0.0f times"%i)
            error=0.0
            for j in range(len(labels)):
                label=labels[j]
                case=cases[j]
                error+=self.back_propagation(case,label,learn)
        print("Finished training！")
        pass
    def display_weight(self):
        print("input_weight[][]:")
        print(self.input_weights)
        print("output_weight[][]:")
        print(self.output_weights)

    def draw(self,labels,results):
        plt.title("Train Result")
        plt.xlabel("step=1.0")
        plt.ylabel("step=0.2")
        len_x=len(labels)
        x1 = np.array([i for i in range(0,len_x)])
        y1 = np.array([v for v in labels])
        y2 = np.array([v for v in results])
        plt.scatter(x1, y1, color="green")
        #draw result scatter
        plt.scatter(x1, y2, color="red")
        plt.show()

    def test(self):
        print("Start to read cases set:")
        f1=open(r"cases.txt","r")
        cases = ast.literal_eval(f1.read())
        print("The cases set has been read！")
        print("Start to read labels set:")
        f2=open(r"labels.txt","r")
        labels=ast.literal_eval(f2.read())
        print("The labels set has been read！")
        f1.close();f2.close()
        self.train(cases,labels,100000,0.05)
        self.display_weight()
        #Test set input feedback neural network to get output(result)
        results=[]
        for case in cases:
            re=self.predict(case)
            results.append(re[:])#re中所有元素作为一个列表添加至result
            #Writing like this(results.append(re[:])) is wrong
        self.draw(labels,results)

if __name__=="__main__":
    nn=BPNeuralNetwork(3,5,1)
    nn.test()
