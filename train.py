import random
def random_num(a,b):
    return a+(b-a)*random.random()

def produce_data(cases,labels):
    for i in range(20):
        x=random_num(0.0,1.0)
        y=random_num(0.0,1.0)
        z=random_num(0.0,1.0)
        cases.append([x,y,z])
        if(x+y>z and x+z>y and y+z>x):
            labels.append([1])
        else:
            labels.append([0])
    f1=open(r"cases.txt","w")
    f2=open(r"labels.txt","w")
    f1.write(str(cases))
    f2.write(str(labels))
    f1.close()
    f2.close()

if __name__=="__main__":
    cases = []
    labels = []
    tests = []
    produce_data(cases,labels)
