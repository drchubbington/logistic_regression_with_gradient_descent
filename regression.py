'''cell 1'''
import matplotlib.pyplot as plt
import numpy as np
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def sigmoid_list(x):
    a = []
    for item in x:
        a.append(1/(1+math.exp(-item)))
    return a

#data
func = {
    0:1,
    10:1,
    -5:1,
    -6:0,
    -10:0,
    -4.8:1,
    -2:1,
    -5.5:1,
    -5.6:0,
    8:1,
    -20:0,
    -5.1:0,
    -5.8:0,
    -15:0,
    4:1,
}

#split for graphing purposes
x = []
y = []
for x_val in func:
    x.append(x_val)
    y.append(func[x_val])
#plot data
plt.plot(x, y, ".r")

'''cell 2'''
#train model
#regression line: f(x)=b0x+b1 --> mx+b
#guess: h(x)=1/(1+e^(-f(x)))
#cost: J=-ylog(h(x))-(1-y)log(1-h(x))
lr=.003
m=0
b=0
#1000 epochs, each epoch loops through data once
for i in range(1000):
    error = 0
    for x_val in func:
        #in case of overflow error (e^x too large)
        try:
            error += -func[x_val]*math.log(sigmoid(m*x_val+b))-(1-func[x_val])*math.log(1-sigmoid(m*x_val+b))
        except:
            pass

        #derivative of error function with respect to m
        d_error_m = -x_val*((math.exp(b)*func[x_val]-math.exp(b))*math.exp(x_val*m)+func[x_val])/(1+math.exp(m*x_val+b))
        
        #derivative of error function with respect to b
        d_error_b = -((math.exp(m*x_val)*func[x_val]-math.exp(m*x_val))*math.exp(b)+func[x_val])/(1+math.exp(m*x_val+b))
        
        m -= lr*d_error_m
        b -= lr*d_error_b
    print(error, m, b, d_error_m, d_error_b)
print(m, b)

'''cell 3'''
x_ = np.linspace(-20,10,1000)
y_ = sigmoid_list(m*x_+b)
plt.plot(x_, y_, '-r')
plt.scatter(x, y)
