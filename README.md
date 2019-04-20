Experiment Report Lab 1——Backpropagation
===

### 1. **Introduction**
In this lab, we design an easy neural network by forward pass and back propagation to update weights and bias.
And using final parameters to fit two datasets( linear and XOR).
The result of this lab looks like good, no matter the loss or the accuracy.
* #### *Sigmoid functions*

    * *Introduction of Sigmoid Function*
    The sigmoid function gives an ‘S’ shaped curve, which is also called logistic function. Transfer the x whose range is $({ - \infty }$ , ${ + \infty })$ to y whose range is $（0，1）$.The function helps a lot to solve the classfication problem.
    Sigmoid function has been used as the activation function of artificial neurons. Sigmoid curve is also similar as cumulative distribution functions (which go from 0 to 1) in statistics. So sigmoid function plays an important role in Deep Learing research and study.
    ==$Sigmoid(x)=\frac{1}{1+e^{-x}}$==
    ![](https://i.imgur.com/xWtwRVF.png)

    * *Classificion of Sigmoid Function*
    Our current prediction function returns a probability score between 0 and 1. In order to map this to a discrete class (true/false, cat/dog), we select a threshold value or tipping point above which we will classify values into class 1 (y=0)and below which we classify values into class 2(y=1). 
    For example, if our threshold was 0.5 and our prediction value is 0.8, we would classify this observation in class1(y = 0). If our prediction was 0.2 we would classify the observation in class 2(y = 1). 
    For multiple classfication we could select the class with the highest predicted probability.
    $$y=\left\{
\begin{array}{rcl}
    0 & {Sigmoid(x)<0.5}\\
    1& {Sigmoid(x) \geq 0.5}\\
    \end{array} \right. $$


    * *Disadvantage of Sigmoid Function*
    When x is close to ${ - \infty }$ or ${ + \infty }$, the change of $Sigmoid(x)$ is not obvious. The 'S'curve tends to be gentle.
    * *Differential Equation of Sigmoid Function*
    ==$DSigmoid(x)={Sigmoid(x)}*({1-Sigmoid(x)})$==



* #### *Neural network*

    ![](https://i.imgur.com/PoTPx7S.png)
    Neural Network is a computational model that is inspired by the way biological neural networks in the human brain process information. Artificial Neural Networks have generated a lot of excitement in Machine Learning research and industry, thanks to many breakthrough results in speech recognition, computer vision and text processing. In this blog post we will try to develop an understanding of a particular type of Artificial Neural Network called the Multi Layer Perceptron.
    In general, we have input layer, hidden layer and output layer. You can set the node number at each layer and the number of hidden layer.
    Neural network is able to solve the binary classifacation, muti classification and regression problems.

* #### *Backpropagation*
    Using the initial parameters by forword pass to get a loss about the model. Then use backpropagation to update w and b. Repeating n times and get the final parameters(w,b) and predict_y.
        
### 3. **Experiment setups**
* #### *Sigmoid functions*
    * Notation
        1. sigmoid(x):Sigmoid Function
        2. derivative_sigmoid(x):Differential Equation of Sigmoid Function
            > The x in derivative_sigmoid(x) is after sigmoid

    * Defination
        1. sigmoid(x) = 1.0/(1.0 +np.exp(-x))
        2. derivative_sigmoid(x) = np.multiply(x,1.0-x)

    * Code
        ```python
        #sigmoid
        def sigmoid(x):
            return 1.0/(1.0 +np.exp(-x))

        def derivative_sigmoid(x):
            return np.multiply(x,1.0-x)

        ```
* #### *Neural network*
    * Framework

        ![](https://i.imgur.com/hXI0oj6.jpg)

     * Notation
        * $w$ : weight
        * $b$ : bias
        * $z$ : w*a
        * $a$ : $sigmoid(z)$  
        * $y$_pred : the predict value of y
        * $d$ :  the delta of z，partial loss/partial z
        * $lr$ : learning rate
        * $dw$ ： gradirnt of w
        * $db$ : gradirn of b


    * Neural Table
        |  | Input Layer | Hidden Layer1 |Hidden Layer2 |Output Layer |
        | -------- | -------- | -------- |-------- |-------- |
        | num of neural    | 2     | 4    |4   |1     |


    * Parameter Table
        |  | 0 | 1 |2 |
        | -------- | -------- | -------- |-------- |-------- |
        | shape of weight | (4,2) | (4,4) |(1,4) |
        | shape of bias | (4,1) | (4,1) |(1,1) |
        | shape of a | (4,1) |(4,1) | (1,1) |
        | shape of z | (4,1) |(4,1) | (1,1) |


* #### *Backpropagation*

    ![](https://i.imgur.com/imgDcQk.jpg)


    ![](https://i.imgur.com/NNkbJhT.jpg)

    * This is my derivation of backpropagation.(I use a 2,2,2,1 example)
        1. initial parameters(w,b) by random
        2. by feed forward to get a,z and loss
        3. by back propagation to get gradient 
        4. Using an optimization algorithm, updating parameters(w,b) at each interation

        ![](https://i.imgur.com/QRpTuRw.jpg)

        * Detail of derivation
          ![](https://i.imgur.com/UaElPtd.jpg)

     *  Function Defination
         *  initial_w_b() ： initial function
            ```python
             def initial_w_b():
                w = [0,0,0]
                b = [0,0,0]
                w[0] = np.random.randn(4,2)
                w[1] = np.random.randn(4,4)
                w[2] = np.random.randn(1,4)

                b[0] = np.random.randn(4,1)
                b[1] = np.random.randn(4,1)
                b[2] = np.random.randn(1,1)
                return w,b
              ```

        * forword(x,w,b) : feed forward function
             ```python
             def forword(x,w,b):
                a = [x.reshape(x.shape[0],1)]
                z = [np.add(np.dot(w[0],a[0]),b[0])]
                for l in range(1,3):
                    a.append(sigmoid(z[l-1]))
                    z.append(np.add(np.dot(w[l],a[l]),b[l]))
                    y_pred = sigmoid(z[2])
                return a, z, y_pred
             ```

         *  back(a,z,y,y_pred,w) : back propagation function
            ```python
             def back(a,z,y,y_pred,w):
                d = [0,0,0]

                d[2] = np.multiply(derivative_sigmoid(y_pred),(y_pred.T[0]-y).reshape(y_pred.shape))

                for l in reversed(range(2)):
                    d[l] = np.multiply(derivative_sigmoid(a[l+1]), np.dot(np.transpose(w[l+1]), d[l+1]))

                return d
            ```
         *  optimize(d,lr,w,a,b) : updating function
            ```python
             def optimize(d,lr,w,a,b):
                dw = [0,0,0]
                db = [0,0,0]
                for l in range(3):
                    #print(d[l].shape)
                    #print(a[l].shape)
                    dw[l] = np.dot(d[l],a[l].transpose())
                    db[l] = d[l]
                    w[l] -= lr * dw[l]
                    b[l] -= lr * db[l]
                return w,b
            ```

        
    
### 3. **Results of  testing**
* #### XOR Data
    ![](https://i.imgur.com/700ed4o.png)

     ![](https://i.imgur.com/OniUJI5.png)


* #### Linear Data
    ![](https://i.imgur.com/cEtXF6D.png)
      ![](https://i.imgur.com/0NCKENQ.jpg)

* #### Code
    ```python
    def lab1(X,Y,lr = 0.05,iteration = 100001):
    w,b = initial_w_b()
    for i in range(iteration):
        Y_pred = np.zeros((X.shape[0],1))
        for j in range(X.shape[0]):
            x = X[j]
            y = Y[j]
            a, z, y_pred = forword(x,w,b)
            Y_pred[j] = y_pred
            d = back(a,z,y,y_pred,w)
            w,b = optimize(d,lr,w,a,b) 
        if i % 5000 == 0:
            Y_pred = Y_pred.reshape(Y.shape)
            print('epoch', i, 'loss :', 1/2*np.sum(np.square(Y-Y_pred)))

        print(Y_pred)

        show_results(X, Y, np.where(Y_pred<0.5,0,1))   


    X,Y = generate_XOR_easy()   
    lab1(X,Y,lr = 0.05,iteration = 100001)
    print('XOR finish')
    X,Y = generate_linear(n=100)
    lab1(X,Y,lr = 0.05,iteration = 100001)
    print('linear finish')

    ```




### 4. **Discussion**
* #### ***Initial parameter***
    In the beginning, I used *np.random.rand* to generate the initial parameter, for the loss of XOR data not reducing sometimes. However after I changed my initial parameter as *np.random.randn*, the loss performed better.
    *np.random.rand* generate samples from (0,1) uniform distribution.*np.random.randn* sample from a univariate “normal” (Gaussian) distribution of mean 0 and variance 1.
    I guess maybe a local min for parameters exist from 0 to 1. So the probility of (0,1)uniform distribution getting local min is higher than gaussian distribution.

* #### Loss differential equation
    After I change the way to generate the initial, my loss still is very high. After I check and check my code. I found some error in my Loss differential equation. At first I forgot to mutiple(-1) using chain rule.[1/2(y-y_pred)^2].

