import theano
import theano.tensor as T
import numpy
import numpy as np
import numpy.random as rng

class classifier(object):

    def __init__(self, **kwargs):
        pass
    
    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])


    def get_updates(self, cost, learning_rate):
       grds     = [T.grad(cost=cost, wrt=param) for param in self.params]
       updates  = [(param, param - learning_rate * grd) for param, grd in zip(self.params, grds)]
       return updates
       
    def errors(self, y):
        return T.mean(T.neq(self.y_pred, y))


    
class LogisticRegression(classifier):

    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]
        self.input = input



class NN(classifier):

    def __init__(self, input, n_in, n_out, n_hidden, n_layers):

        
        W_values = [np.asarray(rng.uniform(low=-numpy.sqrt(6. / (n_in + n_hidden)),
                                           high=numpy.sqrt(6. / (n_in + n_hidden)),
                                           size=(n_in, n_hidden)
                              ), dtype=theano.config.floatX)]

        b_values = [numpy.zeros((n_hidden,), dtype=theano.config.floatX)]

        
        for _ in range(n_layers):
            W_values.append(np.asarray(rng.uniform(low   = -numpy.sqrt(6. / (n_hidden + n_hidden)),
                                                   high  = numpy.sqrt(6. / (n_hidden + n_hidden)),
                                                   size  = (n_hidden, n_hidden)),
                                                   dtype = theano.config.floatX))

            b_values.append(numpy.zeros((n_hidden,), dtype=theano.config.floatX))


            
        W_values.append(np.asarray(rng.uniform(low  = -numpy.sqrt(6. / (n_hidden + n_out)),
                                                   high = numpy.sqrt(6. / (n_hidden + n_out)),
                                                   size = (n_hidden, n_out)),
                                                   dtype= theano.config.floatX))

        b_values.append(numpy.zeros((n_out,), dtype=theano.config.floatX))
        

        self.Ws = [theano.shared(value=W, borrow=True) for W in W_values]
        self.bs = [theano.shared(value=b, borrow=True) for b in b_values]

        self.p_y_given_x = T.nnet.relu(T.dot(input, self.Ws[0]) + self.bs[0])
        
        for cnt_l in range(n_layers):
            self.p_y_given_x = T.nnet.relu(T.dot(self.p_y_given_x, self.Ws[cnt_l+1]) + self.bs[cnt_l+1])  

        self.p_y_given_x = T.nnet.softmax(T.dot(self.p_y_given_x, self.Ws[-1]) + self.bs[-1])
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = self.Ws + self.bs
        self.input = input

        

    
if __name__ == "__main__":

    data = np.load('dataTrain.npy')

    
    trainx = []
    trainy = []
    validx = []
    validy = []

    for gr in data:
        validx.append(theano.shared(np.asarray(gr[0][:,:-1], dtype=theano.config.floatX)))
        validy.append(T.cast(theano.shared(np.asarray(gr[0][:,-1], dtype=int)), 'int32'))
        trainx.append(theano.shared(np.asarray(gr[1][:,:-1], dtype=theano.config.floatX)))
        trainy.append(T.cast(theano.shared(np.asarray(gr[1][:,-1], dtype=int)), 'int32'))
       

    valid_sets_x = validx
    valid_sets_y = validy
    train_sets_x = trainx
    train_sets_y = trainy


    batch_size    = 30
    n_epochs      = 100
    learning_rate = 0.01
    
    index = T.lscalar()
    
    x = T.matrix('x') 
    y = T.ivector('y')

    n_models    = 3
    #classifiers = [LogisticRegression(input=x, n_in=14, n_out=2) for cnt in range(n_models)]
    classifiers = [NN(input=x, n_in=14, n_out=2, n_hidden=100, n_layers=2) for cnt in range(n_models)]
    
    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    costs = [classifier.negative_log_likelihood(y) for classifier in classifiers]

    validate_models = [theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={ x: valid_set_x[index * batch_size: (index + 1) * batch_size], y: valid_set_y[index * batch_size: (index + 1) * batch_size]})
                       for classifier, valid_set_x, valid_set_y in zip(classifiers, valid_sets_x, valid_sets_y)]

    # compute the gradient of cost with respect to theta = (W,b)
    #gs_W = [T.grad(cost=cost, wrt=classifier.W) for cost, classifier in zip (costs, classifiers)]
    #gs_b = [T.grad(cost=cost, wrt=classifier.b) for cost, classifier in zip (costs, classifiers)]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = [classifier.get_updates(cost, learning_rate) for classifier, cost in zip(classifiers, costs)]

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_models = [theano.function(
        inputs=[index],
        outputs=cost,
        updates=update,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    ) for cost, update, train_set_x, train_set_y in zip(costs, updates, train_sets_x, train_sets_y)]
    # end-snippet-3

    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    
    epoch = 0
    while (epoch < n_epochs):
        epoch = epoch + 1

        train_error = [0.]*n_models
        cnt_btc     = 0
        for minibatch_index in range(n_train_batches):
            for cnt_mdl in range(n_models):
                train_error[cnt_mdl] += train_models[cnt_mdl](minibatch_index)
            cnt_btc     += 1
            
        train_error = np.asarray(train_error)/float(cnt_btc)

        cnt_btc_valid = 0
        validation_losses = [0.] * n_models
        for minibatch_index in range(n_valid_batches):
            for cnt_mdl in range(n_models):
                validation_losses[cnt_mdl]  += validate_models[cnt_mdl](minibatch_index)
            cnt_btc_valid += 1

        validation_losses = np.asarray(validation_losses)/float(cnt_btc_valid)

        print('epoch %i' %epoch)
        for cnt_mdl in range(n_models):
            print('Model %i: train error %.4f, validation error %.4f' %(cnt_mdl, train_error[cnt_mdl], validation_losses[cnt_mdl] * 100.))



        


    ## Prediction on the test set
    data_test = np.load('dataTest.npy')
    
    print data_test.shape
    
    output    = 0.
    for cl in classifiers:
        output += cl.p_y_given_x
    output /= float(n_models)
    y_pred  = T.argmax(output, axis=1)
        
    test_models = theano.function(inputs=[x], outputs=y_pred)
    test_prediction = test_models(data_test)
    np.save('testPrediction.npy', test_prediction)
        
