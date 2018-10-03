import numpy as np, sys, time
from sklearn.datasets import make_classification
np.random.seed(4567)

def log(x):
    return 1 / ( 1 + np.exp(-1*x))

def d_log(x):
    return log(x) * (1- log(x))

# Init Data Set
X, Y = make_classification(n_samples=500,n_features=2,class_sep=4, n_repeated = 0,n_redundant=0, n_classes=2,n_informative=2,n_clusters_per_class=2)
Y = np.expand_dims(Y,axis=1)
time.sleep(2)

# Hyper Params
num_epoch = 100
learning_rate = 0.00008
n_value = 0.1

#Dims

DIM = [[2,10],[10,28],[28,1]]

#Weights
Ws = [(np.random.randn(i[0],i[1]) * 0.2) - 0.1 for i in DIM]
W_DNI = [(np.random.randn(i[1],i[1]) * 0.2) - 0.1 for i in DIM]

W_Syth = W_DNI
W_Noise = W_DNI
W_DN = Ws

#Network Inteface

def LayerAct(IN, W): # IN - input layer to level ###### W - DN weights #
    layer = IN.dot(W)
    act = log(layer)
    return list([layer, act, IN])


# act comes from LayerAct, so does layer, prior is last input to level, parDelta is bool to process delta values
def Syth_Grad(act, layer, weight_index, wsyth, prior, parDelta):
    part1 = act.dot(wsyth)
    part2 = d_log(layer)
    part3 = prior
    grad = part3.T.dot(part1 * part2)
    out = [grad,part1]
    W_DN[weight_index] = W_DN[weight_index] + learning_rate * grad
    if parDelta == True:
        delta = (part1 * part2).dot(W_DN[weight_index].T)
        out += [delta]
    if parDelta == False:
        out += [False]
    return out
    

def Syth_Truth(sythLayerIndex, syth, delta, lastAct):
    gt1 = syth - delta
    gt2 = lastAct
    gt = gt2.T.dot(gt1)
    fixed_layer = W_Syth[sythLayerIndex] + learning_rate * gt
    return fixed_layer


Layer_Acts = []
Grad_parts = []

def Activate(IN):
    Cur = IN
    Layer_Acts.append(IN)
    for i in xrange(len(DIM)):
        Layer_n, layer_n_act, prior = LayerAct(Cur, W_DN[i])
        Layer_Acts.append(layer_n_act)
        #print "Act"
        try:
            jay = 0
            syth_grad, grad_part1, delta = Syth_Grad(layer_n_act, Layer_n, i, W_Syth[i], prior, True)
            if delta != False:
                #print "make synth grad"
                Grad_parts.append(grad_part1)
                #print "grad append"
                W_Syth[i-1] = Syth_Truth(i-1, Grad_parts[len(Grad_parts)-2], delta, Layer_Acts[len(Layer_Acts)-2])
                #print "apply synth"
                jay += 1
        except ValueError:
            #print "upper failure"
            if jay < 1:
                syth_grad, grad_part1, delta = Syth_Grad(layer_n_act, Layer_n, i, W_Syth[i], prior, False)
                #print "make synth grad"
                Grad_parts.append(grad_part1)
                #print "grad append"
        Cur = Layer_Acts[len(Layer_Acts)-1]
    cost = np.square(Layer_Acts[len(Layer_Acts)-1] - Y).sum() * 0.5
    W_Syth[len(DIM)-1] = Syth_Truth(len(DIM)-1, Grad_parts[len(Grad_parts)-1], (Layer_Acts[len(Layer_Acts)-1] - Y), Layer_Acts[len(Layer_Acts)-1])
    print cost


while 1:
    Activate(X)


