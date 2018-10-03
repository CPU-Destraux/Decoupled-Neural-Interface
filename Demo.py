
import numpy as np,sys,time
from sklearn.datasets import make_classification
np.random.seed(4567)

def log(x):
    return 1  / ( 1 + np.exp(-1*x))
def d_log(x):
    return log(x) * ( 1- log(x))

# 0. Generate Training Data
X, Y = make_classification(n_samples=500,n_features=2,class_sep=4, n_repeated = 0,n_redundant=0, n_classes=2,n_informative=2,n_clusters_per_class=2)
Y = np.expand_dims(Y,axis=1)

time.sleep(2)

# 1. Declare Hyper Parameters
num_epoch = 100
learning_rate = 0.00008
n_value = 0.1

# 2. Create Weigths and copy them to have the same values
w1 = (np.random.randn(2,10) * 0.2) - 0.1
w2 = (np.random.randn(10,28) * 0.2) - 0.1
w3 = (np.random.randn(28,1) * 0.2) - 0.1

w1_DNI = (np.random.randn(10,10) * 0.2) - 0.1
w2_DNI =  (np.random.randn(28,28) * 0.2) - 0.1
w3_DNI = (np.random.randn(1,1) * 0.2) - 0.1

w1_syth = w1_DNI
w2_syth = w2_DNI
w3_syth = w3_DNI

w1_syth_noise = w1_DNI
w2_syth_noise = w2_DNI
w3_syth_noise = w3_DNI

w1_nn,w2_nn,w3_nn = w1,w2,w3
w1_DN,w2_DN,w3_DN = w1,w2,w3
w1_nn_noise,w2_nn_noise,w3_nn_noise = w1,w2,w3
w1_DN_noise,w2_DN_noise,w3_DN_noise = w1,w2,w3


# ============================================
print("\n------- Case 2: Decoupled Neural Interface ----------------")
for iter in range(num_epoch):
    
    # ------ Normal Layer 1 ---------
    layer_1 = X.dot(w1_DN)
    layer_1_act = log(layer_1)
    # ------ Normal Layer 1 ---------
    # ------ Sythn Layer 1 ---------
    syth_grad_1_part_1 = layer_1_act.dot(w1_syth)
    syth_grad_1_part_2 = d_log(layer_1)
    syth_grad_1_part_3 = X
    syth_grad_1 =   syth_grad_1_part_3.T.dot(syth_grad_1_part_1 * syth_grad_1_part_2)

    w1_DN = w1_DN + learning_rate * syth_grad_1
    # ------ Sythn Layer 1 ---------
    

    # ------ Normal Layer 2 ---------
    layer_2 = layer_1_act.dot(w2_DN)
    layer_2_act = log(layer_2)
    # ------ Normal Layer 2 ---------
    # ------ Sythn Layer  2---------
    syth_grad_2_part_1 = layer_2_act.dot(w2_syth)
    syth_grad_2_part_2 = d_log(layer_2)
    syth_grad_2_part_3 = layer_1_act
    syth_grad_2 =   syth_grad_2_part_3.T.dot(syth_grad_2_part_1 * syth_grad_2_part_2)

    w2_DN = w2_DN + learning_rate * syth_grad_2
    layer_2_delta = (syth_grad_2_part_1 * syth_grad_2_part_2).dot(w2_DN.T)
    # ------ Sythn Layer  2 ---------
    # ------ Layer 1 Groud Truth Weight Update -----
    syth_gt_grad_1_part_1 = syth_grad_1_part_1 - layer_2_delta
    syth_gt_grad_1_part_2 = layer_1_act
    syth_gt_grad_1 = syth_gt_grad_1_part_2.T.dot(syth_gt_grad_1_part_1)
    w1_syth = w1_syth + learning_rate * syth_gt_grad_1
    # ------ Layer 1 Groud Truth Weight Update -----


    # ------ Normal Layer 3 ---------
    layer_3 = layer_2_act.dot(w3_DN)
    layer_3_act = log(layer_3)
    # ------ Normal Layer 3 ---------
    # ------ Sythn Layer  3---------
    syth_grad_3_part_1 = layer_3_act.dot(w3_syth)
    syth_grad_3_part_2 = d_log(layer_3)
    syth_grad_3_part_3 = layer_2_act
    syth_grad_3 =   syth_grad_3_part_3.T.dot(syth_grad_3_part_1 * syth_grad_3_part_2)

    w3_DN = w3_DN + learning_rate * syth_grad_3
    layer_3_delta = (syth_grad_3_part_1 * syth_grad_3_part_2).dot(w3_DN.T)
    # ------ Sythn Layer  3 ---------
    # ------ Layer 2 Groud Truth Weight Update -----
    syth_gt_grad_2_part_1 = syth_grad_2_part_1 - layer_3_delta
    syth_gt_grad_2_part_2 = layer_2_act
    syth_gt_grad_2 = syth_gt_grad_2_part_2.T.dot(syth_gt_grad_2_part_1)
    w2_syth = w2_syth + learning_rate * syth_gt_grad_2
    # ------ Layer 3 Groud Truth Weight Update -----

    cost = np.square(layer_3_act - Y).sum() * 0.5
    #print("Current Iter : ", iter, " current cost : ", cost,end="\r")

    # ------ Layer 3 Groud Truth Weight Update -----
    syth_gt_grad_3_part_1 = syth_grad_3_part_1 - (layer_3_act - Y)
    syth_gt_grad_3_part_2 = layer_3_act
    syth_gt_grad_3 = syth_gt_grad_3_part_2.T.dot(syth_gt_grad_3_part_1)
    w3_syth = w3_syth + learning_rate * syth_gt_grad_3
    # ------ Layer 3 Groud Truth Weight Update -----
