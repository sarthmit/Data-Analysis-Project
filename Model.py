import numpy as np
from scipy.stats import multivariate_normal
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

###################################################################
########################## LOAD DATA ##############################
###################################################################

data = np.loadtxt('swiss-bank.dat',delimiter=' ',skiprows=1)

features = data[:,1:]
labels = np.array(data[:,0],dtype=np.int)
z = np.zeros_like(labels,dtype=np.float)
N = np.shape(labels)[0]
steps = 1000

###################################################################
####################### INITIALIZE PARAMETERS #####################
###################################################################

def label_init():
	pi = np.mean(labels)
	mu_0 = np.mean(features[labels==0,:],axis=0)
	mu_1 = np.mean(features[labels==1,:],axis=0)
	cov_0 = np.cov(features[labels==0,:],rowvar=False)
	cov_1 = np.cov(features[labels==1,:],rowvar=False)
	return pi,mu_0,mu_1,cov_0,cov_1

def no_label_init():
	pi = 0.5
	mean = np.mean(features,axis=0)
	mu_0 = mean + np.random.randn(6)
	mu_1 = mean + np.random.randn(6)
	cov_0 = np.identity(6)
	cov_1 = np.identity(6)
	return pi,mu_0,mu_1,cov_0,cov_1

pi,mu_0,mu_1,cov_0,cov_1 = label_init()

###################################################################
###################### GAUSSIAN MIXTURE MODEL #####################
###################################################################

prev_lik = 0
prev_pi = 0
prev_mu_0 = np.zeros(6)
prev_mu_1 = np.zeros(6)
prev_cov_0 = np.zeros((6,6))
prev_cov_1 = np.zeros((6,6))

thresh = 0.0001

for i in xrange(steps):
	theta_0 = (1-pi)*multivariate_normal(mean=mu_0,cov=cov_0) \
				.pdf(features)
	theta_1 = pi*multivariate_normal(mean=mu_1, cov= cov_1). \
				pdf(features)
	z = theta_1 / (theta_0 + theta_1)
	pi = np.sum(z)/N
	mu_1 = np.sum(z[:,np.newaxis]*features,axis=0) / np.sum(z)
	mu_0 = np.sum((1-z)[:,np.newaxis]*features,axis=0)/np.sum(1-z)
	cov_1 = np.dot(np.transpose(z[:,np.newaxis]*(features-mu_1)) \
				,features-mu_1)/np.sum(z)
	cov_0 = np.dot(np.transpose((1-z)[:,np.newaxis]*(features-mu_0)) \
				,features-mu_0)/np.sum(1-z)
	lik = np.sum(np.log(theta_0 + theta_1))
	if lik - prev_lik < thresh:
		if max(pi-prev_pi,np.max(mu_0-prev_mu_0), \
			np.max(mu_1-prev_mu_1),np.max(cov_0-prev_cov_0), \
			np.max(cov_1-prev_cov_1)) < thresh:
			break
	prev_lik = lik
	prev_pi = pi
	prev_mu_0 = mu_0
	prev_mu_1 = mu_1
	prev_cov_0 = cov_0
	prev_cov_1 = cov_1

z[z<0.5] = 0
z[z>=0.5] = 1
z = np.array(z,np.int)

acc = np.sum(z==labels)*100./np.size(z)
print max(acc,100-acc)

X_embedded = TSNE(n_components=2).fit_transform(features)
plt.rcParams["figure.figsize"] = (8,5)
plt.subplots(1,2)
plt.subplot('121')
plt.scatter(X_embedded[:,0],X_embedded[:,1], s=5, c = labels, \
		 cmap="tab20b")
plt.colorbar()
plt.title('True Labels')
plt.subplot('122')
plt.scatter(X_embedded[:,0],X_embedded[:,1], s=5, c = z, \
		cmap="tab20b")
plt.colorbar()
plt.title('Predicted Labels')
plt.savefig('Plot_GMM.png',bbox_inches='tight')
plt.close()

###################################################################
##################### GENERATIVE CLASIFICATION ####################
###################################################################

arr = np.arange(N/2)
np.random.shuffle(arr)

train_data = np.concatenate([features[arr[:90],:], \
			features[100+arr[:90],:]],axis=0)
train_labels = np.concatenate([labels[arr[:90]], \
			labels[100+arr[:90]]],axis=0)
test_data = np.concatenate([features[arr[90:],:], \
			features[100+arr[90:],:]],axis=0)
test_labels = np.concatenate([labels[arr[90:]], \
			labels[100+arr[90:]]],axis=0)

# Since equal points from both classes taken in training and testing
pi = np.sum(train_labels,dtype=np.float)/np.size(train_labels) 

mu_0 = np.mean(train_data[train_labels==0],axis=0)
mu_1 = np.mean(train_data[train_labels==1],axis=0)

cov_0 = np.cov(train_data[train_labels==0],rowvar=False)
cov_1 = np.cov(train_data[train_labels==1],rowvar=False)

var_0 =multivariate_normal(mean=mu_0,cov=cov_0)
var_1 =multivariate_normal(mean=mu_1,cov=cov_1)

theta_0 = var_0.pdf(test_data)
theta_1 = var_1.pdf(test_data)

z = pi * theta_1 / (pi*theta_1 + (1-pi)*theta_0)

z[z<0.5] = 0
z[z>=0.5] = 1
z = np.array(z,np.int)

print np.sum(z==test_labels)*100./np.size(z)

X_embedded = TSNE(n_components=2).fit_transform(features)
X_embedded = np.concatenate([X_embedded[arr[90:],:], \
				X_embedded[100+arr[90:],:]],axis=0)
plt.rcParams["figure.figsize"] = (8,5)
plt.subplots(1,2)
plt.subplot('121')
plt.scatter(X_embedded[:,0],X_embedded[:,1], s=5, c = test_labels, \
				cmap="tab20b")
plt.colorbar()
plt.title('True Labels')
plt.subplot('122')
plt.scatter(X_embedded[:,0],X_embedded[:,1], s=5, c = z, \
				cmap="tab20b")
plt.colorbar()
plt.title('Predicted Labels')
plt.savefig('Plot_GCM.png',bbox_inches='tight')
plt.close()