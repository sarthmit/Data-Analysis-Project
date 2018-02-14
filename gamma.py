import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import math
import seaborn as sns
import sys
import time

n_samples = 1000
e = np.exp(1)

def gen_normal():
	u = np.random.uniform()
	r = -2*np.log(1-u)
	theta = 2*np.pi * np.random.uniform()
	return np.sqrt(r)*np.cos(theta)

def gen_exp():
	u = np.random.uniform()
	return -np.log(1-u)

def gen_gamma(alpha_I):
	z = 0
	for i in xrange(alpha_I):
		z += gen_exp()
	return z

def check_1(z,alpha,v):
	if z==0:
		return False
	if v <= np.power(z,alpha-1)*np.power(e,-z/2.0)/(np.power(2,alpha-1)*(np.power(1-np.power(e,-z/2.0),alpha-1))):
		return True
	else:
		return False

def algo_1(alpha_I,alpha_F):
	K = alpha_F*e/(alpha_F+e)
	c = 1.0/(math.gamma(alpha_F)*K)

	def accept_reject(z):
		u = np.random.uniform()
		if(z<1):
			if(u <= np.exp(-z)/(c*K*math.gamma(alpha_F))):
				return True
		else:
			if(u <= np.power(z,alpha_F-1)/(c*K*math.gamma(alpha_F))):
				return True
		return False

	samples = []
	reject = 0
	while len(samples) <= n_samples:
		u = np.random.uniform()
		if(u<= K/alpha_F):
			z = np.power(alpha_F*u/K,1.0/alpha_F)
		else:
			z = -np.log(np.exp(-1) - (u/K - 1.0/alpha_F))
		if accept_reject(z):
			z += gen_gamma(alpha_I)
			samples.append(z)
		else:
			reject +=1
	return samples, reject/(1.0*(reject+n_samples))

def algo_2(alpha_I,alpha_F):
	samples = []
	reject = 0
	while len(samples) <= n_samples:
		u = np.random.uniform()
		v = np.random.uniform()
		w = np.random.uniform()
		if(u <= e/(e+alpha_F)):
			p = np.power(v,1.0/alpha_F)
			q = w*np.power(p,alpha_F-1)
		else:
			p = 1 - np.log(v)
			q = w*np.power(e,-p)
		if(q<=np.power(p,alpha_F-1)*np.power(e,-p)):
			samples.append(p+gen_gamma(alpha_I))
		else:
			reject += 1
	return samples, 1.0*reject/(n_samples+reject)

def algo_3(alpha_I,alpha_F):
	alpha = alpha_I+alpha_F
	d = alpha -1.0/3
	c = 1.0/np.sqrt(9*d)
	samples = []
	reject = 0
	while len(samples) <= n_samples:
		x = gen_normal()
		u = np.random.uniform()
		v = np.power(1+x*c,3)
		if v>0 and np.log(u) < 0.5*x*x + d - d*v + d*np.log(v):
			samples.append(d*v)
		else:
			reject +=1
	return samples, 1.0*reject/(n_samples+reject)

def algo_4(alpha_I,alpha_F):
	alpha = alpha_I+alpha_F
	samples = []
	reject = 0
	theta = np.sqrt(np.power(alpha-1,alpha-1)*np.power(e,1-alpha)/math.gamma(alpha))
	lam = np.sqrt(np.power(alpha+1,alpha+1)*np.power(e,-alpha-1)/math.gamma(alpha))
	while len(samples) <= n_samples:
		u = np.random.uniform(0,theta)
		v = np.random.uniform(0,lam)
		if u <= np.sqrt(stats.gamma.pdf((1.0*v)/u,alpha)):
			samples.append((1.0*v)/u)
		else:
			reject +=1
	return samples, 1.0*reject/(n_samples+reject)

def algo_5(alpha_I,alpha_F):
	a = 0.07 + 0.75*np.power(1-alpha_F,0.5)
	b = 1 + np.power(e,-a)*(alpha_F/a)
	samples = []
	reject = 0
	while len(samples) <= n_samples:
		u = np.random.uniform()
		p = b*u
		if p <= 1:
			z = a*np.power(p,1.0/alpha_F)
			u = np.random.uniform()
			if u <= (2-z)/(1.0*(2+z)) or u <= np.power(e,-z):
				samples.append(z+gen_gamma(alpha_I))
			else:
				reject+=1
		else:
			z = -np.log(a*(b-p)/alpha_F)
			y = z/a
			u = np.random.uniform()
			if u*(alpha_F+y-alpha_F*y) < 1 or u <= np.power(y,alpha_F-1):
				samples.append(z+gen_gamma(alpha_I))
			else:
				reject +=1
	return samples, (1.0*reject)/(reject+n_samples)

def algo_6(alpha_I,alpha_F):
	samples = []
	reject = 0
	while len(samples) <= n_samples:
		u = np.random.uniform()
		z = -2*np.log(1-np.power(u,1.0/alpha_F))
		v = np.random.uniform()
		if check_1(z,alpha_F,v):
			samples.append(z+gen_gamma(alpha_I))
		else:
			reject += 1
	return samples, (1.0*reject)/(reject+n_samples)

def algo_7(alpha_I,alpha_F):
	a = np.power(1-np.power(e,-0.5),alpha_F)
	a = a/(a+alpha_F/(np.power(2,alpha_F)*e))
	b = np.power(1-np.power(e,-0.5),alpha_F) + alpha_F/(np.power(2,alpha_F)*e)
	samples = []
	reject = 0
	while len(samples) <= n_samples:
		u = np.random.uniform()
		if u <= a:
			z = -2*np.log(1-np.power(u*b,1.0/alpha_F))
		else:
			z = -np.log(b*(1-u)*np.power(2,alpha_F)/alpha_F)
		v = np.random.uniform()
		if z<=1:
			if check_1(z,alpha_F,v):
				samples.append(z+gen_gamma(alpha_I))
			else:
				reject +=1
		else:
			if v<= np.power(z,alpha_F-1):
				samples.append(z+gen_gamma(alpha_I))
			else:
				reject +=1
	return samples, (1.0*reject)/(n_samples+reject)

def algo_8(alpha_I, alpha_F):
	samples = []
	reject = 0
	d = 1.0334 - 0.0766*np.power(e,2.2942*alpha_F)
	a = np.power(2,alpha_F)*(np.power(1-np.power(e,-d/2.0),alpha_F))
	b = alpha_F*np.power(d,alpha_F-1)*np.power(e,-d)
	c = a+b

	while len(samples) <= n_samples:
		u = np.random.uniform()
		if u <= a/(a+b):
			z = -2*np.log(1-np.power(c*u,1.0/alpha_F)/2.0)
		else:
			z = -np.log(c*(1-u)/(alpha_F*np.power(d,alpha_F-1)))
		v = np.random.uniform()
		if z<=d:
			if check_1(z,alpha_F,v):
				samples.append(z+gen_gamma(alpha_I))
			else:
				reject +=1
		else:
			if v<= np.power(d/(1.0*z),1-alpha_F):
				samples.append(z+gen_gamma(alpha_I))
			else:
				reject +=1
	return samples, (1.0*reject)/(n_samples+reject)

def algo_9(alpha_I, alpha_F):
	alpha = alpha_F+ alpha_I
	def gamma(x):
		return np.power(x,alpha-1)*np.exp(-beta*x)

	def exponential(x,l):
		return (1.0/l)*np.exp(-x/(1.0*l))

	def gen_exp(l):
		u = np.random.uniform()
		return -l*np.log(1-u)

	def acceptance(x,y):
		return min(1,gamma(x) * exponential(y,x) / (gamma(y) * exponential(x,y)))

	y = alpha
	samples = []
	reject = 0
	while len(samples) <= n_samples:
		x = gen_exp(y)
		u = np.random.uniform()
		if(u <= acceptance(x,y)):
			samples.append(x)
			y = x
		else:
			reject +=1
	return samples, (1.0*reject)/(reject+n_samples)

def algo_10(alpha_I,alpha_F):
	alpha = alpha_I+alpha_F
	d = alpha -1.0/3
	c = 1.0/np.sqrt(9*d)
	samples = []
	reject = 0
	while len(samples) <= n_samples:
		x = gen_normal()
		u = np.random.uniform()
		v = np.power(1+x*c,3)
		if v>0:
			if u < 1-0.0331*x*x*x*x or np.log(u) < 0.5*x*x + d - d*v + d*np.log(v):
				samples.append(d*v)
			else:
				reject +=1
		else:
			reject +=1
	return samples, 1.0*reject/(n_samples+reject)

def algo_11(alpha_I,alpha_F):
	alpha = alpha_I+alpha_F
	b = alpha - 1
	A = alpha + b
	s = np.sqrt(A)
	samples = []
	reject = 0
	while len(samples) <= 1000:
		u = np.random.uniform()
		t = s*np.tan(np.pi*(u-0.5))
		x = b + t
		if x<0:
			reject +=1
			continue
		u = np.random.uniform()
		if u > np.power(e,b*np.log((1.0*x)/b) - t + np.log(1+t*t/(1.0*A))):
			reject +=1
		else:
			samples.append(x)
	return samples, 1.0*reject/(n_samples+reject)

alpha_Is = [1,3,100]
alpha_Fs = [.5,.2,.7]
beta = 1
plt.close()

algoS_3 = [0,0,0]
algoS_10 = [0,0,0]

time_10 = [0,0,0]
time_3 = [0,0,0]

for s in xrange(100):
	print s
	for j in xrange(3):
		alpha_F = alpha_Fs[j]
		alpha_I = alpha_Is[j]
		alpha = alpha_F+alpha_I
		print "alpha = ", alpha_I+alpha_F
		print
		for i in [2,9]:
			start_time = time.time()
			samples, reject = getattr(sys.modules[__name__], "algo_%s" %str(i+1))(alpha_I,alpha_F)
			end_time = time.time()
			if i+1 == 3:
				algoS_3[j] += reject
				time_3[j] += end_time-start_time
			if i+1 == 10:
				algoS_10[j] += reject
				time_10[j] += end_time-start_time
			print "Algorithm %d: \t Rejection Rate: %f \t Time Required: %s" %(i+1,reject,end_time-start_time)
			s = np.linspace(0,max(samples),100)
			sns.distplot(samples,kde=False,norm_hist=True,bins=50,label="Empirical Distribution")
			sns.set_style("darkgrid")
			plt.plot(s,stats.gamma.pdf(s,alpha),c='k',label="Probability Density of Gamma(1.5,1)")
			plt.xlabel('X')
			plt.ylabel('Probability')
			plt.legend()
			plt.savefig("Plot_{}.png".format(str(j)+"_"+str(i+1).zfill(3)),bbox_inches='tight')
			plt.close()
		print

print "Algorithm 3"
print np.array(algoS_3)/100.0
print np.array(time_3)/100.0

print "Algorithm 10"
print np.array(algoS_10)/100.0
print np.array(time_10)/100.0