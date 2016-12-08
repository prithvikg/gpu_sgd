import numpy as np
r = 5000
c = 20

x = np.random.uniform(0, 1, (r,c))
w = np.random.randint(1,high=10,size=c)

#print x
#print w

y = np.dot(x,w)


#print "y is"
#print y

sigma = 1
mean = 0
noise = sigma * np.random.randn(10)

#print "y with noise is"
#y = y + noise
#print y


np.savetxt('xmatrix.csv', x, delimiter=',')
y = np.reshape(y, (1,r))
np.savetxt('yvector.csv', y, delimiter=',')
np.savetxt('wsol.csv', w, delimiter=',')

