from sympy import *
import numpy as np

def compJacobian(func, values, idVars):
	#function vector
	numOfFuc = len(func)
	#number of variables
	numOfVar = len(idVars)
	#initialize jacobian matrix
	Jac = np.zeros((numOfFuc, numOfVar))

	#if at this node, assign the key value
	KeyVal = {}
	idx = 0
	for sym in idVars:
		KeyVal.update({sym:values[idx][0]})
		idx += 1

	row = 0
	for func in func:
		for sym in func.atoms(Symbol):
			dif = diff(func, sym)
			idx = 0
			#derivative using diff function, find syms to calculate
			for varAtNode in idVars:
				if varAtNode == sym:
					break
				else:
					idx += 1
			Jac[row][idx] = dif.subs(KeyVal)
		row += 1
	return Jac

def obj_func(func, values, idVars):
	#calculate object function vector
	result = np.zeros((len(func), 1))

	KeyVal = {}
	idx = 0
	for sym in idVars:
		KeyVal.update({sym:values[idx][0]})
		idx += 1

	row = 0
	for func in func:
		result[row][0] = func.subs(KeyVal)
		row += 1
	return result

def LM(func, initial_val, idVars, maxIter = 300, eps1 = 1e-8, eps2 = 1e-8, to = 0.001):
	iter = 0
	v = 2
	x = initial_val
	J = compJacobian(func, initial_val, idVars)
	f = obj_func(func, initial_val, idVars)
	A = J.T.dot(J)
	g = J.T.dot(f)

	gnormal_inf = np.linalg.norm(g, np.inf)
	found = (gnormal_inf <= eps1)
	max = A[0][0]
	for i in range(len(A[0])):
		if A[i][i] >= max:
			max = A[i][i]

	miu = to * max
	I = np.eye(len(A[0]),  dtype = float)
	while((not(found))and(iter < maxIter)):
		iter += 1
		h = np.linalg.solve((A + miu * I), -g)
		if (np.linalg.norm(h) <= eps2 * (eps2 + np.linalg.norm(x))):
			found = True
		else:
			x_new = x+h
			ro = (obj_func(func, x, idVars).T.dot(obj_func(func, x, idVars))-obj_func(func, x_new, idVars).T.dot(obj_func(func, x_new, idVars)))/(h.T.dot((miu*h-g)))
			rho = ro[0][0]
			if rho > 0:
				x = x_new
				J = compJacobian(func, x, idVars)
				f = obj_func(func, x, idVars)
				A = J.T.dot(J)
				g = J.T.dot(f)
				gnormal_inf = np.linalg.norm(g, np.inf)
				found = (gnormal_inf <= eps1)
				max = 1./3.
				if (1 - (2 * rho - 1)) ** 3 > max:
					max = (1-(2*rho-1)) ** 3
				miu = miu * max
				v = 2
			else:
				miu = miu * v
				v = 2 * v
	print("the times of iteration are "+str(iter))
	return x

print(LM(h, initial_val, idVars, 100,  1e-15,  1e-15, 1))
