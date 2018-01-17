import sympy as sp
import numpy as np

C = sp.cos
S = sp.sin

def generate_DH(i, V, n):
    '''
    generate DH transformation matrix according to initial parameters,
    i: position in DH chain,
    V: initial values,
    n: specifies free parameter-creates variable to change offset
    '''
    x = sp.symbols('%c%d' %('tard'[n], i))
    V[n] += x
        
    T = sp.Matrix([
        [C(V[0]),   -S(V[0])*C(V[1]),   S(V[0])*S(V[1]) ,   V[2]*C(V[0])],
        [S(V[0]),   C(V[0])*C(V[1]) ,   -C(V[0])*S(V[1]),   V[2]*S(V[0])],
        [0      ,   S(V[1])         ,   C(V[1])         ,   V[3]        ],
        [0      ,   0               ,   0               ,   1           ]])
       
    return T.evalf(), x

def getJacobian(funcs, vars_):
    m = len(funcs)
    n = len(vars_)
    J = sp.Matrix(np.zeros([m, n]))
    for i in range(m):
        for j in range(n):
            J[i, j] = sp.diff(funcs[i], vars_[j])
    return sp.lambdify(vars_, J)

def createParams(table):
    n = len(table)

    DH = []
    X = []

    F = np.identity(4, dtype=np.float32)
    for i in range(n):
        T, x = generate_DH(i, table[i], 0)
        X.append(x)
        DH.append( sp.lambdify(x, T) )
        F *= T
    
    J = getJacobian(F[:3, :], X)
    
    return DH, J

