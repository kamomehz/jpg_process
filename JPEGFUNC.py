import numpy as np

####　色変換
def grayG(x):
    r=x[:,:,0].reshape(-1)
    g=x[:,:,1].reshape(-1)
    b=x[:,:,2].reshape(-1)
    g=((r**2.2+(1.5*g)**2.2+(0.6*b)**2.2)/(1+1.5**2.2+0.6**2.2))**(1/2.2)
    return g.reshape(x.shape[:2])

def grayL(x):
    r=x[:,:,0].reshape(-1)
    g=x[:,:,1].reshape(-1)
    b=x[:,:,2].reshape(-1)
    g=r*0.3+g*0.59+b*0.11
    return g.reshape(x.shape[:2])

def ycc(x): # in (0,255) range
    r=x[:,:,0]
    g=x[:,:,1]
    b=x[:,:,2]
    y = .299*r + .587*g + .114*b
    cb = 128 -.168736*r -.331364*g + .5*b
    cr = 128 +.5*r - .418688*g - .081312*b
    
    return np.stack([y, cb, cr], 2)

def rgb(x):
    y=x[:,:,0]
    cb=x[:,:,1]
    cr=x[:,:,2]
    r = y + 1.402 * (cr-128)
    g = y - .34414 * (cb-128) -  .71414 * (cr-128)
    b = y + 1.772 * (cb-128)
    out=np.stack([r, g, b], 2)
    out[out>255]=255
    out[out<0]=0
    return out

####　8*8
def cutTo8(cbcr):
    for cutx in range(cbcr.shape[0]//8):
        for cuty in range(cbcr.shape[1]//8):
            cbcrResize=cbcr[cutx*8:cutx*8+8,cuty*8:cuty*8+8].copy()
#             break
            yield [cbcrResize, cutx, cuty]

def normalize(img):
    vmin = np.min(img)
    vmax = np.max(img)
    return (img - vmin) / (vmax - vmin) * 255



#### Trans
def dft(x):
    n=x.size
    An=np.arange(n).reshape(1,-1)
    Ak=np.arange(n).reshape(-1,1)
    w=np.exp(-1j*2*np.pi*An*Ak/n)
    X=np.sum(x*w,axis=1)
    return X

def idft(X):
    n=X.size
    X2=X.real-1j*X.imag
    x=dft(X2)/n
    x2=x.real-1j*x.imag
    return X

def dct0(x): #type2
    n=x.size
    An=np.arange(n).reshape(1,-1)
    Ak=np.arange(n).reshape(-1,1)
    w=np.cos(np.pi/n*(An+0.5)*(Ak))
    X=np.sum(x*w,axis=1)
    X*=np.sqrt(2/n)
    X[0]/=np.sqrt(2)
    return X.real

def idct0(X):
    n=X.size
    An=np.arange(n).reshape(1,-1)
    Ak=np.arange(n).reshape(-1,1)
    w=np.cos(np.pi/n*(An)*(Ak+0.5))
    x=np.sum(X*w,axis=1)
    x*=np.sqrt(2/n)
    x[0]/=np.sqrt(2)
    return x

def dct02ori(img):
    h, w = img.shape
    if ((h-1) & h) or ((w-1) & w):
        print('Image size not a power of 2')
        return img
    res = np.zeros([h, w], 'complex128')
    for i in range(h):
        res[i, :] = dct0(img[i, :])
    for j in range(w):
        res[:, j] = dct0(res[:,j])
    return res.real

def dct02(img):
    n = img.shape[0]
    res = np.zeros([n,n])
    A = np.zeros([n,n])
    for i in range(n):
        A[i,:]=np.cos((np.arange(n)+0.5)*np.pi*i/n)
    A*=np.sqrt(2/n)
    A[0,:]/=np.sqrt(2)
    return A.dot(img).dot(A.T)

def idct02(img):
    n = img.shape[0]
    res = np.zeros([n,n])
    A = np.zeros([n,n])
    for i in range(n):
        A[i,:]=np.cos((np.arange(n)+0.5)*np.pi*i/n)
    A*=np.sqrt(2/n)
    A[0,:]/=np.sqrt(2)
    Ai=np.linalg.inv(A)
    return Ai.dot(img).dot(Ai.T)



def fft(x):
    n=x.size
    if n == 2:
        return [x[0]+x[1],x[0]-x[1]]
    
    x1 = fft(x[::2])
    x2 = fft(x[1::2])
    W = np.exp(-2j*np.pi*np.arange(n//2)/n)
    W_x2= W*x2
    X = np.concatenate([x1+W_x2, x1-W_x2])
    return X

def ifft(X):
    n=X.size
    X2=X.real-1j*X.imag
    x=fft(X2)/n
    x2=x.real-1j*x.imag
    return x2

def fft2(img):
    h, w = img.shape
    if ((h-1) & h) or ((w-1) & w):
        print('Image size not a power of 2')
        return img
    
    img = normalize(img)
    res = np.zeros([h, w], 'complex128')
    for i in range(h):
        res[i, :] = fft(img[i, :])
    for j in range(w):
        res[:, j] = fft(res[:, j])
    return res

def fftshift(img):
    # swap the first and third quadrants, and the second and fourth quadrants
    h, w = img.shape
    h_mid, w_mid = h//2, w//2
    res = np.zeros([h, w], 'complex128')
    res[:h_mid, :w_mid] = img[h_mid:, w_mid:]
    res[:h_mid, w_mid:] = img[h_mid:, :w_mid]
    res[h_mid:, :w_mid] = img[:h_mid, w_mid:]
    res[h_mid:, w_mid:] = img[:h_mid, :w_mid]
    return res



####　量子化
def Qy(Gy):
    y=np.array([16,11,10,16,24,40,51,61,\
                12,12,14,19,26,58,60,55,\
                14,13,16,24,40,57,69,56,\
                14,17,22,29,51,87,80,62,\
                18,22,37,56,68,109,103,77,\
                24,35,55,64,81,104,113,92,\
                49,64,78,87,103,121,120,101,
                72,92,95,98,112,100,103,99]).reshape((8,8))
    return np.round(Gy/y)

def iQy(Q_y):
    y=np.array([16,11,10,16,24,40,51,61,\
                12,12,14,19,26,58,60,55,\
                14,13,16,24,40,57,69,56,\
                14,17,22,29,51,87,80,62,\
                18,22,37,56,68,109,103,77,\
                24,35,55,64,81,104,113,92,\
                49,64,78,87,103,121,120,101,
                72,92,95,98,112,100,103,99]).reshape((8,8))
    return y*Q_y

def Qc(Gc):
    c=np.ones((8,8))*99
    c[:4,0]=[17,18,24,47]
    c[0,:4]=[17,18,24,47]
    c[1,:4]=[18,21,26,66]
    c[1:4,1]=[21,26,66]
    c[2,2]=56    
    return np.round(Gc/c)

def iQc(G_c):
    c=np.ones((8,8))*99
    c[:4,0]=[17,18,24,47]
    c[0,:4]=[17,18,24,47]
    c[1,:4]=[18,21,26,66]
    c[1:4,1]=[21,26,66]
    c[2,2]=56    
    return G_c*c



###

def Zigzag(width=8):
    path=np.arange(1)
    lineNum=width*2-1
    crossLine=np.arange(lineNum)+1
    crossLine[width:]=lineNum-np.arange(lineNum)[width:]
    
    for ind,count in enumerate(crossLine):
        newList=np.arange(ind,ind+(ind)*width,width-1)
        if ind>=width:
            newList=newList[width-count:width]
        path=np.append(path,newList[::(-1)**(ind+1)])
    return path



def huff():
    pass