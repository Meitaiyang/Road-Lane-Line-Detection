# ------------------------------------------------
#
# 
# 
# !!!please run 'pip install -r requestments.txt' first!!!
# 
# 
# --------------------------------------------------
import cv2
import numpy as np
import os 
import pathlib
os.chdir(pathlib.Path(__file__).parent.absolute())
from multiprocessing import Pool

def toGray(img): 
    # change color image to grayscaler
    xs = img.shape[0]
    ys = img.shape[1]
    gray_mat = np.zeros((xs,ys))
    for i in range(xs):
        for j in range(ys):
            gray_mat[i,j] = img[i,j,2]//3 + img[i,j,1]//3 + img[i,j,0]//3
    
    return gray_mat.astype(np.uint8)

def convolution(img,ker): # the convolution method
    xs = img.shape[0]-(ker.shape[0]-1)
    ys = img.shape[1]-(ker.shape[1]-1)
    conv_mat = np.zeros((xs,ys))
    for i in range(xs):
        for j in range(ys):
            conv_mat[i][j] = (img[i:i+ker.shape[0],j:j+ker.shape[1]]*ker).sum()
            
    return conv_mat


def cannyEnhancer(img,n,sig): 
    yd = np.zeros((n,n))
    xd = np.zeros((n,n))
    gau = np.zeros((n,n))
    p = -(n-1)//2
    for x in range(p,p+n):
        for y in range(p,p+n):
            # gaussian kernel filter for image_res1
            gau[x-p,y-p] = (1/(2*np.pi*(sig**2)))*np.exp(-(x**2+y**2)/(2*(sig**2))) 

            # first derivative gaussian kernel with vertical direction
            xd[x-p,y-p] = -(1/(2*np.pi*(sig**4)))*(x*np.exp(-(x**2+y**2)/(2*(sig**2)))) 

            # first derivative gaussian kernel with horizontal direction
            yd[x-p,y-p] = -(1/(2*np.pi*(sig**4)))*(y*np.exp(-(x**2+y**2)/(2*(sig**2))))
    
    jx = convolution(img,xd)
    jy = convolution(img,yd)
    gimg = convolution(img,gau)

    es = np.sqrt(jx**2+jy**2)
    eo = np.arctan2(jx,jy) 
    
    es = (es/es.max())*255 # normalize of es 
    return es,eo,gimg

def nonMaximalSuppression(es,eo,gau):
    # classify all es direction
    canny = gau.copy()
    for i in range(1,eo.shape[0]-1):
        for j in range(1,eo.shape[1]-1):
            if((0 <= eo[i,j] < np.pi / 8) or (15 * np.pi / 8 <= eo[i,j] <= 2 * np.pi)):
                if( es[i,j] < es[i+1,j] or es[i,j] < es[i-1,j]):
                    canny[i,j] = 0
                else:
                    canny[i,j] = es[i,j]

            elif((np.pi / 8 <= eo[i,j] < 3 * np.pi / 8) or (9 * np.pi / 8 <= eo[i,j] < 11 * np.pi / 8)):
                if( es[i,j] < es[i+1,j-1] or es[i,j] < es[i-1,j+1]):
                    canny[i,j] = 0
                else:
                    canny[i,j] = es[i,j]

            elif((3 * np.pi / 8 <= eo[i,j] < 5 * np.pi / 8) or (11 * np.pi / 8 <= eo[i,j] < 13 * np.pi / 8)):
                if( es[i,j] < es[i,j+1] or es[i,j] < es[i,j-1]):
                    canny[i,j] = 0
                else:
                    canny[i,j] = es[i,j]

            else:
                if( es[i,j] < es[i-1,j-1] or es[i,j] < es[i+1,j+1]):
                    canny[i,j] = 0
                else:
                    canny[i,j] = es[i,j]

    return canny

def edge_connection(canny,strong,weak,neighbor_size):
# use two threshold method detecte edge
    n = (neighbor_size-1)//2
    thre = canny.copy()
    for i in range(n,thre.shape[0]-n):
        for j in range(n,thre.shape[1]-n):
            if(thre[i,j] >= strong):
                thre[i,j] = 255
            elif(thre[i,j] <= weak):
                thre[i,j] = 0
            else:
                if(np.max(thre[i-n:i+n+1,j-n:j+n+1]) >= strong):
                    thre[i,j] = 255
                else:
                    thre[i,j] = 0

    return thre

def houghTransform(th):
    y,x = np.where(th == 255)
    A = np.zeros((int(np.sqrt(th.shape[0]**2+th.shape[1]**2)),181))

    thela = [i for i in range(0,181)]
    for i in range(len(x)):

        for the in thela:
            rho = x[i]*np.cos(the*np.pi/180)+y[i]*np.sin(the*np.pi/180)
            rho = len(y) * ( 1.0 * rho ) / len(y)
            A[int(rho),the] += 1
    
    return A

def plotTheLine(A,img,th):
    # plot the line use two-point form
    image = img.copy()
    rho,thela = np.where(A > th)
    
    for i in range(len(rho)):
        y0 = int(rho[i]/np.sin(thela[i]*np.pi/180))
        y1 = int((rho[i]-(np.cos(thela[i]*np.pi/180)*img.shape[1]))/np.sin(thela[i]*np.pi/180))
        cv2.line(image,(0,y0),(img.shape[1],y1),(0,0,255),1) 

    return image

def getfaster(path,folder): 
    # speed up by multiproccessing

    img = cv2.imread(path)
    img_gray = toGray(img)
    es,eo,img_gauss = cannyEnhancer(img_gray,3,1)
    img_canny = nonMaximalSuppression(es,eo,img_gauss)
    img_th = edge_connection(img_canny,100,50,3)
    hough = houghTransform(img_th)
    resault = plotTheLine(hough,img,70)

    cv2.imwrite(folder+'/resault_img1.PNG',img_gauss.astype(np.uint8))
    cv2.imwrite(folder+'/es.PNG',es.astype(np.uint8))
    cv2.imwrite(folder+'/img_canny.PNG',img_canny.astype(np.uint8))
    cv2.imwrite(folder+'/tryCanny.PNG',img_th.astype(np.uint8))
    cv2.imwrite(folder+'/resault_img3.PNG',resault)
    cv2.imwrite(folder+'/Gray.PNG',img_gray)


if __name__ == '__main__':
    
    
    path = ['1.jpg','2.jpg','3.jpg','4.jpg'] #input Image
    path = ["input/"+ p for p in path] 
    
    folder = ['first','second',"third",'forth'] # build output image folder
    folder = ["result/"+ f for f in folder]
    for f in folder:
        pathlib.Path(f"{f}").mkdir(parents=True, exist_ok=True)
        
        
    pool = Pool(len(path)) # multiprocessing
    pool.starmap(getfaster,zip(path,folder))
    

    






