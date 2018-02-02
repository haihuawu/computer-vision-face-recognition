from PIL import Image
import numpy as np

img1name = "subject01.normal.jpg"
img2name = "subject02.normal.jpg"
img3name = "subject03.normal.jpg"
img7name = "subject07.normal.jpg"
img10name = "subject10.normal.jpg"
img11name = "subject11.normal.jpg"
img14name = "subject14.normal.jpg"
img15name = "subject15.normal.jpg"


img1cl = "subject01.centerlight.jpg"
img1h = "subject01.happy.jpg"
img7cl = "subject07.centerlight.jpg"
img7h = "subject07.happy.jpg"
img11cl = "subject11.centerlight.jpg"
img11h = "subject11.happy.jpg"
img12name = "subject12.normal.jpg"
img14h = "subject14.happy.jpg"
img14s = "subject14.sad.jpg"


imgapple = "apple1_gray.jpg"

imgarray = [img1name,img2name,img3name,img7name,img10name,img11name,img14name,img15name]

testimgarray = [img1name,img2name,img3name,img7name,img10name,img11name,img14name,img15name,
                img1cl, img1h, img7cl, img7h, img11cl, img11h, img12name, img14h, img14s, imgapple]

#threshold
T0 = 6.42001150109e+12
T1 = 130964037


trainimgnum = 8
width = 195
height = 231
matrixh = height*width

#show mean face
def show_mean_face():
    X = np.reshape(meanfacearr, (height, width))
    meanface = Image.fromarray(X)
    meanface.show()
    #save mean face
    # if meanface.mode != 'RGB':
    #     meanface = meanface.convert('RGB')
    # meanface.save('meanf.bmp')


#compute mean face
def compute_mean_face():
    #dimens w*h  nums 8
    dimens,nums = arrayA.shape[:2]
    meanarr=[]
    for i in range(dimens):
        meanval = (sum(arrayA[i,:])/nums)
        meanarr.append(meanval)
    return np.array(meanarr)

#compute difference between two arrays
def compute_diff(arr1,arr2):
    return arr1-arr2

#training face substract mean face
def substract_mean_face():
    diffs=[]
    dimens, nums = arrayA.shape[:2]
    for i in range(nums):
        diffs.append(compute_diff(arrayA[:,i],meanfacearr))
        # print(diffs[i])
    return np.array(diffs).T

#8 training images transfer to array
def generate_matrixA():
    #generate matrixA  nn x 8
    arr = []
    # trainimgnum : number of training image
    for i in range(trainimgnum):
        trainimg = Image.open(imgarray[i])
        # transfer image to numpy array
        data = np.asarray(trainimg,np.float64)
        # stack rows together
        arr.append(data.flatten())
        # print(arr[i])

    return np.array(arr).T

#scale eigenface [0,255]
def scale_eigenface(arr):
    X = arr
    minX, maxX = np.min(X), np.max(X)
    # print(minX,maxX)
    X = X - float(minX)
    X = X / float((maxX - minX))
    X = X * 255
    return X

#show eigen face one by one
def show_eigen_face():
    # eigenfset = []
    for x in range(trainimgnum):
        eigenfacearr = eigenface[:,x]
        X = scale_eigenface(eigenfacearr)
        # eigenface[:,x] = X
        X = np.reshape(X, (height, width))
        eigenfaceimg = Image.fromarray(X)
        eigenfaceimg.show()
        # eigenfset.append(eigenfaceimg)

    # for i in range(trainimgnum):
    #     if eigenfset[i].mode != 'RGB':
    #         eigenfset[i] = eigenfset[i].convert('RGB')
    #     eigenfset[i].save('eigenf' + str(i+1)+'.bmp')

#generate face space
def generate_face_space():
    npL = np.dot(arr_diff.T, arr_diff)
    # print(npL)
    eigenValues, eigenVectors = np.linalg.eig(npL)
    # print("below is eigenvector")
    # print(eigenVectors)
    return eigenValues,np.dot(arr_diff,eigenVectors)

#compute weight
def compute_weight(eigenf,img):
    return np.dot(eigenf.T,img)

#reconstruct test image
def reconstruct_img(w):
    return np.dot(eigenface,w)

#compute distance via euclidean distance
def compute_distance(i1,i2):
    diffi = i1-i2
    diffi = diffi**2
    asum = diffi.sum()
    dist = asum**0.5
    return dist

def show_test_img(type):
    for i in range(18):
        if type == 1:
            set = testdiffimgset
        else:
            set = testrecimgset
        arr = np.reshape(set[i], (height, width))
        testdiffimg = Image.fromarray(arr)
        testdiffimg.show()

        if type == 1 and testdiffimg.mode != 'RGB':
            testdiffimg = testdiffimg.convert('RGB')
            testdiffimg.save('testdiffimg' + str(i+1)+'.bmp')
        else:
            testdiffimg = testdiffimg.convert('RGB')
            testdiffimg.save('testconsimg' + str(i + 1) + '.bmp')

#recognition test images
def recognition(imagename):
    img = Image.open(imagename)
    test_arr = np.asarray(img,np.float64).flatten().T
    #subtract mean face
    test_diff = compute_diff(test_arr,meanfacearr)
    testdiffimgset.append(test_diff)


    #compute its projection onto face space
    test_weight = compute_weight(eigenface,test_diff)
    # print("test image PCA coefficient is")
    # print(test_weight.T)

    #reconstruct input face image from eigenfaces
    test_diff_re = reconstruct_img(test_weight)
    testrecimgset.append(test_diff_re)

    #compute d0
    dist0 = compute_distance(test_diff_re,test_diff)
    # print("d0 is " + str(dist0))
    if dist0 > T0:
        print("this image is not a face.")
    #compute di
    darray = []
    for i in range(trainimgnum):
        trainweight = weight.T
        di = compute_distance(test_weight,trainweight[i])
        darray.append(di)
        # print("d" + str(i+1) +" is " + str(di))

    #find minimum dj
    mind = darray[0]
    res = 0
    for i in range(1,trainimgnum):
        if mind > darray[i]:
            mind = darray[i]
            res = i

    print("mind is "+str(mind))
    if mind < T1:
        print(imagename + "    " + imgarray[res])


#main function
if __name__ == '__main__':
    #original matrix A  pq*m
    arrayA = generate_matrixA()
    #mean face array pq*1
    meanfacearr = compute_mean_face()
    show_mean_face()
    #diff between array A and mean face array
    arr_diff = substract_mean_face()

    eigenvalues,eigenface = generate_face_space()
    # print("this is eigenface before scale")
    # print(eigenface)

    # show_eigen_face()
    # print("this is eigenface after scale")
    # print(eigenface)

    weight = compute_weight(eigenface,arr_diff)
    # print("PCA coefficient is")
    # print(weight.T)

    testdiffimgset = []
    testrecimgset = []
    for i in range(18):
        imagen = testimgarray[i]
        # print(str(i+1) + " " + str(imagen) )

        recognition(testimgarray[i])

    # show_test_img(1)
    # show_test_img(2)