import xlwt,xlrd,os
import numpy as np
import matplotlib.pyplot as plt


#activation function
def AcFunc(list):
    Y=[]
    for point in range(len(list)):
        if list[point]>0:
            Y.append(1)
        elif list[point]<0:
            Y.append(0)
    return np.array(Y)
'''_____________________________________Part I____________________________________________'''
#open xlsx
reader=xlrd.open_workbook('Homework4.xlsx')
workbook = xlwt.Workbook(encoding='utf-8')
print("Part.1!污污污污污~在画了！在画了！")

# getsheet
table = reader.sheet_by_index(0)
# getrows&cols
nrows = table.nrows
ncols = table.ncols
# initialize
Mat = []
W = [0, 0, 1.1]
yita = 1
# getMatrix
for row in range(1, nrows):
    Mat.append(table.row_values(row, start_colx=0, end_colx=None))
# format
Mat = np.array(Mat).reshape(nrows - 1, ncols)
X0 = np.ones((nrows - 1, 1))
X1 = np.array(Mat[:, 1]).reshape(nrows - 1, 1)
X2 = np.array(Mat[:, 2]).reshape(nrows - 1, 1)

# get target value
T = np.array(Mat[:, 3])
# set "Once-for-all" X Matrix
X = np.vstack((X0.T, X1.T, X2.T)).T
plt.ion()
# push forward to get Y
iteration=0
count = 0
for circle in range(50):
    plt.cla()
    C=circle%(nrows-1)
    net = np.array(np.dot(X[C], W)).reshape(1,1)
    Y = AcFunc(net)
    # perceptron rules
    if net <= 0 and T[C] == 1:
        W = W + yita * X[C]
    elif net >= 0 and T[C] == 0:
        W = W - yita * X[C]
    else:
        W = W
    # check effect of module
    count = 0
    net = np.array(np.dot(X, W))
    Y = AcFunc(net)
    Target = (-W[1] / W[2]) * X1 - W[0] / W[2]
    plt.scatter(X1[:, 0], X2[:, 0], c=Y[:])
    plt.plot(X1, Target)
    plt.pause(0.2)
    for i in range(len(Y)):
        if Y[i] == T[i]:
            count += 1
    # Expect accuracy equals to 100%
    if count / (nrows - 1) >= 1:
        break
    iteration+=1
plt.ioff()
print("第一个数据集共计迭代训练：",iteration,"次，精确度为"+str(count*100/(nrows-1))+"%")
# plot original map
plt.subplot('121')
plt.title('Original Map of Sheet.0')
plt.scatter(X1[:, 0], X2[:, 0], c=T[:])
plt.grid()
# plot machine's classify map
net = np.array(np.dot(X, W))
Y = AcFunc(net)
plt.subplot('122')
plt.title("Perceptron's Work of Sheet.0")
plt.scatter(X1[:, 0], X2[:, 0], c=Y[:])
Target = (-W[1] / W[2]) * X1 - W[0] / W[2]
plt.plot(X1, Target)
plt.grid()
plt.tight_layout()
print('画完了画完了！')
plt.show()
# build sheet
sheet = workbook.add_sheet('Module.0', cell_overwrite_ok=True)
# write into excel
title = ['Bias', 'W1', 'W2']
for col in range(len(W)):
    sheet.write(0, col, title[col])
for col in range(len(W)):
    sheet.write(1, col, W[col])
# save as excel V2003 which been known as xls,xlsx cannot be write correctly

# ___________________________________Part II____________________________________________
print("Part.2!污污污污污~在画了！在画了！")
for sheet in range(1,3):
    #getsheet
    table= reader.sheet_by_index(sheet)
    #getrows&cols
    nrows = table.nrows
    ncols = table.ncols
    #initialize
    Mat=[]
    W=[0,0,1.1]
    yita=1
    #getMatrix
    for row in range(1,nrows):
        Mat.append(table.row_values(row, start_colx=0, end_colx=None) )
    #format
    Mat=np.array(Mat).reshape(nrows-1,ncols)
    X0=np.ones((nrows-1,1))
    X1=np.array(Mat[:,1]).reshape(nrows-1,1)
    X2=np.array(Mat[:,2]).reshape(nrows-1,1)

    #get target value
    T=np.array(Mat[:,3])
    #set "Once-for-all" X Matrix
    X=np.vstack((X0.T,X1.T,X2.T)).T

    #iterate
    count = 0
    iteration=0
    ExpectAccuracy=[1,1,0.95]
    plt.ion()
    for circle in range(50):
        #push forward to get Y
        net=np.array(np.dot(X,W))
        Y=AcFunc(net)
        plt.cla()
        #pull backward & fix Weight
        for i in range(len(net)):
            #perceptron rules
            if net[i]<=0 and T[i]==1:
                W = W + yita * X[i]
            elif net[i]>=0 and T[i]==0:
                W = W - yita * X[i]
            else:
                W=W

        #check effect of module
        count=0
        net=np.array(np.dot(X,W))
        Y=AcFunc(net)
        Target = (-W[1] / W[2]) * X1 - W[0] / W[2]
        plt.scatter(X1[:, 0], X2[:, 0], c=Y[:])
        plt.plot(X1, Target)
        plt.pause(0.2)
        for i in range(len(Y)):
            if Y[i]==T[i]:
                count+=1
        #Expect accuracy equals to 100%
        if count/(nrows-1)>=ExpectAccuracy[sheet]:
            break
        iteration+=1
    plt.ioff()
    print("数据集"+str(sheet)+"共计迭代训练：",iteration,"次，精确度为"+str(count*100/(nrows-1))+"%")
    # plot original map
    plt.subplot('121')
    plt.title('Original Map of Sheet.' + str(sheet))
    plt.scatter(X1[:, 0], X2[:, 0], c=T[:])
    plt.grid()
    # plot machine's classify map
    net = np.array(np.dot(X, W))
    Y = AcFunc(net)
    plt.subplot('122')
    plt.title("Perceptron's Work of Sheet." + str(sheet))
    plt.scatter(X1[:, 0], X2[:, 0], c=Y[:])
    Target = (-W[1] / W[2]) * X1 - W[0] / W[2]
    plt.plot(X1, Target)
    plt.grid()
    plt.tight_layout()
    print('画完了画完了！')
    plt.show()

    # build sheet
    sheet = workbook.add_sheet('Module.'+str(sheet+1), cell_overwrite_ok=True)
    # write into excel
    title=['Bias','W1','W2']
    for col in range(len(W)):
        sheet.write(0, col, title[col])
    for col in range(len(W)):
        sheet.write(1, col,W[col])
    # save as excel V2003 which been known as xls,xlsx cannot be write correctly
# ___________________________________________Part III_______________________________________
print('\n','Part.3：换用方案B训练非线性模型')
n=0.8
acu=[]
axis=[]

def SIG(x,deriv=False):
    if (deriv == True):
        return (OriginalSIG(x)-OriginalSIG(x)**2)
    return 1/(1+np.exp(-x))
def OriginalSIG(x):
    return 1/(1+np.exp(-x))
#getsheet
table = reader.sheet_by_index(2)
# getrows&cols
nrows = table.nrows
ncols = table.ncols
# initialize
Mat = []
# getMatrix
for row in range(1, nrows):
    Mat.append(table.row_values(row, start_colx=0, end_colx=None))
# format
Mat = np.array(Mat).reshape(nrows - 1, ncols)
X0 = np.ones((nrows - 1, 1))
X1 = np.array(Mat[:, 1]).reshape(nrows - 1, 1)
X2 = np.array(Mat[:, 2]).reshape(nrows - 1, 1)

# get target value
T = np.array(Mat[:, 3]).reshape(nrows-1,1)
# set "Once-for-all" X Matrix
X = np.vstack((X0.T, X1.T, X2.T)).T

sigma=np.array([])
output=np.array([])

np.random.seed(1)
w0 = np.random.random((3,3))
w1 = np.random.random((3,1))

plt.ion()
for i in range(50000):
    L0 = X
    L1 = SIG(L0.dot(w0))
    L2 = SIG(L1.dot(w1))

    L2_error = L2 - T
    acu.append((1 - np.mean(np.abs(L2_error))))
    axis.append(i)
    if(i%100 == 0):
        plt.cla()
        for ele in range(len(L2)):
            if L2[ele] >= 0.5:
                L2[ele] = 1
            else:
                L2[ele] = 0
        plt.scatter(X[:, 1], X[:, 2], c=L2[:,0])
        plt.pause(0.02)
        print('第'+str(i)+'次迭代-精确度：',acu[-1],"%")
    if acu[-1]>0.99:
        break

    L2_delta = L2_error * SIG(L1.dot(w1),deriv = True)

    L1_error = L2_delta.dot(w1.T)
    L1_delta = L1_error * SIG(L0.dot(w0),deriv = True)

    w1 -= n*L1.T.dot(L2_delta)
    w0 -= n*L0.T.dot(L1_delta)
plt.ioff()
plt.cla()
plt.close()
#模型检测对比
L0 = X
L1 = SIG(L0.dot(w0))
L2 = SIG(L1.dot(w1))
L2=np.array(L2[:,0])

for ele in range(len(L2)):
    if L2[ele]>=0.5:
        L2[ele]=1
    else:
        L2[ele]=0

plt.subplot('131')
plt.title('Original Map')
plt.scatter(X[:,1], X[:,2],c=T[:,0])
plt.grid()

plt.subplot('132')
plt.title("Machine's Map")
plt.scatter(X[:,1], X[:,2],c=L2[:])
plt.grid()

plt.subplot('133')
plt.title('Fit Line')
plt.plot(range(len(acu)),acu)
plt.grid()
plt.show()

sheet = workbook.add_sheet('NN_Module', cell_overwrite_ok=True)
sheet.write(0, 0, 'W0')
for row in range(len(w0)):
    for col in range(len(w0[row])):
        sheet.write(row+1, col, w0[row][col])

sheet.write(len(w0)+2, 0, 'W1')
for row in range(len(w1)):
    for col in range(len(w1[row])):
        sheet.write(row+len(w0)+3, col, w1[row][col])
print('在写了！在写了！')
workbook.save('Answer.xls')
print('写好了！写好了！')
os.startfile(str('Answer.xls'))
