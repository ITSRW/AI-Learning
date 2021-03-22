import matplotlib.pyplot as plt
import numpy as np
import os,time
import xlrd
import xlwt
from mpl_toolkits.mplot3d import Axes3D

'''初始化'''
reader = xlrd.open_workbook('Homework8.xlsx')
workbook = xlwt.Workbook(encoding='utf-8')
sheetname=''
Max_iteration=0
Yita=0
R=0
Lambda=0
Yita_min=0
Mat = []
nrows = 0
ncols = 0
W=[]


'''输入参数'''
while True:
    print('请设置初始化参数，按照η、R、λ、η下限、最大迭代次数的顺序输入，用空格分隔：')
    print('（默认与推荐输入：0.1 8 0.995 0.001 100，直接回车使用默认输入）')
    parameters=input()
    paralist=[]
    if parameters=='':
        parameters='0.1 8 0.995 0.001 100'
    try:
        paralist=parameters.split(' ')
        Yita=(float)(paralist[0])
        R=(float)(paralist[1])
        Lambda=(float)(paralist[2])
        Yita_min=(float)(paralist[3])
        Max_iteration=(int)(paralist[4])
        break
    except:
        continue
while True:
    print('请输入Sheet名称：')
    sheetname=input()
    '''数据集读取(sheet选择)'''
    try:
        table= reader.sheet_by_name(sheetname)
        nrows = table.nrows
        ncols = table.ncols
        for row in range(1, nrows):
            Mat.append(table.row_values(row, start_colx=1, end_colx=None))
        Mat = np.array(Mat).reshape(nrows - 1, ncols-1)
        break
    except:
        continue

netrow=20
netcol=15

'''初始化：建立初始群心立方和群心计数器'''
for row in range(netrow):
    W.append((np.random.random((netcol, ncols-1))).tolist())##????是否规范?
W=np.array(W)


dis=0
if len(Mat[0])==3:
    ax = plt.figure().add_subplot(111, projection = '3d')
'''迭代：按照公式修正群心立方中的各项属性值（即权值，亦即多维坐标值）'''
if len(Mat[0])==1 or len(Mat[0])==2 or len(Mat[0])==3:
    plt.ion()
time_start = time.time()
for T in range(Max_iteration):
    if len(Mat[0])==1 or len(Mat[0])==2 or len(Mat[0])==3:
        plt.cla()
    for point in range(len(Mat)):
        dis=float('inf')
        logr=-1
        logc=-1
        for DI in range(len(W)):
            for DII in range(len(W[DI])):
                temp=np.sum((Mat[point] - W[DI][DII])**2)**0.5
                if temp<=dis:
                    dis=temp
                    logr=DI
                    logc=DII
        for DI in range(len(W)):
            for DII in range(len(W[DI])):
                W[DI][DII] += Yita * np.exp(-(((logr-DI)**2+(logc-DII)**2)/R**2))*(Mat[point]-W[DI][DII])
    if Yita*Lambda<=Yita_min:
        Yita=Yita_min
    else:
        Yita*=Lambda
    if len(Mat[0]) == 1:
        plt.scatter(Mat[:, 0],np.array([0]*len(Mat[:, 0])))
        plt.scatter(W[:, :, 0].reshape(netrow*netcol, 1),np.array([0]*netrow*netcol))
        plt.title('Iteration:' + str(T + 1) + ';Min-Dis:' + str(dis))
        plt.pause(0.01)
    if len(Mat[0]) == 2:
        plt.scatter(Mat[:,0],Mat[:,1])
        plt.scatter(W[:,:,0].reshape(netrow*netcol,1),W[:,:,1].reshape(netrow*netcol,1))
        plt.title('Iteration:'+str(T+1)+';Min-Dis:'+str(dis))
        plt.pause(0.01)
    if len(Mat[0]) == 3:
        ax.scatter(Mat[:,0], Mat[:,1], Mat[:,2], c='b', marker='^')
        ax.scatter(W[:,:,0].reshape(netrow*netcol,1), W[:,:,1].reshape(netrow*netcol,1), W[:,:,2].reshape(netrow*netcol,1), c='r', marker='^')
        plt.title('Iteration:'+str(T+1)+';Min-Dis:'+str(dis))
        plt.pause(0.01)
        # plt.savefig('C:\\Users\\83810\\Desktop\\1\\'+str(T)+'.PNG')
    if (T+1)%10==0:
        print('Iteration:'+str(T+1)+'；Min-Dis:'+str(dis))
time_end = time.time()
print("训练用时：",time_end-time_start,"秒")
if len(Mat[0])==1 or len(Mat[0])==2 or len(Mat[0])==3:
    plt.ioff()
    plt.show()



'''完成迭代，通过计算欧氏距离将所有数据归入拓扑空间'''
count=np.zeros((netrow,netcol))
for point in range(len(Mat)):
    logr = -1
    logc = -1
    min = float("inf")
    dis = 0
    for DI in range(len(W)):
        for DII in range(len(W[DI])):
            dis=np.sum((W[DI][DII] - Mat[point]) ** 2) ** 0.5
            if min>=dis:
                logr=DI
                logc=DII
                min=dis
    if logr!=-1 and logc!=-1:
        count[logr][logc]+=1



'''输出拓扑空间'''
map=[]
print(count)
for row in range(len(count)):
    for col in range(len(count[row])):
        if count[row][col]!=0:
            map.append([row,col])
map=np.array(map)
plt.scatter(map[:,0],map[:,1])
plt.show()



'''输出群心'''
sheet = workbook.add_sheet('HW8_Weight', cell_overwrite_ok=True)
sheet.write(0, 0, '群心行号')
sheet.write(0, 1, '群心列号')
for index in range(len(Mat[0])):
    sheet.write(0, 1+(index+1), '属性'+str(index+1))
sheet.write(0, len(Mat[0])+2, '归入点计数')

POINT=1
for row in range(len(W)):
    for col in range(len(W[row])):
        sheet.write(POINT, 0, row)
        sheet.write(POINT, 1, col)
        for p in range(len(W[row][col])):
            sheet.write(POINT, p+2, W[row][col][p])
        sheet.write(POINT, len(W[row][col])+2, count[row][col])
        POINT+=1

workbook.save('HW8_Weight.xls')
print('写入完毕！正在打开...')
os.startfile(str('HW8_Weight.xls'))