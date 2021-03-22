import matplotlib.pyplot as plt
import numpy as np
import os,time
import xlrd,xlwt
from PIL import Image

'''初始化'''
order=0
reader = xlrd.open_workbook('Homework9.xlsx')
workbook = xlwt.Workbook(encoding='utf-8')
sheetname=''
setname=''
plus=''
Sample = []
SetI=[]
SetII=[]
nrows = 0
ncols = 0
W=[]

'''读取样本和数据集'''
while True:
    print('请选择任务：输入1训练In6_Memorizing，输入其他训练In25_Memorizing')
    order=input()
    print('请输入样本名称：（针对In6_Memorizing请输入“In6_Associating”，针对In25_Memorizing请输入“In25_Associating”或“In25_Associating_2”）')
    sheet=input()
    print('请输入最大迭代次数：')
    Max_iteration=(int)(input())
    '''数据集读取(sheet选择)'''
    if order=='1':
        sheetname='In6_Memorizing'
        setname=sheet
    else:
        sheetname='In25_Memorizing'
        setname=sheet

    try:
        table= reader.sheet_by_name(sheetname)
        nrows = table.nrows
        ncols = table.ncols
        for row in range(1, nrows):
            Sample.append(table.row_values(row, start_colx=1, end_colx=None))
        Sample = np.array(Sample).reshape(nrows - 1, ncols-1)

        table= reader.sheet_by_name(setname)
        nrows = table.nrows
        ncols = table.ncols
        for row in range(1, nrows):
            SetI.append(table.row_values(row, start_colx=1, end_colx=None))
        SetI = np.array(SetI).reshape(nrows - 1, ncols-1)

        break
    except:
        continue
print('训练样本：',Sample)

newimg = Image.new("RGB", (100*len(Sample[0]),len(Sample)*100))
for r in range(len(Sample)):
    for c in range(len(Sample[r])):
        if Sample[r][c]==-1:
            for rpoint in range(100):
                for cpoint in range(100):
                    newimg.putpixel((c*100+cpoint,r*100+rpoint), (255,255,255))
newimg.save("原始样本.jpg")
os.startfile(str("原始样本.jpg"))

'''权值生成'''
W=np.zeros((len(Sample[0]),len(Sample[0])))
for pointI in range(len(Sample[0])):
    for pointII in range(pointI+1,len(Sample[0])):
        W[pointI][pointII]=W[pointII][pointI]=np.sum(Sample[:,pointI]*Sample[:,pointII])
print('权值矩阵：',W)


'''输入样本集进行迭代'''
count=1
start=Sample.copy()
Min_Mat=[]
Min_Energy=[np.float('inf')]*len(SetI)
while True:
    count+=1
    net=np.dot(SetI,W)
    netout=net.copy()

    Energy=-0.5*np.sum(SetI*netout,axis=1)
    if (Energy<=Min_Energy).all:
        Min_Energy=Energy
        Min_Mat=netout

    for row in range(len(net)):
        for col in range(len(net[row])):
            if net[row][col]>0:
                netout[row][col]=1
            elif net[row][col]<0:
                netout[row][col]=-1
            elif net[row][col]==0:
                netout[row][col]=SetI[row][col]
    if (netout==start).all():
        break
    SetI=netout.copy()
    if count>=Max_iteration:
        break

'''输出结果并作图'''
print('共计迭代',count,'次，','Min-Eneergy:',Min_Energy)
print('输出图像：\n',Min_Mat)

newimg = Image.new("RGB", (100*len(Sample[0]),len(Sample)*100))
for r in range(len(Min_Mat)):
    for c in range(len(Min_Mat[r])):
        if Min_Mat[r][c]==-1:
            for rpoint in range(100):
                for cpoint in range(100):
                    newimg.putpixel((c*100+cpoint,r*100+rpoint), (255,255,255))

newimg.save("判定结果.jpg")
os.startfile(str('判定结果.jpg'))