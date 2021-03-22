import matplotlib.pyplot as plt
import numpy as np
import os,time
import xlrd
import xlwt

'''//-----------------代码块0：激活函数-----------------------'''
# #sigmoid 激活函数 & 反向传播时的导函数
# def sig(x, deriv=False):
#     if (deriv == True):
#         return (sig(x) - sig(x) ** 2)
#     return 1 / (1 + np.exp(-x))

# tanh 激活函数
def sig(x,deriv=False):
    if (deriv == True):
        return (1-sig(x)**2)
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

# #Relu 激活函数
# def sig(x,deriv=False):
#     if (deriv == True):
#         np.where(x > 0, 1, 0)
#     np.where(x > 0, x, 0)
#     return (x)
'''---------------------------------------------------------//'''



'''//------------------代码块1：初始化变量集------------------'''
#初始化指令容器 & 初始化节点数量容器 & 初始化前后学习率 & 初始化惯性率 & 初始化目标相对精度
order=''
setNN = ''
yitaI=0
yitaII=0
alpha=0
target=''

#初始化表格选择器
sheetname=''
Drawsheetname=''

#初始化输入边界
inputleftbond=0
inputrightbond=0
inlen=0

#初始化输出边界
outputleftbond=0
outputrightbond=0
outlen=0

# 权重容器
w10 = []
w11 = []

w20 = []
w21 = []

w30 = []
w31 = []

# 样本集文件与权重输出文件预加载、数据输出指针初始化
reader = xlrd.open_workbook('Homework7.xlsx')
workbook = xlwt.Workbook(encoding='utf-8')
logNNI=logNNII=logNNIII=0
INI=INII=INIII=0
OUI=OUII=OUIII=0
'''-------------------------------------------------------------//'''



'''//--------------------代码块2：整活区------------------'''
#开始整活
while order != '0':
    '''//--------------------代码块2.1：容器与变量集还原-----------------'''
    order = ''
    setNN = ''
    target = ''
    sheetname = ''
    Drawsheetname = ''
    yitaI = 0
    yitaII = 0
    alpha = 0

    iteration = 0
    itelim=0
    L0, L1, L2 = [], [], []
    Error = [0]

    #打点精确计
    loss = [1]
    lossvector=[1,1,1,1,1,1,1,1]
    lossI = [1]
    lossII = [1]

    #迭代轴容器
    AXIS = [1]

    #Δ记录器
    deltaOutputs = []
    deltaHid = []
    '''---------------------------------------------------------------------//'''



    '''//--------------------代码块2.2：模式选择---------------------------'''
    print('请选择训练任务，训练 In1Out1 请Call 1，训练In1Out8请Call 2，训练分类问题请Call 3，退出请Call 0：')
    while order == '':
        order = input()
    #任务一
    if order=='1':
        sheetname='In1Out1'
        Drawsheetname='Drawing_In1Out1'

        inputleftbond = 1
        inputrightbond = 2
        INI=inlen=inputrightbond-inputleftbond

        outputleftbond = 2
        outputrightbond = 3
        OUI=outlen=outputrightbond-outputleftbond
    #任务二
    elif order=='2':
        sheetname = 'In1Out8'
        Drawsheetname = 'Drawing_In1Out8'

        inputleftbond = 1
        inputrightbond = 2
        INII=inlen=inputrightbond-inputleftbond

        outputleftbond = 2
        outputrightbond = 10
        OUII=outlen=outputrightbond-outputleftbond
    #任务三
    elif order=='3':
        sheetname = 'Classification'
        Drawsheetname = 'Classification'

        inputleftbond = 1
        inputrightbond = 3
        INIII=inlen=inputrightbond-inputleftbond

        outputleftbond = 3
        outputrightbond = 5
        OUIII=outlen=outputrightbond-outputleftbond
    #其他输入
    else:
        break
    '''---------------------------------------------------//'''



    '''//---------------代码块2.3：样本数据集导入--------------'''
    # getsheet
    tableI = reader.sheet_by_name(sheetname)
    tableII = reader.sheet_by_name(Drawsheetname)

    #提取随机的样本集
    nrows = tableI.nrows
    ncols = tableI.ncols
    Mat = []
    for row in range(1, nrows):
        Mat.append(tableI.row_values(row, start_colx=0, end_colx=None))

    Mat = np.array(Mat).reshape(nrows - 1, ncols)
    X0 = np.ones((nrows - 1, 1))
    X1 = np.array(Mat[:, inputleftbond:inputrightbond]).reshape(nrows - 1, inlen)
    T = np.array(Mat[:, outputleftbond:outputrightbond])
    X = np.vstack((X0.T, X1.T)).T

    #提取有序的样本集
    nrows = tableII.nrows
    ncols = tableII.ncols
    Mat = []
    for row in range(1, nrows):
        Mat.append(tableII.row_values(row, start_colx=0, end_colx=None))

    Mat = np.array(Mat).reshape(nrows - 1, ncols)
    DrawX0 = np.ones((nrows - 1, 1))
    DrawX1 = np.array(Mat[:, inputleftbond:inputrightbond]).reshape(nrows - 1, inlen)
    DrawT =  np.array(Mat[:, outputleftbond:outputrightbond])
    DrawX = np.vstack((DrawX0.T, DrawX1.T)).T
    '''--------------------------------------------------------//'''



    '''//-------------------------代码块2.4：设定输入性参量--------------------------------'''
    # 设定节点个数
    print('初始化！自定义隐藏层神经元个数：')
    while setNN == '':
        setNN = (input())
        setNN = (int)(setNN)
        if order=='1':
            logNNI=setNN
        elif order=='2':
            logNNII=setNN
        elif order=='3':
            logNNIII=setNN
        if setNN<=0:
            setNN=''

    # 设定第一层学习率
    print('请设置输入-隐藏层学习率：（建议输入：一对一问题：0.2；一对八问题：0.3；分类问题：0.4；）')
    while yitaI == 0:
        yitaI = (float)(input())
        if yitaI<=0 or yitaI>=1:
            yitaI=0

    # 设定第一层学习率
    print('请设置隐藏层-输出层学习率：（建议输入：一对一问题：0.01；一对八问题：0.02；分类问题：0.01；）')
    while yitaII == 0:
        yitaII = (float)(input())
        if yitaII<=0 or yitaII>=1:
            yitaII=0

    # 设定惯性率
    print('请设置惯性率：（建议输入：一对一问题：0.1；一对八问题：0.1；分类问题：0.2；）')
    while alpha == 0:
        alpha = (float)(input())
        if alpha<=0 or alpha>=1:
            alpha=0

    #设定目标精度
    print('希望达到的MAE精度：(输入(0,1)之间的数)')
    while target == '':
        target = (input())
    target= (float)(target)
    if target<=0 or target>=1:
        continue

    print('设置最大迭代次数，超过该迭代次数后停止迭代：')
    while itelim == 0:
        itelim = (input())
    itelim= (int)(itelim)
    if itelim<=0:
        continue
    '''---------------------------------------------------------------------------------------//'''



    '''//-----------------------代码块2.5：权值矩阵与矩阵修正量容器初始化-----------------------'''
    # 随机产生两个权值矩阵（采取数据标准化映射方法把随机数从[0,1]映射到[-1,1]）
    w0 = np.random.random((1+inlen, setNN)) * 2 - 1
    w0[0,:]=np.random.random()* 2 - 1
    w1 = np.random.random((setNN + 1, outlen)) * 2 - 1
    w1[0,:]=np.random.random()* 2 - 1

    # w0=np.array([[0.1,0.3],
    #     [0.2,0.4]])
    # w1=np.array([[0.1],
    #     [0.2],
    #     [0.3]])

    #打印矩阵
    print('W0:\n',w0)
    print('W1:\n',w1)

    delta_W0=np.zeros((1+inlen, setNN))
    delta_W1=np.zeros((setNN + 1, outlen))
    '''-----------------------------------------------------------------------------------//'''



    '''！！！****************************代码块2.6：Adaline训练主程式*****************************！！！'''
    time_start = time.time()
    plt.ion()
    while (loss[-1])>target:#使用相对精度控制迭代
        plt.clf()
        '''//-----------------------代码块2.6.1：单次整体样本训练-----------------------'''
        for sample in range(len(X)):
            '''//-----------------------代码块2.6.1.1：前向传播-----------------------'''
            L0 = X[sample]
            L1 = sig(L0.dot(w0))
            L1 = np.insert(L1,0,1)
            L2 = (L1.dot(w1))
            # print("前向输出结果",L2)
            '''-----------------------------------------------------------------------//'''



            '''//-----------------------代码块2.6.1.2：反向传播-----------------------'''
            #计算偏差
            Error=T[sample]-L2

            #计算δ
            deltaOutputs = Error.copy()
            # print("输出层δ",deltaOutputs)
            deltaHid=np.multiply(deltaOutputs.dot(w1.T)[1:],sig(L0.dot(w0),deriv=True))
            # print("隐藏层δ",deltaHid)

            #计算 Δw

            delta_W1 = alpha * delta_W1  + yitaII * np.dot(np.mat(L1).T,np.mat(deltaOutputs))
            # for i in range(setNN + 1):
            #     delta_W1[i] = alpha * delta_W1[i]+ yitaII * deltaOutputs * L1[i]
            #     # for j in range(outlen):
            #     #     delta_W1[i][j] =alpha*delta_W1[i][j] + yitaII*deltaOutputs[j]*L1[i]

            delta_W0 = alpha * delta_W0 + yitaI * np.dot(np.mat(L0).T,np.mat(deltaHid))
            # for i in range(inlen+1):
            #     delta_W0[i] = alpha * delta_W0[i] + yitaI * deltaHid * L0[i]
            #     # for j in range(setNN):
            #     #     delta_W0[i][j] =alpha*delta_W0[i][j] + yitaI*deltaHid[j]*L0[i]

            #模型修正
            w1+=delta_W1
            w0+=delta_W0
            '''---------------------------------------------------------------------------//'''
        '''---------------------------------------------------------------------------//'''



        '''//-----------------------代码块2.6.2：模型校验与动态绘图-----------------------'''
        #产生用于检验与作图的输出值矩阵
        L0 = DrawX #有序样本用于作图
        L1 = sig(L0.dot(w0))
        # 隐藏层的Bias乘数向量塞进隐藏层
        L1 = np.vstack((X0.T, L1.T)).T
        # 第二段批量前向传播得出最终结果（一个向量）
        L2 = (L1.dot(w1))

        #针对任务1：一对一问题 的模型检验与动态绘图
        if order == '1':
            AXIS.append(iteration)
            loss.append(np.mean(np.abs(L2[:,0].T - DrawT[:,0])))
            if iteration == 1:
                del AXIS[0]
                del loss[0]
            # 画图（用来观察模型效果）
            plt.subplot('121')
            plt.plot(DrawX1, DrawT)
            plt.title("View Of Model In Epoch"+str(iteration))
            plt.plot(DrawX1, L2)
            plt.subplot('122')
            plt.title("MAE:"+str(round(loss[-1],4)))
            plt.plot(AXIS,loss)
            plt.tight_layout()
            plt.pause(0.01)
            if iteration % 10 == 0:
                print("第", iteration, "次迭代，", "相对MAE：", loss[-1])
            w10 = w0
            w11 = w1

        #针对 任务2：一对八问题 的模型检验与动态绘图
        elif order=='2':
            AXIS.append(iteration)
            if iteration == 1:
                del AXIS[0]
            #获取八个图中最大的Loss并转为精确度
            for i in range(outlen):
                lossvector[i]=np.average(np.abs(L2-DrawT)[:,i])
            loss.append(np.array(lossvector).max())

            plt.subplot("241")
            plt.title("Epoch:"+str(iteration)+";MAE:"+str(round(lossvector[0],4)))
            plt.plot(DrawX[:, 1], DrawT[:, 0])
            plt.plot(DrawX[:, 1], L2[:, 0])
            plt.subplot("242")
            plt.title("Epoch:"+str(iteration)+";MAE:"+str(round(lossvector[1],4)))
            plt.plot(DrawX[:, 1], DrawT[:, 1])
            plt.plot(DrawX[:, 1], L2[:, 1])
            plt.subplot("243")
            plt.title("Epoch:"+str(iteration)+";MAE:"+str(round(lossvector[2],4)))
            plt.plot(DrawX[:, 1], DrawT[:, 2])
            plt.plot(DrawX[:, 1], L2[:,2])
            plt.subplot("244")
            plt.title("Epoch:"+str(iteration)+";MAE:"+str(round(lossvector[3],4)))
            plt.plot(DrawX[:, 1], DrawT[:, 3])
            plt.plot(DrawX[:, 1], L2[:, 3])
            plt.subplot("245")
            plt.title("Epoch:"+str(iteration)+";MAE:"+str(round(lossvector[4],4)))
            plt.plot(DrawX[:, 1], DrawT[:, 4])
            plt.plot(DrawX[:, 1], L2[:, 4])
            plt.subplot("246")
            plt.title("Epoch:"+str(iteration)+";MAE:"+str(round(lossvector[5],4)))
            plt.plot(DrawX[:, 1], DrawT[:, 5])
            plt.plot(DrawX[:, 1], L2[:, 5])
            plt.subplot("247")
            plt.title("Epoch:"+str(iteration)+";MAE:"+str(round(lossvector[6],4)))
            plt.plot(DrawX[:, 1], DrawT[:, 6])
            plt.plot(DrawX[:, 1], L2[:, 6])
            plt.subplot("248")
            plt.title("Epoch:"+str(iteration)+";MAE:"+str(round(lossvector[7],4)))
            plt.plot(DrawX[:, 1], DrawT[:, 7])
            plt.plot(DrawX[:, 1], L2[:, 7])
            plt.tight_layout()
            plt.pause(0.01)

            if iteration%10==0:
                print("第", iteration, "次迭代，", "最低MAE精度：",loss[-1])
            w20 = w0
            w21 = w1

        #针对 任务3：分类问题 的模型检验与动态绘图
        elif order=='3':
            for i in range(len(L2)):
                for j in range(len(L2[i])):
                    if L2[i][j] >= 0.5:
                        L2[i][j] = 0.9
                    else:
                        L2[i][j] = 0.1
            AXIS.append(iteration)
            lossI.append(np.mean(np.abs(L2[:, 0].T - T[:, 0])))
            lossII.append(np.mean(np.abs(L2[:, 1].T - T[:, 1])))
            loss.append(0.5*(lossI[-1]+lossII[-1]))
            if iteration == 1:
                del AXIS[0]
                del lossI[0]
                del lossII[0]
            plt.subplot("231")
            plt.title("T1")
            plt.scatter(X1[:, 0], X1[:, 1], c=T[:, 0])
            plt.subplot("232")
            plt.title("Model Epoch"+str(iteration))
            plt.scatter(X1[:, 0], X1[:, 1], c=L2[:, 0])
            plt.subplot("233")
            plt.title("MAE:"+str(round(lossI[-1],4)))
            plt.plot(AXIS, lossI)
            plt.subplot("234")
            plt.title("T2")
            plt.scatter(X1[:,0],X1[:,1],c=T[:,1])
            plt.subplot("235")
            plt.title("Model Epoch"+str(iteration))
            plt.scatter(X1[:,0],X1[:,1],c=L2[:,1])
            plt.subplot("236")
            plt.title("MAE:"+str(round(lossII[-1],4)))
            plt.plot(AXIS, lossII)
            plt.tight_layout()
            plt.pause(0.01)
            if iteration % 10 == 0:
                print("第", iteration, "次迭代，", "相对MAE：", loss[-1])
            w30 = w0
            w31 = w1

        if iteration >= itelim:
            break
        else:
            iteration += 1
        '''---------------------------------------------------------------------------//'''
    time_end = time.time()
    '''！！！******************************************************************************************！！！'''



    '''//--------------------代码块2.7：控制台输出本次任务训练结果--------------------------'''
    print("w0最终权重:",w0)
    print("w1最终权重:",w1)
    print("\n训练完成!共计迭代",iteration,"次，(最低)MAE精度：",(loss[-1]))
    print("本次训练时长：",time_end-time_start,"秒")
    plt.ioff()
    plt.show()
    '''----------------------------------------------------------------------------------//'''



'''//-----------------------------代码块3：权值数据整合写入文件--------------------------------'''
print('正在写入存档...')
# build sheet

sheet1 = workbook.add_sheet('Mission1_W0', cell_overwrite_ok=True)
sheet2 = workbook.add_sheet('Mission1_W1', cell_overwrite_ok=True)

for row in range(INI+1):
    for col in range(logNNI):
        sheet1.write(row, col, w10[row][col])
for row in range(logNNI+1):
    for col in range(OUI):
        sheet2.write(row, col, w11[row][col])


sheet3 = workbook.add_sheet('Mission2_W0', cell_overwrite_ok=True)
sheet4 = workbook.add_sheet('Mission2_W1', cell_overwrite_ok=True)
for row in range(INII+1):
    for col in range(logNNII):
        sheet3.write(row, col, w20[row][col])
for row in range(logNNII+1):
    for col in range(OUII):
        sheet4.write(row, col, w21[row][col])


sheet5 = workbook.add_sheet('Mission3_W0', cell_overwrite_ok=True)
sheet6 = workbook.add_sheet('Mission3_W1', cell_overwrite_ok=True)
for row in range(INIII+1):
    for col in range(logNNIII):
        sheet5.write(row, col, w30[row][col])
for row in range(logNNIII+1):
    for col in range(OUIII):
        sheet6.write(row, col, w31[row][col])
# save as excel V2003 which been known as xls,xlsx cannot be write correctly
workbook.save('Weights.xls')
print('写入完毕！正在打开...')
os.startfile(str('Weights.xls'))
'''----------------------------------------------------------------------------------//'''