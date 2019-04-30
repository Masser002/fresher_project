from sklearn.tree import DecisionTreeClassifier
from memlonData import menlonData
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

#加载数据
data_x=menlonData.get_trainX(2)
data_y=menlonData.get_trainY()
#创建分类器，取属性为2
clf=DecisionTreeClassifier()
clf=clf.fit(data_x,data_y)
score=cross_val_score(clf,data_x,data_y,cv=5)
print(score)
#结果处理，准确率，召回率，评估分数
print(classification_report([1,0,0,1],[0,0,0,1]))
#创建2维数组
g_data=[[],[]]
b_data=[[],[]]
i=-1
for y in data_y:
    i=i+1
    if y==0:
        b_data[0].append(data_x[i][0])
        b_data[1].append(data_x[i][1])
    else :
        g_data[0].append(data_x[i][0])
        g_data[1].append(data_x[i][1])
#可视化
plt.scatter(g_data[0],g_data[1],color='blue',linewidths='4')
plt.scatter(b_data[0],b_data[1],color='red',linewidths='0.5')
#坐标轴修改
plt.xticks([1,2,3],['green','black','white'])
plt.show()
