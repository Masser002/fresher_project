from  PIL import Image
import numpy as np
from sklearn.cluster import KMeans

im=Image.open("Free.gif.Gif")
matrix=np.asarray(im)
matrix_2=matrix.reshape(2048*1365,3)
clf=KMeans(n_clusters=3)
clf=clf.fit(matrix_2)
pre=clf.predict(matrix_2)
new=np.empty((2048*1365,3),dtype=np.uint8)
new_2=np.empty((2048*1365,3),dtype=np.uint8)
new_3=np.copy(new)
new_4=np.copy(new)
new_5=np.copy(new)
for i in range(len(pre)):
    if pre[i]==0:
        new[i]=matrix_2[i]
    elif pre[i]==3:
        new_4[i]=matrix_2[i]
    else:
        new_5[i]=matrix_2[i]

new=new.reshape(1365,2048,3)
new_4=new_4.reshape(1365,2048,3)
t=Image.fromarray(new)
new_5=new_5.reshape(1365,2048,3)
t.show()
t3=Image.fromarray(new_4)

t4=Image.fromarray(new_5)

t4.show()