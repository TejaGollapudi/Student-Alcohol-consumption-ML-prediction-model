
import pandas as pd
###################################################################
#to merge dataframes 
"""
df1 = pd.read_csv('blatantmerge-graderemoved.csv')
df2=df1.drop_duplicates(keep=False)
print(df2)
print(df2.shape)
df2.to_csv('mergedng.csv', encoding='utf-8', index=False)
"""



###################################################################
#to convert yes and no to binary
"""
df=pd.read_csv('student-por.csv')
print(df.iloc[:,15:23])
df.iloc[:,15:23] = df.iloc[:,15:23].replace({'yes': 1, 'no': 0})
print(df.iloc[:,15:23])
df.to_csv('student-por-yn2binary.csv',encoding='utf-8',index=False)
"""



###################################################################
#to dummy code the data
"""

df=pd.read_csv('student-por-yn2binary.csv')
df2=pd.get_dummies(df.iloc[:,0:2])
df3=pd.get_dummies(df.iloc[:,3:6])
df4=pd.get_dummies(df.iloc[:,8:12])
df=pd.concat([df4,df],axis=1)
df=pd.concat([df3,df],axis=1)
df=pd.concat([df2,df],axis=1)



print(df.head())

df.to_csv('student-por-yn2binary-dummycoding.csv',encoding='utf-8',index=False)
"""


###################################################################
# correlation
"""
df=pd.read_csv('blatantmerge.csv')
print(df.corr())
"""
