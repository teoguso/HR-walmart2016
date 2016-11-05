
# coding: utf-8

# In[2]:

from sklearn.externals import joblib
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from sklearn.preprocessing import MultiLabelBinarizer


# In[3]:

pipe = joblib.load("src/pipeline.pkl")


# In[8]:

best = joblib.load("src/best.pkl")


# In[9]:

best


# In[27]:



# In[39]:

df_test = pd.read_table("data/test.tsv")


# In[40]:

empty_lines = df_test[df_test['Product Long Description'].isnull()].index.tolist()
df_test = df_test[-df_test['Product Long Description'].isnull()]
print(empty_lines)


# In[24]:

soups = [BeautifulSoup(item, "lxml").get_text(separator="\n") for item in df_test['Product Long Description']]


# In[29]:

prediction = best.predict(soups)


# ### We have to find the inverse transform for the labels

# In[25]:

# The direct transform is calculated as follows:


# In[28]:

df = pd.read_table("data/train.tsv")
df['label'] = df['tag'].apply(lambda x: np.array(eval(x)))
mlb = MultiLabelBinarizer()


# In[41]:

mlb.fit(df['label'])
tag_prediction = mlb.inverse_transform(prediction)


# In[46]:

tag_prediction = [list(tup) for tup in tag_prediction]
tag_prediction[:5]


# ### For lack of a better idea, we'll use the mode for all empty fields

# In[55]:

mode = df.label.mode().values


# In[62]:

mode = int(mode)


# In[68]:

tag_aug_pr = []
for tags in tag_prediction:
    if tags == []:
        tag_aug_pr.append([mode])
    else:
        tag_aug_pr.append(tags)
tag_aug_pr[:10]


# In[71]:

# Now we have to insert the completely 
# empty fields we ditched at the beginning.


# In[72]:

for i in empty_lines:
    tag_aug_pr.insert(i, [mode])


# In[73]:

len(tag_aug_pr)


# In[77]:

len(df_test['item_id'])


# In[78]:

df_test = pd.read_table("data/test.tsv")
len(df_test)


# In[80]:

# out_df = pd.DataFrame(df_test['item_id'], pd.Series(tag_aug_pr, name="tag"))
d = dict(
item_id = df_test['item_id'],
tag = pd.Series(np.array(tag_aug_pr)) 
)


# In[81]:

out_df = pd.DataFrame(d)


# In[87]:

out_df.to_csv("tags.tsv", sep="\t", index=False)


# In[ ]:



