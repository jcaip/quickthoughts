#!/usr/bin/env python
# coding: utf-8

# In[2]:


from eval import *
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics.cluster import *
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import fetch_20newsgroups
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess

import pandas as pd
from tqdm import tqdm
from scipy.linalg import orth
import csv
torch.cuda.set_device(7)


# In[3]:


taboola_data_dir = "/local/diq/all/StepContent__nocontent_title_desc"
step_lines = "/local/diq/all/StepLines__nocontent_title_desc"

import os
#print(os.listdir(step_lines))

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

for file in os.listdir(step_lines):
    sc_len = file_len("{}/{}".format(taboola_data_dir, file))
    sl_len = file_len("{}/{}".format(step_lines, file))
    #print(sc_len, sl_len)
    assert sc_len == sl_len


# In[4]:


taxonomies = set()
#hack for the moment (jupuyter nb in folder)
for file in tqdm(os.listdir(step_lines)):
    with open("{}/{}".format(taboola_data_dir, file), "r") as f:
        data = set(map(lambda json_obj: json_obj.get('first_level_taxonomy'), map(lambda x: json.loads(x), f.readlines())))
        taxonomies= taxonomies.union(data)

print(taxonomies)
mapping = dict([(name, i) for i, name in enumerate(taxonomies)])
print(mapping)


# In[16]:





# In[5]:


text = []
labels = []

for file in tqdm(sorted(os.listdir(step_lines))):
    data = pd.read_csv("{}/{}".format(step_lines, file), header=None, sep='\t', quoting=csv.QUOTE_NONE)[2].tolist()
    text+=data
    
    with open("{}/{}".format(taboola_data_dir, file), "r") as f:
        data = map(lambda json_obj: mapping.get(json_obj.get('first_level_taxonomy'), -1), map(lambda x: json.loads(x), f.readlines()))
        labels += list(data)
        
    #print(len(text), len(labels))
    assert(len(text) == len(labels))

print(len(labels))
print(len(text))

test_batch_size=5000
size = len(labels)
print("Loaded dataset {} with total lines: {}".format("TABOOLA", size))


# In[6]:


#load qt model
checkpoint_dir = '/home/jcjessecai/workspace/taboola/quickthoughts/checkpoints'
with open("{}/config.json".format(checkpoint_dir)) as fp:
    CONFIG = json.load(fp)

WV_MODEL = api.load(CONFIG['embedding'])
qt = QuickThoughts(WV_MODEL, hidden_size=CONFIG['hidden_size'])
trained_params = torch.load("{}/checkpoint_latest.pth".format(checkpoint_dir))
qt.load_state_dict(trained_params['state_dict'])
qt = qt.cuda()
qt.eval()
print("Restored successfully from {}".format(checkpoint_dir))


# In[7]:


data = list(map(lambda x: torch.LongTensor(prepare_sequence(x, WV_MODEL.vocab, no_zeros=True)), mapping.keys()))
packed = safe_pack_sequence(data).cuda()
            
vec_categories = qt(packed).cpu().detach().numpy()
print(vec_categories)


# In[8]:


#encode data
def make_batch(j):
    """Processes one test batch of the test datset"""
    stop_idx = min(size, j+test_batch_size)
    batch_text, batch_labels  = text[j:stop_idx], labels[j:stop_idx]
    data = list(map(lambda x: torch.LongTensor(prepare_sequence(x, WV_MODEL.vocab, no_zeros=True)), batch_text))
    for i in data:
        if len(i) == 0:
            print(i)
            input()
    packed = safe_pack_sequence(data).cuda()
    return qt(packed).cpu().detach().numpy()

feature_list = [make_batch(i) for i in range(0, size, test_batch_size)]
print("Processed {:5d} batches of size {:5d}".format(len(feature_list), test_batch_size))
qt_features = np.concatenate(feature_list)
print("Test feature matrix of shape: {}".format(qt_features.shape))


# In[9]:


taboola_model_dir = "/home/jcjessecai/workspace/taboola_model"
d2v = Doc2Vec.load("{}/{}".format(taboola_model_dir, "gensim_doc2vec.model"))


# In[10]:


d2v_features = np.vstack([d2v.infer_vector(simple_preprocess(doc)) for doc in text])
print(d2v_features.shape)


# In[11]:


#first we compare embedding performance by fitting binary classifier on top
from sklearn import linear_model

X_train, X_test, y_train, y_test = train_test_split(d2v_features, labels)
clf = linear_model.SGDClassifier(loss='log', max_iter=1000, n_jobs=20)
clf.fit(X_train, y_train)
train_acc = clf.score(X_train, y_train)
test_acc = clf.score(X_test, y_test)
print("Fit logistic model on d2v with train acc: {:.2%} test acc: {:.2%}".format(train_acc, test_acc))

X_train, X_test, y_train, y_test = train_test_split(qt_features, labels)
clf = linear_model.SGDClassifier(loss='log', max_iter=1000, n_jobs=20)
clf.fit(X_train, y_train)
train_acc = clf.score(X_train, y_train)
test_acc = clf.score(X_test, y_test)
print("Fit logistic model on qt with train acc: {:.2%} test acc: {:.2%}".format(train_acc, test_acc))


# In[ ]:




