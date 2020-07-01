#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[16]:


# first approach over dataset
black_friday.info()

    


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[3]:


def q1():
    # Retorne aqui o resultado da questão 1.
    # (537577, 12)
    return black_friday.shape
    


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[5]:


def q2():
    # Retorne aqui o resultado da questão 2.
    # 545 resultados distintos
    #return black_friday[(black_friday['Gender'] == 'F') & (black_friday['Age']=='26-35')]['User_ID'].nunique()
    return black_friday[(black_friday['Gender'] == 'F') & (black_friday['Age']=='26-35')]['User_ID'].size # 49348
    


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[6]:


def q3():
    # Retorne aqui o resultado da questão 3.
    # 5891
    return black_friday['User_ID'].nunique()
    


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[7]:


def q4():
    # Retorne aqui o resultado da questão 4.
    # 3
    return black_friday.dtypes.nunique()
    


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[8]:


def q5():
    # Retorne aqui o resultado da questão 5.
    # 0.6944102891306734
    return (black_friday.isnull().sum()/len(black_friday)).max()
    


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[9]:


def q6():
    # Retorne aqui o resultado da questão 6.
    # 373299
    return int(black_friday.isnull().sum().max())
    


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[10]:


def q7():
    # Retorne aqui o resultado da questão 7.
    # 16.0
    df = black_friday.dropna()
    return df['Product_Category_3'].value_counts().index[0]
    


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[11]:


def q8():
    # Retorne aqui o resultado da questão 8.
    # 0.38479390362696736
    from sklearn import preprocessing
    
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(black_friday[['Purchase']])
    df = pd.DataFrame(x_scaled)
    return float(df.mean()[0])
    


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[12]:


def q9():
    # Retorne aqui o resultado da questão 9.
    # 348631
    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler()
    # Fit your data on the scaler object
    scaled_df = scaler.fit_transform(black_friday[['Purchase']])
    scaled_df = pd.DataFrame(scaled_df)
    return int(scaled_df[(scaled_df[0]>= -1) & (scaled_df[0]<= 1)].size)

    


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[13]:


def q10():
    # Retorne aqui o resultado da questão 10.
    # True
    cond = (black_friday['Product_Category_2'].isna() == True) & (black_friday['Product_Category_3'].isna() == True)
    cond2 = black_friday['Product_Category_2'].isna()
    t = black_friday[['Product_Category_2','Product_Category_3']]#.where(cond,) # 166986
    
    if(len(t[cond] == t[cond2])):
        return (True)
    else:
        return (False)
    

