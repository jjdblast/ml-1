# -*- coding: utf-8 -*-
"""
Created on Thu Jul 03 15:19:15 2014

@author: lianchao
"""

'''
Lesson
--`shelve` files are not cross-platform, `pickle` is.
However, using pickle might be time consuming when update.

--`sqlite3`.

'''

import shelve, pickle, sqlite3
import pylab as pl
import numpy as np
from types import ModuleType


def save_obj(obj_name_list, dictt, fname='./cache.db'):
    conn = sqlite3.connect(fname)
    conn.execute('''create table if not exists
                pyobj (key text primary key, value text)''')

    for key in obj_name_list:
        query = '''insert or replace into pyobj values (:key, :value)'''
        value_str = sqlite3.Binary(pickle.dumps(dictt[key]))
        conn.execute(query, (key, value_str))
        print 'save %s' % key
    conn.commit()
    conn.close()

def load_obj(obj_name_list, dictt, fname='./cache.db'):
    conn = sqlite3.connect(fname)
    cur = conn.cursor()
    if obj_name_list is None:
        cur.execute('select key from pyobj')
        obj_name_list = [e[0] for e in cur]
    query = 'select * from pyobj where key in ({seq})'.format(
            seq = ','.join(['?'] * len(obj_name_list)))
    cur.execute(query, obj_name_list)
    for k, v in cur:
        dictt[str(k)] = pickle.loads(str(v))
        print 'load %s' % k
    conn.close()

def query_obj(fname='./cache.db'):
    conn = sqlite3.connect(fname)
    cur = conn.cursor()
    query = 'select key from pyobj'
    cur.execute(query)
    for k in cur:
        print str(k[0])
    conn.close()

def remove_obj(obj_name_list, fname='./cache.db'):
    conn = sqlite3.connect(fname)
    cur = conn.cursor()
    if obj_name_list is None:
        conn.execute('drop table pyobj')
        conn.commit()
        conn.close()
        return
    query = 'delete from pyobj where key in ({seq})'.format(
            seq = ','.join(['?'] * len(obj_name_list)))
    cur.execute(query, obj_name_list)
    conn.commit()
    conn.close()

####################################################################################################
# plot
def group_plot(G, vns, kind='hist', bins=50, **kwargs):
    nc = (len(vns)+1)/2
    nr = len(vns)/nc + 1
    fig, ax = pl.subplots(nrows=nr, ncols=nc)
    if nr == 1:
        ax = ax[np.newaxis, :]
    for i in range(len(vns)):
        if kind == 'hist':
            G[vns[i]].hist(ax=ax[i/nc,i%nc], bins=bins, **kwargs)
        elif kind == 'kde':
            G[vns[i]].plot(ax=ax[i/nc,i%nc], kind='kde', **kwargs)
        pl.title(vns[i])

####################################################################################################
# ensemble
def weighted_ensemble(mds, df, vns, y, kind='ave', iter=200):
    p_mds = [md.predict_proba(df[vns]) for md in mds]
    p = None

    if kind == 'ave':
        ws = np.ones(shape=(1,len(mds)))
    elif kind == 'rand':
        ws = np.random.uniform(size=(iter, len(mds)))

    maxx = 0
    weight = None
    for w in ws:
        p = sum(map(lambda p_md, i: p_md*i, p_mds, w))
        pred = p.argmax(axis=1) + 1
        rate = (pred == y).sum() / float(df.shape[0])
        if rate > maxx:
            maxx = rate
            weight = w
    return maxx, weight

if __name__ == '__main__':
    import numpy as np
    x = np.random.uniform(size=(10,10))
    y = 2

    save_obj(['x', 'y'], locals())

    del x, y
    load_obj(None, locals())
    print x, y
