import pandas as pd
import numpy as np
import json
import random
import seaborn as sns
import os
import os.path
import matplotlib.pyplot as plt
import ast
from textwrap import TextWrapper

from sklearn.cluster import DBSCAN
#from sklearn import metrics
#from sklearn.datasets import make_blobs

class Inspector:

    def results_for_run(run_config:pd.Series) -> pd.DataFrame:
        ensemble_id = run_config.ensemble_id
        run_id = run_config.run_id
        max_t = run_config.max_t
        n_agents = run_config.n_agents
        
        df_results_run = pd.read_csv(os.path.join('data',ensemble_id, run_id+'.csv'),index_col=['step','agent'], skipinitialspace=True)
        df_results_run['run_id']=run_id
        df_results_run['ensemble_id']=ensemble_id
        df_results_run = df_results_run.reset_index()

        df_results_run.rename(columns={'pst':'post'},inplace=True)
        
        # post-process data
        for col in ['perspective', 'peers']:
            df_results_run[col] = df_results_run[col].apply(ast.literal_eval)
        for col in ['post']:
            df_results_run[col] = df_results_run[col].fillna("{'text':''}").apply(ast.literal_eval)
        
        # add further simple metrics
        df_results_run['len_peers']=df_results_run.peers.apply(len)
        df_results_run['len_persp']=df_results_run.perspective.apply(len)
        df_results_run['new_posts']=df_results_run.perspective.apply(lambda p: len([x for x in p if x['tst']==0]))

        # add opinion volatility
        opinions = df_results_run.polarity.tolist()
        #print(opinions)
        shifted_opinions = ([0]*n_agents) + opinions[:-n_agents] 
        #print(shifted_opinions)
        df_results_run['delta_polarity'] = [x-y for x,y in zip(opinions,shifted_opinions)]
        
        return df_results_run



    def detailed_plots(config = None, data = None, title : str = None, y : str = 'polarity', highlight_agents : [int] = None, highlight_range : [int] = [], clusters = [], ylim = None, xlim = None, aspect=2.8, legend=True):
        if title==None:
            title= "%s | %s" % (config.run_id, y)
        
        # load data
        plot_data = data[~data.step.isin(list(range(config.n_initial_posts)))].loc[:]
                
        if highlight_agents != None:
            plot_data['size'] = plot_data.agent.apply(lambda i: 10 if i in highlight_agents else 2)
            
        # xrange
        if xlim!=None:
            xlim=xlim
            xticks=list(range(150))
        else:
            xlim=[config.n_initial_posts,149]
            xticks=[5*i for i in range(31)]
            
        if clusters!=[]:
            palette = sns.color_palette("husl", len(set(clusters)))
            col = lambda i: palette[clusters[i]] if clusters[i]>=0 else 'lightgray'
            palette = {i:col(i) for i in range(config.n_agents)}
        else:
            x0 = plot_data[plot_data.step==config.n_initial_posts]['polarity'].tolist()
            palette = sns.color_palette("Spectral", as_cmap=False, n_colors=len(x0))
            agents_sorted = [i for _,i in sorted(zip(x0,range(len(x0))))]
            palette = {j:palette[i] for i,j in enumerate(agents_sorted)}
        
        sns.set_style("whitegrid")
        g = sns.relplot(
            x='step',
            y=y,
            hue="agent", 
            size='size' if highlight_agents!=None else None, 
            data=plot_data.reset_index(), 
            kind='line', 
            palette=palette, 
            aspect=aspect,
            legend=legend
        )
        g.set(title=title, ylim=ylim, xticks=xticks,xlim=xlim)
        
        # highlight
        if highlight_range != []:
            plt.axvspan(highlight_range[0], highlight_range[1], color='purple', alpha=0.3)

        return g



    def get_persp(data = None, display_agent=1, step=10):
        df = data[data.agent.eq(display_agent)][['step','perspective']]
        def expand_persp(row):
            persp = row.perspective
            if row.step<max(data.step):
                next_perspective = [p['pst'] for p in data[data.agent.eq(display_agent) & data.step.eq(row.step+1)].perspective.item()]  
            else:
                next_perspective = []
            def expand_post(p):
                p_r = p
                p_r['newly_created'] = (p_r['pst'][0]==row.step-1)
                p_r['will_be_dropped'] = not p_r['pst'] in next_perspective 
                return p_r
            persp = [expand_post(p) for p in persp]
            return persp
        df['persp_exp'] = df.apply(expand_persp, axis=1)
        df = pd.concat([
            df.step,
            df.persp_exp.apply(pd.Series)
        ],axis=1)
        
        
        
        df.set_index('step', inplace=True)
        
        results_grouped = data.groupby(['step','agent'])
        
        #get_post = lambda si: TextWrapper(width=20).wrap(results_grouped.get_group(si)['post'].item().get('text'))
        get_post = lambda si: (results_grouped.get_group(si)['post'].item().get('text')).replace('\\','')
        
        df_text = df.applymap(lambda x: get_post(x.get('pst')) if type(x)==dict else x)
        #df_text = df_text.apply(pd.Series.str.wrap(10))        
        
        def format_cells(s, threshold=0.5):
            if type(s)!=dict:
                return ''
            format = 'column-width: 100px; overflow-wrap: break-word; word-wrap: break-word;'
            if  s['pst'][-1]==display_agent:
                format = format + ' font-weight:bold;'
            if s['tst']==0:
                format = format + ' color:Orange;'
            if  s['pst'][-1]==display_agent and s['newly_created']:
                format = format + ' text-decoration: underline;'
            if  s['will_be_dropped']:
                format = format + ' color:SlateBlue;'                
            return format

        df_styled = df_text[step-1:step+2].style.apply(lambda x: df[step-1:step+2].applymap(format_cells),axis=None) 

        return df_styled 



    def cluster_labels(data=None, span : int = 3, t : int = -1):
        max_step = t if t>span else max(data.step.tolist())
        X = data[data.step.le(max_step) & data.step.gt(max_step-span)].groupby(['step'])
        X = [data.polarity[g] for g in X.groups.values()]
        X = np.array(X)
        X = X.transpose()
        # Compute DBSCAN
        db_scan = DBSCAN(eps=0.03, min_samples=3).fit(X)
        core_samples_mask = np.zeros_like(db_scan.labels_, dtype=bool)
        core_samples_mask[db_scan.core_sample_indices_] = True
        labels = db_scan.labels_
        return labels