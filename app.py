import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from inspector import Inspector

st.title('Artificial Deliberating Agents Inspector')

st.markdown('Inspect the simulation runs underpinning the paper *Natural-language Simulations of Argumentative Opinion Dynamics* ([Betz 2021](https://arxiv.org)).')

# Sidebar

st.sidebar.subheader('Simulation run')

filter_update = st.sidebar.selectbox(
    'Filter for update mechanism:',
     ["all","1","2","3"])

filter_agent_type = st.sidebar.selectbox(
    'Filter for agent type:',
     ["all","l","gc","generating (narrow)"])

run_id = st.sidebar.selectbox(
    'Simulation run to inspect:',
     ['20210409-01_60-5','20210409-01_60-5'])

st.sidebar.subheader('Perspective of agent')

agent_focus = st.sidebar.selectbox('Select agent:' , list(range(20)))

step_focus = st.sidebar.slider('Select time step:' , min_value=6 , max_value=148 , value=148 , step=1)

st.sidebar.subheader('Clustering')

show_clusters = st.sidebar.checkbox('Compute clusters')

if show_clusters:
    step_clustering = st.sidebar.slider('Evaluate clustering at time step:' , min_value=5 , max_value=149 , value=149 , step=1, key='cluster_step')


# Load data

df_config = pd.Series(
    {
        'run_id':'20210409-01_60-5',
        'ensemble_id':'20210409-01',
        'n_initial_posts':5,
        'n_agents':20,
        'max_t':150
    }
)
df_results = Inspector.results_for_run(df_config)


# Perspective Table

html_perspective = Inspector.get_persp(
    data=df_results, 
    display_agent=agent_focus, 
    step=step_focus).set_properties(**{'font-size': '9pt', 'font-family': 'Calibri', 'width': '200px'}).render()
html_perspective = """
<p style='font-size:9pt;font-family:Calibri;'>
<span style="color:Orange">newly added</span> | 
<span style="color:SlateBlue">marked for removal</span> | 
<b>written by agent</b> | 
<u>newly generated</u>
</p>
"""+html_perspective


# Cluster analysis

if show_clusters:
    cluster_labels = Inspector.cluster_labels(data=df_results, span = 3, t = step_clustering)
    #df_results['cluster']=df_results.agent.apply(lambda i: cluster_labels[i])
else:
    cluster_labels = []


# MAIN PAGE

st.subheader('Opinion trajectories of all agents')

fig = Inspector.detailed_plots(
    config=df_config,
    data=df_results,
    highlight_agents=[agent_focus],
    highlight_range=[step_focus-1,step_focus+1],
    clusters=cluster_labels,
    legend=None
)
st.pyplot(fig)

#st.write(df_results.head())



st.subheader('Perspective of the selected agent')



#st.dataframe(Inspector.get_persp(data=df_results, display_agent=agent_focus, step=step_focus))

components.html(html_perspective, height=600, scrolling=True)



st.subheader('Parameters')





