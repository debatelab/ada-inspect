import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from inspector import Inspector

# config
result_path = "data/20210409-01"
df_config = pd.read_csv("data/20210409-01_config.csv")
df_filtered = df_config


st.title('Artificial Deliberating Agents Inspector')

st.markdown('Inspect the simulation runs underpinning the paper *Natural-Language Multi-Agent Simulations of Argumentative Opinion Dynamics* ([Betz 2021](http://arxiv.org/abs/2104.06737)).')

# Sidebar

st.sidebar.subheader('Simulation run')

filter_update = st.sidebar.selectbox(
    'Filter for update mechanism:',
     ["any","random/all","random/bounded confidence","confirmation bias/all","homophily/all"])

if filter_update!="any":
    # filter by peer selection method
    if filter_update=="random/bounded confidence":
        df_filtered = df_filtered[df_filtered.peer_selection_method.eq('bounded_confidence')]
    else:
        df_filtered = df_filtered[df_filtered.peer_selection_method.eq('all_neighbors')]
    # filter by perspective updating
    if filter_update=='confirmation bias/all':
        df_filtered = df_filtered[df_filtered.perspective_expansion_method.eq('confirmation_bias_lazy')]
    elif filter_update=='homophily/all':
        df_filtered = df_filtered[df_filtered.perspective_expansion_method.eq('homophily')]
    else:
        df_filtered = df_filtered[df_filtered.perspective_expansion_method.eq('random')]

filter_agent_type = st.sidebar.selectbox(
    'Filter for agent type:',
     ["any","listening","generating (creative)","generating (narrow)"])

# filter by peer selection method
if filter_agent_type!="any":
    if filter_agent_type=="listening":
        df_filtered = df_filtered[df_filtered.agent_type.eq('listening')]
    elif filter_agent_type=="generating (creative)":
        df_filtered = df_filtered[df_filtered.agent_type.eq('generating') & df_filtered['temp/top_p'].eq('[1.4, 0.95]')]
    elif filter_agent_type=="generating (narrow)":
        df_filtered = df_filtered[df_filtered.agent_type.eq('generating') & df_filtered['temp/top_p'].eq('[1.0, 0.5]')]

run_id = st.sidebar.selectbox(
    'Simulation run to inspect:',
     df_filtered.run_id.to_list())

st.sidebar.subheader('Perspective of agent')

agent_focus = st.sidebar.selectbox('Select agent:' , list(range(20)))

step_focus = st.sidebar.slider('Select time step:' , min_value=6 , max_value=148 , value=148 , step=1)

st.sidebar.subheader('Clustering')

show_clusters = st.sidebar.checkbox('Compute clusters')

if show_clusters:
    step_clustering = st.sidebar.slider('Evaluate clustering at time step:' , min_value=5 , max_value=149 , value=149 , step=1, key='cluster_step')



# Load data


configrun = df_filtered.loc[df_filtered['run_id'] == run_id].iloc[0]
df_results = Inspector.results_for_run(configrun)


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
    config=configrun,
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

configrun



