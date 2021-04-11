import streamlit as st

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
     ["a","b"])

st.sidebar.write('You selected:', run_id)

st.sidebar.subheader('Perspective of agent')

agent_focus = st.sidebar.selectbox('Select agent:' , list(range(20)))

step_focus = st.sidebar.slider('Select time step:' , min_value=5 , max_value=149 , value=149 , step=1)

st.sidebar.subheader('Clustering')

show_clusters = st.sidebar.checkbox('Show clusters')

if show_clusters:
    step_clustering = st.sidebar.slider('Select time step:' , min_value=5 , max_value=149 , value=149 , step=1, key='cluster_step')



# MAIN

st.subheader('Opinion trajectories of all agents')

run_id

st.subheader('Perspective of a selected agent')

agent_focus



st.subheader('Parameters')





