from collections import Counter
import io
import zipfile
import streamlit as st
import streamlit.components.v1 as components
import datetime
import os
import snowflake.connector
import pandas as pd
import datetime
import json
import xlwings as xw
from spark_utils import get_spark, reg, cleanse_val, plants, cleanse_col, single, del_spark, get_mermaid, get_po_rcpt_data_for_part, get_snowflake_data, get_make_buy_parts, make_graph, get_top_level_parts, get_cache_folder_and_filenames, mermaid, tags_style, make_gantt, mine_save_lags, group_states
from simutils import run_simulation, init_part_data
import random
import simpy
import pickle
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# Set wide layout
st.set_page_config(layout="wide")
# Sidebar
st.sidebar.title("Demand Schedule Planning")

plant_col, end_col = st.sidebar.columns(2)
selected_plant = plant_col.selectbox(
    "Plant", plants, index=plants.index("3100"))

# End Item Part input
placeholder = end_col.empty()
with placeholder:
    end_item_part = ""
    placeholder.text_input("End Item Part", end_item_part,
                           placeholder="Enter end item part number")

# Calculate default values for Scheduled Delivery Date and Demand Delivery Date
current_year = datetime.date.today().year
scheduled_delivery_date = datetime.date(current_year + 3, 12, 31)
demand_delivery_date = datetime.date.today()

date1, date2 = st.sidebar.columns(2)
# Scheduled Delivery Date input
scheduled_delivery_date = date1.date_input(
    "Scheduled Delivery Date", scheduled_delivery_date)

# Demand Delivery Date input
demand_delivery_date = date2.date_input(
    "Demand Load Date", demand_delivery_date)

bom_df = pd.read_csv('bom_df_data.csv') if os.path.exists(
    'bom_df_data.csv') else None
    
upload_div = st.sidebar.expander("Upload BOM", expanded=(bom_df is None))
# File Upload
uploaded_file = upload_div.file_uploader(
    "Upload BOM Excel File", type=["xlsx"])

# Check if a file was uploaded
if uploaded_file:
    # Save the uploaded file to a cache folder
    cache_folder = "uploads"
    os.makedirs(cache_folder, exist_ok=True)
    uploaded_file_path = os.path.join(cache_folder, uploaded_file.name)
    with open(uploaded_file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Read the uploaded Excel file and extract data
    try:
        # Define your Excel processing logic here
        book = xw.Book(uploaded_file_path)
        sheet_name = "Indented BOM"
        sheet = book.sheets[sheet_name]

        # Read the first and second rows as separate header lines
        # Assumes headers are in the first row
        header_rows = sheet['A1:AS2'].value
        # Combine the two header rows into a single header row
        combined_header = [f"{h1}\n{h2}" if h1 and h2 else h1 or h2 for h1, h2 in zip(
            header_rows[0], header_rows[1])]

        bom_df = sheet['A2'].expand().options(
            pd.DataFrame, chunksize=10_000).value.reset_index()
        bom_df.columns = map(cleanse_col, combined_header)
        bom_df.dropna(subset=['nha', 'part'], axis=0, inplace=True)
        bom_df.to_csv('bom_df_data.csv', index=False)
        book.close()
        # Now you have the DataFrame bom_df with the data from the Excel file
        st.sidebar.success(
            "Excel file uploaded and data extracted successfully.")
    except Exception as e:
        st.sidebar.error(f"Error processing the Excel file: {e}")

cache_folder, po_data_file, rcpt_data_file = get_cache_folder_and_filenames(
    selected_plant,)
graph = None
sample_drawing = {}
if bom_df is not None and not bom_df.empty:
    bom_data_div = st.sidebar.expander("BOM Data", expanded=(bom_df is not None))
    makes, buys = get_make_buy_parts(bom_df)
    graph = make_graph(bom_df)
    top_level_parts = get_top_level_parts(graph)
    tab3, tab1, tab2, tab4, tab5, tab6, tab7 = bom_data_div.tabs(
        ["Top Level Parts", "Buys", "Makes", "BOM Preview", "PO Data", "Receipt Data", "Delta Dist"])
    bom_data_div.markdown(
        tags_style,
        unsafe_allow_html=True,
    )

    def display_tags(tab, tag_list):
        if tag_list:
            tag_html = '<div class="tags-container">'
            for tag in tag_list:
                tag_html += f'<div class="tag">{tag}</div>'
            tag_html += '</div>'
            tab.markdown(tag_html, unsafe_allow_html=True)
        else:
            tab.write("No tags to display")
    with tab2:
        if makes:
            display_tags(tab2, makes)
        else:
            tab2.write("No makes found.")

    with tab1:
        if buys:
            display_tags(tab1, buys)
            tab1.markdown("---")
            if tab1.button("Get Delta Data"):
                progress_bar = tab1.empty()
                progress_bar.write("Fetching data from Snowflake...")
                dist_file, po_data_file, rcpt_data_file = get_po_rcpt_data_for_part(
                    selected_plant, buys)
                progress_bar.write(
                    f"Data cached. {os.path.basename(dist_file)}")
            else:
                dist_file, po_data_file, rcpt_data_file = get_po_rcpt_data_for_part(
                    selected_plant, buys)
        else:
            tab1.write("No buys found.")

    with tab3:
        if top_level_parts:
            if tab3.button(top_level_parts[0]):
                with placeholder:
                    end_item_part = top_level_parts[0]
                    placeholder.text_input(
                        "End Item Part", end_item_part, placeholder="Enter end item part number")
        else:
            tab3.write("No top-level parts found.")

    with tab4:
        tab4.write(bom_df.head(1000))

    if not po_data_file or not rcpt_data_file:
        with tab5:
            tab5.write("No data to display")
        with tab6:
            tab6.write("No data to display")

    if po_data_file:
        with tab5:
            if os.path.exists(po_data_file):
                tab5.dataframe(pd.read_parquet(po_data_file))

    if rcpt_data_file:
        with tab6:
            if os.path.exists(rcpt_data_file):
                tab6.dataframe(pd.read_parquet(rcpt_data_file))

    
    if dist_file:
        with tab7:
            if os.path.exists(dist_file):
                with open(dist_file, 'rb') as file:
                    part_po_histogram_estimates = pickle.load(file)
                selected_buy_part = st.selectbox(
                    "Buy Part:", list(part_po_histogram_estimates.keys()))
                # Create a Streamlit placeholder for the histogram chart
                histogram_placeholder = tab7.empty()
                data_placeholder = tab7.empty()
                prob_placeholder = tab7.empty()
                draw_values = tab7.empty()
                
                # Function to update the placeholder with a histogram chart
                def update_histogram(part_name):
                    if part_name in part_po_histogram_estimates:
                        on_time_delivery_probability = part_po_histogram_estimates[part_name]['OnTimeDeliveryProbability']
                        data, histogram, bins = part_po_histogram_estimates[part_name]['Histogram']
                        fig = px.bar(
                            data, title=f'Histogram for {part_name}')
                        fig.update_layout(showlegend=False)
                        data_placeholder.table(data)
                        histogram_placeholder.plotly_chart(fig, use_container_width=True)
                        if on_time_delivery_probability is not None:
                            on_time_delivery_percentage = round(on_time_delivery_probability * 100, 2)
                            prob_placeholder.write(f'On-Time Delivery Probability for {selected_buy_part}: {on_time_delivery_percentage}%')
                            data_placeholder.table(data)
                        else:
                            prob_placeholder.write(f'On-Time Delivery Probability not found for {selected_buy_part}')
                            data_placeholder.table(data)
                        # Calculate KDE estimate for future simulations
                        data = data.reset_index().set_index('Delta')
                        # draw_values.write(data)
                        original_data = np.array(data.Quantity.apply(lambda x: list(range(0, x))).explode().index.tolist()).reshape(-1, 1)
                        num_bins = 100
                        hist, bin_edges = np.histogram(original_data, bins=num_bins, density=True)
                        cdf = np.cumsum(hist * np.diff(bin_edges))
                        sampled_data = np.interp(np.random.rand(10), cdf, bin_edges[:-1]).round()
                        draw_values.write(str(list(map(int,sampled_data.tolist()))))
                        sample_drawing[part_name] = lambda n: np.interp(np.random.rand(n), cdf, bin_edges[:-1]).round()
                # Update the histogram based on the selected part
                update_histogram(selected_buy_part)

sim_div = st.sidebar.expander(
    "Simulation Parameters", expanded=bom_df is not None)
num_sim_runs = sim_div.slider(
    "Number of Simulations", min_value=1, max_value=1000, value=100)

# Main content
st.title("Simulation Planning Tool")
if graph:
    graphviz, simulationtab, gantt_tab, path_tab, runlogs = st.tabs(
        ["Assembly", "Planning Simulation", "Gantt", "Most Traversed Paths", "Run Logs"])
    with graphviz:
        viz = graphviz.expander("Assembly", expanded=True)
        with viz:
            mermaid(*get_mermaid(graph))
        code = graphviz.expander("Code", expanded=False)
        code.markdown(f"""```mermaid
{get_mermaid(graph)[0]}```
""")
    with runlogs:
        logs_expander = runlogs.empty()
        completion_expander = runlogs.empty()
        critical_expander = runlogs.empty()
        delivery_expander = runlogs.empty()
        
    with path_tab:
        paths_display = path_tab.empty()
        mermaid_paths = path_tab.empty()
                
    with simulationtab:
        if sim_div.button("Run Simulation"):
            part_data = init_part_data(bom_df)
            # Call the simulation function and get the results
            progress_bar = sim_div.progress(0)
            progress_text = sim_div.empty()

            logs_data, completion_data, delivery_data, critical_data, critical_paths = None, None, None, None, []
            for i in range(1, num_sim_runs + 1):
                initial_guess = [random.uniform(1, 300)
                                 for _ in range(len(graph.nodes))]
                completion_times, delivery_times, critical_part, log_df, critical_path = run_simulation(
                    initial_guess, top_level_parts, graph, part_data, makes, buys, deltas=sample_drawing, env=simpy.Environment(), counter=i)

                completion_data = completion_times if completion_data is None else pd.concat(
                    [completion_data, completion_times], ignore_index=True).reset_index(drop=True)
                delivery_data = delivery_times if delivery_data is None else pd.concat(
                    [delivery_data, delivery_times], ignore_index=True).reset_index(drop=True)
                critical_data = critical_part if critical_data is None else pd.concat(
                    [critical_data, critical_part], ignore_index=True).reset_index(drop=True)
                logs_data = log_df if logs_data is None else pd.concat(
                    [logs_data, log_df], ignore_index=True).reset_index(drop=True)
                critical_paths.append(critical_path)
                logs_df = logs_expander.expander("Logs", expanded=True)
                logs_df.dataframe(logs_data, use_container_width=True)
                completion_df = completion_expander.expander(
                    "Completion Times", expanded=False)
                completion_df.dataframe(
                    completion_data, use_container_width=True)
                critical_df = critical_expander.expander(
                    "Critical Parts", expanded=False)
                critical_df.dataframe(critical_data, use_container_width=True)
                delivery_df = delivery_expander.expander(
                    "Delivery Times", expanded=False)
                delivery_df.dataframe(delivery_data, use_container_width=True)

                progress_bar.progress(
                    int(min(i, num_sim_runs)*100.0/num_sim_runs))
                progress_text.text(f"{i} of {num_sim_runs} runs completed")

    # Save data to session state
    if 'logs_data' in vars():
        st.session_state.logs_data = logs_data
        st.session_state.completion_data = completion_data
        st.session_state.critical_data = critical_data
        st.session_state.delivery_data = delivery_data
        st.session_state.path_data = critical_paths

    # Display the download button based on session state
    if hasattr(st.session_state, 'logs_data') and st.session_state.logs_data is not None:
        def get_data(df):
            log_buffer = io.StringIO()
            df.to_csv(log_buffer, index=False)
            return log_buffer.getvalue().encode('utf-8')

        # Create a zip archive
        if hasattr(st.session_state, 'logs_data'):
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.writestr('simlogs.csv', get_data(
                    st.session_state.logs_data))
                zipf.writestr('completion_times.csv',
                              get_data(st.session_state.completion_data))
                zipf.writestr('critical_parts.csv',
                              get_data(st.session_state.critical_data))
                zipf.writestr('delivery_times.csv',
                              get_data(st.session_state.delivery_data))

        if hasattr(st.session_state, 'logs_data'):
            logs_df = logs_expander.expander("Logs", expanded=True)
            logs_df.dataframe(st.session_state.logs_data,
                              use_container_width=True)
            completion_df = completion_expander.expander(
                "Completion Times", expanded=False)
            completion_df.dataframe(
                st.session_state.completion_data, use_container_width=True)
            critical_df = critical_expander.expander(
                "Critical Parts", expanded=False)
            critical_df.dataframe(
                st.session_state.critical_data, use_container_width=True)
            delivery_df = delivery_expander.expander(
                "Delivery Times", expanded=False)
            delivery_df.dataframe(
                st.session_state.delivery_data, use_container_width=True)

        sim_div.download_button(
            label='Download All Data',
            data=zip_buffer.getvalue(),
            file_name='data.zip',
            key='download-zip'
        )

    with simulationtab:
        if hasattr(st.session_state, 'critical_data'):
            critical_data = st.session_state.critical_data
            critical_data = critical_data[critical_data.Make_Buy == "BUY"]
            part_occurrence = critical_data['Predecessor'].value_counts(
            ).reset_index()
            part_occurrence.columns = ['Critical Part', 'Occurrence']

            # Calculate the total number of runs (X)
            # Replace with the actual number of runs
            total_runs = len(critical_data.Run.unique())

            def format_percentage(value):
                return f"{value:.2f}%"
            # Calculate the percentage metric for each part
            part_occurrence['Percentage'] = (
                part_occurrence['Occurrence'] / total_runs) * 100

            # Sort the parts by occurrence in descending order
            sorted_parts = part_occurrence.sort_values(
                by='Occurrence', ascending=False)
            sorted_parts['String Percentage'] = sorted_parts['Percentage'].apply(
                format_percentage)

            fig = px.bar(
                sorted_parts,
                x='Critical Part',
                y='String Percentage',
                labels={'Percentage': 'Percentage (%)'},
                title='Barchart'
            )

            # Customize the appearance of the plot
            fig.update_xaxes(title_text='Critical Part')
            fig.update_yaxes(title_text='Percentage (%)')
            # Rotate x-axis labels for better readability
            fig.update_layout(xaxis_tickangle=-45)

            simulationtab.subheader("Critical Parts Summary")
            fig1, fig2 = simulationtab.columns(2)
            # Display the plot in Streamlit
            fig1.plotly_chart(fig, use_container_width=True)
            # Display the pie chart in Streamlit
            fig2.plotly_chart(px.treemap(
                sorted_parts,
                values='Percentage',
                path=['Critical Part'],
                title='Piechart',
                labels={'Percentage': 'Percentage (%)'}
            ), use_container_width=True)

            simulationtab.subheader("Critical Parts Supportive Data")
            simulationtab.dataframe(sorted_parts, use_container_width=True)

    with gantt_tab:
        if hasattr(st.session_state, 'logs_data'):
            logs_data = st.session_state.logs_data
            completion_data = st.session_state.completion_data
            # Make a dropdown to select the Run of the simulation to collect logs for...
            run = gantt_tab.slider("Select a Run", min_value=1, max_value=len(logs_data.Run.unique()), key="run_key")
            # .selectbox(
            #     "Run", logs_data.Run.unique(), index=0)
            # Filter for specific Run data
            logs_data = logs_data[(logs_data.Run == run) & (
                logs_data.State.isin(["PART_STATE.PDT", "PART_STATE.IHPT", "PART_STATE.FIN"]))]
            logs_data['Time'] = (pd.Timestamp(demand_delivery_date) + pd.to_timedelta(logs_data['Time'].astype(float), unit='D'))
            gantt_chart_code = make_gantt(
                graph, makes + top_level_parts, group_states(logs_data))
            with gantt_tab:
                st.components.v1.html(gantt_chart_code, width=1400, height=1400, scrolling=True)
                # st.markdown(f"```html\n{gantt_chart_code}```")
    
    def most_freq_path_viz(df):
        # Sort the DataFrame by 'Count' column in descending order
        df = df.sort_values(by='Count', ascending=False)
        
        # Create a list of unique part names from all paths
        unique_parts = set(part for path in df['Critical_Path'] for part in path)
        
        # Create a dictionary to store edge counts
        edge_counts = {}
        
        # Iterate through the DataFrame and count edge occurrences
        for path, count in zip(df['Critical_Path'], df['Count']):
            for i in range(len(path) - 1):
                edge = (path[i], path[i + 1])
                if edge in edge_counts:
                    edge_counts[edge] += count
                else:
                    edge_counts[edge] = count
        
        # Generate Mermaid-compatible graph definition
        mermaid_graph = "graph TD;\n"
        for part in unique_parts:
            mermaid_graph += f"""{part}("{part}")\n"""
        
        for edge, count in edge_counts.items():
            mermaid_graph += f"""{edge[0]} -.->|"  {count}  "| {edge[1]}\n"""
        return mermaid_graph

    with path_tab:
        if hasattr(st.session_state, 'path_data'):
            critical_paths = st.session_state.path_data
            if critical_paths:
                critical_paths = [tuple(path) for path in critical_paths]
                path_dfs = pd.DataFrame(Counter(critical_paths).most_common(), columns=['Critical_Path', 'Count'])
                paths_display.dataframe(path_dfs)
                with mermaid_paths:
                    mr_code = most_freq_path_viz(path_dfs)
                    mermaid(mr_code, height=300)
                if path_tab.button("Redraw Flowmap", key=str(abs(hash(mermaid_paths)))):
                    with mermaid_paths:
                        mr_code = most_freq_path_viz(path_dfs)
                        mermaid(mr_code, height=1500)
                    