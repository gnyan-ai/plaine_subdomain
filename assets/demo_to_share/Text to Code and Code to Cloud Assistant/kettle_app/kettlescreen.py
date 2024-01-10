import pandas as pd
import streamlit as st
from streamlit_tree_select import tree_select
import json
import os
import time
from pygments import lexers
from pygments.util import ClassNotFound
from kettleutils import generate_repeatable_id, invoke_chain, preseeded_questions_list
import threading

# Make this function async


def run_full_chain_async(root_dir, query):
    try:
        invoke_chain(query)
    except Exception as e:
        print(f"Error in full_chain for {query}: {e}")


def guess_language_from_filename(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            # Try to guess the lexer based on file name
            lexer = lexers.get_lexer_for_filename(file_path, code=content)
            return lexer.name
    except ClassNotFound:
        # Default to plain text if no lexer is found
        return "Text"
    except FileNotFoundError:
        # Handle file not found error
        return "File not found"


# Make wide and white theme only
st.set_page_config(layout="wide", page_title="kettle.gnyan.ai",
                   initial_sidebar_state="expanded",)

scol1, scol2 = st.sidebar.columns([5, 10])
with scol1:
    scol1.image("http://192.168.27.10:8000/static/images/logo.svg", width=75)
with scol2:
    scol2.subheader("Welcome to Kettle!\nTurn your text into code", divider=False)
st.sidebar.markdown("---")

# Render the HTML with unsafe_allow_html set to True
st.title("Kettle :keyboard:")

st.subheader("What do you wish to build today?", divider=True)

if 'user_input' not in st.session_state:
    st.session_state['user_input'] = "Create a streamlit folium geo heatmap to show car rentals, reservations, fleet levels cluster markers in the US. Add date filters and state filters in the sidebar."

if 'init_done' not in st.session_state or not st.session_state.init_done:
    st.session_state['init_done'] = False
    st.session_state['folder_path'] = None

print(st.session_state)
# Create a text area to enter the query
user_input = st.text_area("", st.session_state['user_input'], height=125)

# Build some shortcut buttons in three columns -- preferable a flex flow layout -- to auto-populate the text area from preseeded questions
buttons, submit = st.columns([10, 1])

with buttons:
    button_col1, button_col2, button_col3, button_col4 = buttons.columns(4)
    button_shortcuts = list(preseeded_questions_list.keys())
    with button_col1:
        if st.button(button_shortcuts[0], type="secondary"):
            st.session_state['user_input'] = preseeded_questions_list[button_shortcuts[0]]
            st.rerun()
        if st.button(button_shortcuts[1], type="secondary"):
            st.session_state['user_input'] = preseeded_questions_list[button_shortcuts[1]]
            st.rerun()
    with button_col2:
        if st.button(button_shortcuts[2], type="secondary"):
            st.session_state['user_input'] = preseeded_questions_list[button_shortcuts[2]]
            st.rerun()
        if st.button(button_shortcuts[3], type="secondary"):
            st.session_state['user_input'] = preseeded_questions_list[button_shortcuts[3]]
            st.rerun()
    with button_col3:
        if st.button(button_shortcuts[4], type="secondary"):
            st.session_state['user_input'] = preseeded_questions_list[button_shortcuts[4]]
            st.rerun()
        if st.button(button_shortcuts[5], type="secondary"):
            st.session_state['user_input'] = preseeded_questions_list[button_shortcuts[5]]
            st.rerun()
    with button_col4:
        if st.button(button_shortcuts[6], type="secondary"):
            st.session_state['user_input'] = preseeded_questions_list[button_shortcuts[6]]
            st.rerun()
        if st.button(button_shortcuts[7], type="secondary"):
            st.session_state['user_input'] = preseeded_questions_list[button_shortcuts[7]]
            st.rerun()

with submit:
    if st.button("Submit", type="primary"):
        user_input = user_input
        # Generate a unique directory name based on the query
        root_dir = generate_repeatable_id(user_input)
        if not os.path.exists(root_dir):
            os.makedirs(root_dir, exist_ok=True)
            thread = threading.Thread(
                target=run_full_chain_async, args=(root_dir, user_input))
            thread.start()
        # Initialize the folder path if not already done
        folder_name = root_dir
        if folder_name:
            original_folder_path = st.session_state.get('folder_path')
            st.session_state['folder_path'] = folder_name
            st.session_state['init_done'] = True
            if folder_name != original_folder_path:
                st.rerun()


def is_folder_modified_recently(folder_path, minutes=10):
    """Check if the folder was modified in the last 'minutes' minutes."""
    current_time = time.time()
    last_modified_time = os.path.getmtime(folder_path)
    return not os.path.isfile(f"{folder_path}/.success") and (current_time - last_modified_time) < (minutes * 60)


def folder_structure(path, files=[]):
    result = {"label": os.path.basename(path), "value": path, "children": []}
    try:
        items = os.listdir(path)
    except OSError:
        items = []

    for item in items:
        item_path = os.path.join(path, item)

        if os.path.isdir(item_path):
            result["children"].append(folder_structure(item_path, files)[0])
        elif os.path.isfile(item_path) and os.path.getsize(item_path) > 0:
            result["children"].append({"label": item, "value": item_path})
            # Check file size is greater than 0
            files.append(item_path)

    return result, files


@st.cache_data(show_spinner=True)  # Cache the file reading
def read_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read(100*1024)  # Read only first 4096 characters
    except UnicodeDecodeError:  # Handle non-textual file
        return None


# Use the initialized folder path or a default if not available
folder_path = st.session_state.get('folder_path')
if folder_path:
    with st.sidebar:
        if is_folder_modified_recently(folder_path):
            if st.button('Refresh Folder Tree'):
                st.rerun()
        folder_structure_data, files_data = folder_structure(folder_path)
        return_select = tree_select([folder_structure_data], show_expand_all=True,
                                    only_leaf_checkboxes=False, checked=files_data[:48], key='tree_select')
        # Add return select to session state
        st.session_state['return_select'] = return_select

    # Read return_select from session state
    return_select = st.session_state.get('return_select', None)
    if return_select and return_select['checked']:
        st.subheader("Text to Code", divider=True)
        for file_selected in [item_path for item_path in sorted(return_select['checked']) if os.path.isfile(item_path) and os.path.getsize(item_path) > 0]:
            file_content = read_file(file_selected)

            if file_content:
                # Display the contents of each file in an expander
                # Add file icon/emoji to streamlit
                with st.expander(f"**{file_selected}**", expanded=True):
                    language = guess_language_from_filename(
                        file_selected).lower()
                    st.markdown(f"```{language}\n{file_content}\n```")

    if return_select and return_select['checked']:
        if st.sidebar.button("Commit to Git"):
            progress_text = st.sidebar.empty()  # Placeholder for text
            progress_text.write("Committing to Git.... Please wait...")
            my_bar = st.sidebar.progress(0)
            for percent_complete in range(100):
                time.sleep(0.03)  # Simulate a task
                my_bar.progress(percent_complete + 1, text=f"{percent_complete + 1}%")
            progress_text.write("Commit completed!")
            time.sleep(1)
            my_bar.empty()
            progress_text.empty()
