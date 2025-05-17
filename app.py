import glob
import streamlit as st
from PIL import Image
import base64
import json
from cv_information_extraction import cv_information_extraction as cv_ext
from streamlit_option_menu import option_menu
import streamlit as st
import subprocess
import streamlit as st
from ultralytics import YOLO
import zipfile
import rarfile
import os
import shutil
import yaml

rarfile.UNRAR_TOOL = r"C:\Program Files\WinRAR\UnRAR.exe"

cv_detection = cv_ext()

root_datasets = r"D:\PFE\detect and recognize\main\uploaded_dataset"


def config_yaml(yaml_path):
    try:
        # Ensure the YAML file exists
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"❌ The file {yaml_path} does not exist.")

        # Ensure the file is not read-only
        if not os.access(yaml_path, os.W_OK):
            os.chmod(yaml_path, 0o666)  # Make it readable and writable

        with open(yaml_path, "r") as file:
            data = yaml.safe_load(file)

        # Get the root directory of the YAML file
        root = os.path.dirname(yaml_path)
        data["test"] = os.path.join(root, "test")
        data["train"] = os.path.join(root, "train")
        data["val"] = os.path.join(root, "valid")

        # Save the updated YAML file
        with open(yaml_path, "w") as file:
            yaml.dump(data, file, default_flow_style=False)

        print("✅ data.yaml successfully updated:", data)
        return yaml_path

    except FileNotFoundError as ex:
        print(ex)
        return None
    except PermissionError as ex:
        print(f"❌ Permission Error: {ex}.")
        print("⚠️ Please ensure the file is not open in another program and you have full control over it.")
        return None
    except Exception as ex:
        print(f"❌ Unexpected Error: {ex}")
        return None


def extract_dataset(file):
    extraction_path = r"D:\\PFE\\detect and recognize\\main\\uploaded_dataset\\" + \
        os.path.splitext(file.name)[0]

    # Check if the extraction path already exists and contains data.yaml (indicating it is already extracted)
    if os.path.exists(extraction_path) and os.path.isfile(os.path.join(extraction_path, "data.yaml")):
        st.info("✅ Dataset already extracted.")
        return os.path.join(extraction_path, "data.yaml")

    # Always create the extraction directory (clear existing)
    if os.path.exists(extraction_path):
        shutil.rmtree(extraction_path)
    os.makedirs(extraction_path, exist_ok=True)

    st.info("Extracting dataset... Please wait.")
    file_ext = os.path.splitext(file.name)[1].lower()

    try:
        if file_ext == ".zip":
            with zipfile.ZipFile(file, 'r') as zip_ref:
                zip_ref.extractall(extraction_path)
            st.success("✅ ZIP file extracted successfully.")
        elif file_ext == ".rar":
            with rarfile.RarFile(file, 'r') as rar_ref:
                rar_ref.extractall(extraction_path)
            st.success("✅ RAR file extracted successfully.")
        else:
            st.error("❌ Unsupported file format. Please upload ZIP or RAR files.")
            return None
    except Exception as e:
        st.error(f"❌ Error during extraction: {e}")
        return None

    # Detect the root directory of the extracted files
    extracted_files = os.listdir(extraction_path)
    if len(extracted_files) == 1 and os.path.isdir(os.path.join(extraction_path, extracted_files[0])):
        root = os.path.join(extraction_path, extracted_files[0])
    else:
        root = extraction_path

    return config_yaml(os.path.join(root, "data.yaml"))


def getAvailable_datasets():
    global root_datasets
    # Get all directories in the specified path with full paths
    return [os.path.abspath(os.path.join(root_datasets, d))
            for d in os.listdir(root_datasets)
            if os.path.isdir(os.path.join(root_datasets, d))]


def list_all_trained_models(metrics=False):
    # Define the base path for training runs
    root_path = r'D:\PFE\runs\detect'
    # Check if the directory exists
    if not os.path.exists(root_path):
        # st.warning("⚠️ No training runs found.")
        return []

    # Find all best.pt files in each train directory
    best_models = glob.glob(os.path.join(root_path, "train*/weights/best.pt"))
    if metrics == True:
        # Get back one directory for each path
        return [os.path.dirname(os.path.dirname(path)) for path in best_models]

    if not best_models:
        # st.warning("⚠️ No trained models (best.pt) found.")
        return []

    return best_models


# ---------------------- Run Extraction Method ----------------------


def Run_Extraction(uploaded_files, show_boxes=True, filter=False, target_classes=None, search_query=None, selected_model="Default Model"):
    image_inputs = []
    valid_files = []

    for file in uploaded_files:
        if file.type.startswith("image"):
            try:
                image = Image.open(file).convert("RGB")
                image_inputs.append(image)
                valid_files.append(file)
            except Exception as e:
                st.warning(f"Failed to read {file.name}: {e}")
        else:
            st.warning(f"{file.name} is not an image. Skipping.")

    if not image_inputs:
        st.info("No valid images uploaded.")
        return
    if selected_model == "Default Model":
        results = cv_detection.detect_and_ocr_batch(
            image_inputs)
    else:
        results = cv_detection.detect_and_ocr_batch(
            image_inputs, selected_model=selected_model)\

    st.session_state.extraction_result = {
        "results": results,
        "files": [f.name for f in valid_files],
        "images": image_inputs
    }

    Display_Extraction(results, valid_files, image_inputs,
                       show_boxes, filter, target_classes, search_query)

# ---------------------- Display Extraction Results ----------------------


def Display_Extraction(results, valid_files, image_inputs, show_boxes, filter=False, target_classes=None, search_query=None):
    for idx, (json_info, image_det, personal_image) in enumerate(results):
        file = valid_files[idx]

        if filter:
            matched = False
            if target_classes and search_query and search_query.strip() != "":
                for target_class in target_classes:
                    if target_class in json_info:
                        value = json_info[target_class]
                        if search_query.lower() in value.lower():
                            matched = True
                            break
                if not matched:
                    continue

        st.markdown("---")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader(f"📌 Document {idx+1}")
            st.write(f"Filename: `{file.name}`")
            st.image(
                image_inputs[idx], caption="Original Document", use_container_width=True)

            if show_boxes:
                st.image(image_det, caption="📦 Layout Detection Preview",
                         use_container_width=True)

            if personal_image is not None:
                st.image(personal_image, caption="🖼️ Personal Image",
                         use_container_width=True)

        with col2:
            st.subheader("🧠 Extracted Content")
            extracted_text = json_info.get('Name', 'No name extracted')
            st.markdown(extracted_text, unsafe_allow_html=True)

            if st.session_state.enable_download:
                with st.expander("🧾 Preview Extracted CV (Structured Format)", expanded=True):
                    st.json(json_info)

                def create_download_link(data_dict, filename):
                    json_str = json.dumps(data_dict, indent=2)
                    b64 = base64.b64encode(json_str.encode()).decode()
                    href = f'<a href="data:file/json;base64,{b64}" download="{filename}_extracted.json">📥 Download Extracted JSON</a>'
                    return href

                st.markdown(create_download_link(
                    json_info, file.name), unsafe_allow_html=True)


# ---------------------- APP CONFIG ----------------------
st.set_page_config(page_title="Document Information Extractor", layout="wide")
st.title("📄 Information Extraction from Scanned Documents 'CV'")
st.caption(
    "Built with layout analysis (YOLOv11), OCR (Tesseract), and NLP — UI by Streamlit")
# ---------------------- SIDEBAR ----------------------
with st.sidebar:
    page = option_menu(
        menu_title="Navigation",
        options=["Upload & Extract", "View Uploaded Images", "Model Training"],
        icons=["cloud-upload", "image"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px"},
            "icon": {"color": "#4285f4", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "2px", "--hover-color": "#eee"},
        },
    )

# ---------------------- SESSION STATE ----------------------
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "enable_download" not in st.session_state:
    st.session_state.enable_download = True
if "extraction_result" not in st.session_state:
    st.session_state.extraction_result = None

# ---------------------- PAGE 1: Upload & Extract ----------------------
if page == "Upload & Extract":
    show_boxes = st.sidebar.checkbox("Show Detected Layout Boxes", value=True)
    filter_cv = st.sidebar.checkbox("Filter Resumes (CV)", value=False)
    st.session_state.enable_download = st.sidebar.checkbox(
        "Enable Export", value=True)

    default_cv_sections = [
        "Name", "Languages", "Resume", "Skills", "Profil", "Experience",
        "Interests", "Education", "Certifications", "Extracurricular", "Contact", "Projects"
    ]

    if "selected_sections" not in st.session_state:
        st.session_state.selected_sections = default_cv_sections[3:4]

    if filter_cv:
        st.sidebar.markdown("### 🎯 Resume Filter Settings")
        with st.sidebar.expander("🔎 Custom Resume Filter", expanded=True):
            st.caption(
                "Type a keyword and select the CV sections you want to filter by:")
            search_query = st.text_input("Keyword", "")
            col1, col2, col3 = st.columns(3)
            if col1.button("✅ Select All"):
                st.session_state.selected_sections = default_cv_sections.copy()
            if col2.button("❌ Deselect All"):
                st.session_state.selected_sections = []
            if col3.button("🚀 Run filter"):
                st.session_state.trigger_filter = True
            else:
                st.session_state.trigger_filter = False

            selected_sections = []
            for section in default_cv_sections:
                checked = section in st.session_state.selected_sections
                new_checked = st.checkbox(section, value=checked, key=section)
                if new_checked:
                    selected_sections.append(section)
            st.session_state.selected_sections = selected_sections
    else:
        search_query = ""
        selected_sections = default_cv_sections.copy()

    uploaded_files = st.file_uploader("📤 Upload one or more scanned documents (Image or PDF)",
                                      type=["jpg", "jpeg", "png", "pdf"],
                                      accept_multiple_files=True)

    existing_files = {f.name: f for f in st.session_state.uploaded_files}
    new_files = {f.name: f for f in uploaded_files}
    combined_files = {**existing_files, **new_files}
    st.session_state.uploaded_files = list(combined_files.values())
    uploaded_files = st.session_state.uploaded_files

    # 📌 Displaying Available Trained Models
    st.subheader("📌 Available Trained Models (best.pt)")
    trained_models = ["Default Model"]
    trained_models += list_all_trained_models()

    if trained_models:
        selected_model = st.selectbox(
            "Select a model to use:", trained_models)
        st.success(f"✅ Selected Model: {selected_model}")
    else:
        st.warning("⚠️ No trained models found.")

    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) uploaded.")
        if filter_cv and st.session_state.trigger_filter:
            Run_Extraction(uploaded_files, show_boxes, filter=True,
                           target_classes=st.session_state.selected_sections,
                           search_query=search_query)
        elif st.button("🚀 Run Extraction"):
            Run_Extraction(uploaded_files, show_boxes,
                           selected_model=selected_model)

        # If results already exist in session, display them
        elif st.session_state.extraction_result:
            prev = st.session_state.extraction_result
            files_map = {f.name: f for f in uploaded_files}
            valid_files = [files_map[name]
                           for name in prev["files"] if name in files_map]
            Display_Extraction(
                prev["results"], valid_files, prev["images"], show_boxes)
    else:
        st.info("⬆️ Please upload one or more scanned documents to begin.")

# ---------------------- PAGE 2: View Uploaded Images ----------------------
elif page == "View Uploaded Images":
    st.title("🖼️ Uploaded Document Previews")
    if st.session_state.uploaded_files:
        uploaded_files = st.session_state.uploaded_files
        rows = (len(uploaded_files) + 3) // 4
        for i in range(rows):
            cols = st.columns(4)
            for j in range(4):
                idx = i * 4 + j
                if idx < len(uploaded_files):
                    file = uploaded_files[idx]
                    with cols[j]:
                        if file.type.startswith("image"):
                            image = Image.open(file)
                            st.image(image, use_container_width=True)
                        else:
                            st.info("📄 PDF file")
    else:
        st.warning("No files uploaded yet. Go to the Upload page first.")


elif page == "Model Training":
    st.header("🚀 Upload Your YOLO Dataset (ZIP or RAR)")

    with st.expander("📂 Dataset Structure (YOLO Format)", expanded=False):
        st.markdown(
            '''
            ```json
            {
                "train": {
                    "images": "path/to/train/images/",
                    "labels": "path/to/train/labels/"
                },
                "val": {
                    "images": "path/to/val/images/",
                    "labels": "path/to/val/labels/"
                },
                "test": {
                    "images": "path/to/test/images/",
                    "labels": "path/to/test/labels/"
                },
                "data.yaml": "Configuration file (classes, paths)"
            }
            ```

            ### ⚡ Instructions:
            - Upload a single ZIP or RAR file.
            - The `data.yaml` must correctly define paths to `train`, `val`, and `test`.
            - Ensure your images and labels follow YOLO format.
            ''')

    dataset_file = st.file_uploader("Upload YOLO Dataset (ZIP or RAR)", type=[
                                    "zip", "rar"], accept_multiple_files=False)

    if dataset_file:
        st.info(f"Uploaded file: {dataset_file.name}")
        yaml_path = extract_dataset(dataset_file)

        st.success("Dataset extracted successfully!")

        st.subheader("Training Settings")
        # img_size = st.number_input(
        #     "Image Size", min_value=256, max_value=1024, value=640)
        epochs = st.number_input(
            "Number of Epochs", min_value=1, max_value=100, value=2)
        batch_size = st.number_input(
            "Batch Size", min_value=1, max_value=100, value=16)

        start_training = st.button("🚀 Start Training")

        if start_training:
            model = YOLO("yolo11n.pt")
            model.train(data=yaml_path, imgsz=640,
                        epochs=epochs, batch=batch_size)
            st.success("Training Completed!")
    else:
        datasets = getAvailable_datasets()
        if datasets:
            selected_dataset = st.selectbox(
                "Select a dataset to train:", datasets)
            st.success(f"✅ Selected Model: {selected_dataset}")
            st.subheader("Training Settings")
            # img_size = st.number_input(
            #     "Image Size", min_value=256, max_value=1024, value=640)
            epochs = st.number_input(
                "Number of Epochs", min_value=1, max_value=100, value=2)
            batch_size = st.number_input(
                "Batch Size", min_value=1, max_value=100, value=16)

            start_training = st.button("🚀 Start Training")
            config_yaml(os.path.join(os.path.join(root_datasets,
                        selected_dataset), "data.yaml"))
            if start_training:
                model = YOLO("yolo11n.pt")
                model.train(data=selected_dataset+"\\data.yaml", imgsz=640,
                            epochs=epochs, batch=batch_size)
                st.success("Training Completed!")

    # Display final performance (Multiple images in latest training run)
    metrics = list_all_trained_models()
    # 📌 Displaying Available Trained Models
    st.subheader("📌 Available Trained Models")
    trained_models = list_all_trained_models(metrics=True)

    if trained_models:
        selected_model = st.selectbox(
            "Select a model to use:", trained_models)
        st.success(f"✅ Selected Model: {selected_model}")
        st.subheader("📊 Training Performance")
        result_images = glob.glob(os.path.join(selected_model, "*.png"))

        if result_images:
            for img_path in result_images:
                st.image(
                    img_path, caption=f"📊 {os.path.basename(img_path)}")
        else:
            st.warning(
                "⚠️ No results images found in the latest training run.")
    else:
        st.warning("⚠️ No trained models found.")


# streamlit run "D:\PFE\detect and recognize\main\app.py" --server.maxUploadSize 1024
