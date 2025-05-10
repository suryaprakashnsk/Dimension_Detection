import streamlit as st
import torch
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError
import os
import subprocess # For Detectron2 installation check
import sys # For Detectron2 installation check

# --- Detectron2 Imports (handle potential import errors) ---
d2_imported_successfully = False
try:
    import detectron2
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.utils.visualizer import Visualizer, ColorMode
    from detectron2.data import MetadataCatalog
    from detectron2.structures import Boxes # For Bounding Boxes
    d2_imported_successfully = True
    print("Detectron2 utilities imported successfully.")
except ImportError:
    st.error("Detectron2 not found or not installed correctly. Please ensure it's installed in your environment.")
    print("❌ Failed to import Detectron2 utilities.")
except Exception as e:
    st.error(f"An error occurred during Detectron2 imports: {e}")
    print(f"❌ An error occurred during Detectron2 imports: {e}")


# --- PyTorch Model Imports ---
from torchvision import models as torchvision_models
import torch.nn as nn

# ------------------------------
# Configuration
# ------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CNN_INPUT_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# Ensure this path is correct for your environment
MODEL_PATH = r"pix3d_dimension_estimator_mask_crop.pth"
OUTPUT_DIR = 'streamlit_d2_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------
# Dimension Estimation CNN
# ------------------------------
def create_dimension_estimator_cnn_for_inference(num_outputs=4):
    model = torchvision_models.resnet50(weights=None) # Load architecture only
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_outputs) # Outputs L, W, H, V
    )
    return model

@st.cache_resource
def load_dimension_model():
    model = None
    if not os.path.exists(MODEL_PATH):
        st.error(f"Dimension estimation model not found at {MODEL_PATH}. Please check the path.")
        return None
    try:
        model = create_dimension_estimator_cnn_for_inference()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        print(f"Dimension estimation model loaded from {MODEL_PATH}")
    except Exception as e:
        st.error(f"Error loading dimension estimation model: {e}")
        return None
    return model

# ------------------------------
# Detectron2 Model
# ------------------------------
@st.cache_resource
def load_detectron2_model():
    if not d2_imported_successfully:
        return None, None
    try:
        cfg = get_cfg()
        # Example: Mask R-CNN for instance segmentation
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set threshold for detection
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        if not torch.cuda.is_available():
            cfg.MODEL.DEVICE = "cpu"
        else:
            cfg.MODEL.DEVICE = "cuda" # Explicitly set
        predictor = DefaultPredictor(cfg)
        print("Detectron2 predictor created.")
        return predictor, cfg
    except Exception as e:
        st.error(f"Error loading Detectron2 model: {e}")
        return None, None

# ------------------------------
# Helper Functions
# ------------------------------
def get_largest_instance_index(instances):
    """Returns the index of the largest instance based on mask area or box area."""
    if not len(instances):
        return -1 # No instances

    if instances.has("pred_masks"):
        areas = instances.pred_masks.sum(dim=(1,2)) # Sum of True pixels in boolean mask
        if len(areas) > 0:
            return areas.argmax().item()
    elif instances.has("pred_boxes"):
        boxes_tensor = instances.pred_boxes.tensor
        areas = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
        if len(areas) > 0:
            return areas.argmax().item()
    return 0 # Default to first instance if area calculation fails or no masks/boxes

def crop_from_mask(image_np_rgb, mask_tensor):
    """Crops an object from an image using a boolean mask tensor."""
    mask_np = mask_tensor.cpu().numpy().astype(np.uint8) # Ensure mask is on CPU and uint8
    if mask_np.sum() == 0: return None # Empty mask

    rows = np.any(mask_np, axis=1)
    cols = np.any(mask_np, axis=0)
    if not np.any(rows) or not np.any(cols): return None

    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    # Add padding to the bounding box from mask
    padding = 5
    ymin = max(0, ymin - padding)
    xmin = max(0, xmin - padding)
    ymax = min(image_np_rgb.shape[0] - 1, ymax + padding)
    xmax = min(image_np_rgb.shape[1] - 1, xmax + padding)


    if ymin >= ymax or xmin >= xmax : return None
    cropped_image = image_np_rgb[ymin:ymax+1, xmin:xmax+1, :]
    return cropped_image

def predict_dimensions_cnn(image_patch_np_rgb, model):
    """Predicts dimensions using the custom CNN."""
    if model is None:
        return {"L": "N/A", "W": "N/A", "H": "N/A", "V": "N/A", "Note": "DimCNN not loaded"}
    try:
        if image_patch_np_rgb.dtype != np.uint8:
            image_patch_np_rgb = image_patch_np_rgb.astype(np.uint8)

        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((CNN_INPUT_SIZE, CNN_INPUT_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        input_tensor = transform(image_patch_np_rgb).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = model(input_tensor)
        dims = pred.squeeze().cpu().tolist()

        if not isinstance(dims, list): dims = [dims]
        while len(dims) < 4: dims.append(0.0) # Pad if model outputs fewer

        # Assuming model was trained to output in meters, convert to cm for display
        L_cm = dims[0] * 100
        W_cm = dims[1] * 100
        H_cm = dims[2] * 100
        V_cm3 = dims[3] * 1_000_000 # Convert m^3 to cm^3

        return {
            "Length (cm)": f"{L_cm:.1f}",
            "Width (cm)": f"{W_cm:.1f}",
            "Height (cm)": f"{H_cm:.1f}",
            "Volume (cm³)": f"{V_cm3:.1f}",
            "Note": "CustomCNN (Pix3D Scale)"
        }
    except Exception as e:
        print(f"Error in predict_dimensions_cnn: {e}")
        return {"L": "N/A", "W": "N/A", "H": "N/A", "V": "N/A", "Note": "CNN Predict Error"}

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(layout="wide", page_title="Object Dimension Estimator")
st.title("📦 Object Dimension & Volume Estimation")
st.write("Upload an image. The system will detect objects using Detectron2, draw bounding boxes and masks, and estimate dimensions for the largest detected object using a custom-trained CNN.")

# Load models
dim_model = load_dimension_model()
if d2_imported_successfully:
    d2_predictor, d2_cfg = load_detectron2_model()
    if d2_cfg is not None:
        # Attempt to get metadata, handle potential KeyErrors
        try:
            d2_metadata = MetadataCatalog.get(d2_cfg.DATASETS.TRAIN[0] if d2_cfg.DATASETS.TRAIN else "coco_2017_val")
        except KeyError:
            st.warning("Default COCO metadata not found. Trying 'coco_2017_train'. Class names might be generic if this also fails.")
            try:
                d2_metadata = MetadataCatalog.get("coco_2017_train")
            except KeyError:
                st.warning("Could not load standard COCO metadata. Using dummy metadata.")
                dummy_name = "streamlit_dummy_coco_dataset_main"
                if dummy_name not in MetadataCatalog.list():
                    MetadataCatalog.get(dummy_name).thing_classes = [f"class_{i}" for i in range(80)] # COCO has 80 classes
                d2_metadata = MetadataCatalog.get(dummy_name)
    else:
        d2_metadata = None # Set to None if cfg is None
else:
    d2_predictor = None
    d2_cfg = None
    d2_metadata = None


uploaded_file = st.file_uploader("Upload a single image (JPG/PNG)", accept_multiple_files=False, type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    st.subheader(f"🖼️ Processing: {uploaded_file.name}")
    try:
        image_pil = Image.open(uploaded_file).convert("RGB")
        image_np_rgb = np.array(image_pil) # Convert PIL to NumPy RGB
        image_bgr = cv2.cvtColor(image_np_rgb, cv2.COLOR_RGB2BGR) # OpenCV BGR for Detectron2
    except UnidentifiedImageError:
        st.error("Cannot identify image file. Please upload a valid image.")
        image_bgr = None
    except Exception as e:
        st.error(f"Error loading image: {e}")
        image_bgr = None

    if image_bgr is not None:
        st.image(image_pil, caption="Uploaded Image", use_container_width=True)

        if d2_predictor is None or dim_model is None:
            st.error("One or more models (Detectron2, Dimension CNN) failed to load. Cannot process.")
        else:
            with st.spinner("Detecting objects and estimating dimensions..."):
                # --- Detectron2 Inference ---
                outputs = d2_predictor(image_bgr) # Detectron2 expects BGR
                instances = outputs["instances"].to("cpu")

                if len(instances) == 0:
                    st.warning("No objects detected by Detectron2.")
                else:
                    # --- Visualization with Bounding Boxes and Masks ---
                    # Create a copy for drawing Detectron2's full visualization
                    viz_image_bgr = image_bgr.copy()
                    v = Visualizer(viz_image_bgr[:, :, ::-1], metadata=d2_metadata, scale=0.8, instance_mode=ColorMode.IMAGE_BW)
                    out_vis = v.draw_instance_predictions(instances)
                    annotated_img_d2_rgb = out_vis.get_image()[:, :, ::-1] # Visualizer gives RGB

                    st.image(annotated_img_d2_rgb, caption="Detectron2 Detections (Masks & Boxes)", use_container_width=True)

                    # --- Process the largest detected instance for dimension estimation ---
                    largest_idx = get_largest_instance_index(instances)
                    if largest_idx != -1:
                        instance = instances[largest_idx]

                        class_name = "Unknown"
                        if instance.has("pred_classes") and d2_metadata and hasattr(d2_metadata, 'thing_classes'):
                            class_id = instance.pred_classes.item()
                            if class_id < len(d2_metadata.thing_classes):
                                class_name = d2_metadata.thing_classes[class_id]
                        score = instance.scores.item() if instance.has("scores") else 0.0
                        st.write(f"**Processing largest detected object:** {class_name} (Confidence: {score:.2f})")

                        # --- Crop from Mask for Custom CNN ---
                        if instance.has("pred_masks"):
                            mask_tensor = instance.pred_masks[0] # Get the mask for the largest instance
                            # Crop from the RGB numpy array
                            object_crop_rgb = crop_from_mask(image_np_rgb, mask_tensor)

                            if object_crop_rgb is not None and object_crop_rgb.shape[0] > 0 and object_crop_rgb.shape[1] > 0:
                                st.image(object_crop_rgb, caption="Cropped Object Patch for Dimension CNN", width=250) # Smaller display

                                # --- Predict Dimensions with Custom CNN ---
                                dims = predict_dimensions_cnn(object_crop_rgb, dim_model)
                                st.write("📏 **Predicted Dimensions (from Custom CNN):**")
                                st.json(dims)
                            else:
                                st.error("Could not crop a valid object patch from the mask.")
                        else:
                            st.warning("No segmentation mask found for the largest instance. Cannot estimate dimensions with custom CNN.")
                    else:
                        st.info("Could not determine the largest object to process for dimensions.")
    else:
        if uploaded_file: # If a file was uploaded but image_bgr is None
             st.error("Image could not be loaded for processing.")


# --- Status Footer ---
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ System Status")
st.sidebar.markdown(f"**Processing Device:** `{DEVICE}`")
st.sidebar.markdown(f"**Detectron2 Predictor:** `{'Loaded' if d2_predictor else 'Not Loaded'}`")
st.sidebar.markdown(f"**Dimension CNN:** `{'Loaded' if dim_model else 'Not Loaded'}`")
if not os.path.exists(MODEL_PATH):
    st.sidebar.warning(f"Dimension CNN weights file not found at the specified path.")
if __name__ == "__main__":
    main()  # or whatever function runs the Streamlit app


